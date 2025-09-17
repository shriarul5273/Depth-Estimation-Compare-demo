import glob
import gradio as gr
import matplotlib
import numpy as np
from PIL import Image
import torch
import tempfile
import os
import logging
import gc
from typing import Optional, Tuple
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from depth_anything_v2.dpt import DepthAnythingV2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
"""

# Device detection with fallback
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
logging.info(f"Using device: {DEVICE}")

# Model configurations for Depth Anything V2
MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

# Available model variants with display names
MODEL_VARIANTS = {
    'vits': {
        'display_name': 'ViT-Small (Fastest, Lower Quality)',
        'checkpoint': 'checkpoints/depth_anything_v2_vits.pth'
    },
    'vitb': {
        'display_name': 'ViT-Base (Balanced Speed/Quality)',
        'checkpoint': 'checkpoints/depth_anything_v2_vitb.pth'
    },
    'vitl': {
        'display_name': 'ViT-Large (High Quality, Recommended)',
        'checkpoint': 'checkpoints/depth_anything_v2_vitl.pth'
    },
    'vitg': {
        'display_name': 'ViT-Giant (Highest Quality, Slowest)',
        'checkpoint': 'checkpoints/depth_anything_v2_vitg.pth'
    }
}

# Global variables for model caching
_cached_model = None
_cached_device = None
_cached_model_selection = None

def check_gpu_memory():
    """Check and log current GPU memory usage"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logging.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB, Total: {total:.2f}GB")
            return allocated, reserved, max_allocated, total
    except RuntimeError as e:
        logging.warning(f"Failed to get GPU memory info: {e}")
    return None, None, None, None

def get_paginated_examples(examples: list, page: int = 0, per_page: int = 8) -> tuple:
    """Get paginated examples with navigation info"""
    total_pages = (len(examples) - 1) // per_page + 1 if examples else 0
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, len(examples))
    
    current_examples = examples[start_idx:end_idx]
    has_prev = page > 0
    has_next = page < total_pages - 1
    
    return current_examples, total_pages, has_prev, has_next

def aggressive_cleanup():
    """Perform aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Performed memory cleanup")

def get_available_models() -> dict:
    """Get all available models with their display names"""
    available_models = {}
    
    # All models are available since we can download them from HF Hub
    for variant, info in MODEL_VARIANTS.items():
        available_models[info['display_name']] = {
            'variant': variant,
            'checkpoint': info['checkpoint'],  # Keep for backwards compatibility
            'config': MODEL_CONFIGS[variant]
        }
        logging.info(f"Available model: {info['display_name']} (variant: {variant})")
    
    return available_models

def get_model_from_selection(model_selection: str) -> Tuple[str, dict]:
    """Get model variant and config from selection"""
    available_models = get_available_models()
    
    if model_selection in available_models:
        model_info = available_models[model_selection]
        return model_info['variant'], model_info['config'], model_info['checkpoint']
    
    # Fallback to default if selection not found
    logging.warning(f"Model selection '{model_selection}' not found, using default")
    return 'vitl', MODEL_CONFIGS['vitl'], 'checkpoints/depth_anything_v2_vitl.pth'

def load_model(model_selection: str) -> Tuple[DepthAnythingV2, str]:
    """Load and cache the selected model"""
    global _cached_model, _cached_device, _cached_model_selection
    
    # Check if we already have the right model cached
    if (_cached_model is not None and 
        _cached_model_selection == model_selection and 
        _cached_device == DEVICE):
        logging.info(f"Using cached model: {model_selection}")
        return _cached_model, _cached_device
    
    # Clear previous model if any
    if _cached_model is not None:
        logging.info("Clearing previous model from cache...")
        del _cached_model
        _cached_model = None
        aggressive_cleanup()
    
    try:
        # Get model info
        variant, config, checkpoint_path = get_model_from_selection(model_selection)
        
        logging.info(f"Loading model: {model_selection} (variant: {variant})")
        
        # Download model from Hugging Face Hub if not already cached locally
        try:
            # Map variant to model names used in HF Hub
            model_name_mapping = {
                'vits': 'Small',
                'vitb': 'Base', 
                'vitl': 'Large',
                'vitg': 'Giant'
            }
            
            model_name = model_name_mapping.get(variant, 'Large')  # Default to Large
            filename = f"depth_anything_v2_{variant}.pth"
            
            # Try to download from HF Hub first
            try:
                filepath = hf_hub_download(
                    repo_id=f"depth-anything/Depth-Anything-V2-{model_name}", 
                    filename=filename, 
                    repo_type="model"
                )
                logging.info(f"Downloaded model from HF Hub: {filepath}")
                checkpoint_path = filepath
            except Exception as e:
                logging.warning(f"Failed to download from HF Hub: {e}")
                # Fallback to local checkpoint if it exists
                if not os.path.exists(checkpoint_path):
                    raise FileNotFoundError(f"Neither HF Hub download nor local checkpoint available: {checkpoint_path}")
                logging.info(f"Using local checkpoint: {checkpoint_path}")
        
        except Exception as e:
            logging.error(f"Error in model download/loading: {e}")
            raise
        
        # Create model
        model = DepthAnythingV2(**config)
        
        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        # Move to device and set to eval mode
        model = model.to(DEVICE).eval()
        
        # Cache the model
        _cached_model = model
        _cached_device = DEVICE
        _cached_model_selection = model_selection
        
        logging.info(f"✅ Model loaded successfully: {model_selection}")
        check_gpu_memory()
        
        return model, DEVICE
        
    except Exception as e:
        logging.error(f"Failed to load model {model_selection}: {e}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

def predict_depth(image: np.ndarray, model_selection: str) -> np.ndarray:
    """Predict depth using the selected model"""
    try:
        # Load model (uses cache if available)
        model, device = load_model(model_selection)
        
        # Predict depth
        depth = model.infer_image(image[:, :, ::-1])  # BGR to RGB conversion
        
        return depth
        
    except Exception as e:
        logging.error(f"Depth prediction failed: {e}")
        raise

def process_image(model_selection: str, image: np.ndarray, 
                 progress: gr.Progress = gr.Progress()) -> Tuple[Optional[tuple], Optional[str], Optional[str], str]:
    """
    Main processing function for depth estimation
    """
    if image is None:
        return None, None, None, "❌ Please upload an image."
    
    try:
        progress(0.1, desc=f"Loading model ({model_selection})...")
        
        # Get model info for status
        variant, _, _ = get_model_from_selection(model_selection)
        
        progress(0.3, desc="Running depth inference...")
        
        # Make a copy of the original image
        original_image = image.copy()
        h, w = image.shape[:2]
        
        # Predict depth
        depth = predict_depth(image, model_selection)
        
        progress(0.7, desc="Creating visualizations...")
        
        # Create raw depth file
        raw_depth = Image.fromarray(depth.astype('uint16'))
        tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp_raw_depth.name)
        
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        colored_depth = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
        
        # Create grayscale depth file
        gray_depth = Image.fromarray(depth_normalized)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        gray_depth.save(tmp_gray_depth.name)
        
        progress(1.0, desc="Complete!")
        
        # Create slider output
        slider_output = (original_image, colored_depth)
        
        # Create status message
        min_depth = depth.min()
        max_depth = depth.max()
        mean_depth = depth.mean()
        
        # Get memory info
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated(0) / 1024**3
            max_memory = torch.cuda.max_memory_allocated(0) / 1024**3
            memory_info = f" | GPU: {current_memory:.2f}GB/{max_memory:.2f}GB peak"
        else:
            memory_info = " | CPU processing"
        
        status = f"""✅ Processing successful!
🔧 Model: {variant.upper()}{memory_info}
📊 Depth Statistics:
   • Range: {min_depth:.2f} - {max_depth:.2f}
   • Mean: {mean_depth:.2f}
   • Input size: {w}×{h}
   • Device: {DEVICE}"""
        
        return slider_output, tmp_gray_depth.name, tmp_raw_depth.name, status
        
    except Exception as e:
        logging.error(f"Image processing failed: {e}")
        # Clean up on error
        aggressive_cleanup()
        return None, None, None, f"❌ Error: {str(e)}"

def create_app() -> gr.Blocks:
    """Create the Gradio application"""
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        logging.warning("No model checkpoints found!")
        # Add dummy entries for interface
        available_models = {
            "No models found - please check checkpoints folder": {
                'variant': 'vitl',
                'checkpoint': 'checkpoints/depth_anything_v2_vitl.pth',
                'config': MODEL_CONFIGS['vitl']
            }
        }
    
    model_choices = list(available_models.keys())
    default_model = model_choices[0]
    
    # Try to find ViT-Large as default if available
    for choice in model_choices:
        if "ViT-Large" in choice:
            default_model = choice
            break
    
    title = "# Depth Anything V2 - Enhanced"
    description = """Enhanced demo for **Depth Anything V2** with model selection.
Please refer to the [paper](https://arxiv.org/abs/2406.09414), [project page](https://depth-anything-v2.github.io), or [github](https://github.com/DepthAnything/Depth-Anything-V2) for more details."""
    
    with gr.Blocks(
        css=css,
        title="Depth Anything V2 - Enhanced",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown(title)
        gr.Markdown(description)
        
        # Instructions section
        with gr.Accordion("📋 Instructions", open=False):
            gr.Markdown("""
            ## 🚀 How to Use This Demo
            
            1. **Select Model**: Choose the model variant that best fits your needs:
               - **ViT-Small**: Fastest processing, lower quality
               - **ViT-Base**: Balanced speed and quality
               - **ViT-Large**: High quality, recommended for most uses
               - **ViT-Giant**: Highest quality, slowest processing
            
            2. **Upload Image**: Upload any image in common formats (JPEG, PNG, etc.)
            
            3. **Process**: Click "Compute Depth" to generate the depth map
            
            4. **View Results**: 
               - Interactive slider to compare original and depth map
               - Download grayscale and raw depth maps
            
            ### 📊 Model Comparison
            - **Speed**: ViT-S > ViT-B > ViT-L > ViT-G
            - **Quality**: ViT-G > ViT-L > ViT-B > ViT-S
            - **Memory Usage**: ViT-G > ViT-L > ViT-B > ViT-S
            
            ### 🔧 Technical Notes
            - Models are cached for faster switching
            - GPU acceleration when available
            - Supports various image formats and sizes
            """)
        
        # Model selection
        with gr.Row():
            model_selector = gr.Dropdown(
                choices=model_choices,
                value=default_model,
                label="🎯 Select Model Variant",
                info="Choose the Depth Anything V2 model variant",
                interactive=True
            )
        
        gr.Markdown("### Depth Prediction Demo")
        
        with gr.Row():
            input_image = gr.Image(
                label="Input Image", 
                type='numpy', 
                elem_id='img-display-input'
            )
            depth_image_slider = ImageSlider(
                label="Depth Map with Slider View", 
                elem_id='img-display-output', 
                position=0.5
            )
        
        submit = gr.Button(
            value="🚀 Compute Depth",
            variant="primary",
            size="lg"
        )
        
        with gr.Row():
            gray_depth_file = gr.File(
                label="📥 Grayscale depth map", 
                elem_id="download"
            )
            raw_file = gr.File(
                label="📥 16-bit raw output (disparity)", 
                elem_id="download"
            )
        
        status_text = gr.Textbox(
            label="📊 Processing Status",
            interactive=False,
            lines=6
        )
        
        # Example images - Paginated
        example_files = glob.glob('assets/examples/*')
        if example_files:
            # Sort files for consistent ordering
            example_files = sorted(example_files)
            
            # Show first 8 examples
            page1_examples = example_files[:8] if len(example_files) > 8 else example_files
            gr.Examples(
                examples=[[f] for f in page1_examples],
                inputs=[input_image],
                label=f"📋 Example Images (1-{len(page1_examples)} of {len(example_files)})"
            )
            
            # Show remaining examples if there are more than 8
            if len(example_files) > 8:
                page2_examples = example_files[8:16] if len(example_files) > 16 else example_files[8:]
                gr.Examples(
                    examples=[[f] for f in page2_examples],
                    inputs=[input_image],
                    label=f"📋 More Examples ({9}-{len(page2_examples)+8} of {len(example_files)})"
                )
                
                # Show final batch if there are more than 16
                if len(example_files) > 16:
                    page3_examples = example_files[16:]
                    gr.Examples(
                        examples=[[f] for f in page3_examples],
                        inputs=[input_image],
                        label=f"📋 Additional Examples ({17}-{len(example_files)} of {len(example_files)})"
                    )
        
        # Event handlers
        submit.click(
            fn=process_image,
            inputs=[model_selector, input_image],
            outputs=[depth_image_slider, gray_depth_file, raw_file, status_text],
            show_progress=True
        )
        
        # Footer
        with gr.Accordion("📖 Citation & Info", open=False):
            gr.Markdown("""
            ### 📄 Citation
            
            If you use Depth Anything V2 in your research, please cite:
            
            ```bibtex
            @article{depth_anything_v2,
              title={Depth Anything V2},
              author={Lihe Yang and Bingyi Kang and Zilong Huang and Zhen Zhao and Xiaogang Xu and Jiashi Feng and Hengshuang Zhao},
              journal={arXiv:2406.09414},
              year={2024}
            }
            ```
            
            ### 🔗 Links
            - [Paper](https://arxiv.org/abs/2406.09414)
            - [Project Page](https://depth-anything-v2.github.io)
            - [GitHub Repository](https://github.com/DepthAnything/Depth-Anything-V2)
            
            ### ⚡ Performance Notes
            - Enhanced with model caching for faster switching
            - GPU memory optimization
            - Support for multiple model variants
            """)
    
    return app

def main():
    """Main function to launch the app"""
    
    logging.info("🚀 Starting Enhanced Depth Anything V2 App...")
    
    # Check available models
    available_models = get_available_models()
    logging.info(f"Found {len(available_models)} available models")
    
    try:
        # Create and launch app
        logging.info("Creating Gradio app...")
        app = create_app()
        logging.info("✅ Gradio app created successfully")
        
        # Launch app
        app.queue().launch(
            share=False,
            show_error=True
        )
        
    except Exception as e:
        logging.error(f"Failed to launch app: {e}")
        raise

if __name__ == "__main__":
    main()
