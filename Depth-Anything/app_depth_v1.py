import gradio as gr
import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
import tempfile
from gradio_imageslider import ImageSlider

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

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
"""

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configurations - supports different model variants
MODEL_CONFIGS = {
    "vits14": {
        "model_name": "LiheYoung/depth_anything_vits14",
        "display_name": "Depth Anything ViT-S (Small, Fastest)",
        "description": "Smallest and fastest model variant"
    },
    "vitb14": {
        "model_name": "LiheYoung/depth_anything_vitb14", 
        "display_name": "Depth Anything ViT-B (Base, Balanced)",
        "description": "Balanced model with good speed/quality tradeoff"
    },
    "vitl14": {
        "model_name": "LiheYoung/depth_anything_vitl14",
        "display_name": "Depth Anything ViT-L (Large, Best Quality)",
        "description": "Largest model with best quality (default)"
    }
}

# Global model cache
current_model = None
current_model_name = None
cached_models = {}  # Store all downloaded models

title = "# Depth Anything with Model Selection"
description = """Official demo for **Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data** with multiple model variants.

You can choose between different model sizes for speed vs quality tradeoffs:
- **ViT-S**: Fastest inference, good for real-time applications
- **ViT-B**: Balanced performance and quality 
- **ViT-L**: Best quality, slower inference

Please refer to our [paper](https://arxiv.org/abs/2401.10891), [project page](https://depth-anything.github.io), or [github](https://github.com/LiheYoung/Depth-Anything) for more details."""

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
])

def get_memory_status():
    """Get current memory usage status"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            return f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached, {total_memory:.2f}GB total"
        else:
            return "Running on CPU"
    except:
        return "Memory status unavailable"

def download_all_models():
    """Download and cache all model variants at startup"""
    global cached_models
    
    print("🔄 Downloading all Depth Anything model variants...")
    print("This may take a few minutes depending on your internet connection...")
    
    for key, config in MODEL_CONFIGS.items():
        try:
            print(f"📥 Downloading {config['display_name']}...")
            model = DepthAnything.from_pretrained(config['model_name']).to(DEVICE).eval()
            cached_models[key] = model
            print(f"✅ {config['display_name']} downloaded and cached successfully")
        except Exception as e:
            print(f"❌ Failed to download {config['display_name']}: {e}")
            cached_models[key] = None
    
    print(f"🎉 Model download complete! {len([m for m in cached_models.values() if m is not None])}/{len(MODEL_CONFIGS)} models cached successfully.")
    return cached_models

def load_model(model_selection):
    """Load the selected model variant from cache"""
    global current_model, current_model_name
    
    # Find the model key from the display name
    selected_key = None
    for key, config in MODEL_CONFIGS.items():
        if config["display_name"] == model_selection:
            selected_key = key
            break
    
    if selected_key is None:
        # Fallback to vitl14 if not found
        selected_key = "vitl14"
    
    # Check if we need to switch to a different model
    if current_model_name != selected_key:
        print(f"🔄 Switching to model: {MODEL_CONFIGS[selected_key]['display_name']}")
        
        # Get model from cache
        if selected_key in cached_models and cached_models[selected_key] is not None:
            current_model = cached_models[selected_key]
            current_model_name = selected_key
            print(f"✅ Model {selected_key} loaded from cache successfully")
        else:
            # Fallback: download model if not in cache
            print(f"⚠️ Model {selected_key} not in cache, downloading...")
            try:
                current_model = DepthAnything.from_pretrained(MODEL_CONFIGS[selected_key]['model_name']).to(DEVICE).eval()
                cached_models[selected_key] = current_model
                current_model_name = selected_key
                print(f"✅ Model {selected_key} downloaded and loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load model {selected_key}: {e}")
                # Fallback to any available cached model
                for fallback_key, fallback_model in cached_models.items():
                    if fallback_model is not None:
                        current_model = fallback_model
                        current_model_name = fallback_key
                        print(f"🔄 Using fallback model: {fallback_key}")
                        break
    
    return current_model

@torch.no_grad()
def predict_depth(model, image):
    return model(image)

def on_submit(model_selection, image):
    if image is None:
        return None, None
    
    # Load the selected model
    try:
        model = load_model(model_selection)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
    original_image = image.copy()

    h, w = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

    depth = predict_depth(model, image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    raw_depth = Image.fromarray(depth.cpu().numpy().astype('uint16'))
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    raw_depth.save(tmp.name)

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]

    return [(original_image, colored_depth), tmp.name]

# Download and cache all models at startup
print("🚀 Initializing Depth Anything with all model variants...")
cached_models = download_all_models()

# Set default model to the first successfully cached model
default_model_key = None
for key in ["vitl14", "vitb14", "vits14"]:  # Priority order
    if key in cached_models and cached_models[key] is not None:
        default_model_key = key
        break

if default_model_key:
    current_model = cached_models[default_model_key]
    current_model_name = default_model_key
    print(f"🎯 Default model set to: {MODEL_CONFIGS[default_model_key]['display_name']}")
else:
    print("❌ No models were successfully cached!")
    current_model = None
    current_model_name = None

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model Selection")
            model_selector = gr.Dropdown(
                choices=[config["display_name"] for config in MODEL_CONFIGS.values()],
                value=MODEL_CONFIGS[default_model_key]["display_name"] if default_model_key else MODEL_CONFIGS["vitl14"]["display_name"],
                label="Choose Model Variant",
                info="Select the model size based on your speed/quality requirements"
            )
            
            # Add model info display
            initial_info = f"**Selected Model**: {MODEL_CONFIGS[default_model_key]['description']}" if default_model_key else "**Selected Model**: Unknown"
            model_info = gr.Markdown(initial_info)
            
            # Add memory status display
            memory_status = gr.Markdown(f"**Memory Status**: {get_memory_status()}")
            
            def update_model_info(selection):
                info_text = "**Selected Model**: Unknown"
                for key, config in MODEL_CONFIGS.items():
                    if config["display_name"] == selection:
                        cached_status = "✅ Cached" if key in cached_models and cached_models[key] is not None else "❌ Not cached"
                        info_text = f"**Selected Model**: {config['description']} ({cached_status})"
                        break
                
                memory_text = f"**Memory Status**: {get_memory_status()}"
                return info_text, memory_text
            
            model_selector.change(update_model_info, inputs=[model_selector], outputs=[model_info, memory_status])

    gr.Markdown("### Depth Prediction Demo")
    gr.Markdown("You can slide the output to compare the depth prediction with input image")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
    
    raw_file = gr.File(label="16-bit raw depth (can be considered as disparity)")
    
    submit = gr.Button("Submit", variant="primary")

    submit.click(on_submit, inputs=[model_selector, input_image], outputs=[depth_image_slider, raw_file])
    
    # Examples section
    if os.path.exists('assets/examples'):
        example_files = os.listdir('assets/examples')
        example_files.sort()
        example_files = [os.path.join('assets/examples', filename) for filename in example_files]
        
        examples = gr.Examples(
            examples=example_files, 
            inputs=[input_image], 
            outputs=[depth_image_slider, raw_file], 
            fn=lambda img: on_submit(model_selector.value, img), 
            cache_examples=False,
            label="Example Images"
        )
    
    # Model comparison section
    with gr.Accordion("📊 Model Comparison & Cache Status", open=False):
        # Create cache status dynamically
        cache_status_md = "### 📦 Cached Models Status\n"
        for key, config in MODEL_CONFIGS.items():
            status = "✅ Cached" if key in cached_models and cached_models[key] is not None else "❌ Not cached"
            cache_status_md += f"- **{config['display_name']}**: {status}\n"
        
        cache_status_md += f"\n**Total Models Cached**: {len([m for m in cached_models.values() if m is not None])}/{len(MODEL_CONFIGS)}\n"
        cache_status_md += f"**Current Memory**: {get_memory_status()}\n\n"
        
        gr.Markdown(cache_status_md)
        
        gr.Markdown("""
        ### 📈 Model Performance Comparison
        | Model | Parameters | Speed | Quality | Use Case |
        |-------|------------|-------|---------|----------|
        | ViT-S | ~25M | Fastest | Good | Real-time applications |
        | ViT-B | ~97M | Medium | Better | Balanced performance |
        | ViT-L | ~335M | Slower | Best | High-quality results |
        
        **Note**: All models are pre-downloaded and cached for instant switching!  
        **Processing times** are approximate and depend on hardware and image resolution.
        """)
        
        # Add refresh button for memory status
        def refresh_status():
            updated_status_md = "### 📦 Cached Models Status\n"
            for key, config in MODEL_CONFIGS.items():
                status = "✅ Cached" if key in cached_models and cached_models[key] is not None else "❌ Not cached"
                updated_status_md += f"- **{config['display_name']}**: {status}\n"
            
            updated_status_md += f"\n**Total Models Cached**: {len([m for m in cached_models.values() if m is not None])}/{len(MODEL_CONFIGS)}\n"
            updated_status_md += f"**Current Memory**: {get_memory_status()}\n\n"
            return updated_status_md
        
        refresh_btn = gr.Button("🔄 Refresh Status", size="sm")
        status_display = gr.Markdown(cache_status_md)
        refresh_btn.click(refresh_status, outputs=[status_display])
    
    # Citation section
    with gr.Accordion("📖 Citation", open=False):
        gr.Markdown("""
        ```bibtex
        @article{depthanything,
            title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
            author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
            journal={arXiv:2401.10891},
            year={2024}
        }
        ```
        """)

if __name__ == '__main__':
    demo.queue().launch()
