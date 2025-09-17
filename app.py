"""
Depth Anything Comparison Demo (v1 vs v2) - ZeroGPU Version

Compare different Depth Anything models (v1 and v2) side-by-side or with a slider using Gradio.
Optimized for HuggingFace Spaces with ZeroGPU support.
"""

import os
import sys
import logging
import gc
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download
import spaces

# Import v1 and v2 model code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))

# v1 imports
from depth_anything.dpt import DepthAnything as DepthAnythingV1
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

# v2 imports
from depth_anything_v2.dpt import DepthAnythingV2

import matplotlib

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device selection - ZeroGPU will handle GPU allocation
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model configs
V1_MODEL_CONFIGS = {
    "vits14": {
        "model_name": "LiheYoung/depth_anything_vits14",
        "display_name": "Depth Anything v1 ViT-S (Small, Fastest)"
    },
    "vitb14": {
        "model_name": "LiheYoung/depth_anything_vitb14",
        "display_name": "Depth Anything v1 ViT-B (Base, Balanced)"
    },
    "vitl14": {
        "model_name": "LiheYoung/depth_anything_vitl14",
        "display_name": "Depth Anything v1 ViT-L (Large, Best Quality)"
    }
}

V2_MODEL_CONFIGS = {
    'vits': {
        'display_name': 'Depth Anything v2 ViT-Small',
        'checkpoint': 'Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth',
        'features': 64, 'out_channels': [48, 96, 192, 384]
    },
    'vitb': {
        'display_name': 'Depth Anything v2 ViT-Base',
        'checkpoint': 'Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth',
        'features': 128, 'out_channels': [96, 192, 384, 768]
    },
    'vitl': {
        'display_name': 'Depth Anything v2 ViT-Large',
        'checkpoint': 'Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth',
        'features': 256, 'out_channels': [256, 512, 1024, 1024]
    }
}

# Model cache - cleared after each inference for ZeroGPU
_v1_models = {}
_v2_models = {}

# v1 transform
v1_transform = Compose([
    Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

def load_v1_model(key: str):
    """Load v1 model with memory management for ZeroGPU"""
    if key in _v1_models:
        return _v1_models[key]
    
    # Clear cache to free memory
    clear_model_cache()
    
    model = DepthAnythingV1.from_pretrained(V1_MODEL_CONFIGS[key]["model_name"]).to(DEVICE).eval()
    _v1_models[key] = model
    return model

def load_v2_model(key: str):
    """Load v2 model with memory management for ZeroGPU"""
    if key in _v2_models:
        return _v2_models[key]
    
    # Clear cache to free memory
    clear_model_cache()
    
    config = V2_MODEL_CONFIGS[key]
    model = DepthAnythingV2(encoder=key, features=config['features'], out_channels=config['out_channels'])
    
    # Try to download from HF Hub first, fallback to local checkpoint
    try:
        # Map variant to model names used in HF Hub
        model_name_mapping = {
            'vits': 'Small',
            'vitb': 'Base', 
            'vitl': 'Large'
        }
        
        model_name = model_name_mapping.get(key, 'Large')  # Default to Large
        filename = f"depth_anything_v2_{key}.pth"
        
        # Try to download from HF Hub first
        try:
            filepath = hf_hub_download(
                repo_id=f"depth-anything/Depth-Anything-V2-{model_name}", 
                filename=filename, 
                repo_type="model"
            )
            logging.info(f"Downloaded V2 model from HF Hub: {filepath}")
            checkpoint_path = filepath
        except Exception as e:
            logging.warning(f"Failed to download V2 model from HF Hub: {e}")
            # Fallback to local checkpoint
            checkpoint_path = config['checkpoint']
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Neither HF Hub download nor local checkpoint available: {checkpoint_path}")
            logging.info(f"Using local V2 checkpoint: {checkpoint_path}")
        
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    except Exception as e:
        logging.error(f"Failed to load V2 model {key}: {e}")
        raise
    
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    _v2_models[key] = model
    return model

def clear_model_cache():
    """Clear model cache to free GPU memory for ZeroGPU"""
    global _v1_models, _v2_models
    for model in _v1_models.values():
        del model
    for model in _v2_models.values():
        del model
    _v1_models.clear()
    _v2_models.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def predict_v1(model, image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = v1_transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        depth = model(image)
        depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    return depth.cpu().numpy()

def predict_v2(model, image: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        depth = model.infer_image(image[:, :, ::-1])  # BGR to RGB
    return depth

def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    colored = (cmap(depth_uint8)[:, :, :3] * 255).astype(np.uint8)
    return colored

def get_model_choices() -> List[Tuple[str, str]]:
    choices = []
    for k, v in V1_MODEL_CONFIGS.items():
        choices.append((v['display_name'], f'v1_{k}'))
    for k, v in V2_MODEL_CONFIGS.items():
        choices.append((v['display_name'], f'v2_{k}'))
    return choices

@spaces.GPU
def run_model(model_key: str, image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Run model inference with ZeroGPU optimization"""
    try:
        if model_key.startswith('v1_'):
            key = model_key[3:]
            model = load_v1_model(key)
            depth = predict_v1(model, image)
            label = V1_MODEL_CONFIGS[key]['display_name']
        else:
            key = model_key[3:]
            model = load_v2_model(key)
            depth = predict_v2(model, image)
            label = V2_MODEL_CONFIGS[key]['display_name']
        
        colored = colorize_depth(depth)
        return colored, label
    finally:
        # Clean up GPU memory after inference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@spaces.GPU
def compare_models(image, model1: str, model2: str, progress=gr.Progress()) -> Tuple[np.ndarray, str]:
    """Compare two models with ZeroGPU optimization"""
    if image is None:
        return None, "❌ Please upload an image."

    try:
        # Convert image to numpy array if needed
        if isinstance(image, str):
            # If it's a file path
            image = cv2.imread(image)
        elif hasattr(image, 'save'):
            # If it's a PIL Image
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        progress(0.1, desc=f"Running {model1}")
        out1, label1 = run_model(model1, image)
        progress(0.5, desc=f"Running {model2}")
        out2, label2 = run_model(model2, image)
        
        h, w = out1.shape[:2]
        canvas = np.ones((h + 40, w * 2 + 20, 3), dtype=np.uint8) * 255
        canvas[40:40+h, 10:10+w] = out1
        canvas[40:40+h, w+20:w*2+20] = out2
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        size1 = cv2.getTextSize(label1, font, font_scale, thickness)[0]
        size2 = cv2.getTextSize(label2, font, font_scale, thickness)[0]
        cv2.putText(canvas, label1, (10 + (w - size1[0]) // 2, 28), font, font_scale, (0,0,0), thickness)
        cv2.putText(canvas, label2, (w+20 + (w - size2[0]) // 2, 28), font, font_scale, (0,0,0), thickness)
        
        progress(1.0, desc="Done")
        return canvas, f"**{label1}** vs **{label2}**"
    
    finally:
        # Clean up GPU memory after inference
        clear_model_cache()

@spaces.GPU
def slider_compare(image, model1: str, model2: str, progress=gr.Progress()):
    """Slider comparison with ZeroGPU optimization"""
    if image is None:
        return None, "❌ Please upload an image."

    try:
        # Convert image to numpy array if needed
        if isinstance(image, str):
            # If it's a file path
            image = cv2.imread(image)
        elif hasattr(image, 'save'):
            # If it's a PIL Image
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        progress(0.1, desc=f"Running {model1}")
        out1, label1 = run_model(model1, image)
        progress(0.5, desc=f"Running {model2}")
        out2, label2 = run_model(model2, image)
        
        def add_label(img, label):
            h, w = img.shape[:2]
            canvas = np.ones((h+40, w, 3), dtype=np.uint8) * 255
            canvas[40:, :] = img
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            cv2.putText(canvas, label, ((w-size[0])//2, 28), font, font_scale, (0,0,0), thickness)
            return canvas
        
        return (add_label(out1, label1), add_label(out2, label2)), f"Slider: **{label1}** vs **{label2}**"
    
    finally:
        # Clean up GPU memory after inference
        clear_model_cache()

@spaces.GPU
def single_inference(image, model: str, progress=gr.Progress()):
    """Single model inference with ZeroGPU optimization"""
    if image is None:
        return None, "❌ Please upload an image."

    try:
        # Store original image for slider comparison
        original_image = None
        
        # Convert image to numpy array if needed
        if isinstance(image, str):
            # If it's a file path
            original_image = cv2.imread(image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
            image = cv2.imread(image)
        elif hasattr(image, 'save'):
            # If it's a PIL Image
            original_image = np.array(image)  # PIL images are already in RGB
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # If it's already a numpy array (from Gradio)
            original_image = np.array(image)  # Keep original in RGB
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        progress(0.1, desc=f"Running {model}")
        depth_result, label = run_model(model, image)
        
        # Convert depth result back to RGB for slider (depth_result is already in RGB from colorize_depth)
        depth_result_rgb = depth_result  # colorize_depth already returns RGB
        
        progress(1.0, desc="Done")
        return (original_image, depth_result_rgb), f"**Original** vs **{label}**"
    
    finally:
        # Clean up GPU memory after inference
        clear_model_cache()

def get_example_images() -> List[str]:
    import re

    def natural_sort_key(filename):
        """Sort filenames with numbers naturally (demo1, demo2, ..., demo10, demo11)"""
        # Split by numbers and convert numeric parts to integers for proper sorting
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

    # Try both v1 and v2 examples
    examples = []
    for ex_dir in ["assets/examples", "Depth-Anything/assets/examples", "Depth-Anything-V2/assets/examples"]:
        ex_path = os.path.join(os.path.dirname(__file__), ex_dir)
        if os.path.exists(ex_path):
            # Get all image files and sort them naturally
            all_files = [f for f in os.listdir(ex_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            sorted_files = sorted(all_files, key=natural_sort_key)
            files = [os.path.join(ex_path, f) for f in sorted_files]
            examples.extend(files)
    return examples

def get_paginated_examples(examples: List[str], page: int = 0, per_page: int = 6) -> Tuple[List[str], int, bool, bool]:
    """Get paginated examples with navigation info"""
    total_pages = (len(examples) - 1) // per_page + 1 if examples else 0
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, len(examples))
    
    current_examples = examples[start_idx:end_idx]
    has_prev = page > 0
    has_next = page < total_pages - 1
    
    return current_examples, total_pages, has_prev, has_next

def create_app():
    model_choices = get_model_choices()
    default1 = model_choices[0][1]
    default2 = model_choices[1][1]
    
    with gr.Blocks(title="Depth Anything v1 vs v2 Comparison", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Depth Anything v1 vs v2 Comparison
        Compare different Depth Anything models (v1 and v2) side-by-side or with a slider.
        
        ⚡ **Running on ZeroGPU** - GPU resources are allocated automatically for inference.
        """)
        
        with gr.Tabs():
            with gr.Tab("🎚️ Slider Comparison"):
                with gr.Row():
                    img_input2 = gr.Image(label="Input Image")
                    with gr.Column():
                        m1s = gr.Dropdown(choices=model_choices, label="Model A", value=default1)
                        m2s = gr.Dropdown(choices=model_choices, label="Model B", value=default2)
                        btn2 = gr.Button("Slider Compare", variant="primary")
                slider = gr.ImageSlider(label="Model Comparison Slider")
                slider_status = gr.Markdown()
                btn2.click(slider_compare, inputs=[img_input2, m1s, m2s], outputs=[slider, slider_status], show_progress=True)

                # Examples for slider comparison
                ex_imgs = get_example_images()
                if ex_imgs:
                    def slider_example_fn(image):
                        return slider_compare(image, default1, default2)
                    examples2 = gr.Examples(examples=ex_imgs, inputs=[img_input2], outputs=[slider, slider_status], fn=slider_example_fn)

            with gr.Tab("🔍 Method Comparison"):
                with gr.Row():
                    img_input = gr.Image(label="Input Image")
                    with gr.Column():
                        m1 = gr.Dropdown(choices=model_choices, label="Model 1", value=default1)
                        m2 = gr.Dropdown(choices=model_choices, label="Model 2", value=default2)
                        btn = gr.Button("Compare", variant="primary")
                out_img = gr.Image(label="Comparison Result")
                out_status = gr.Markdown()
                btn.click(compare_models, inputs=[img_input, m1, m2], outputs=[out_img, out_status], show_progress=True)

                # Examples for method comparison
                if ex_imgs:
                    def compare_example_fn(image):
                        return compare_models(image, default1, default2)
                    examples = gr.Examples(examples=ex_imgs, inputs=[img_input], outputs=[out_img, out_status], fn=compare_example_fn)

            with gr.Tab("� Single Model"):
                with gr.Row():
                    img_input3 = gr.Image(label="Input Image")
                    with gr.Column():
                        m_single = gr.Dropdown(choices=model_choices, label="Model", value=default1)
                        btn3 = gr.Button("Run", variant="primary")
                single_slider = gr.ImageSlider(label="Original vs Depth")
                out_single_status = gr.Markdown()
                btn3.click(single_inference, inputs=[img_input3, m_single], outputs=[single_slider, out_single_status], show_progress=True)

                # Examples for single model
                if ex_imgs:
                    def single_example_fn(image):
                        return single_inference(image, default1)
                    examples3 = gr.Examples(examples=ex_imgs, inputs=[img_input3], outputs=[single_slider, out_single_status], fn=single_example_fn)

        gr.Markdown("""
        ---
        **References:**
        - **v1**: [Depth Anything v1](https://github.com/LiheYoung/Depth-Anything)
        - **v2**: [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2)
        
        **Note**: This app uses ZeroGPU for efficient GPU resource management. Models are loaded on-demand and GPU memory is automatically cleaned up after each inference.
        """)
    
    return app

def main():
    logging.info("🚀 Starting Depth Anything Comparison App on ZeroGPU...")
    app = create_app()
    app.queue().launch(show_error=True)

if __name__ == "__main__":
    main()
