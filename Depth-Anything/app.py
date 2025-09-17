"""
Depth Anything Comparison Demo (v1 vs v2)

Compare different Depth Anything models (v1 and v2) side-by-side or with a slider using Gradio.
Inspired by the Stereo Matching Methods Comparison Demo.
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

# Import v1 and v2 model code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "depth_anything"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../Depth-Anything-V2/depth_anything_v2"))

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

# Device selection
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
        'checkpoint': '../Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth',
        'features': 64, 'out_channels': [48, 96, 192, 384]
    },
    'vitb': {
        'display_name': 'Depth Anything v2 ViT-Base',
        'checkpoint': '../Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth',
        'features': 128, 'out_channels': [96, 192, 384, 768]
    },
    'vitl': {
        'display_name': 'Depth Anything v2 ViT-Large',
        'checkpoint': '../Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth',
        'features': 256, 'out_channels': [256, 512, 1024, 1024]
    }
}

# Model cache
_v1_models = {}
_v2_models = {}

# v1 transform
v1_transform = Compose([
    Resize(width=518, height=518, resize_target=False, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

def load_v1_model(key: str):
    if key in _v1_models:
        return _v1_models[key]
    model = DepthAnythingV1.from_pretrained(V1_MODEL_CONFIGS[key]["model_name"]).to(DEVICE).eval()
    _v1_models[key] = model
    return model

def load_v2_model(key: str):
    if key in _v2_models:
        return _v2_models[key]
    config = V2_MODEL_CONFIGS[key]
    model = DepthAnythingV2(encoder=key, features=config['features'], out_channels=config['out_channels'])
    state_dict = torch.load(config['checkpoint'], map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    _v2_models[key] = model
    return model

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

def run_model(model_key: str, image: np.ndarray) -> Tuple[np.ndarray, str]:
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

def compare_models(image: np.ndarray, model1: str, model2: str, progress=gr.Progress()) -> Tuple[np.ndarray, str]:
    if image is None:
        return None, "❌ Please upload an image."
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

def slider_compare(image: np.ndarray, model1: str, model2: str, progress=gr.Progress()):
    if image is None:
        return None, "❌ Please upload an image."
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

def single_inference(image: np.ndarray, model: str, progress=gr.Progress()):
    if image is None:
        return None, "❌ Please upload an image."
    progress(0.1, desc=f"Running {model}")
    out, label = run_model(model, image)
    progress(1.0, desc="Done")
    return out, f"**{label}**"

def get_example_images() -> List[str]:
    ex_dir = os.path.join(os.path.dirname(__file__), "assets/examples")
    if not os.path.exists(ex_dir):
        return []
    files = [os.path.join(ex_dir, f) for f in sorted(os.listdir(ex_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return files[:6]

def create_app():
    model_choices = get_model_choices()
    default1 = model_choices[0][1]
    default2 = model_choices[1][1]
    with gr.Blocks(title="Depth Anything v1 vs v2 Comparison", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Depth Anything v1 vs v2 Comparison
        Compare different Depth Anything models (v1 and v2) side-by-side or with a slider.
        """)
        with gr.Tabs():
            with gr.Tab("🔍 Method Comparison"):
                with gr.Row():
                    img_input = gr.Image(label="Input Image", type="numpy")
                    with gr.Column():
                        m1 = gr.Dropdown(choices=model_choices, label="Model 1", value=default1)
                        m2 = gr.Dropdown(choices=model_choices, label="Model 2", value=default2)
                        btn = gr.Button("Compare", variant="primary")
                out_img = gr.Image(label="Comparison Result")
                out_status = gr.Markdown()
                btn.click(compare_models, inputs=[img_input, m1, m2], outputs=[out_img, out_status], show_progress=True)
                # Examples
                ex_imgs = get_example_images()
                if ex_imgs:
                    gr.Examples(examples=[[f] for f in ex_imgs], inputs=[img_input], label="Example Images")
            with gr.Tab("🎚️ Slider Comparison"):
                with gr.Row():
                    img_input2 = gr.Image(label="Input Image", type="numpy")
                    with gr.Column():
                        m1s = gr.Dropdown(choices=model_choices, label="Model A", value=default1)
                        m2s = gr.Dropdown(choices=model_choices, label="Model B", value=default2)
                        btn2 = gr.Button("Slider Compare", variant="primary")
                slider = gr.ImageSlider(label="Model Comparison Slider")
                slider_status = gr.Markdown()
                btn2.click(slider_compare, inputs=[img_input2, m1s, m2s], outputs=[slider, slider_status], show_progress=True)
                if ex_imgs:
                    gr.Examples(examples=[[f] for f in ex_imgs], inputs=[img_input2], label="Example Images")
            with gr.Tab("🎯 Single Model"):
                with gr.Row():
                    img_input3 = gr.Image(label="Input Image", type="numpy")
                    m_single = gr.Dropdown(choices=model_choices, label="Model", value=default1)
                    btn3 = gr.Button("Run", variant="primary")
                out_single = gr.Image(label="Depth Result")
                out_single_status = gr.Markdown()
                btn3.click(single_inference, inputs=[img_input3, m_single], outputs=[out_single, out_single_status], show_progress=True)
                if ex_imgs:
                    gr.Examples(examples=[[f] for f in ex_imgs], inputs=[img_input3], label="Example Images")
        gr.Markdown("""
        ---
        - **v1**: [Depth Anything v1](https://github.com/LiheYoung/Depth-Anything)
        - **v2**: [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2)
        """)
    return app

def main():
    logging.info("🚀 Starting Depth Anything Comparison App...")
    app = create_app()
    app.queue().launch(show_error=True)

if __name__ == "__main__":
    main()
