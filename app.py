"""
Depth Estimation Comparison Demo (ZeroGPU)

Compare 28 depth estimation models from 12 families side-by-side or with a slider using Gradio.
Powered by the depth_estimation package. Optimized for HuggingFace Spaces with ZeroGPU support.
"""

import os
import logging
import gc
import inspect
from typing import Tuple, Dict, List
import numpy as np
import cv2
import gradio as gr
import spaces
import torch
from PIL import Image
from depth_estimation import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, str] = {
    # Depth Anything v1 — Relative
    "depth-anything-v1-vits":               "Depth Anything v1 ViT-S [Relative]",
    "depth-anything-v1-vitb":               "Depth Anything v1 ViT-B [Relative]",
    "depth-anything-v1-vitl":               "Depth Anything v1 ViT-L [Relative]",
    # Depth Anything v2 — Relative
    "depth-anything-v2-vits":               "Depth Anything v2 ViT-S [Relative]",
    "depth-anything-v2-vitb":               "Depth Anything v2 ViT-B [Relative]",
    "depth-anything-v2-vitl":               "Depth Anything v2 ViT-L [Relative]",
    # Depth Anything v3
    "depth-anything-v3-small":              "Depth Anything v3 Small [Relative+Metric]",
    "depth-anything-v3-base":               "Depth Anything v3 Base [Relative+Metric]",
    "depth-anything-v3-large":              "Depth Anything v3 Large [Relative+Metric]",
    "depth-anything-v3-giant":              "Depth Anything v3 Giant [Relative+Metric]",
    "depth-anything-v3-nested-giant-large": "Depth Anything v3 Nested Giant Large [Relative]",
    "depth-anything-v3-metric-large":       "Depth Anything v3 Metric Large [Metric]",
    "depth-anything-v3-mono-large":         "Depth Anything v3 Mono Large [Relative+Metric]",
    # ZoeDepth — Metric
    "zoedepth":                             "Intel ZoeDepth [Metric]",
    # MiDaS — Relative
    "midas-dpt-large":                      "MiDaS DPT-Large [Relative]",
    "midas-dpt-hybrid":                     "MiDaS DPT-Hybrid [Relative]",
    "midas-beit-large":                     "MiDaS BEiT-Large [Relative]",
    # Apple DepthPro — Metric
    "depth-pro":                            "Apple DepthPro [Metric]",
    # Pixel-Perfect Depth — Relative
    "pixel-perfect-depth":                  "Pixel-Perfect Depth [Relative]",
    # Marigold-DC — Depth Completion
    "marigold-dc":                          "Marigold-DC [Depth Completion]",
    # MoGe — Metric
    "moge-v1":                              "MoGe v1 [Metric]",
    "moge-v2-vitl":                         "MoGe v2 ViT-L [Metric]",
    "moge-v2-vitl-normal":                  "MoGe v2 ViT-L Normal [Metric]",
    "moge-v2-vitb-normal":                  "MoGe v2 ViT-B Normal [Metric]",
    "moge-v2-vits-normal":                  "MoGe v2 ViT-S Normal [Metric]",
    # OmniVGGT — Metric
    "omnivggt":                             "OmniVGGT [Metric]",
    # VGGT — Metric
    "vggt":                                 "VGGT [Metric]",
    "vggt-commercial":                      "VGGT (Commercial) [Metric]",
}

DA3_MODEL_KEYS: List[str] = [k for k in MODEL_REGISTRY if k.startswith("depth-anything-v3-")]

# ── Pipeline cache ─────────────────────────────────────────────────────────────
_pipeline_cache: Dict[str, object] = {}


def get_or_load_pipeline(model_string: str):
    if model_string not in _pipeline_cache:
        clear_model_cache()
        logging.info(f"Loading pipeline: {model_string}")
        _pipeline_cache[model_string] = pipeline("depth-estimation", model=model_string)
    return _pipeline_cache[model_string]


def clear_model_cache():
    global _pipeline_cache
    _pipeline_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_pil(image: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL RGB Image."""
    if image.ndim == 2:
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


# ── Inference ──────────────────────────────────────────────────────────────────
@spaces.GPU
def run_model(model_key: str, image: np.ndarray) -> Tuple[np.ndarray, str]:
    try:
        pipe = get_or_load_pipeline(model_key)
        result = pipe(_to_pil(image))
        return result.colored_depth, MODEL_REGISTRY[model_key]
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@spaces.GPU
def compare_models(image, model1: str, model2: str, progress=gr.Progress()) -> Tuple[np.ndarray, str]:
    """Compare two models with ZeroGPU optimization"""
    if image is None:
        return None, "❌ Please upload an image."

    try:
        if isinstance(image, str):
            image = cv2.imread(image)
        elif hasattr(image, 'save'):
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
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
        cv2.putText(canvas, label1, (10 + (w - size1[0]) // 2, 28), font, font_scale, (0, 0, 0), thickness)
        cv2.putText(canvas, label2, (w+20 + (w - size2[0]) // 2, 28), font, font_scale, (0, 0, 0), thickness)

        progress(1.0, desc="Done")
        return canvas, f"**{label1}** vs **{label2}**"
    finally:
        clear_model_cache()


@spaces.GPU
def slider_compare(image, model1: str, model2: str, progress=gr.Progress()):
    """Slider comparison with ZeroGPU optimization"""
    if image is None:
        return None, "❌ Please upload an image."

    try:
        if isinstance(image, str):
            image = cv2.imread(image)
        elif hasattr(image, 'save'):
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
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
            cv2.putText(canvas, label, ((w-size[0])//2, 28), font, font_scale, (0, 0, 0), thickness)
            return canvas

        return (add_label(out1, label1), add_label(out2, label2)), f"Slider: **{label1}** vs **{label2}**"
    finally:
        clear_model_cache()


@spaces.GPU
def single_inference(image, model: str, progress=gr.Progress()):
    """Single model inference with ZeroGPU optimization"""
    if image is None:
        return None, "❌ Please upload an image."

    try:
        original_image = None

        if isinstance(image, str):
            original_image = cv2.imread(image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            image = cv2.imread(image)
        elif hasattr(image, 'save'):
            original_image = np.array(image)
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            original_image = np.array(image)
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        progress(0.1, desc=f"Running {model}")
        depth_result, label = run_model(model, image)
        progress(1.0, desc="Done")
        return (original_image, depth_result), f"**Original** vs **{label}**"
    finally:
        clear_model_cache()


@spaces.GPU
def da3_single_inference(image, model: str, progress=gr.Progress()):
    if image is None:
        return None, "❌ Please upload an image."

    try:
        if isinstance(image, str):
            original_rgb = np.array(Image.open(image).convert("RGB"))
            np_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        elif hasattr(image, "save"):
            original_rgb = np.array(image.convert("RGB"))
            np_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        else:
            original_rgb = np.array(image)
            np_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR) if original_rgb.ndim == 3 else original_rgb

        if np_bgr is None:
            raise gr.Error("Invalid image input.")

        progress(0.1, desc=f"Running {model}")
        depth_colored, label = run_model(model, np_bgr)
        progress(1.0, desc="Done")

        info_lines = [
            f"**Model:** `{label}`",
            f"**Package model string:** `{model}`",
            f"**Device:** `{DEVICE}`",
            f"**Output shape:** `{depth_colored.shape}`",
        ]
        return (original_rgb, depth_colored), "\n".join(info_lines)
    finally:
        clear_model_cache()


def get_model_choices() -> List[Tuple[str, str]]:
    return [(display, key) for key, display in MODEL_REGISTRY.items()]


def get_example_images() -> List[str]:
    import re

    def natural_sort_key(filename):
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

    ex_path = os.path.join(os.path.dirname(__file__), "assets", "examples")
    if not os.path.exists(ex_path):
        return []
    all_files = [f for f in os.listdir(ex_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return [os.path.join(ex_path, f) for f in sorted(all_files, key=natural_sort_key)]


def get_paginated_examples(examples: List[str], page: int = 0, per_page: int = 6) -> Tuple[List[str], int, bool, bool]:
    total_pages = (len(examples) - 1) // per_page + 1 if examples else 0
    start_idx = page * per_page
    end_idx = min(start_idx + per_page, len(examples))
    current_examples = examples[start_idx:end_idx]
    has_prev = page > 0
    has_next = page < total_pages - 1
    return current_examples, total_pages, has_prev, has_next


def create_app():
    model_choices = get_model_choices()
    default1 = next((v for _, v in model_choices if v == "depth-anything-v2-vitl"), model_choices[0][1])
    default2 = next((v for _, v in model_choices if v == "pixel-perfect-depth"), model_choices[1][1])
    da3_choices = [(MODEL_REGISTRY[k], k) for k in DA3_MODEL_KEYS]
    da3_default = next((k for k in DA3_MODEL_KEYS if k == "depth-anything-v3-large"), DA3_MODEL_KEYS[0])
    example_images = get_example_images()

    blocks_kwargs = {"title": "Depth Estimation Comparison"}
    try:
        if "theme" in inspect.signature(gr.Blocks.__init__).parameters and hasattr(gr, "themes"):
            blocks_kwargs["theme"] = gr.themes.Soft()
    except (ValueError, TypeError):
        pass

    with gr.Blocks(**blocks_kwargs) as app:
        gr.Markdown("""
        # Depth Estimation Comparison
        Compare **28 depth estimation models** from 12 families side-by-side or with a slider.
        Powered by the [`depth_estimation`](https://pypi.org/project/depth-estimation/) package.

        ⚡ **Running on ZeroGPU** — GPU resources are allocated automatically for inference.
        """)

        with gr.Tabs():
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
                if example_images:
                    gr.Examples(examples=example_images, inputs=[img_input2])

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
                if example_images:
                    gr.Examples(examples=example_images, inputs=[img_input])

            with gr.Tab("📷 Single Model"):
                with gr.Row():
                    img_input3 = gr.Image(label="Input Image", type="numpy")
                    with gr.Column():
                        m_single = gr.Dropdown(choices=model_choices, label="Model", value=default1)
                        btn3 = gr.Button("Run", variant="primary")
                single_slider = gr.ImageSlider(label="Original vs Depth")
                out_single_status = gr.Markdown()
                btn3.click(single_inference, inputs=[img_input3, m_single], outputs=[single_slider, out_single_status], show_progress=True)
                if example_images:
                    gr.Examples(examples=example_images, inputs=[img_input3])

            with gr.Tab("🧪 Depth Anything v3"):
                with gr.Row():
                    da3_img_input = gr.Image(label="Input Image", type="numpy")
                    with gr.Column():
                        da3_model_dropdown = gr.Dropdown(
                            choices=da3_choices, label="DA3 Model", value=da3_default
                        )
                        da3_btn = gr.Button("Run DA3", variant="primary")
                da3_slider = gr.ImageSlider(label="Original vs Depth")
                da3_info = gr.Markdown()
                da3_btn.click(
                    da3_single_inference,
                    inputs=[da3_img_input, da3_model_dropdown],
                    outputs=[da3_slider, da3_info],
                    show_progress=True,
                )
                if example_images:
                    gr.Examples(examples=example_images, inputs=[da3_img_input])

        gr.Markdown("""
        ---
        **Supported Models (28 variants · 12 families):**
        - **Depth Anything v1** — ViT-S / ViT-B / ViT-L
        - **Depth Anything v2** — ViT-S / ViT-B / ViT-L
        - **Depth Anything v3** — Small / Base / Large / Giant / Nested Giant Large / Metric Large / Mono Large
        - **Intel ZoeDepth** — metric depth fine-tuned on NYU + KITTI
        - **MiDaS** — DPT-Large / DPT-Hybrid / BEiT-Large
        - **Apple DepthPro** — sharp metric monocular depth in < 1 s
        - **Pixel-Perfect Depth** — diffusion + MoGe metric alignment
        - **Marigold-DC** — diffusion-based depth completion
        - **MoGe** — v1 & v2 metric geometry (ViT-S/B/L + normal variants)
        - **OmniVGGT** — metric depth ViT-L
        - **VGGT / VGGT-Commercial** — metric depth from `facebook/VGGT-1B`

        Powered by [`depth_estimation`](https://pypi.org/project/depth-estimation/) · ZeroGPU manages GPU allocation automatically.
        """)

    return app


def main():
    logging.info("🚀 Starting Depth Estimation Comparison App on ZeroGPU...")
    app = create_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
