from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image

# Add src directory to Python path to import depth_anything_3 module
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SOURCES: Dict[str, str] = {
    "Depth Anything v3 Nested Giant Large": "depth-anything/DA3NESTED-GIANT-LARGE",
    "Depth Anything v3 Giant": "depth-anything/DA3-GIANT",
    "Depth Anything v3 Large": "depth-anything/DA3-LARGE",
    "Depth Anything v3 Base": "depth-anything/DA3-BASE",
    "Depth Anything v3 Small": "depth-anything/DA3-SMALL",
    "Depth Anything v3 Metric Large": "depth-anything/DA3METRIC-LARGE",
    "Depth Anything v3 Mono Large": "depth-anything/DA3MONO-LARGE",
}
_MODEL_CACHE: Dict[str, DepthAnything3] = {}


def _load_model(model_label: str) -> DepthAnything3:
    repo_id = MODEL_SOURCES[model_label]
    if repo_id not in _MODEL_CACHE:
        model = DepthAnything3.from_pretrained(repo_id)
        model = model.to(device=DEVICE)
        model.eval()
        _MODEL_CACHE[repo_id] = model
    return _MODEL_CACHE[repo_id]


def _prep_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def run_inference(
    model_label: str,
    image: Optional[np.ndarray],
) -> tuple[Tuple[np.ndarray, np.ndarray], str]:
    if image is None:
        raise gr.Error("Upload an image before running inference.")
    rgb = _prep_image(image)
    model = _load_model(model_label)
    prediction = model.inference(
        image=[Image.fromarray(rgb)],
        process_res=None,
        process_res_method="keep",
    )
    depth_map = prediction.depth[0]
    depth_vis = visualize_depth(depth_map, cmap="Spectral")
    processed_rgb = (
        prediction.processed_images[0]
        if prediction.processed_images is not None
        else rgb
    )
    slider_value: Tuple[np.ndarray, np.ndarray] = (processed_rgb, depth_vis)
    lines = [
        f"**Model:** `{MODEL_SOURCES[model_label]}`",
        f"**Device:** `{DEVICE}`",
        f"**Depth shape:** `{tuple(prediction.depth.shape)}`",
    ]
    if prediction.extrinsics is not None:
        lines.append(f"**Extrinsics shape:** `{prediction.extrinsics.shape}`")
    if prediction.intrinsics is not None:
        lines.append(f"**Intrinsics shape:** `{prediction.intrinsics.shape}`")
    return slider_value, "\n".join(lines)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Depth Anything v3 - Any Size Demo") as demo:
        gr.Markdown(
            """
            ## Depth Anything v3 (Any-Size Demo)
            Upload an image, pick a pretrained model, and compare RGB against the inferred depth.
            """
        )
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_SOURCES.keys()),
                value="Depth Anything v3 Large",
                label="Model",
            )
        image_input = gr.Image(type="numpy", label="Input Image", image_mode="RGB")
        run_button = gr.Button("Run Inference", variant="primary")
        with gr.Row():
            comparison_slider = gr.ImageSlider(label="RGB vs Depth")
        info_panel = gr.Markdown()
        run_button.click(
            fn=run_inference,
            inputs=[model_dropdown, image_input],
            outputs=[comparison_slider, info_panel],
        )
    return demo


def main() -> None:
    app = build_app()
    app.queue(max_size=8).launch()


if __name__ == "__main__":
    main()
