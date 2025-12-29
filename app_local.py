"""
Depth Estimation Comparison Demo (Depth Anything v1/v2 + Pixel-Perfect Depth)

Compare Depth Anything models (v1 and v2) and Pixel-Perfect Depth side-by-side or with a slider using Gradio.
Inspired by the Stereo Matching Methods Comparison Demo.
"""

from __future__ import annotations

import os
import sys
import logging
import tempfile
import shutil
import inspect
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2
import gradio as gr
from huggingface_hub import hf_hub_download
import open3d as o3d
import trimesh
from PIL import Image

# Import v1 and v2 model code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything-V2"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Depth-Anything-3-anysize", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DepthPro"))

# v1 imports
from depth_anything.dpt import DepthAnything as DepthAnythingV1
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

# v2 imports
from depth_anything_v2.dpt import DepthAnythingV2

import matplotlib

# Depth Anything v3 imports
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.visualize import visualize_depth

# DepthPro imports
import depth_pro

# ZoeDepth imports (Intel)
from transformers import pipeline as hf_pipeline

# Pixel-Perfect Depth imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Pixel-Perfect-Depth"))
from ppd.utils.set_seed import set_seed
from ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppd.utils.depth2pcd import depth2pcd
from moge.model.v2 import MoGeModel
from ppd.models.ppd import PixelPerfectDepth

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device selection
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DEVICE = torch.device(DEVICE)

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

DA3_MODEL_SOURCES = {
    "nested_giant_large": {
        "display_name": "Depth Anything v3 Nested Giant Large",
        "repo_id": "depth-anything/DA3NESTED-GIANT-LARGE",
    },
    "giant": {
        "display_name": "Depth Anything v3 Giant",
        "repo_id": "depth-anything/DA3-GIANT",
    },
    "large": {
        "display_name": "Depth Anything v3 Large",
        "repo_id": "depth-anything/DA3-LARGE",
    },
    "base": {
        "display_name": "Depth Anything v3 Base",
        "repo_id": "depth-anything/DA3-BASE",
    },
    "small": {
        "display_name": "Depth Anything v3 Small",
        "repo_id": "depth-anything/DA3-SMALL",
    },
    "metric_large": {
        "display_name": "Depth Anything v3 Metric Large",
        "repo_id": "depth-anything/DA3METRIC-LARGE",
    },
    "mono_large": {
        "display_name": "Depth Anything v3 Mono Large",
        "repo_id": "depth-anything/DA3MONO-LARGE",
    },
}

# Model cache
_v1_models = {}
_v2_models = {}
_da3_models: Dict[str, DepthAnything3] = {}
_depth_pro_cache = {
    "model": None,
    "transform": None,
}
_zoedepth_model = None
_midas_model = None
_midas_v2_model = None
_midas_v3_model = None

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


def load_da3_model(key: str) -> DepthAnything3:
    if key in _da3_models:
        return _da3_models[key]
    repo_id = DA3_MODEL_SOURCES[key]["repo_id"]
    model = DepthAnything3.from_pretrained(repo_id)
    model = model.to(device=TORCH_DEVICE)
    model.eval()
    _da3_models[key] = model
    return model


def load_depth_pro_model():
    """
    Load the DepthPro model and preprocessing transforms.
    Uses caching to avoid reloading the model on every inference.
    Loads weights from HuggingFace apple/DepthPro depth_pro.pt
    """
    global _depth_pro_cache
    
    if _depth_pro_cache["model"] is None:
        logging.info(f"Loading DepthPro model on {TORCH_DEVICE}...")
        
        # Download checkpoint from HuggingFace
        checkpoint_path = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            repo_type="model"
        )
        logging.info(f"DepthPro checkpoint downloaded to: {checkpoint_path}")
        
        # Create the checkpoints directory and symlink in current working directory
        # depth_pro.create_model_and_transforms() looks for ./checkpoints/depth_pro.pt
        local_checkpoint_dir = "./checkpoints"
        local_checkpoint_path = os.path.join(local_checkpoint_dir, "depth_pro.pt")
        
        if not os.path.exists(local_checkpoint_path):
            os.makedirs(local_checkpoint_dir, exist_ok=True)
            # Create symlink to the HuggingFace cached file
            os.symlink(checkpoint_path, local_checkpoint_path)
            logging.info(f"Created symlink: {local_checkpoint_path} -> {checkpoint_path}")
        
        # Create model and transforms
        model, transform = depth_pro.create_model_and_transforms(device=TORCH_DEVICE)
        model = model.to(TORCH_DEVICE)
        model.eval()
        
        _depth_pro_cache["model"] = model
        _depth_pro_cache["transform"] = transform
        
        logging.info("DepthPro model loaded successfully!")
    
    return _depth_pro_cache["model"], _depth_pro_cache["transform"]


def predict_depth_pro(image_bgr: np.ndarray) -> np.ndarray:
    """
    Run DepthPro inference on an image.
    
    Args:
        image_bgr: Input image in BGR format (numpy array)
    
    Returns:
        Depth map as numpy array
    """
    model, transform = load_depth_pro_model()
    
    # Save image temporarily for depth_pro.load_rgb
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_path = tmp_file.name
        # Convert BGR to RGB and save
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(image_rgb).save(temp_path)
    
    try:
        # Load and preprocess the image
        image, _, f_px = depth_pro.load_rgb(temp_path)
        
        # Apply transform
        image = transform(image)
        
        # Move to device
        image = image.to(TORCH_DEVICE)
        
        # Run inference
        with torch.no_grad():
            prediction = model.infer(image, f_px=f_px)
        
        # Extract depth in meters
        depth = prediction["depth"].cpu().numpy()
        
    finally:
        # Clean up temp file
        os.unlink(temp_path)
    
    return depth


def load_zoedepth_model():
    """
    Load the Intel ZoeDepth model using HuggingFace transformers pipeline.
    ZoeDepth is fine-tuned on NYU and KITTI datasets for metric depth estimation.
    """
    global _zoedepth_model
    
    if _zoedepth_model is None:
        logging.info(f"Loading ZoeDepth model on {TORCH_DEVICE}...")
        
        # Use the transformers pipeline for depth estimation
        _zoedepth_model = hf_pipeline(
            task="depth-estimation",
            model="Intel/zoedepth-nyu-kitti",
            device=0 if DEVICE == 'cuda' else -1
        )
        
        logging.info("ZoeDepth model loaded successfully!")
    
    return _zoedepth_model


def predict_zoedepth(image_bgr: np.ndarray) -> np.ndarray:
    """
    Run ZoeDepth inference on an image.
    
    Args:
        image_bgr: Input image in BGR format (numpy array)
    
    Returns:
        Depth map as numpy array
    """
    model = load_zoedepth_model()
    
    # Convert BGR to RGB PIL Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Run inference
    outputs = model(pil_image)
    
    # Get depth map and convert to numpy
    depth = np.array(outputs["depth"])
    
    # Resize to match input image size if needed
    h, w = image_bgr.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return depth


def load_midas_model():
    """
    Load the MiDaS DPT-Large model using HuggingFace transformers pipeline.
    MiDaS produces high-quality relative depth maps.
    """
    global _midas_model
    
    if _midas_model is None:
        logging.info(f"Loading MiDaS DPT-Large model on {TORCH_DEVICE}...")
        
        _midas_model = hf_pipeline(
            task="depth-estimation",
            model="Intel/dpt-large",
            device=0 if DEVICE == 'cuda' else -1
        )
        
        logging.info("MiDaS DPT-Large model loaded successfully!")
    
    return _midas_model


def load_midas_v2_model():
    """
    Load the MiDaS v2.1 Small model using HuggingFace transformers pipeline.
    Smaller and faster variant of MiDaS.
    """
    global _midas_v2_model
    
    if _midas_v2_model is None:
        logging.info(f"Loading MiDaS v2.1 Small model on {TORCH_DEVICE}...")
        
        _midas_v2_model = hf_pipeline(
            task="depth-estimation",
            model="Intel/dpt-hybrid-midas",
            device=0 if DEVICE == 'cuda' else -1
        )
        
        logging.info("MiDaS v2.1 model loaded successfully!")
    
    return _midas_v2_model


def predict_midas(image_bgr: np.ndarray) -> np.ndarray:
    """
    Run MiDaS DPT-Large inference on an image.
    
    Args:
        image_bgr: Input image in BGR format (numpy array)
    
    Returns:
        Depth map as numpy array
    """
    model = load_midas_model()
    
    # Convert BGR to RGB PIL Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Run inference
    outputs = model(pil_image)
    
    # Get depth map and convert to numpy
    depth = np.array(outputs["depth"])
    
    # Resize to match input image size if needed
    h, w = image_bgr.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return depth


def predict_midas_v2(image_bgr: np.ndarray) -> np.ndarray:
    """
    Run MiDaS v2.1 Small inference on an image.
    
    Args:
        image_bgr: Input image in BGR format (numpy array)
    
    Returns:
        Depth map as numpy array
    """
    model = load_midas_v2_model()
    
    # Convert BGR to RGB PIL Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Run inference
    outputs = model(pil_image)
    
    # Get depth map and convert to numpy
    depth = np.array(outputs["depth"])
    
    # Resize to match input image size if needed
    h, w = image_bgr.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return depth


def load_midas_v3_model():
    """
    Load the MiDaS v3.1 DPT-BEiT-Large model using HuggingFace transformers pipeline.
    Latest and most accurate MiDaS model.
    """
    global _midas_v3_model
    
    if _midas_v3_model is None:
        logging.info(f"Loading MiDaS v3.1 DPT-BEiT-Large model on {TORCH_DEVICE}...")
        
        _midas_v3_model = hf_pipeline(
            task="depth-estimation",
            model="Intel/dpt-beit-large-512",
            device=0 if DEVICE == 'cuda' else -1
        )
        
        logging.info("MiDaS v3.1 model loaded successfully!")
    
    return _midas_v3_model


def predict_midas_v3(image_bgr: np.ndarray) -> np.ndarray:
    """
    Run MiDaS v3.1 DPT-BEiT-Large inference on an image.
    
    Args:
        image_bgr: Input image in BGR format (numpy array)
    
    Returns:
        Depth map as numpy array
    """
    model = load_midas_v3_model()
    
    # Convert BGR to RGB PIL Image
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Run inference
    outputs = model(pil_image)
    
    # Get depth map and convert to numpy
    depth = np.array(outputs["depth"])
    
    # Resize to match input image size if needed
    h, w = image_bgr.shape[:2]
    if depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return depth


def _prep_da3_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def run_da3_inference(model_key: str, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, str]:
    model = load_da3_model(model_key)
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = _prep_da3_image(rgb)
    prediction = model.inference(
        image=[Image.fromarray(rgb)],
        process_res=None,
        process_res_method="keep",
    )
    depth_map = prediction.depth[0]
    depth_vis = visualize_depth(depth_map, cmap="Spectral")
    processed_rgb = (
        prediction.processed_images[0]
        if getattr(prediction, "processed_images", None) is not None
        else rgb
    )
    processed_rgb = np.clip(processed_rgb, 0, 255).astype(np.uint8)
    target_h, target_w = image.shape[:2]
    if depth_vis.shape[:2] != (target_h, target_w):
        depth_vis = cv2.resize(depth_vis, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    if processed_rgb.shape[:2] != (target_h, target_w):
        processed_rgb = cv2.resize(processed_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    label = DA3_MODEL_SOURCES[model_key]["display_name"]
    info_lines = [
        f"**Model:** `{label}`",
        f"**Repo:** `{DA3_MODEL_SOURCES[model_key]['repo_id']}`",
        f"**Device:** `{str(TORCH_DEVICE)}`",
        f"**Depth shape:** `{tuple(prediction.depth.shape)}`",
    ]
    if getattr(prediction, "extrinsics", None) is not None:
        info_lines.append(f"**Extrinsics shape:** `{prediction.extrinsics.shape}`")
    if getattr(prediction, "intrinsics", None) is not None:
        info_lines.append(f"**Intrinsics shape:** `{prediction.intrinsics.shape}`")
    info_text = "\n".join(info_lines)
    return depth_vis, processed_rgb, info_text, label


def da3_single_inference(image, model: str, progress=gr.Progress()):
    if image is None:
        return None, "❌ Please upload an image."

    if isinstance(image, str):
        np_image = cv2.imread(image)
    elif hasattr(image, "save"):
        np_image = np.array(image)
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    else:
        np_image = np.array(image)
        if len(np_image.shape) == 3 and np_image.shape[2] == 3:
            np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

    if np_image is None:
        raise gr.Error("Invalid image input.")

    key = model[4:] if model.startswith("da3_") else model

    progress(0.1, desc=f"Running {model}")
    depth_vis, processed_rgb, info_text, label = run_da3_inference(key, np_image)
    progress(1.0, desc="Done")
    return (processed_rgb, depth_vis), info_text

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


# Pixel-Perfect Depth setup -------------------------------------------------
set_seed(666)
PPD_DEFAULT_STEPS = 20
PPD_TEMP_ROOT = Path(tempfile.gettempdir()) / "ppd"

_ppd_model: Optional[PixelPerfectDepth] = None
_moge_model: Optional[MoGeModel] = None
_ppd_cmap = matplotlib.colormaps.get_cmap('Spectral')


def load_ppd_model() -> PixelPerfectDepth:
    global _ppd_model
    if _ppd_model is None:
        model = PixelPerfectDepth(sampling_steps=PPD_DEFAULT_STEPS)
        ckpt_path = hf_hub_download(
            repo_id="gangweix/Pixel-Perfect-Depth",
            filename="ppd.pth",
            repo_type="model"
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model = model.to(TORCH_DEVICE).eval()
        _ppd_model = model
    return _ppd_model


def load_moge_model() -> MoGeModel:
    global _moge_model
    if _moge_model is None:
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval()
        model = model.to(TORCH_DEVICE)
        _moge_model = model
    return _moge_model


def _ensure_ppd_temp_dir(session_hash: str) -> Path:
    PPD_TEMP_ROOT.mkdir(exist_ok=True)
    output_path = PPD_TEMP_ROOT / session_hash
    shutil.rmtree(output_path, ignore_errors=True)
    output_path.mkdir(exist_ok=True, parents=True)
    return output_path


def _normalize_depth_to_rgb(depth: np.ndarray) -> np.ndarray:
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5) * 255.0
    depth_vis = depth_vis.astype(np.uint8)
    colored = (_ppd_cmap(depth_vis)[:, :, :3] * 255).astype(np.uint8)
    return colored


def pixel_perfect_depth_inference(
    image_bgr: np.ndarray,
    denoise_steps: int,
    apply_filter: bool,
    request: Optional[gr.Request] = None,
    generate_assets: bool = True
):
    if image_bgr is None:
        return None, None, []

    ppd_model = load_ppd_model()
    moge_model = load_moge_model()

    H, W = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # PixelPerfectDepth expects BGR input similar to original demo
    with torch.no_grad():
        depth_rel, resize_image = ppd_model.infer_image(image_bgr, sampling_steps=denoise_steps)
    resize_H, resize_W = resize_image.shape[:2]

    # MoGe expects RGB tensor
    rgb_tensor = torch.tensor(cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB) / 255, dtype=torch.float32, device=TORCH_DEVICE).permute(2, 0, 1)
    with torch.no_grad():
        metric_depth, mask, intrinsics = moge_model.infer(rgb_tensor)
    metric_depth[~mask] = metric_depth[mask].max()

    # Align relative depth to metric using RANSAC
    metric_depth_aligned = recover_metric_depth_ransac(depth_rel, metric_depth, mask)
    intrinsics[0, 0] *= resize_W
    intrinsics[1, 1] *= resize_H
    intrinsics[0, 2] *= resize_W
    intrinsics[1, 2] *= resize_H

    depth_full = cv2.resize(metric_depth_aligned, (W, H), interpolation=cv2.INTER_LINEAR)
    colored_depth = _normalize_depth_to_rgb(depth_full)

    if not generate_assets:
        return (image_rgb, colored_depth), None, []

    pcd = depth2pcd(
        metric_depth_aligned,
        intrinsics,
        color=cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB),
        input_mask=mask,
        ret_pcd=True
    )
    if apply_filter:
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

    session_hash = getattr(request, "session_hash", "default")
    output_dir = _ensure_ppd_temp_dir(session_hash)

    # Save artifacts
    ply_path = output_dir / "pointcloud.ply"
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32))
    o3d.io.write_point_cloud(ply_path.as_posix(), pcd)
    vertices = np.asarray(pcd.points)
    vertex_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    mesh = trimesh.PointCloud(vertices=vertices, colors=vertex_colors)
    glb_path = output_dir / "pointcloud.glb"
    mesh.export(glb_path.as_posix())

    raw_depth_path = output_dir / "raw_depth.npy"
    np.save(raw_depth_path.as_posix(), depth_full)

    split_region = np.ones((image_bgr.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([image_bgr, split_region, colored_depth[:, :, ::-1]])
    vis_path = output_dir / "image_depth_vis.png"
    cv2.imwrite(vis_path.as_posix(), combined_result)

    available_files = [
        path.as_posix()
        for path in [vis_path, raw_depth_path, ply_path]
        if path.exists()
    ]

    return (image_rgb, colored_depth), glb_path.as_posix(), available_files


def get_model_choices() -> List[Tuple[str, str]]:
    choices = []
    for k, v in V1_MODEL_CONFIGS.items():
        choices.append((v['display_name'], f'v1_{k}'))
    for k, v in V2_MODEL_CONFIGS.items():
        choices.append((v['display_name'], f'v2_{k}'))
    for k, v in DA3_MODEL_SOURCES.items():
        choices.append((v['display_name'], f'da3_{k}'))
    choices.append(("Pixel-Perfect Depth", "ppd"))
    choices.append(("AppleDepthPro", "depthpro"))
    choices.append(("Intel ZoeDepth", "zoedepth"))
    choices.append(("MiDaS v3.0 (DPT-Large)", "midas"))
    choices.append(("MiDaS v3.0 (DPT-Hybrid)", "midas_v2"))
    choices.append(("MiDaS v3.1 (BEiT-Large)", "midas_v3"))
    return choices

def run_model(model_key: str, image: np.ndarray) -> Tuple[np.ndarray, str]:
    if model_key.startswith('v1_'):
        key = model_key[3:]
        model = load_v1_model(key)
        depth = predict_v1(model, image)
        label = V1_MODEL_CONFIGS[key]['display_name']
    elif model_key.startswith('v2_'):
        key = model_key[3:]
        model = load_v2_model(key)
        depth = predict_v2(model, image)
        label = V2_MODEL_CONFIGS[key]['display_name']
    elif model_key.startswith('da3_'):
        key = model_key[4:]
        depth_vis, _, _, label = run_da3_inference(key, image)
        return depth_vis, label
    elif model_key == 'ppd':
        slider_data, _, _ = pixel_perfect_depth_inference(
            image,
            denoise_steps=PPD_DEFAULT_STEPS,
            apply_filter=False,
            request=None,
            generate_assets=False
        )
        depth = slider_data[1]
        label = "Pixel-Perfect Depth"
        return depth, label
    elif model_key == 'depthpro':
        depth = predict_depth_pro(image)
        label = "AppleDepthPro"
        colored = colorize_depth(depth)
        return colored, label
    elif model_key == 'zoedepth':
        depth = predict_zoedepth(image)
        label = "Intel ZoeDepth"
        colored = colorize_depth(depth)
        return colored, label
    elif model_key == 'midas':
        depth = predict_midas(image)
        label = "MiDaS v3.0 (DPT-Large)"
        colored = colorize_depth(depth)
        return colored, label
    elif model_key == 'midas_v2':
        depth = predict_midas_v2(image)
        label = "MiDaS v3.0 (DPT-Hybrid)"
        colored = colorize_depth(depth)
        return colored, label
    elif model_key == 'midas_v3':
        depth = predict_midas_v3(image)
        label = "MiDaS v3.1 (BEiT-Large)"
        colored = colorize_depth(depth)
        return colored, label
    else:
        raise ValueError(f"Unknown model key: {model_key}")
    colored = colorize_depth(depth)
    return colored, label

def compare_models(image, model1: str, model2: str, progress=gr.Progress()) -> Tuple[np.ndarray, str]:
    if image is None:
        return None, "❌ Please upload an image."

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

def slider_compare(image, model1: str, model2: str, progress=gr.Progress()):
    if image is None:
        return None, "❌ Please upload an image."

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

def single_inference(image, model: str, progress=gr.Progress()):
    if image is None:
        return None, "❌ Please upload an image."

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

def get_example_images() -> List[str]:
    import re

    def natural_sort_key(filename):
        """Sort filenames with numbers naturally (demo1, demo2, ..., demo10, demo11)"""
        # Split by numbers and convert numeric parts to integers for proper sorting
        return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', filename)]

    # Try both v1 and v2 examples
    examples = []
    for ex_dir in [
        "assets/examples",
        "Depth-Anything/assets/examples",
        "Depth-Anything-V2/assets/examples",
        "Depth-Anything-3-anysize/assets/examples",
        "Pixel-Perfect-Depth/assets/examples",
    ]:
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
    da3_choices = [(cfg['display_name'], f"da3_{key}") for key, cfg in DA3_MODEL_SOURCES.items()]
    if not da3_choices:
        raise ValueError("Depth Anything v3 models are not configured.")
    da3_default = da3_choices[2][1] if len(da3_choices) > 2 else da3_choices[0][1]
    example_images = get_example_images()
    blocks_kwargs = {"title": "Depth Anything v1 vs v2 Comparison"}
    try:
        if "theme" in inspect.signature(gr.Blocks.__init__).parameters and hasattr(gr, "themes"):
            # Use theme only when the installed gradio version accepts it.
            blocks_kwargs["theme"] = gr.themes.Soft()
    except (ValueError, TypeError):
        pass
    with gr.Blocks(**blocks_kwargs) as app:
        gr.Markdown("""
        # Depth Estimation Comparison
        Compare Depth Anything v1, Depth Anything v2, and Pixel-Perfect Depth side-by-side or with a slider.
        """)
        with gr.Tabs():  # Select the first tab (Slider Comparison) by default
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
                    def slider_example_fn(image):
                        return slider_compare(image, default1, default2)
                    gr.Examples(examples=example_images, inputs=[img_input2], outputs=[slider, slider_status], fn=slider_example_fn)
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
                    def compare_example_fn(image):
                        return compare_models(image, default1, default2)
                    gr.Examples(examples=example_images, inputs=[img_input], outputs=[out_img, out_status], fn=compare_example_fn)
            with gr.Tab("📷 Single Model"):
                with gr.Row():
                    img_input3 = gr.Image(label="Input Image", type="numpy")
                    m_single = gr.Dropdown(choices=model_choices, label="Model", value=default1)
                    btn3 = gr.Button("Run", variant="primary")
                single_slider = gr.ImageSlider(label="Original vs Depth")
                out_single_status = gr.Markdown()
                btn3.click(single_inference, inputs=[img_input3, m_single], outputs=[single_slider, out_single_status], show_progress=True)

                if example_images:
                    def single_example_fn(image):
                        return single_inference(image, default1)
                    gr.Examples(examples=example_images, inputs=[img_input3], outputs=[single_slider, out_single_status], fn=single_example_fn)
        gr.Markdown("""
        ---
        - **v1**: [Depth Anything v1](https://github.com/LiheYoung/Depth-Anything)
        - **v2**: [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2)
        - **v3**: [Depth Anything v3](https://github.com/ByteDance-Seed/Depth-Anything-3) & [Depth-Anything-3-anysize](https://github.com/shriarul5273/Depth-Anything-3-anysize)
        - **PPD**: [Pixel-Perfect Depth](https://github.com/gangweix/pixel-perfect-depth)
        - **DepthPro**: [Apple DepthPro](https://github.com/apple/ml-depth-pro) - Sharp Monocular Metric Depth
        - **ZoeDepth**: [Intel ZoeDepth](https://huggingface.co/Intel/zoedepth-nyu-kitti) - Zero-shot Metric Depth Estimation
        - **MiDaS v3.0**: [Intel DPT-Large](https://huggingface.co/Intel/dpt-large) - Robust Monocular Depth Estimation
        - **MiDaS v3.0**: [Intel DPT-Hybrid](https://huggingface.co/Intel/dpt-hybrid-midas) - Fast Monocular Depth Estimation
        - **MiDaS v3.1**: [Intel DPT-BEiT-Large](https://huggingface.co/Intel/dpt-beit-large-512) - State-of-the-art Depth Estimation
        """)
    return app

def main():
    logging.info("🚀 Starting Depth Anything Comparison App...")
    app = create_app()
    app.queue().launch(show_error=True)

if __name__ == "__main__":
    main()
