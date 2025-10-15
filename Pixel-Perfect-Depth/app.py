import gradio as gr
import cv2
import matplotlib
import numpy as np
import os
import time
from PIL import Image
import torch
import torch.nn.functional as F
import open3d as o3d
import trimesh
import tempfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from gradio_imageslider import ImageSlider
from huggingface_hub import hf_hub_download

from ppd.utils.set_seed import set_seed
from ppd.utils.align_depth_func import recover_metric_depth_ransac
from ppd.utils.depth2pcd import depth2pcd
from moge.model.v2 import MoGeModel 
from ppd.models.ppd import PixelPerfectDepth

try:
    import spaces
    HUGGINFACE_SPACES_INSTALLED = True
except ImportError:
    HUGGINFACE_SPACES_INSTALLED = False

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 100vh;
}
#img-display-output {
    max-height: 100vh;
}
#download {
    height: 62px;
}

#img-display-output .image-slider-image {
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}
"""

set_seed(666)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_steps = 20
model = PixelPerfectDepth(sampling_steps=default_steps)
ckpt_path = hf_hub_download(
    repo_id="gangweix/Pixel-Perfect-Depth",
    filename="ppd.pth",
    repo_type="model"
)
state_dict = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model = model.eval()
model = model.to(DEVICE)

moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").eval()
moge_model = moge_model.to(DEVICE)


def main(share=True):
    print("Initializing Pixel-Perfect Depth Demo...")

    cmap = matplotlib.colormaps.get_cmap('Spectral')

    title = "# Pixel-Perfect Depth"
    description = """Official demo for **Pixel-Perfect Depth**.
    Please refer to our [paper](https://arxiv.org/pdf/2510.07316), [project page](https://pixel-perfect-depth.github.io), and [github](https://github.com/gangweix/pixel-perfect-depth) for more details."""

    @(spaces.GPU if HUGGINFACE_SPACES_INSTALLED else (lambda x: x))
    def predict_depth(image, denoise_steps):
        depth, resize_image = model.infer_image(image, sampling_steps=denoise_steps)
        return depth, resize_image

    @(spaces.GPU if HUGGINFACE_SPACES_INSTALLED else (lambda x: x))
    def predict_moge_depth(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image / 255, dtype=torch.float32, device=DEVICE).permute(2, 0, 1)
        metric_depth, mask, intrinsics = moge_model.infer(image)
        metric_depth[~mask] = metric_depth[mask].max()
        return metric_depth, mask, intrinsics

    def on_submit(image, denoise_steps, apply_filter, request: gr.Request = None):

        H, W = image.shape[:2]
        ppd_depth, resize_image = predict_depth(image[:, :, ::-1], denoise_steps)
        resize_H, resize_W = resize_image.shape[:2]

        # moge provide metric depth and intrinsics
        moge_depth, mask, intrinsics = predict_moge_depth(resize_image)

        # relative depth -> metric depth
        metric_depth = recover_metric_depth_ransac(ppd_depth, moge_depth, mask)
        intrinsics[0, 0] *= resize_W 
        intrinsics[1, 1] *= resize_H
        intrinsics[0, 2] *= resize_W
        intrinsics[1, 2] *= resize_H

        # metric depth -> point cloud
        pcd = depth2pcd(metric_depth, intrinsics, color=cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB), input_mask=mask, ret_pcd=True)
        if apply_filter:
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            pcd = pcd.select_by_index(ind)

        tempdir = Path(tempfile.gettempdir(), 'ppd')
        tempdir.mkdir(exist_ok=True)
        output_path = Path(tempdir, request.session_hash)
        shutil.rmtree(output_path, ignore_errors=True)
        output_path.mkdir(exist_ok=True, parents=True)
        
        ply_path = os.path.join(output_path, 'pointcloud.ply')

        # save pcd to temporary .ply
        pcd.points = o3d.utility.Vector3dVector(
            np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32)
        )
        o3d.io.write_point_cloud(ply_path, pcd)
        vertices = np.asarray(pcd.points)
        vertex_colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        mesh = trimesh.PointCloud(vertices=vertices, colors=vertex_colors)
        glb_path = os.path.join(output_path, 'pointcloud.glb')
        mesh.export(glb_path)


        # save raw depth (npy)
        depth = cv2.resize(ppd_depth, (W, H), interpolation=cv2.INTER_LINEAR)
        raw_depth_path = os.path.join(output_path, 'raw_depth.npy')
        np.save(raw_depth_path, depth)

        depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5) * 255.0
        depth_vis = depth_vis.astype(np.uint8)
        colored_depth = (cmap(depth_vis)[:, :, :3] * 255).astype(np.uint8)

        split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([image[:, :, ::-1], split_region, colored_depth[:, :, ::-1]])

        vis_path = os.path.join(output_path, 'image_depth_vis.png')
        cv2.imwrite(vis_path, combined_result)
    
        file_names = ["image_depth_vis.png", "raw_depth.npy", "pointcloud.ply"]
            
        download_files = [
            (output_path / name).as_posix()
            for name in file_names
            if (output_path / name).exists()
        ]

        return [(image, colored_depth), glb_path, download_files]


    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(title)
        gr.Markdown(description)
        gr.Markdown("### Point Cloud & Depth Prediction demo")

        with gr.Row():
            # Left: input image + settings
            with gr.Column():
                input_image = gr.Image(label="Input Image", image_mode="RGB", type='numpy', elem_id='img-display-input')
                with gr.Accordion(label="Settings", open=False):
                    denoise_steps = gr.Slider(label="Denoising Steps", minimum=1, maximum=100, value=20, step=1)
                    apply_filter = gr.Checkbox(label="Apply filter points", value=True)
                submit_btn = gr.Button(value="Predict")

            # Right: 3D point cloud + depth
            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("3D View"):
                        model_3d = gr.Model3D(display_mode="solid", label="3D Point Map", clear_color=[1,1,1,1], height="60vh")
                    with gr.Tab("Depth"):
                        depth_map = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
                    with gr.Tab("Download"):
                        download_files = gr.File(type='filepath', label="Download Files")

        submit_btn.click(
            fn=lambda: [None, None, None, "", "", ""],
            outputs=[depth_map, model_3d, download_files]
        ).then(
            fn=on_submit,
            inputs=[input_image, denoise_steps, apply_filter],
            outputs=[depth_map, model_3d, download_files]
        )

        example_files = os.listdir('assets/examples')
        example_files.sort()
        example_files = [os.path.join('assets/examples', filename) for filename in example_files]
        examples = gr.Examples(
            examples=example_files, 
            inputs=input_image, 
            outputs=[depth_map, model_3d, download_files], 
            fn=on_submit,
            cache_examples=False
        )
        
    demo.queue().launch(share=share) 

if __name__ == '__main__':
    main(share=True)