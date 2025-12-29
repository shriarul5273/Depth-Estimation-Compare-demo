"""
Depth Estimation using Apple's DepthPro Model - Gradio App

Depth Pro: Sharp Monocular Metric Depth in Less Than a Second
This Gradio app provides an interactive web interface for depth estimation.
"""

import os
import sys
import tempfile

# Add depth_pro from root folder to Python path
DEPTH_PRO_PATH = os.path.dirname(os.path.abspath(__file__))
if DEPTH_PRO_PATH not in sys.path:
    sys.path.insert(0, DEPTH_PRO_PATH)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Gradio
import torch
import gradio as gr
from huggingface_hub import hf_hub_download

# Import depth_pro module (now from ml-depth-pro/src)
import depth_pro


# Global model cache
_model_cache = {
    "model": None,
    "transform": None,
    "device": None
}

# Path to checkpoint - will be downloaded from HuggingFace
CHECKPOINT_PATH = None


def download_checkpoint():
    """
    Download the DepthPro checkpoint from HuggingFace.
    
    Returns:
        str: Path to the downloaded checkpoint file
    """
    global CHECKPOINT_PATH
    
    if CHECKPOINT_PATH is None or not os.path.exists(CHECKPOINT_PATH):
        print("Downloading DepthPro checkpoint from HuggingFace...")
        CHECKPOINT_PATH = hf_hub_download(
            repo_id="apple/DepthPro",
            filename="depth_pro.pt",
            repo_type="model"
        )
        print(f"Checkpoint downloaded to: {CHECKPOINT_PATH}")
    
    return CHECKPOINT_PATH


def load_model():
    """
    Load the DepthPro model and preprocessing transforms.
    Uses caching to avoid reloading the model on every inference.
    Loads weights from HuggingFace apple/DepthPro depth_pro.pt
    
    Returns:
        model: Loaded DepthPro model
        transform: Preprocessing transform
        device: Device model is loaded on
    """
    if _model_cache["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading DepthPro model on {device}...")
        
        # Download checkpoint from HuggingFace
        checkpoint_path = download_checkpoint()
        
        # Create the checkpoints directory and symlink to downloaded file
        local_checkpoint_dir = "./checkpoints"
        local_checkpoint_path = os.path.join(local_checkpoint_dir, "depth_pro.pt")
        
        if not os.path.exists(local_checkpoint_path):
            os.makedirs(local_checkpoint_dir, exist_ok=True)
            # Create symlink to the HuggingFace cached file
            os.symlink(checkpoint_path, local_checkpoint_path)
            print(f"Created symlink: {local_checkpoint_path} -> {checkpoint_path}")
        
        # Create model and transforms - it will now find the checkpoint
        model, transform = depth_pro.create_model_and_transforms(device=device)
        
        model = model.to(device)
        model.eval()
        
        # Cache the model
        _model_cache["model"] = model
        _model_cache["transform"] = transform
        _model_cache["device"] = device
        
        print("Model loaded successfully!")
    
    return _model_cache["model"], _model_cache["transform"], _model_cache["device"]


def create_depth_colormap(depth: np.ndarray, colormap: str = "turbo") -> Image.Image:
    """
    Create a colorized depth map image.
    
    Args:
        depth: Depth map array in meters
        colormap: Matplotlib colormap name
    
    Returns:
        PIL Image with colorized depth
    """
    # Normalize depth to 0-1 range
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)
    
    # Convert to uint8 RGB
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    
    return Image.fromarray(depth_colored)


def create_grayscale_depth(depth: np.ndarray) -> Image.Image:
    """
    Create a grayscale depth map image.
    
    Args:
        depth: Depth map array in meters
    
    Returns:
        PIL Image with grayscale depth
    """
    # Normalize depth to 0-255 range
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_gray = (depth_normalized * 255).astype(np.uint8)
    
    return Image.fromarray(depth_gray)


def estimate_depth(input_image: Image.Image, colormap: str = "turbo"):
    """
    Estimate depth from an input image using DepthPro.
    
    Args:
        input_image: PIL Image input
        colormap: Colormap for visualization
    
    Returns:
        Tuple of (colored_depth, grayscale_depth, info_text)
    """
    if input_image is None:
        return None, None, "Please upload an image."
    
    try:
        # Load model
        model, transform, device = load_model()
        
        # Save image temporarily for depth_pro.load_rgb
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            temp_path = tmp_file.name
            input_image.save(temp_path)
        
        try:
            # Load and preprocess the image
            image, _, f_px = depth_pro.load_rgb(temp_path)
            
            # Apply transform
            image = transform(image)
            
            # Move to device
            image = image.to(device)
            
            # Run inference
            with torch.no_grad():
                prediction = model.infer(image, f_px=f_px)
            
            # Extract results
            depth = prediction["depth"].cpu().numpy()  # Depth in meters
            focallength_px = prediction["focallength_px"]  # Focal length in pixels
            
            # Handle tensor focal length
            if hasattr(focallength_px, 'item'):
                focallength_px = focallength_px.item()
            
        finally:
            # Clean up temp file
            os.unlink(temp_path)
        
        # Create visualizations
        colored_depth = create_depth_colormap(depth, colormap)
        grayscale_depth = create_grayscale_depth(depth)
        
        # Create info text
        info_text = f"""### Depth Estimation Results

| Metric | Value |
|--------|-------|
| **Minimum Depth** | {depth.min():.3f} m |
| **Maximum Depth** | {depth.max():.3f} m |
| **Mean Depth** | {depth.mean():.3f} m |
| **Estimated Focal Length** | {focallength_px:.2f} px |
| **Depth Map Resolution** | {depth.shape[1]} x {depth.shape[0]} |
"""
        
        return colored_depth, grayscale_depth, info_text
        
    except Exception as e:
        error_msg = f"Error during depth estimation: {str(e)}"
        print(error_msg)
        return None, None, error_msg


def create_comparison_figure(
    original: Image.Image,
    depth_colored: Image.Image,
    colormap: str = "turbo"
) -> Image.Image:
    """
    Create a side-by-side comparison figure.
    
    Args:
        original: Original input image
        depth_colored: Colorized depth map
        colormap: Colormap used (for colorbar)
    
    Returns:
        PIL Image with comparison
    """
    if original is None or depth_colored is None:
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis("off")
    
    # Depth map
    axes[1].imshow(depth_colored)
    axes[1].set_title("Depth Map", fontsize=14, fontweight='bold')
    axes[1].axis("off")
    
    plt.tight_layout()
    
    # Save to buffer
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        temp_path = tmp_file.name
        plt.savefig(temp_path, dpi=150, bbox_inches="tight", facecolor='white')
        plt.close(fig)
        comparison_image = Image.open(temp_path)
        comparison_image.load()
        os.unlink(temp_path)
    
    return comparison_image


def process_image(input_image: Image.Image, colormap: str):
    """
    Main processing function for Gradio interface.
    
    Args:
        input_image: Input PIL Image
        colormap: Selected colormap
    
    Returns:
        Tuple of outputs for Gradio
    """
    if input_image is None:
        return None, None, None, "Please upload an image to begin."
    
    # Estimate depth
    colored_depth, grayscale_depth, info_text = estimate_depth(input_image, colormap)
    
    # Create comparison
    comparison = create_comparison_figure(input_image, colored_depth, colormap)
    
    return colored_depth, grayscale_depth, comparison, info_text


# Available colormaps for depth visualization
COLORMAPS = [
    "turbo",
    "viridis", 
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "jet",
    "hot",
    "cool",
    "rainbow"
]


def create_gradio_app():
    """Create and configure the Gradio application."""
    
    with gr.Blocks(
        title="DepthPro - Monocular Depth Estimation"
    ) as app:
        # Header
        gr.Markdown("""
        # 🔬 Depth Pro: Monocular Depth Estimation

        **Sharp Monocular Metric Depth in Less Than a Second**

        Upload an image to estimate its depth map using Apple's DepthPro model. 
        The model produces high-resolution depth maps with metric scale (in meters) 
        without requiring camera intrinsics.

        ---
        """)

        with gr.Row():
            # Left column - Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    height=350
                )

                colormap = gr.Dropdown(
                    choices=COLORMAPS,
                    value="turbo",
                    label="Depth Colormap",
                    info="Select colormap for depth visualization"
                )

                submit_btn = gr.Button(
                    "🚀 Estimate Depth",
                    variant="primary",
                    size="lg"
                )

                gr.Markdown("""
                **Tips:**
                - Works best with natural outdoor/indoor scenes
                - Higher resolution images produce more detailed depth maps
                - Processing time: ~0.3s on GPU, longer on CPU
                """)

            # Right column - Outputs
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")

                with gr.Tabs():
                    with gr.TabItem("🎨 Colored Depth"):
                        colored_output = gr.Image(
                            label="Colored Depth Map",
                            type="pil",
                            height=400
                        )

                    with gr.TabItem("⬛ Grayscale Depth"):
                        grayscale_output = gr.Image(
                            label="Grayscale Depth Map",
                            type="pil",
                            height=400
                        )

                    with gr.TabItem("📊 Comparison"):
                        comparison_output = gr.Image(
                            label="Side-by-Side Comparison",
                            type="pil",
                            height=400
                        )

                info_output = gr.Markdown(
                    value="Upload an image and click 'Estimate Depth' to see results.",
                    label="Depth Statistics"
                )

        # Footer
        gr.Markdown("""
        ---

        ### About DepthPro

        Depth Pro is a foundation model for zero-shot metric monocular depth estimation developed by Apple.

        **Key Features:**
        - 🎯 **Metric depth**: Absolute scale in meters, no camera intrinsics needed
        - ⚡ **Fast**: 2.25 megapixel depth map in 0.3 seconds
        - 🔍 **Sharp**: High-frequency details and precise boundaries
        - 📐 **Focal length estimation**: Automatic from single image

        **Citation:**
        ```
        @article{Bochkovskii2024:arxiv,
          author = {Aleksei Bochkovskii et al.},
          title = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
          journal = {arXiv},
          year = {2024}
        }
        ```
        """)

        # Event handlers
        submit_btn.click(
            fn=process_image,
            inputs=[input_image, colormap],
            outputs=[colored_output, grayscale_output, comparison_output, info_output]
        )

        # Also trigger on image upload
        input_image.change(
            fn=process_image,
            inputs=[input_image, colormap],
            outputs=[colored_output, grayscale_output, comparison_output, info_output]
        )

        # Update on colormap change (if image exists)
        colormap.change(
            fn=process_image,
            inputs=[input_image, colormap],
            outputs=[colored_output, grayscale_output, comparison_output, info_output]
        )
    
    return app


def main():
    """Launch the Gradio application."""
    print("Starting DepthPro Gradio App...")
    print("Loading model on startup (this may take a moment)...")
    
    # Pre-load model
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")
        print("Model will be loaded on first inference.")
    
    # Create and launch app
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {max-width: 1200px !important}
        .output-image {min-height: 300px}
        """
    )


if __name__ == "__main__":
    main()
