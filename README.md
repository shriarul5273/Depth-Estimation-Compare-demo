---
title: Depth Estimation Compare Demo
emoji: 👀
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.46.0
app_file: app.py
pinned: false
---

# Depth Estimation Comparison Demo

A ZeroGPU-friendly Gradio interface for comparing **Depth Anything v1**, **Depth Anything v2**, and **Pixel-Perfect Depth (PPD)** on the same image. Switch between side-by-side layouts, a slider overlay, or single-model inspection to understand how different pipelines perceive scene geometry.

## 🚀 Highlights
- **Three interactive views**: draggable slider, labeled side-by-side comparison, and original vs depth for any single model.
- **Multi-family depth models**: run ViT variants from Depth Anything v1/v2 alongside Pixel-Perfect Depth with MoGe metric alignment.
- **ZeroGPU aware**: on-demand loading, model cache clearing, and torch CUDA cleanup keep GPU usage inside HuggingFace Spaces limits.
- **Curated examples**: reusable demo images sourced from each model family plus local assets to quickly validate behaviour.

## 🔍 Supported Pipelines
- **Depth Anything v1** (`LiheYoung/depth_anything_*`): ViT-S/B/L with fast transformer backbones and colorized outputs via `Spectral_r` colormap.
- **Depth Anything v2** (`Depth-Anything-V2/checkpoints/*.pth`): ViT-Small/Base/Large with HF Hub fallback, configurable feature channels, and improved edge handling.
- **Pixel-Perfect Depth**: Diffusion-based relative depth refined by the **MoGe** metric surface model and RANSAC alignment to recover metric depth; customizable denoising steps.

## 🖥️ App Experience
- **Slider Comparison**: drag between two predictions with automatically labeled overlays.
- **Method Comparison**: view models side-by-side with synchronized layout and captions rendered in OpenCV.
- **Single Model**: inspect the RGB input versus one model output using the Gradio `ImageSlider` component.
- **Example Gallery**: natural-number sorting across `assets/examples`, `Depth-Anything/assets/examples`, `Depth-Anything-V2/assets/examples`, and `Pixel-Perfect-Depth/assets/examples`.

## 📦 Installation & Setup

### Local Development
1. **Clone & enter**:
   ```bash
   git clone <repository-url>
   cd Depth-Estimation-Compare-demo
   ```
2. **Install dependencies** (includes `gradio`, `torch`, `gradio_imageslider`, `open3d`, `scikit-learn`, and MoGe utilities):
   ```bash
   pip install -r requirements.txt
   ```
3. **Model assets**:
   - Depth Anything v1 checkpoints stream automatically from the HuggingFace Hub.
   - Download Depth Anything v2 weights into `Depth-Anything-V2/checkpoints/` if they are not already present (`depth_anything_v2_vits.pth`, `depth_anything_v2_vitb.pth`, `depth_anything_v2_vitl.pth`).
   - Pixel-Perfect Depth pulls the diffusion checkpoint (`ppd.pth`) from `gangweix/Pixel-Perfect-Depth` on first use and loads MoGe weights (`Ruicheng/moge-2-vitl-normal`).
4. **Run the app**:
   ```bash
   python app_local.py   # Local UI with live reload tweaks
   python app.py         # ZeroGPU-ready launch script
   ```

### HuggingFace Spaces (ZeroGPU)
1. Push the repository contents to a Gradio Space.
2. Select the **ZeroGPU** hardware preset.
3. The app will download required checkpoints on demand and aggressively free memory after each inference via `clear_model_cache()`.

## 📁 Project Structure
```
Depth-Estimation-Compare-demo/
├── app.py                 # ZeroGPU deployment entrypoint
├── app_local.py           # Local-friendly launch script
├── requirements.txt       # Python dependencies (Gradio, Torch, PPD stack)
├── assets/
│   └── examples/          # Shared demo imagery
├── Depth-Anything/        # Depth Anything v1 implementation + utilities
├── Depth-Anything-V2/     # Depth Anything v2 implementation & checkpoints
├── Pixel-Perfect-Depth/   # Pixel-Perfect Depth diffusion + MoGe helpers
└── README.md              # You are here
```

## ⚙️ Configuration Notes
- Model dropdown labels come from `V1_MODEL_CONFIGS`, `V2_MODEL_CONFIGS`, and the PPD entry in `app.py`.
- `clear_model_cache()` resets every model and flushes CUDA to respect ZeroGPU constraints.
- Pixel-Perfect Depth inference aligns relative depth to metric scale through `recover_metric_depth_ransac()` for consistent visualization.
- Depth visualizations use a normalized `Spectral_r` colormap; PPD uses a dedicated matplotlib colormap for metric maps.

## 📊 Performance Expectations
- **Depth Anything v1**: ViT-S ~1–2 s, ViT-B ~2–4 s, ViT-L ~4–8 s (image dependent).
- **Depth Anything v2**: similar to v1 with improved sharpness; HF downloads add one-time startup overhead.
- **Pixel-Perfect Depth**: diffusion + metric refinement typically takes longer (10–20 denoise steps) but returns metrically-aligned depth suitable for downstream 3D tasks.

## 🎯 Usage Tips
- Mix-and-match any two models in comparison tabs to highlight qualitative differences.
- Use the Single Model tab to corroborate PPD metric depth versus RGB input.
- Leverage the provided examples to benchmark indoor/outdoor, lighting extremes, and complex geometry scenarios before running custom images.

## 🤝 Contributing
Enhancements are welcome—new model backends, visualization modes, or memory optimizations are especially valuable for ZeroGPU deployments. Please follow the coding style in `app.py` and keep documentation in sync with new capabilities.

## 📚 References
- [Depth Anything v1](https://github.com/LiheYoung/Depth-Anything)
- [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Pixel-Perfect Depth](https://github.com/gangweix/pixel-perfect-depth)
- [MoGe](https://huggingface.co/Ruicheng/moge-2-vitl-normal)

## 📄 License
- Depth Anything v1: MIT License
- Depth Anything v2: Apache 2.0 License
- Pixel-Perfect Depth: see upstream repository for licensing
- Demo scaffolding in this repo: MIT License (follow individual component terms)

---

Built as a hands-on playground for exploring modern monocular depth estimators. Adjust tabs, compare outputs, and plug results into your 3D workflows.
