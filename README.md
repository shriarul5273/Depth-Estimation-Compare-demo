---
title: Depth Estimation Compare Demo
emoji: 👀
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
---

# Depth Estimation Comparison Demo


**🌐 Live demo on Hugging Face Spaces (ZeroGPU):** [Depth-Estimation-Compare-demo](https://huggingface.co/spaces/shriarul5273/Depth-Estimation-Compare-demo)


A Gradio interface for comparing **Depth Anything v1**, **Depth Anything v2**, **Depth Anything v3 (AnySize)**, **Pixel-Perfect Depth (PPD)**, **AppleDepthPro**, **Intel ZoeDepth**, and **MiDaS** on the same image. Switch between side-by-side layouts, a slider overlay, single-model inspection, or a dedicated v3 tab to understand how different pipelines perceive scene geometry. Two entrypoints are provided:

- `app_local.py` – full-featured local runner with minimal memory constraints.
- `app.py` – ZeroGPU-aware build tuned for HuggingFace Spaces with aggressive cache management.

## 🚀 Highlights
- **Four interactive experiences**: draggable slider, labeled side-by-side comparison, original-vs-depth slider, and a Depth Anything v3 tab with RGB vs depth visualization + metadata.
- **Multi-family depth models**: run ViT variants from Depth Anything v1/v2/v3 alongside Pixel-Perfect Depth with MoGe metric alignment, AppleDepthPro for sharp monocular metric depth, Intel ZoeDepth for zero-shot metric depth, and MiDaS for robust relative depth.
- **ZeroGPU aware**: `app.py` performs on-demand loading, cache clearing, and CUDA cleanup to stay within HuggingFace Spaces limits, while `app_local.py` keeps models warm for faster iteration.
- **Curated examples**: reusable demo images sourced from each model family (`assets/examples`, `Depth-Anything*/assets/examples`, `Depth-Anything-3-anysize/assets/examples`, `Pixel-Perfect-Depth/assets/examples`).

## 🔍 Supported Pipelines
- **Depth Anything v1** (`LiheYoung/depth_anything_*`): ViT-S/B/L with fast transformer backbones and colorized outputs via `Spectral_r` colormap.
- **Depth Anything v2** (`Depth-Anything-V2/checkpoints/*.pth` or HF Hub mirrors): ViT-Small/Base/Large with configurable feature channels and improved edge handling.
- **Depth Anything v3 (AnySize)** (`depth-anything/DA3*` via bundled AnySize fork): Nested, giant, large, base, small, mono, and metric variants with native-resolution inference and automatic padding/cropping.
- **Pixel-Perfect Depth**: Diffusion-based relative depth refined by the **MoGe** metric surface model and RANSAC alignment to recover metric depth; customizable denoising steps.
- **AppleDepthPro** (`apple/DepthPro`): Apple's foundation model for zero-shot metric monocular depth estimation producing sharp, high-resolution depth maps with absolute scale in meters and automatic focal length estimation—all in under a second.
- **Intel ZoeDepth** (`Intel/zoedepth-nyu-kitti`): Zero-shot metric depth estimation model fine-tuned on NYU and KITTI datasets; extends the DPT framework for absolute depth with state-of-the-art results via the HuggingFace transformers pipeline.
- **MiDaS v1/v2/v3** (`Intel/dpt-large`, `Intel/dpt-hybrid-midas`, `Intel/dpt-beit-large-512`): Intel's robust monocular depth estimation models; v1 DPT-Large provides high-quality relative depth, v2 DPT-Hybrid offers a faster/lighter alternative, and v3 BEiT-Large delivers the best accuracy—all via the transformers pipeline.

## 🖥️ App Experience
- **Slider Comparison**: drag between any two predictions with automatically labeled overlays.
- **Method Comparison**: view models side-by-side with synchronized layout and captions rendered in OpenCV.
- **Single Model**: inspect the RGB input versus one model output using the Gradio `ImageSlider` component.

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
3. **Install the AnySize fork** (required for Depth Anything v3 tab):
   ```bash
   pip install -e Depth-Anything-3-anysize/.[all]
   ```
4. **Model assets**:
   - Depth Anything v1 checkpoints stream automatically from the HuggingFace Hub.
   - Download Depth Anything v2 weights into `Depth-Anything-V2/checkpoints/` if they are not already present (`depth_anything_v2_vits.pth`, `depth_anything_v2_vitb.pth`, `depth_anything_v2_vitl.pth`).
   - Depth Anything v3 models download via the bundled AnySize API from `depth-anything/*` repositories at inference time; no manual checkpoints required.
   - Pixel-Perfect Depth pulls the diffusion checkpoint (`ppd.pth`) from `gangweix/Pixel-Perfect-Depth` on first use and loads MoGe weights (`Ruicheng/moge-2-vitl-normal`).
   - AppleDepthPro downloads `depth_pro.pt` from `apple/DepthPro` on HuggingFace Hub on first use; a symlink is created in `DepthPro/checkpoints/` for subsequent runs.
   - Intel ZoeDepth downloads automatically via the HuggingFace transformers pipeline from `Intel/zoedepth-nyu-kitti` on first use.
5. **Run the app**:
   ```bash
   python app_local.py   # Local UI with v3 tab and warm caches
   python app.py         # ZeroGPU-ready launch script (loads models on demand)
   ```

### HuggingFace Spaces (ZeroGPU)
1. Push the repository contents to a Gradio Space.
2. Select the **ZeroGPU** hardware preset.
3. The app downloads required checkpoints (Depth Anything v1/v2/v3, PPD, MoGe, AppleDepthPro, ZoeDepth) on demand and aggressively frees memory via `clear_model_cache()` between requests.

## 📁 Project Structure
```
Depth-Estimation-Compare-demo/
├── app.py                        # ZeroGPU deployment entrypoint (includes v3 tab)
├── app_local.py                  # Local-friendly launch script (full feature set)
├── requirements.txt              # Python dependencies (Gradio, Torch, PPD stack)
├── assets/
│   └── examples/                 # Shared demo imagery
├── Depth-Anything/               # Depth Anything v1 implementation + utilities
├── Depth-Anything-V2/            # Depth Anything v2 implementation & checkpoints
├── Depth-Anything-3-anysize/     # Bundled AnySize fork powering Depth Anything v3 tab
│   ├── app.py                    # Standalone AnySize Gradio demo (optional)
│   ├── depth3_anysize.py         # Scripted inference example
│   ├── pyproject.toml            # Editable install metadata
│   ├── requirements.txt          # AnySize-specific dependencies
│   └── src/depth_anything_3/     # AnySize API, configs, and model code
├── DepthPro/                     # Apple DepthPro implementation
│   ├── app.py                    # Standalone DepthPro Gradio demo (optional)
│   ├── depth_pro/                # DepthPro model code and utilities
│   └── checkpoints/              # Symlinked checkpoint from HuggingFace cache
├── Pixel-Perfect-Depth/          # Pixel-Perfect Depth diffusion + MoGe helpers
└── README.md                     # You are here
```

## ⚙️ Configuration Notes
- Model dropdown labels come from `V1_MODEL_CONFIGS`, `V2_MODEL_CONFIGS`, `DA3_MODEL_SOURCES`, plus the PPD, AppleDepthPro, ZoeDepth, and MiDaS entries in both apps.
- `clear_model_cache()` resets every model family (v1/v2/v3/PPD/DepthPro/ZoeDepth/MiDaS) and flushes CUDA to respect ZeroGPU constraints in `app.py`.
- Depth Anything v3 inference leverages the AnySize API (`process_res=None`, `process_res_method="keep"`) to preserve native resolution and returns processed RGB/depth pairs.
- Pixel-Perfect Depth inference aligns relative depth to metric scale through `recover_metric_depth_ransac()` for consistent visualization.
- AppleDepthPro returns metric depth in meters with automatic focal length estimation; no camera intrinsics required.
- Intel ZoeDepth uses the HuggingFace transformers depth-estimation pipeline for easy integration and consistent inference.
- Depth visualizations use a normalized `Spectral_r` colormap; PPD uses a dedicated matplotlib colormap for metric maps.

## 📊 Performance Expectations
- **Depth Anything v1**: ViT-S ~1–2 s, ViT-B ~2–4 s, ViT-L ~4–8 s (image dependent).
- **Depth Anything v2**: similar to v1 with improved sharpness; HF downloads add one-time startup overhead.
- **Depth Anything v3**: nested/giant models are heavier (expect longer cold starts), while base/small options are close to v2 latency when running at native resolution.
- **Pixel-Perfect Depth**: diffusion + metric refinement typically takes longer (10–20 denoise steps) but returns metrically-aligned depth suitable for downstream 3D tasks.
- **AppleDepthPro**: produces 2.25 megapixel depth maps in ~0.3 s on GPU; delivers sharp boundaries and metric scale without requiring camera intrinsics.
- **Intel ZoeDepth**: fast inference via the transformers pipeline; produces metric depth maps fine-tuned for indoor (NYU) and outdoor (KITTI) scenes.
- **MiDaS**: v1 (DPT-Large) offers high-quality relative depth; v2 (DPT-Hybrid) is faster for real-time use; v3 (BEiT-Large) provides the best accuracy at higher compute cost.

## 🎯 Usage Tips
- Mix-and-match any two models in comparison tabs to highlight qualitative differences.
- Use the Single Model tab to corroborate PPD metric depth versus RGB input.
- Leverage the provided examples to benchmark indoor/outdoor, lighting extremes, and complex geometry scenarios before running custom images.

## 🤝 Contributing
Enhancements are welcome—new model backends, visualization modes, or memory optimizations are especially valuable for ZeroGPU deployments. Please follow the coding style in `app.py` and keep documentation in sync with new capabilities.

## 📚 References
- [Depth Anything v1](https://github.com/LiheYoung/Depth-Anything)
- [Depth Anything v2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Depth Anything 3 AnySize Fork](https://github.com/ByteDance-Seed/Depth-Anything-3) (see bundled `Depth-Anything-3-anysize` directory)
- [Pixel-Perfect Depth](https://github.com/gangweix/pixel-perfect-depth)
- [AppleDepthPro](https://github.com/apple/ml-depth-pro) - Sharp Monocular Metric Depth in Less Than a Second
- [Intel ZoeDepth](https://huggingface.co/Intel/zoedepth-nyu-kitti) - Zero-shot Transfer by Combining Relative and Metric Depth
- [MiDaS](https://github.com/isl-org/MiDaS) - Robust Monocular Depth Estimation

## 📄 License
- Depth Anything v1: MIT License
- Depth Anything v2: Apache 2.0 License
- Pixel-Perfect Depth: see upstream repository for licensing
- AppleDepthPro: see [Apple ML License](https://github.com/apple/ml-depth-pro/blob/main/LICENSE)
- Intel ZoeDepth: MIT License
- MiDaS: MIT License
- Demo scaffolding in this repo: MIT License (follow individual component terms)

---

Built as a hands-on playground for exploring modern monocular depth estimators. Adjust tabs, compare outputs, and plug results into your 3D workflows.