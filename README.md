---
title: Depth Anything Compare Demo
emoji: 👀
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.46.0
app_file: app.py
pinned: false
---

# Depth Anything v1 vs v2 Comparison Demo

A comprehensive comparison tool for **Depth Anything v1** and **Depth Anything v2** models, built with Gradio and optimized for HuggingFace Spaces with ZeroGPU support.

## 🚀 Features

### Three Comparison Modes

1. **🎚️ Slider Comparison**: Interactive side-by-side comparison with a draggable slider
2. **🔍 Method Comparison**: Traditional side-by-side view with model labels
3. **🔬 Single Model**: Run individual models for detailed analysis

### Supported Models

#### Depth Anything v1
- **ViT-S (Small)**: Fastest inference, good quality
- **ViT-B (Base)**: Balanced speed and quality
- **ViT-L (Large)**: Best quality, slower inference

#### Depth Anything v2
- **ViT-Small**: Enhanced small model with improved accuracy
- **ViT-Base**: Balanced performance with v2 improvements
- **ViT-Large**: State-of-the-art depth estimation quality

## 🖼️ Example Images

The demo includes 20+ carefully selected example images showcasing various scenarios:
- Indoor and outdoor scenes
- Different lighting conditions
- Various object types and compositions
- Challenging depth estimation scenarios

## 🛠️ Technical Details

### Architecture
- **Framework**: Gradio 4.0+ with modern UI components
- **Backend**: PyTorch with CUDA acceleration
- **Deployment**: ZeroGPU-optimized for HuggingFace Spaces
- **Memory Management**: Automatic model loading/unloading for efficient GPU usage

### ZeroGPU Optimizations
- `@spaces.GPU` decorators for GPU-intensive functions
- Automatic memory cleanup between inferences
- On-demand model loading to prevent OOM errors
- Efficient resource allocation and deallocation

### Depth Visualization
- **Colormap**: Spectral_r colormap for intuitive depth representation
- **Normalization**: Min-max scaling for consistent visualization
- **Resolution**: Maintains original image aspect ratios

## 📦 Installation & Setup

### Local Development

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Depth-Anything-Compare-demo
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download model checkpoints** (for local usage):
```bash
# Depth Anything v1 models are downloaded automatically from HuggingFace Hub
# For v2 models, download checkpoints to Depth-Anything-V2/checkpoints/
```

4. **Run locally**:
```bash
python app_local.py  # For local development
python app.py        # For ZeroGPU deployment
```

### HuggingFace Spaces Deployment

This app is optimized for HuggingFace Spaces with ZeroGPU. Simply:

1. Upload the repository to your HuggingFace Space
2. Set hardware to "ZeroGPU"
3. The app will automatically handle GPU allocation and model loading

## 📁 Project Structure

```
Depth-Anything-Compare-demo/
├── app.py                 # ZeroGPU-optimized main application
├── app_local.py          # Local development version
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── assets/
│   └── examples/        # Example images for testing
├── Depth-Anything/      # Depth Anything v1 implementation
│   ├── depth_anything/
│   │   ├── dpt.py      # v1 model architecture
│   │   └── util/       # v1 utilities and transforms
│   └── torchhub/       # Required dependencies
└── Depth-Anything-V2/   # Depth Anything v2 implementation
    ├── depth_anything_v2/
    │   ├── dpt.py      # v2 model architecture
    │   └── dinov2_layers/ # DINOv2 components
    └── assets/
        └── examples/    # v2-specific examples
```

## 🔧 Configuration

### Model Configuration
Models are configured in the respective config dictionaries:
- `V1_MODEL_CONFIGS`: HuggingFace Hub model identifiers
- `V2_MODEL_CONFIGS`: Local checkpoint paths and architecture parameters

### Environment Variables
- `DEVICE`: Automatically detects CUDA availability
- GPU memory is managed automatically by ZeroGPU

## 📊 Performance

### Inference Times (Approximate)
- **ViT-S models**: ~1-2 seconds
- **ViT-B models**: ~2-4 seconds  
- **ViT-L models**: ~4-8 seconds

*Times vary based on image resolution and GPU availability*

### Memory Usage
- Optimized for ZeroGPU's memory constraints
- Automatic model unloading prevents OOM errors
- Efficient batch processing for multiple comparisons

## 🎯 Usage Examples

### Compare v1 vs v2 Models
1. Upload an image or select from examples
2. Choose models from both v1 and v2 families
3. Click "Compare" or "Slider Compare"
4. Analyze the depth estimation differences

### Analyze Single Model Performance
1. Select "Single Model" tab
2. Choose any available model
3. Upload image and click "Run"
4. Examine detailed depth map output

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional model variants
- New visualization options
- Performance optimizations
- UI/UX enhancements

## 📚 References

- **Depth Anything v1**: [LiheYoung/Depth-Anything](https://github.com/LiheYoung/Depth-Anything)
- **Depth Anything v2**: [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **Original Papers**: 
  - [Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data](https://arxiv.org/abs/2401.10891)
  - [Depth Anything V2: More Efficient, Better Supervised](https://arxiv.org/abs/2406.09414)

## 📄 License

This project combines implementations from:
- Depth Anything v1: MIT License
- Depth Anything v2: Apache 2.0 License
- Demo code: MIT License

Please check individual component licenses for specific terms.

## 🙏 Acknowledgments

- Original Depth Anything authors and contributors
- HuggingFace team for Spaces and ZeroGPU infrastructure
- Gradio team for the excellent UI framework

---

**Note**: This is a demonstration/comparison tool. For production use of the Depth Anything models, please refer to the original repositories and follow their recommended practices.
