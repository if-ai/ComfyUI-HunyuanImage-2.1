# ComfyUI-HunyuanImage-2.1

Advanced ComfyUI custom nodes for HunyuanImage 2.1 - a high-resolution (2K) text-to-image diffusion model with 17B parameters.

## Features

### 🎯 Core Capabilities
- **Text-to-Image Generation**: Generate high-quality 2K images from text prompts
- **Prompt Enhancement**: Automatic prompt improvement using MLLM reprompting
- **Image Refinement**: Enhance existing images with the refiner model
- **Multi-language Support**: Chinese and English text prompts

### ⚡ Advanced Features
- **Dtype Selection**: Choose from fp32, fp16, bf16, fp8 quantization
- **Torch Compile Support**: Optimize with eager, inductor, or max-autotune backends
- **Quantization Methods**: INT8, INT4, NF4, FP8, GPTQ, AWQ support
- **Memory Optimization**: CPU offloading, attention slicing, VAE tiling
- **Attention Optimizations**: Flash Attention, xFormers, PyTorch SDPA

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-HunyuanImage-2.1.git
```

2. Install dependencies:
```bash
cd ComfyUI-HunyuanImage-2.1
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

3. Model Management:

### Automatic Download (Recommended)
Models will be automatically downloaded on first use when you load a workflow. They will be stored in ComfyUI's standard model directories:
- **DiT models**: `ComfyUI/models/diffusion_models/hunyuanimage-v2.1/`
- **VAE models**: `ComfyUI/models/vae/hunyuanimage-v2.1/`
- **Text encoders**: `ComfyUI/models/text_encoders/hunyuanimage-v2.1/`

### Manual Download
If you prefer to download models manually:
```bash
# Main DiT model
huggingface-cli download tencent/HunyuanImage-2.1 --local-dir ComfyUI/models/diffusion_models/hunyuanimage-v2.1

# VAE
huggingface-cli download tencent/HunyuanImage-2.1 --local-dir ComfyUI/models/vae/hunyuanimage-v2.1

# Text encoders
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ComfyUI/models/text_encoders/hunyuanimage-v2.1/llm
huggingface-cli download google/byt5-small --local-dir ComfyUI/models/text_encoders/hunyuanimage-v2.1/byt5-small
modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ComfyUI/models/text_encoders/hunyuanimage-v2.1/Glyph-SDXL-v2
```

## Nodes

### Basic Nodes

#### 🔧 HunyuanImage Model Loader
Loads the HunyuanImage model with dtype and compilation options.
- **Inputs**: Model selection, dtype, device, offloading options, compile settings
- **Outputs**: Pipeline, model info

#### ✨ HunyuanImage Prompt Enhancer
Enhances prompts using the MLLM reprompt model.
- **Inputs**: Pipeline, prompt text
- **Outputs**: Enhanced prompt, original prompt

#### 🎨 HunyuanImage Sampler
Main text-to-image generation node.
- **Inputs**: Pipeline, prompts, dimensions, steps, cfg, seed, enhancement options
- **Outputs**: Latent, image, generation info

#### 🔧 HunyuanImage Refiner
Refines existing images for better quality.
- **Inputs**: Pipeline, image, prompts, refinement settings
- **Outputs**: Refined image

#### 📦 HunyuanImage VAE Encode/Decode
Convert between images and latent representations.

### Advanced Nodes

#### 🚀 HunyuanImage Advanced Loader
Advanced model loading with comprehensive quantization and optimization options.
- **Quantization**: INT8, INT4, NF4, FP8, GPTQ, AWQ methods
- **Optimization**: Flash Attention, xFormers, SDPA, channels-last format
- **Compilation**: TorchScript, TensorRT, ONNX runtime backends
- **Memory**: Sequential CPU offload, attention/VAE slicing and tiling

#### 💾 HunyuanImage Memory Manager
Manage GPU memory and model placement.
- **Actions**: Clear cache, offload to CPU, load to GPU, memory stats

## Usage Examples

### Basic Text-to-Image Workflow
1. Add **HunyuanImage Model Loader** node
2. Connect to **HunyuanImage Sampler** node
3. Set your prompt and generation parameters
4. Connect output to **Preview Image** node

### With Prompt Enhancement
1. Load model with **HunyuanImage Model Loader**
2. Add **HunyuanImage Prompt Enhancer** node
3. Connect enhanced prompt to **HunyuanImage Sampler**
4. View results with **Preview Image**

### Advanced Quantized Workflow
1. Use **HunyuanImage Advanced Loader** with:
   - `dit_quant_method`: "int8"
   - `compile_backend`: "inductor"
   - `compile_mode`: "max-autotune"
2. Connect to sampling nodes as usual
3. Add **HunyuanImage Memory Manager** for memory control

## Performance Tips

### For Best Quality
- Use bf16 or fp16 dtype
- Enable reprompt and refiner
- Set steps to 50+
- Use guidance scale 3.5-7.0

### For Fastest Generation
- Use INT8 quantization
- Enable torch compile with inductor backend
- Disable refiner
- Use distilled model variant
- Enable Flash Attention

### For Low Memory
- Enable all offloading options
- Use INT4 or NF4 quantization
- Enable attention and VAE slicing
- Use sequential CPU offload

## Model Variants

- **hunyuanimage-v2.1**: Full 17B parameter model
- **hunyuanimage-v2.1-distilled**: Faster distilled version

## Supported Resolutions

- Square: 512x512 to 2048x2048
- Landscape: 16:9, 3:2, 4:3 ratios
- Portrait: 9:16, 2:3, 3:4 ratios

## Hardware Requirements

- **Minimum**: 24GB VRAM (with optimizations)
- **Recommended**: 40GB+ VRAM for 2K generation
- **CPU Mode**: Available but very slow

## Troubleshooting

### Out of Memory
1. Enable all offloading options
2. Use INT8 or INT4 quantization
3. Reduce resolution
4. Enable attention slicing

### Slow Generation
1. Enable torch compile
2. Use distilled model
3. Reduce inference steps
4. Enable Flash Attention

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
```

## Credits

Based on HunyuanImage-2.1 by Tencent
- Paper: [HunyuanImage-2.1: An Efficient Diffusion Model for High-Resolution Text-to-Image Generation](https://arxiv.org/abs/2412.00000)
- Original repo: [tencent/HunyuanImage](https://github.com/tencent/HunyuanImage)

## License

This project follows the license of the original HunyuanImage model.