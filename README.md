# ComfyUI-HunyuanImage-2.1

Advanced ComfyUI custom nodes for HunyuanImage 2.1 - a high-resolution (2K) text-to-image diffusion model with 17B parameters.

## Features

### 🎯 Core Capabilities
- **Text-to-Image Generation**: Generate high-quality 2K images from text prompts
- **Prompt Enhancement**: Automatic prompt improvement using MLLM reprompting
- **Image Refinement**: Enhance existing images with the refiner model
- **Multi-language Support**: Chinese and English text prompts

### ⚡ Optimization Features
- **Dtype Selection**: fp32, fp16, bf16, fp8_e5m2, fp8_e4m3fn
- **FP8 Models**: Pre-quantized FP8 models for 40% memory savings
- **Torch Compile Support**: inductor and cudagraphs backends
- **Memory Optimization**: CPU offloading for models and VAE
- **Modular Pipeline**: Separate model loading for better memory management

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/if-ai/ComfyUI-HunyuanImage-2.1.git
```

2. Install dependencies:

### Automatic Installation (Recommended)
```bash
cd ComfyUI-HunyuanImage-2.1
python install.py
```

### Manual Installation
```bash
cd ComfyUI-HunyuanImage-2.1
pip install -r requirements.txt
pip install -e .  # Install hyimage package
pip install flash-attn>=2.7.3 --no-build-isolation  # Optional but recommended
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


Place models in the following structure:

```
ComfyUI/models/
├── diffusion_models/
│   ├── hunyuanimage-v2.1/
│   ├── hunyuanimage-v2.1-distilled/
│   ├── hunyuanimage-v2.1-fp8/
│   ├── hunyuanimage-v2.1-distilled-fp8/
│   ├── hunyuanimage-refiner/
│   └── hunyuanimage-refiner-fp8/
├── vae/
│   └── hunyuanimage-v2.1/
└── text_encoders/
    └── hunyuanimage-v2.1/
```
## Nodes

### Modular Pipeline Nodes (Recommended)

The modular pipeline allows separate loading of models for better memory management. Example workflow: `example_workflows/hunyuan_modular_basic_workflow.json`

#### 🎨 HunyuanImage DiT Loader V2
Load the Diffusion Transformer model.
- **Model variants**: 
  - hunyuanimage-v2.1 (base)
  - hunyuanimage-v2.1-distilled
  - hunyuanimage-v2.1-fp8 (40% smaller)
  - hunyuanimage-v2.1-distilled-fp8
- **Datatypes**: bf16, fp16, fp32, fp8_e5m2, fp8_e4m3fn
- **Features**: CPU offloading, auto-download

#### 📝 HunyuanImage Text Encoder Loader V2
Simple text encoder with optional PromptEnhancer.
- **Modes**: default, promptenhancer_int8
- **Features**: CPU offloading

#### 🔤 HunyuanImage Dual Text Encoder Loader
Dual encoder system (experimental).
- **Components**: RePrompt + ByT5/Glyph
- **Note**: Currently in development

#### 🖼️ HunyuanImage VAE Loader/Encoder/Decoder V2
Separate VAE operations.
- **Loader**: Load VAE model with CPU offload option
- **Encoder**: Convert images to latents
- **Decoder**: Convert latents to images

#### 🎯 HunyuanImage Sampler V2
Generate latents from text (memory efficient).
- **Schedulers**: dpm-solver, euler
- **Resolution**: Up to 2048x2048
- **Output**: Latents only (use VAE Decode for images)

#### ✨ HunyuanImage Refiner Loader/Sampler V2
Optional refinement stage.
- **Models**: 
  - hunyuanimage-refiner
  - hunyuanimage-refiner-fp8
- **Input**: Latents from base sampler
- **Output**: Refined latents

#### ⚡ HunyuanImage Torch Compile
Optimize models with PyTorch compilation.
- **Backends**: inductor, cudagraphs
- **Modes**: default, reduce-overhead, max-autotune


## Usage Examples

### Basic Modular Workflow (Recommended)
1. **Load Models Separately**:
   - **DiT Loader V2**: Select model (base/distilled/fp8)
   - **VAE Loader V2**: Enable CPU offload if needed
   - **Text Encoder Loader V2**: Choose default or promptenhancer_int8

2. **Generate Image**:
   - Connect all to **Sampler V2**
   - Sampler outputs latents only
   - Add **VAE Decode V2** to see images

3. **Optional Refinement**:
   - Add **Refiner Loader V2** (optional FP8)
   - Pass latents through **Refiner Sampler V2**
   - Decode with **VAE Decode V2**

### FP8 Optimized Workflow (24GB VRAM)
1. **DiT Loader V2**: `hunyuanimage-v2.1-distilled-fp8` + `fp8_e4m3fn`
2. **Refiner Loader V2**: `hunyuanimage-refiner-fp8`
3. **Optional**: Add **Torch Compile** node with inductor backend
4. Result: 40% memory savings vs fp16

### With Dual Text Encoder (Experimental)
1. Load **Dual Text Encoder Loader**
2. Connect to **Prompt Encoder Dual**
3. Use conditioning with sampler

## Performance Tips

### For Best Quality
- Use bf16 or fp16 dtype
- Enable PromptEnhancer if available
- Use refiner for final polish
- Set steps to 20-50
- Use guidance scale 3.5-5.0

### For Fastest Generation
- Use FP8 quantized models
- Use distilled variant
- Enable torch compile (inductor)
- Skip refiner
- Reduce steps to 15-20

### For Low Memory (24GB VRAM)
- Use FP8 models (40% memory reduction)
- Enable CPU offloading on all loaders
- Use modular pipeline (latent-only output)
- Decode VAE separately

## Model Variants

### Base Models
- **hunyuanimage-v2.1**: Full model (34.9GB)
- **hunyuanimage-v2.1-distilled**: Faster distilled version
- **hunyuanimage-refiner**: Enhancement model

### FP8 Quantized Models (40% smaller)
- **hunyuanimage-v2.1-fp8**: Base in FP8
- **hunyuanimage-v2.1-distilled-fp8**: Distilled in FP8  
- **hunyuanimage-refiner-fp8**: Refiner in FP8

## Supported Resolutions

- Square: 512x512 to 2048x2048
- Landscape: 16:9, 3:2, 4:3 ratios
- Portrait: 9:16, 2:3, 3:4 ratios

## Hardware Requirements

- **Minimum**: 24GB VRAM (with FP8 models and offloading)
- **Recommended**: 40GB+ VRAM for full fp16 2K generation
- **Tested**: RTX 3090 24GB with FP8 models works well

## Troubleshooting

### Import Error: "HunyuanImagePipelineConfig is not defined"
This error indicates the hyimage package is not properly installed:

1. **Run the installation script:**
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-HunyuanImage-2.1
   python install.py
   ```

2. **Or install manually:**
   ```bash
   pip install -e .  # Install hyimage package in editable mode
   pip install -r requirements.txt
   ```

3. **Restart ComfyUI** after installation

4. **Verify installation:**
   ```bash
   python -c "from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipelineConfig; print('Success!')"
   ```

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

### Model Download Issues
If automatic download fails:
1. Check internet connection
2. Ensure huggingface-cli is installed: `pip install huggingface-hub[cli]`
3. Try manual download with the commands in the Installation section
4. Check disk space (requires ~50GB for all models)

## Credits

Based on HunyuanImage-2.1 by Tencent
- Paper: [HunyuanImage-2.1: An Efficient Diffusion Model for High-Resolution Text-to-Image Generation](https://arxiv.org/abs/2412.00000)
- Original repo: [tencent/HunyuanImage](https://github.com/tencent/HunyuanImage)

## License

This project follows the license of the original HunyuanImage model.