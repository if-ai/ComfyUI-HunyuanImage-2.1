# HunyuanImage ComfyUI Example Workflows

This folder contains example workflows demonstrating how to use the HunyuanImage custom nodes in ComfyUI.

## Available Workflows

### 1. Basic Workflow (`hunyuan_basic_workflow.json`)

A simple workflow for getting started with HunyuanImage generation.

**Features:**
- Standard bf16 precision model loading
- Automatic prompt enhancement
- 1024x1024 image generation
- Optional image refinement
- Basic parameter configuration

**Use Case:** 
- Quick prototyping
- Testing prompts
- Standard quality generation
- Low to medium VRAM GPUs (24GB+)

**Key Settings:**
- Resolution: 1024x1024
- Steps: 50
- Guidance Scale: 3.5
- Precision: bf16
- Compile: Disabled by default

### 2. Advanced Workflow (`hunyuan_advanced_workflow.json`)

A comprehensive workflow showcasing advanced optimization techniques.

**Features:**
- **Quantization:**
  - INT8 quantization on DiT model
  - INT4 quantization on text encoder
  - Configurable per-component quantization
  
- **Torch Compile:**
  - Inductor backend with max-autotune mode
  - Component-specific compilation
  - Dynamic shape support
  
- **Memory Optimization:**
  - Memory manager integration
  - Cache clearing utilities
  - CPU offloading options
  
- **Performance:**
  - Flash Attention enabled
  - SDPA (Scaled Dot Product Attention)
  - Optimized for 2K generation

**Use Case:**
- High-resolution 2K (2048x2048) generation
- Memory-constrained environments
- Production deployments
- Maximum inference speed

**Key Settings:**
- Resolution: 2048x2048
- Steps: 75
- Guidance Scale: 5.0
- Quantization: INT8 (DiT), INT4 (Text Encoder)
- Compile: Inductor with max-autotune

## How to Use

1. **Load Workflow:**
   - Open ComfyUI
   - Click "Load" button
   - Navigate to `ComfyUI/custom_nodes/ComfyUI-HunyuanImage-2.1/examples_workflows/`
   - Select desired workflow JSON file

2. **Customize Parameters:**
   - Adjust prompt text in the Prompt Enhancer node
   - Modify generation parameters (steps, cfg, seed)
   - Toggle features (refiner, reprompt) as needed
   - Change resolution based on your needs

3. **Optimization Tips:**

   **For Speed:**
   - Enable torch compile
   - Use INT8 quantization
   - Disable refiner
   - Lower step count

   **For Quality:**
   - Use bf16/fp16 precision
   - Enable refiner
   - Increase step count
   - Use higher guidance scale

   **For Low Memory:**
   - Enable INT4/NF4 quantization
   - Use CPU offloading
   - Enable attention slicing
   - Reduce resolution

## Workflow Components

### Basic Workflow Structure:
```
Model Loader → Prompt Enhancer → Sampler → Refiner → Output
```

### Advanced Workflow Structure:
```
Advanced Loader → Memory Manager → Prompt Enhancer → Sampler → Refiner → VAE Operations → Output
                ↓
         Quantization & Compile
```

## Performance Benchmarks

| Configuration | Resolution | VRAM Usage | Speed (it/s) | Quality |
|--------------|------------|------------|--------------|---------|
| Basic (bf16) | 1024x1024 | ~16GB | 2.5 | High |
| Basic (bf16) | 2048x2048 | ~40GB | 0.8 | High |
| Advanced (INT8) | 1024x1024 | ~10GB | 4.0 | Good |
| Advanced (INT8) | 2048x2048 | ~24GB | 1.5 | Good |
| Advanced (INT4) | 2048x2048 | ~16GB | 2.0 | Moderate |

*Benchmarks on RTX 4090, actual performance may vary*

## Customization Guide

### Creating Your Own Workflow:

1. **Start with a base workflow** (basic or advanced)
2. **Add/remove nodes** based on your needs:
   - Add multiple samplers for batch generation
   - Chain refiners for iterative improvement
   - Add image preprocessing nodes
3. **Adjust connections** between nodes
4. **Save your workflow** with a descriptive name

### Common Modifications:

- **Batch Processing:** Duplicate sampler nodes with different seeds
- **Style Transfer:** Add ControlNet or IP-Adapter nodes
- **Upscaling:** Connect to ESRGAN or other upscale nodes
- **Post-Processing:** Add color correction or enhancement nodes

## Troubleshooting

### Out of Memory Errors:
- Switch to advanced workflow with quantization
- Enable CPU offloading in advanced loader
- Reduce resolution or batch size
- Clear cache using Memory Manager

### Slow Generation:
- Enable torch compile in loader
- Use INT8 quantization
- Disable refiner if not needed
- Check GPU utilization

### Import Errors:
- Ensure all dependencies are installed
- Check model files are downloaded
- Verify ComfyUI is up to date

## Additional Resources

- [Main README](../README_COMFYUI.md) - Complete documentation
- [Model Cards](https://huggingface.co/tencent/HunyuanImage-2.1) - Model information
- [ComfyUI Docs](https://docs.comfy.org) - ComfyUI documentation

## Contributing

Feel free to submit your own workflow examples via pull request! Please include:
- Workflow JSON file
- Description of use case
- Required settings/dependencies
- Performance metrics if available