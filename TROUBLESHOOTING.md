# Troubleshooting Guide

## FP8 Model Download Issues

If the FP8 model download gets stuck at 94% or fails to complete:

### Option 1: Manual Download with huggingface-cli

```bash
# For distilled FP8 model
huggingface-cli download drbaph/HunyuanImage-2.1_fp8 hunyuanimage2.1-distilled_fp8_e4m3fn.safetensors \
  --local-dir ComfyUI/models/diffusion_models/hunyuanimage-v2.1-distilled-fp8

# For refiner FP8 model  
huggingface-cli download drbaph/HunyuanImage-2.1_fp8 hunyuanimage-refiner_fp8_e4m3fn.safetensors \
  --local-dir ComfyUI/models/diffusion_models/hunyuanimage-refiner-fp8

# For base FP8 model
huggingface-cli download drbaph/HunyuanImage-2.1_fp8 hunyuanimage2.1_fp8_e4m3fn.safetensors \
  --local-dir ComfyUI/models/diffusion_models/hunyuanimage-v2.1-fp8
```

### Option 2: Direct Download Links

Download these files manually and place them in the specified directories:

**Distilled FP8 (17.5GB):**
- URL: https://huggingface.co/drbaph/HunyuanImage-2.1_fp8/resolve/main/hunyuanimage2.1-distilled_fp8_e4m3fn.safetensors
- Place in: `ComfyUI/models/diffusion_models/hunyuanimage-v2.1-distilled-fp8/`

**Refiner FP8 (10.9GB):**
- URL: https://huggingface.co/drbaph/HunyuanImage-2.1_fp8/resolve/main/hunyuanimage-refiner_fp8_e4m3fn.safetensors
- Place in: `ComfyUI/models/diffusion_models/hunyuanimage-refiner-fp8/`

**Base FP8 (20.4GB):**
- URL: https://huggingface.co/drbaph/HunyuanImage-2.1_fp8/resolve/main/hunyuanimage2.1_fp8_e4m3fn.safetensors
- Place in: `ComfyUI/models/diffusion_models/hunyuanimage-v2.1-fp8/`

### Option 3: Use the Download Script

```bash
cd ComfyUI/custom_nodes/ComfyUI-HunyuanImage-2.1
python download_fp8.py
```

## VAE Files in Wrong Location

If you get "config.json not found" errors:

### Check VAE file location:
```bash
# Windows PowerShell
tree /f D:\ComfyUI\models\vae\hunyuanimage-v2.1

# Linux/Mac
find ComfyUI/models/vae/hunyuanimage-v2.1 -type f
```

### Correct structure should be:
```
ComfyUI/models/vae/hunyuanimage-v2.1/
└── vae_2_1/
    ├── config.json
    └── pytorch_model.ckpt
```

### Fix incorrect structure:
```bash
# If files are in root directory, move them:
cd ComfyUI/models/vae/hunyuanimage-v2.1
mkdir -p vae_2_1
mv config.json pytorch_model.ckpt vae_2_1/

# If files are in nested vae/vae_2_1, move them up:
cd ComfyUI/models/vae/hunyuanimage-v2.1
mv vae/vae_2_1/* vae_2_1/
rmdir vae/vae_2_1 vae
```

## Permission Denied Errors

If you see permission errors with `.claude` directory:

```bash
# Remove the problematic directory
rm -rf ComfyUI/custom_nodes/ComfyUI-HunyuanImage-2.1/.claude
```

## ComfyUI Hot Reload Issues

If hot reload causes problems during download:

1. **Disable hot reload temporarily:**
   - Remove or disable ComfyUI-HotReloadHack custom node
   - Or stop ComfyUI while downloading models

2. **Restart ComfyUI after model downloads:**
   ```bash
   # Kill ComfyUI process
   pkill -f "python.*main.py"
   
   # Start fresh
   cd ComfyUI
   python main.py --listen
   ```

## Memory Issues with FP8 Models

If you run out of memory even with FP8 models:

1. **Enable CPU offloading** on all loader nodes
2. **Use modular pipeline** - samplers output latents only, decode separately
3. **Reduce resolution** - start with 1024x1024 instead of 2048x2048
4. **Use distilled models** - they're faster and use less memory

## Import Errors

If you get "hyimage module not found" errors:

```bash
cd ComfyUI/custom_nodes/ComfyUI-HunyuanImage-2.1
pip install -e .
```

## Workflow Loading Issues

If example workflows don't load:

1. Ensure all nodes are properly installed
2. Check that models are downloaded
3. Try the basic workflow first: `example_workflows/hunyuan_modular_basic_workflow.json`