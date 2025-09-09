"""
ComfyUI nodes for HunyuanImage 2.1
Advanced text-to-image generation with dtype selection, torch compile, and prompt enhancement
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from PIL import Image
import folder_paths
import comfy.utils

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline, HunyuanImagePipelineConfig
    from hyimage.diffusion.pipelines.hunyuanimage_refiner_pipeline import HunyuanImageRefinerPipeline
    from hyimage.common.constants import PRECISION_TO_TYPE
    from hyimage.models.model_zoo import HUNYUANIMAGE_REPROMPT
except ImportError as e:
    print(f"[HunyuanImage] Error importing HunyuanImage modules: {e}")
    print("[HunyuanImage] Please ensure all dependencies are installed with: pip install -r requirements.txt")
    HunyuanImagePipeline = None
    HunyuanImagePipelineConfig = None
    HunyuanImageRefinerPipeline = None
    PRECISION_TO_TYPE = None
    HUNYUANIMAGE_REPROMPT = None

try:
    from .model_manager import model_manager
except ImportError as e:
    print(f"[HunyuanImage] Error importing model_manager: {e}")
    model_manager = None

# Define dtype options
DTYPE_OPTIONS = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e4m3fn": torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else None,
    "fp8_e5m2": torch.float8_e5m2 if hasattr(torch, 'float8_e5m2') else None,
}

# Torch compile backends
COMPILE_BACKENDS = ["eager", "inductor", "aot_eager", "cudagraphs", "none"]

# Torch compile modes
COMPILE_MODES = ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs", "none"]


class HunyuanImageModelLoader:
    """Load HunyuanImage model with dtype selection and torch compile options"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["hunyuanimage-v2.1", "hunyuanimage-v2.1-distilled"], {"default": "hunyuanimage-v2.1"}),
                "dtype": (list(DTYPE_OPTIONS.keys()), {"default": "bf16"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "auto_download": ("BOOLEAN", {"default": True}),
                "enable_dit_offloading": ("BOOLEAN", {"default": True}),
                "enable_reprompt_offloading": ("BOOLEAN", {"default": True}),
                "enable_refiner_offloading": ("BOOLEAN", {"default": True}),
                "compile_backend": (COMPILE_BACKENDS, {"default": "inductor"}),
                "compile_mode": (COMPILE_MODES, {"default": "max-autotune"}),
                "compile_dit": ("BOOLEAN", {"default": False}),
                "compile_vae": ("BOOLEAN", {"default": False}),
                "compile_text_encoder": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_PIPELINE", "HUNYUAN_MODEL_INFO")
    RETURN_NAMES = ("pipeline", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "HunyuanImage"
    
    def load_model(self, model_name, dtype, device, auto_download, enable_dit_offloading, 
                  enable_reprompt_offloading, enable_refiner_offloading,
                  compile_backend, compile_mode, compile_dit, compile_vae, compile_text_encoder):
        """Load the HunyuanImage model with specified configuration"""
        
        # Check if required modules are imported
        if HunyuanImagePipelineConfig is None or HunyuanImagePipeline is None:
            raise ImportError(
                "[HunyuanImage] Required modules not loaded. Please ensure:\n"
                "1. All dependencies are installed: pip install -r requirements.txt\n"
                "2. The hyimage package is properly installed\n"
                "3. You may need to restart ComfyUI after installation"
            )
        
        # Ensure models are available
        if model_manager:
            print(f"[HunyuanImage] Checking models for {model_name}")
            if not model_manager.ensure_models(model_name, auto_download=auto_download):
                raise RuntimeError(f"Failed to load models for {model_name}. Enable auto_download or download manually.")
            
            # Set model root to ComfyUI paths
            model_root = model_manager.get_model_root(model_name)
            os.environ['HUNYUANIMAGE_V2_1_MODEL_ROOT'] = model_root
            print(f"[HunyuanImage] Using model root: {model_root}")
        
        # Get dtype
        torch_dtype = DTYPE_OPTIONS.get(dtype)
        if torch_dtype is None:
            print(f"[HunyuanImage] Warning: dtype {dtype} not available, falling back to bf16")
            torch_dtype = torch.bfloat16
        
        # Prepare config
        use_distilled = "distilled" in model_name
        dtype_str = "bf16" if torch_dtype == torch.bfloat16 else "fp16" if torch_dtype == torch.float16 else "fp32"
        
        config = HunyuanImagePipelineConfig.create_default(
            version="v2.1",
            use_distilled=use_distilled,
            torch_dtype=dtype_str,
            device=device,
            enable_dit_offloading=enable_dit_offloading,
            enable_reprompt_model_offloading=enable_reprompt_offloading,
            enable_refiner_offloading=enable_refiner_offloading
        )
        
        # Load pipeline
        print(f"[HunyuanImage] Loading {model_name} with dtype={dtype}, device={device}")
        pipeline = HunyuanImagePipeline.from_pretrained(
            model_name=model_name,
            config=config,
            device=device
        )
        
        # Apply torch compile if requested
        if compile_backend != "none" and compile_mode != "none":
            compile_config = {
                "backend": compile_backend,
                "mode": compile_mode,
                "fullgraph": False,
                "dynamic": True
            }
            
            if compile_dit and hasattr(pipeline, 'dit') and pipeline.dit is not None:
                print(f"[HunyuanImage] Compiling DiT with backend={compile_backend}, mode={compile_mode}")
                pipeline.dit = torch.compile(pipeline.dit, **compile_config)
            
            if compile_vae and hasattr(pipeline, 'vae') and pipeline.vae is not None:
                print(f"[HunyuanImage] Compiling VAE with backend={compile_backend}, mode={compile_mode}")
                pipeline.vae.encoder = torch.compile(pipeline.vae.encoder, **compile_config)
                pipeline.vae.decoder = torch.compile(pipeline.vae.decoder, **compile_config)
            
            if compile_text_encoder and hasattr(pipeline, 'text_encoder') and pipeline.text_encoder is not None:
                print(f"[HunyuanImage] Compiling text encoder with backend={compile_backend}, mode={compile_mode}")
                if hasattr(pipeline.text_encoder, 'model'):
                    pipeline.text_encoder.model = torch.compile(pipeline.text_encoder.model, **compile_config)
        
        # Prepare model info
        model_info = {
            "model_name": model_name,
            "dtype": dtype,
            "device": device,
            "use_distilled": use_distilled,
            "compile_backend": compile_backend,
            "compile_mode": compile_mode,
            "dit_compiled": compile_dit and compile_backend != "none",
            "vae_compiled": compile_vae and compile_backend != "none",
            "text_encoder_compiled": compile_text_encoder and compile_backend != "none"
        }
        
        print(f"[HunyuanImage] Model loaded successfully")
        return (pipeline, model_info)


class HunyuanImagePromptEnhancer:
    """Enhance prompts using the HunyuanImage reprompt model"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HUNYUAN_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": "A cat sitting on a table"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "original_prompt")
    FUNCTION = "enhance_prompt"
    CATEGORY = "HunyuanImage"
    
    def enhance_prompt(self, pipeline, prompt):
        """Enhance the prompt using the reprompt model"""
        
        try:
            # Move models to appropriate devices
            if hasattr(pipeline, 'dit') and pipeline.dit is not None:
                pipeline.to('cpu')
            
            # Use reprompt model
            print(f"[HunyuanImage] Enhancing prompt: {prompt[:50]}...")
            enhanced_prompt = pipeline.reprompt_model.predict(prompt)
            print(f"[HunyuanImage] Enhanced prompt generated")
            
            return (enhanced_prompt, prompt)
            
        except Exception as e:
            print(f"[HunyuanImage] Error enhancing prompt: {e}")
            return (prompt, prompt)


class HunyuanImageSampler:
    """Main text-to-image generation node for HunyuanImage"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HUNYUAN_PIPELINE",),
                "positive": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 50, "min": 10, "max": 100}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "use_reprompt": ("BOOLEAN", {"default": True}),
                "use_refiner": ("BOOLEAN", {"default": False}),
                "refiner_steps": ("INT", {"default": 4, "min": 1, "max": 20}),
                "shift": ("INT", {"default": 4, "min": 0, "max": 10}),
            },
            "optional": {
                "model_info": ("HUNYUAN_MODEL_INFO",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "IMAGE", "STRING")
    RETURN_NAMES = ("latent", "image", "info")
    FUNCTION = "generate"
    CATEGORY = "HunyuanImage"
    
    def generate(self, pipeline, positive, negative, width, height, steps, cfg, seed, 
                use_reprompt, use_refiner, refiner_steps, shift, model_info=None):
        """Generate image using HunyuanImage pipeline"""
        
        try:
            # Clear cache before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Handle seed
            if seed == -1:
                import random
                seed = random.randint(100000, 999999)
            
            # Move pipeline to appropriate device
            device = model_info.get("device", "cuda") if model_info else "cuda"
            if hasattr(pipeline, 'refiner_pipeline') and pipeline.refiner_pipeline is not None:
                pipeline.refiner_pipeline.to('cpu')
            pipeline.to(device)
            
            # Generate image
            print(f"[HunyuanImage] Generating image: {width}x{height}, steps={steps}, cfg={cfg}, seed={seed}")
            print(f"[HunyuanImage] Options: reprompt={use_reprompt}, refiner={use_refiner}")
            
            # Call pipeline
            image = pipeline(
                prompt=positive,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                seed=seed,
                use_reprompt=use_reprompt,
                use_refiner=use_refiner,
                shift=shift
            )
            
            # Convert PIL image to tensor
            if isinstance(image, Image.Image):
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
            else:
                image_tensor = image
            
            # Create latent representation (placeholder - actual latent would come from VAE encoder)
            latent = {"samples": torch.zeros((1, 4, height // 8, width // 8))}
            
            # Create info string
            info = json.dumps({
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "width": width,
                "height": height,
                "reprompt": use_reprompt,
                "refiner": use_refiner,
                "model": model_info.get("model_name", "unknown") if model_info else "unknown"
            })
            
            print(f"[HunyuanImage] Image generated successfully")
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (latent, image_tensor, info)
            
        except Exception as e:
            print(f"[HunyuanImage] Error generating image: {e}")
            import traceback
            traceback.print_exc()
            # Return empty tensors on error
            empty_latent = {"samples": torch.zeros((1, 4, height // 8, width // 8))}
            empty_image = torch.zeros((1, height, width, 3))
            error_info = json.dumps({"error": str(e)})
            return (empty_latent, empty_image, error_info)


class HunyuanImageRefiner:
    """Refine existing images using HunyuanImage refiner"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HUNYUAN_PIPELINE",),
                "image": ("IMAGE",),
                "positive": ("STRING", {"multiline": True, "default": "Make the image more detailed and high quality"}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 20}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "shift": ("INT", {"default": 5, "min": 0, "max": 10}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("refined_image",)
    FUNCTION = "refine"
    CATEGORY = "HunyuanImage"
    
    def refine(self, pipeline, image, positive, negative, steps, cfg, seed, shift):
        """Refine an image using the refiner pipeline"""
        
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Convert tensor to PIL image
            if isinstance(image, torch.Tensor):
                # Assume image is in [B, H, W, C] format with values 0-1
                image_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
            else:
                pil_image = image
            
            # Get dimensions
            width, height = pil_image.size
            
            # Handle seed
            if seed == -1:
                import random
                seed = random.randint(100000, 999999)
            
            # Move models
            pipeline.to('cpu')
            pipeline.refiner_pipeline.to('cuda')
            
            print(f"[HunyuanImage] Refining image: {width}x{height}, steps={steps}")
            
            # Refine image
            refined_image = pipeline.refiner_pipeline(
                image=pil_image,
                prompt=positive,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg,
                shift=shift,
                seed=seed
            )
            
            # Convert back to tensor
            if isinstance(refined_image, Image.Image):
                refined_np = np.array(refined_image).astype(np.float32) / 255.0
                refined_tensor = torch.from_numpy(refined_np)[None,]
            else:
                refined_tensor = refined_image
            
            print(f"[HunyuanImage] Image refined successfully")
            
            # Clear memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return (refined_tensor,)
            
        except Exception as e:
            print(f"[HunyuanImage] Error refining image: {e}")
            import traceback
            traceback.print_exc()
            return (image,)


class HunyuanImageVAEDecode:
    """Decode latent to image using HunyuanImage VAE"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HUNYUAN_PIPELINE",),
                "latent": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "HunyuanImage"
    
    def decode(self, pipeline, latent):
        """Decode latent representation to image"""
        
        try:
            # Get latent tensor
            latent_tensor = latent["samples"]
            
            # Decode using VAE
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                print(f"[HunyuanImage] Decoding latent to image")
                with torch.no_grad():
                    image = pipeline.vae.decode(latent_tensor)
                
                # Convert to proper format
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float()
            else:
                print(f"[HunyuanImage] Warning: VAE not available, returning placeholder")
                # Return placeholder image
                b, c, h, w = latent_tensor.shape
                image = torch.ones((b, h * 8, w * 8, 3))
            
            return (image,)
            
        except Exception as e:
            print(f"[HunyuanImage] Error decoding latent: {e}")
            # Return placeholder on error
            return (torch.ones((1, 512, 512, 3)),)


class HunyuanImageVAEEncode:
    """Encode image to latent using HunyuanImage VAE"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HUNYUAN_PIPELINE",),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "HunyuanImage"
    
    def encode(self, pipeline, image):
        """Encode image to latent representation"""
        
        try:
            # Convert image to proper format
            if isinstance(image, torch.Tensor):
                # Assume [B, H, W, C] format with values 0-1
                image_tensor = image.permute(0, 3, 1, 2).to(pipeline.device)
            else:
                # Convert PIL to tensor
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)[None,].to(pipeline.device)
            
            # Normalize to [-1, 1]
            image_tensor = image_tensor * 2 - 1
            
            # Encode using VAE
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                print(f"[HunyuanImage] Encoding image to latent")
                with torch.no_grad():
                    latent = pipeline.vae.encode(image_tensor)
                
                latent_dict = {"samples": latent}
            else:
                print(f"[HunyuanImage] Warning: VAE not available, returning placeholder")
                # Return placeholder latent
                b, c, h, w = image_tensor.shape
                latent_dict = {"samples": torch.zeros((b, 4, h // 8, w // 8))}
            
            return (latent_dict,)
            
        except Exception as e:
            print(f"[HunyuanImage] Error encoding image: {e}")
            # Return placeholder on error
            return ({"samples": torch.zeros((1, 4, 64, 64))},)


# Import advanced nodes
try:
    from .nodes_advanced import ADVANCED_NODE_CLASS_MAPPINGS, ADVANCED_NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    ADVANCED_NODE_CLASS_MAPPINGS = {}
    ADVANCED_NODE_DISPLAY_NAME_MAPPINGS = {}

# Node registration
NODE_CLASS_MAPPINGS = {
    "HunyuanImageModelLoader": HunyuanImageModelLoader,
    "HunyuanImagePromptEnhancer": HunyuanImagePromptEnhancer,
    "HunyuanImageSampler": HunyuanImageSampler,
    "HunyuanImageRefiner": HunyuanImageRefiner,
    "HunyuanImageVAEDecode": HunyuanImageVAEDecode,
    "HunyuanImageVAEEncode": HunyuanImageVAEEncode,
    **ADVANCED_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImageModelLoader": "HunyuanImage Model Loader",
    "HunyuanImagePromptEnhancer": "HunyuanImage Prompt Enhancer",
    "HunyuanImageSampler": "HunyuanImage Sampler",
    "HunyuanImageRefiner": "HunyuanImage Refiner",
    "HunyuanImageVAEDecode": "HunyuanImage VAE Decode",
    "HunyuanImageVAEEncode": "HunyuanImage VAE Encode",
    **ADVANCED_NODE_DISPLAY_NAME_MAPPINGS,
}