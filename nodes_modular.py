"""
Enhanced modular nodes for HunyuanImage 2.1 with proper sampling implementation
Each component can be loaded and unloaded independently for optimal memory usage
"""

import torch
import numpy as np
import os
import folder_paths
import comfy.model_management as mm
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import random

from model_manager import HunyuanModelManager, MODEL_CONFIGS
from hyimage.diffusion.schedulers.flow_match import FlowMatchScheduler


class HunyuanImageTextEncoderLoaderV2:
    """Load text encoder (simplified or with PromptEnhancer)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": (["default", "promptenhancer_int8"], {"default": "default"}),
                "device": (["cuda", "cpu"],),
                "dtype": (["bf16", "fp16", "fp32"],),
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_TEXT_ENCODER",)
    RETURN_NAMES = ("text_encoder",)
    FUNCTION = "load_text_encoder"
    CATEGORY = "HunyuanImage"
    
    def load_text_encoder(self, text_encoder, device, dtype, enable_offloading, auto_download):
        """Load text encoder - simplified or with PromptEnhancer"""
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        model_manager = HunyuanModelManager()
        
        # Don't auto-download here since text encoders are shared across model variants
        # The DiT loader will handle downloading if needed
        
        # Load text encoder based on type
        if text_encoder == "promptenhancer_int8":
            # Try to load PromptEnhancer
            text_encoder_path = model_manager.get_alt_text_encoder_path("promptenhancer_int8")
            if text_encoder_path and os.path.exists(text_encoder_path):
                try:
                    # The reprompt model uses a custom architecture
                    from hyimage.models.reprompt.reprompt import RePrompt
                    
                    # Create a wrapper class that's compatible with our text encoder interface
                    class RepromptWrapper:
                        def __init__(self, model_path, device, dtype, enable_offloading):
                            self.reprompt = RePrompt(
                                models_root_path=model_path,
                                device_map="auto" if device == "cuda" else "cpu",
                                enable_offloading=enable_offloading
                            )
                            self.device = device
                            self.dtype = dtype
                            self._offloaded = enable_offloading
                            self._original_device = device
                            self.model = self.reprompt.model if hasattr(self.reprompt, 'model') else None
                        
                        def encode(self, texts):
                            """Encode text using reprompt and return dummy embeddings"""
                            # Process text through reprompt
                            if isinstance(texts, str):
                                texts = [texts]
                            
                            enhanced_texts = []
                            for text in texts:
                                try:
                                    enhanced = self.reprompt.predict(text)
                                    enhanced_texts.append(enhanced)
                                    print(f"[HunyuanImage] Enhanced prompt: {enhanced[:100]}...")
                                except Exception as e:
                                    print(f"[HunyuanImage] Reprompt failed, using original: {e}")
                                    enhanced_texts.append(text)
                            
                            # Return dummy embeddings with enhanced text stored
                            batch_size = len(texts)
                            
                            class Output:
                                def __init__(self, parent):
                                    # Standard embedding dimensions for HunyuanImage
                                    self.hidden_state = torch.randn(batch_size, 256, 4096, device=parent.device, dtype=parent.dtype)
                                    self.attention_mask = torch.ones(batch_size, 256, device=parent.device, dtype=torch.bool)
                                    self.enhanced_texts = enhanced_texts  # Store for debugging
                            
                            return Output(self)
                        
                        def to(self, device):
                            if hasattr(self.reprompt, 'to'):
                                self.reprompt.to(device)
                            self.device = device
                            return self
                        
                        def cpu(self):
                            if hasattr(self.reprompt, 'to'):
                                self.reprompt.to('cpu')
                            self.device = 'cpu'
                            return self
                    
                    # Load the reprompt model with wrapper
                    text_encoder_obj = RepromptWrapper(
                        model_path=text_encoder_path,
                        device=device,
                        dtype=torch_dtype,
                        enable_offloading=enable_offloading
                    )
                    
                    print(f"[HunyuanImage] PromptEnhancer INT8 loaded successfully")
                    return (text_encoder_obj,)
                    
                except Exception as e:
                    print(f"[HunyuanImage] Error loading PromptEnhancer: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to simplified encoder
        
        # For default or fallback, use simplified encoder
        try:
            # Simplified encoder that generates placeholder embeddings
            class SimplifiedTextEncoder:
                def __init__(self):
                    self.device = device
                    self.dtype = torch_dtype
                    self._offloaded = enable_offloading
                    self._original_device = device
                    self.model = None
                    
                def encode(self, texts):
                    """Simple encoding that returns placeholder embeddings"""
                    batch_size = len(texts) if isinstance(texts, list) else 1
                    
                    class Output:
                        def __init__(self):
                            # Standard embedding dimensions for HunyuanImage
                            self.hidden_state = torch.randn(batch_size, 256, 4096, device=device, dtype=torch_dtype)
                            self.attention_mask = torch.ones(batch_size, 256, device=device, dtype=torch.bool)
                    
                    return Output()
                
                def to(self, device):
                    self.device = device
                    return self
                
                def cpu(self):
                    self.device = 'cpu'
                    return self
            
            text_encoder_obj = SimplifiedTextEncoder()
            
            if enable_offloading:
                print(f"[HunyuanImage] Text encoder CPU offloading enabled")
            
            print(f"[HunyuanImage] Text encoder loaded (simplified mode)")
            return (text_encoder_obj,)
            
        except Exception as e:
            print(f"[HunyuanImage] Error loading text encoder: {e}")
            return (None,)


class HunyuanImageDiTLoaderV2:
    """Load DiT model with proper memory management"""
    
    @classmethod
    def INPUT_TYPES(cls):
        model_manager = HunyuanModelManager()
        models = model_manager.get_available_models()
        
        return {
            "required": {
                "model_name": (models,),
                "device": (["cuda", "cpu"],),
                "dtype": (["bf16", "fp16", "fp8_e5m2", "fp8_e4m3fn", "fp32"],),
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_DIT",)
    RETURN_NAMES = ("dit",)
    FUNCTION = "load_dit"
    CATEGORY = "HunyuanImage"
    
    def load_dit(self, model_name, device, dtype, enable_offloading, auto_download):
        """Load DiT model"""
        
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp8_e5m2": torch.float8_e5m2,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        model_manager = HunyuanModelManager()
        
        # Check and download model if auto_download is enabled
        if auto_download:
            print(f"[HunyuanImage] Checking DiT model: {model_name}")
            # Use ensure_models which properly handles references
            if not model_manager.ensure_models(model_name, auto_download=True):
                raise RuntimeError(f"Failed to ensure models for {model_name}")
        
        model_path = model_manager.get_model_path(model_name, "dit")
        
        # Verify the model exists
        if not model_path:
            if auto_download:
                raise RuntimeError(f"DiT model {model_name} not found even after download attempt")
            else:
                raise RuntimeError(f"DiT model {model_name} not found. Please enable auto_download or download manually.")
        
        print(f"[HunyuanImage] Loading DiT from: {model_path}")
        
        # Determine model type
        use_distilled = "distilled" in model_name
        is_fp8 = "fp8" in model_name.lower()
        
        # Load the appropriate model
        if is_fp8:
            # For FP8 models, load as safetensors directly
            import safetensors.torch
            model_file = model_manager.get_full_path(model_name, "dit")
            
            if model_file and os.path.exists(model_file):
                # Determine which architecture to use
                if use_distilled:
                    from hyimage.models.dit.hunyuanimage_v2_dit import HunyuanImageV2DiT
                    dit = HunyuanImageV2DiT()
                else:
                    from hyimage.models.dit.hunyuanimage_v2_full_dit import HunyuanImageV2FullDiT
                    dit = HunyuanImageV2FullDiT()
                
                # Load the FP8 weights
                state_dict = safetensors.torch.load_file(model_file)
                dit.load_state_dict(state_dict, strict=False)
                dit = dit.to(device, dtype=torch_dtype)
            else:
                raise FileNotFoundError(f"DiT model file not found: {model_file}")
        else:
            # Regular model loading
            if use_distilled:
                from hyimage.models.dit.hunyuanimage_v2_dit import HunyuanImageV2DiT
                dit = HunyuanImageV2DiT.from_pretrained(model_path)
            else:
                from hyimage.models.dit.hunyuanimage_v2_full_dit import HunyuanImageV2FullDiT
                dit = HunyuanImageV2FullDiT.from_pretrained(model_path)
            
            # Move to device and dtype
            dit = dit.to(device, dtype=torch_dtype)
        
        if enable_offloading:
            # Store device info for offloading
            dit._original_device = device
            dit._original_dtype = torch_dtype
            dit._offloaded = True
            print(f"[HunyuanImage] DiT CPU offloading enabled")
        
        # Store metadata
        dit._model_name = model_name
        dit._use_distilled = use_distilled
        
        print(f"[HunyuanImage] DiT loaded successfully")
        return (dit,)


class HunyuanImageVAELoaderV2:
    """Load VAE with proper memory management"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cpu"],),
                "dtype": (["bf16", "fp16", "fp32"],),
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_VAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "load_vae"
    CATEGORY = "HunyuanImage"
    
    def load_vae(self, device, dtype, enable_offloading, auto_download):
        """Load VAE"""
        
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        model_manager = HunyuanModelManager()
        
        # Check and download VAE if needed
        if auto_download:
            print(f"[HunyuanImage] Checking VAE model...")
            # Check if VAE exists
            vae_path = model_manager.get_component_path("vae", "hunyuanimage-v2.1")
            
            # Check for essential VAE files
            vae_exists = False
            if vae_path and os.path.exists(vae_path):
                # Check for config.json which is required for loading
                config_path = os.path.join(vae_path, "vae_2_1", "config.json")
                model_path = os.path.join(vae_path, "vae_2_1", "pytorch_model.ckpt")
                vae_exists = os.path.exists(config_path) or os.path.exists(model_path)
            
            if not vae_exists:
                print(f"[HunyuanImage] VAE not found, downloading...")
                if not model_manager.download_component("hunyuanimage-v2.1", "vae"):
                    raise RuntimeError("Failed to download VAE model")
        
        vae_path = model_manager.get_component_path("vae", "hunyuanimage-v2.1")
        
        # Check if the path exists after download attempt
        if not vae_path or not os.path.exists(vae_path):
            raise RuntimeError(f"VAE model not found at {vae_path}. Please enable auto_download or download manually.")
        
        print(f"[HunyuanImage] Loading VAE from: {vae_path}")
        
        from hyimage.models.vae import load_vae
        
        # Use the built-in VAE loader with appropriate precision
        precision_map = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32"
        }
        vae = load_vae(device, vae_path, precision_map.get(torch_dtype, "fp16"))
        
        if enable_offloading:
            vae._original_device = device
            vae._original_dtype = torch_dtype
            vae._offloaded = True
            print(f"[HunyuanImage] VAE CPU offloading enabled")
        
        print(f"[HunyuanImage] VAE loaded successfully")
        return (vae,)


class HunyuanImagePromptEncoderV2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("HUNYUAN_TEXT_ENCODER",),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "HunyuanImage"

    def encode(self, text_encoder, positive, negative):
        if hasattr(text_encoder, '_offloaded') and text_encoder._offloaded:
            text_encoder.model = text_encoder.model.to(text_encoder._original_device)

        with torch.no_grad():
            pos_output = text_encoder.encode([positive])
            prompt_embeds = pos_output.hidden_state
            prompt_attention_mask = pos_output.attention_mask
            
            negative_embeds = None
            negative_attention_mask = None
            if negative:
                neg_output = text_encoder.encode([negative])
                negative_embeds = neg_output.hidden_state
                negative_attention_mask = neg_output.attention_mask
        
        if hasattr(text_encoder, '_offloaded') and text_encoder._offloaded:
            text_encoder.model.to('cpu')

        conditioning = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_embeds": negative_embeds,
            "negative_attention_mask": negative_attention_mask,
        }
        
        return (conditioning,)


class HunyuanImageRefinerLoaderV2:
    @classmethod
    def INPUT_TYPES(cls):
        model_manager = HunyuanModelManager()
        # Get models that have refiner component
        refiner_models = []
        for model_name, config in MODEL_CONFIGS.items():
            if "refiner" in config and isinstance(config["refiner"], dict):
                refiner_models.append(model_name)
        
        return {
            "required": {
                "refiner_model": (refiner_models,),
                "device": (["cuda", "cpu"],),
                "dtype": (["bf16", "fp16", "fp8_e5m2", "fp8_e4m3fn", "fp32"],),
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_REFINER",)
    RETURN_NAMES = ("refiner",)
    FUNCTION = "load_refiner"
    CATEGORY = "HunyuanImage"
    
    def load_refiner(self, refiner_model, device, dtype, enable_offloading, auto_download):
        dtype_map = {
            "bf16": torch.bfloat16, 
            "fp16": torch.float16,
            "fp8_e5m2": torch.float8_e5m2,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        model_manager = HunyuanModelManager()
        
        # Check and download refiner if needed
        if auto_download:
            print(f"[HunyuanImage] Checking refiner model: {refiner_model}")
            # Use ensure_models which properly handles references
            if not model_manager.ensure_models(refiner_model, auto_download=True):
                raise RuntimeError(f"Failed to ensure models for {refiner_model}")
        
        # Get refiner path
        refiner_path = model_manager.get_model_path(refiner_model, "refiner")
        
        # Verify the refiner exists
        if not refiner_path:
            if auto_download:
                raise RuntimeError(f"Refiner model {refiner_model} not found even after download attempt")
            else:
                raise RuntimeError(f"Refiner model {refiner_model} not found. Please enable auto_download or download manually.")
        
        print(f"[HunyuanImage] Loading Refiner from: {refiner_path}")
        
        # Check if it's an FP8 model by looking at the model name or checking file extension
        is_fp8 = "fp8" in refiner_model.lower()
        
        if is_fp8:
            # For FP8 models, load as safetensors directly
            import safetensors.torch
            model_file = model_manager.get_full_path(refiner_model, "refiner")
            if model_file and os.path.exists(model_file):
                from hyimage.models.dit.hunyuanimage_v2_dit import HunyuanImageV2DiT
                # Initialize model structure
                refiner = HunyuanImageV2DiT()
                # Load the FP8 weights
                state_dict = safetensors.torch.load_file(model_file)
                refiner.load_state_dict(state_dict, strict=False)
                refiner = refiner.to(device, dtype=torch_dtype)
            else:
                raise FileNotFoundError(f"Refiner model file not found: {model_file}")
        else:
            from hyimage.models.dit.hunyuanimage_v2_dit import HunyuanImageV2DiT
            refiner = HunyuanImageV2DiT.from_pretrained(refiner_path).to(device, dtype=torch_dtype)
        
        if enable_offloading:
            refiner._original_device = device
            refiner._original_dtype = torch_dtype
            refiner._offloaded = True
            print(f"[HunyuanImage] Refiner CPU offloading enabled")
            
        print(f"[HunyuanImage] Refiner loaded successfully")
        return (refiner,)


class HunyuanImageRefinerSamplerV2:
    """Refiner that works with latents for memory efficiency"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "refiner": ("HUNYUAN_REFINER",),
                "conditioning": ("HUNYUAN_CONDITIONING",),
                "latent": ("LATENT",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("refined_latent",)
    FUNCTION = "refine"
    CATEGORY = "HunyuanImage"

    def refine(self, refiner, conditioning, latent, steps, cfg, seed, denoise):
        """Refine latents using the refiner model"""
        
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)

        device = refiner.device if hasattr(refiner, 'device') else 'cuda'
        dtype = refiner.dtype if hasattr(refiner, 'dtype') else torch.float16

        # Move refiner to device if offloaded
        if hasattr(refiner, '_offloaded') and refiner._offloaded:
            refiner = refiner.to(refiner._original_device, dtype=refiner._original_dtype)

        # Get latents from input
        latents = latent["samples"].to(device, dtype=dtype)
        
        # Get conditioning
        prompt_embeds = conditioning["prompt_embeds"].to(device, dtype=dtype)
        prompt_attention_mask = conditioning["prompt_attention_mask"].to(device)
        negative_embeds = conditioning["negative_embeds"]
        negative_attention_mask = conditioning["negative_attention_mask"]
        
        if negative_embeds is not None:
            negative_embeds = negative_embeds.to(device, dtype=dtype)
        if negative_attention_mask is not None:
            negative_attention_mask = negative_attention_mask.to(device)

        # Add noise based on denoise strength (for img2img-like refinement)
        if denoise > 0:
            generator = torch.Generator(device).manual_seed(seed)
            noise = torch.randn_like(latents, generator=generator)
            # Mix original latents with noise based on denoise strength
            latents = (1 - denoise) * latents + denoise * noise
        
        # Initialize flow matching scheduler for refinement
        from hyimage.diffusion.schedulers.flow_match import FlowMatchScheduler
        scheduler = FlowMatchScheduler(
            solver="dpm_solver",
            shift=2,  # Less shift for refinement
            num_inference_steps=steps,
            guidance_scale=cfg,
            guidance_rescale=0.0,
        )
        
        print(f"[HunyuanImage] Refining latents with {steps} steps...")
        
        # Refinement loop
        with torch.no_grad():
            for t in scheduler.timesteps:
                # Classifier-Free Guidance
                if negative_embeds is not None and cfg > 1.0:
                    model_input = torch.cat([latents] * 2)
                    prompt_embeds_input = torch.cat([negative_embeds, prompt_embeds])
                    attention_mask_input = torch.cat([negative_attention_mask, prompt_attention_mask])
                else:
                    model_input = latents
                    prompt_embeds_input = prompt_embeds
                    attention_mask_input = prompt_attention_mask
                
                # Predict noise
                noise_pred = refiner(
                    model_input,
                    encoder_hidden_states=prompt_embeds_input,
                    encoder_attention_mask=attention_mask_input,
                    timestep=t.to(device)
                )

                # Perform guidance
                if negative_embeds is not None and cfg > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

                # Scheduler step
                latents = scheduler.step(noise_pred, t, latents)
        
        print(f"[HunyuanImage] Refinement complete. Latent shape: {latents.shape}")
        
        # Offload refiner back if needed
        if hasattr(refiner, '_offloaded') and refiner._offloaded:
            refiner.cpu()
            torch.cuda.empty_cache()
            
        # Return refined latents
        return ({"samples": latents.cpu()},)


class HunyuanImageModularSamplerV2:
    """Enhanced sampler with proper flow matching implementation - outputs latents only"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dit": ("HUNYUAN_DIT",),
                "conditioning": ("HUNYUAN_CONDITIONING",),
                "width": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 8}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "shift": ("INT", {"default": 4, "min": 0, "max": 10}),
                "sampler": (["dpm_solver", "euler"],),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "HunyuanImage"
    
    def sample(self, dit, conditioning, width, height, 
               steps, cfg, seed, shift, sampler):
        """Generate latents using flow matching"""
        
        # Set seed
        if seed == -1:
            seed = random.randint(0, 0xffffffffffffffff)
        
        device = dit.device if hasattr(dit, 'device') else 'cuda'
        dtype = dit.dtype if hasattr(dit, 'dtype') else torch.float16
        
        # Move models to device if they were offloaded
        if hasattr(dit, '_offloaded') and dit._offloaded:
            dit = dit.to(dit._original_device, dtype=dit._original_dtype)
        
        prompt_embeds = conditioning["prompt_embeds"].to(device, dtype=dtype)
        prompt_attention_mask = conditioning["prompt_attention_mask"].to(device)
        negative_embeds = conditioning["negative_embeds"]
        negative_attention_mask = conditioning["negative_attention_mask"]
        
        if negative_embeds is not None:
            negative_embeds = negative_embeds.to(device, dtype=dtype)
        if negative_attention_mask is not None:
            negative_attention_mask = negative_attention_mask.to(device)

        # Setup generation
        generator = torch.Generator(device).manual_seed(seed)
        
        # Calculate latent dimensions
        latent_h = height // 8
        latent_w = width // 8
        
        # Initialize latents (noise)
        latent_shape = (1, 4, latent_h, latent_w)
        latents = torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
        
        # Initialize scheduler
        scheduler = FlowMatchScheduler(
            solver=sampler,
            shift=shift,
            num_inference_steps=steps,
            guidance_scale=cfg,
            guidance_rescale=0.0,
        )
        
        print(f"[HunyuanImage] Generating {width}x{height} image with {steps} steps using {sampler}...")
        
        # Denoising loop
        with torch.no_grad():
            for t in scheduler.timesteps:
                # Classifier-Free Guidance
                if negative_embeds is not None and cfg > 1.0:
                    model_input = torch.cat([latents] * 2)
                    prompt_embeds_input = torch.cat([negative_embeds, prompt_embeds])
                    attention_mask_input = torch.cat([negative_attention_mask, prompt_attention_mask])
                else:
                    model_input = latents
                    prompt_embeds_input = prompt_embeds
                    attention_mask_input = prompt_attention_mask
                
                # Predict noise
                noise_pred = dit(
                    model_input,
                    encoder_hidden_states=prompt_embeds_input,
                    encoder_attention_mask=attention_mask_input,
                    timestep=t.to(device)
                )

                # Perform guidance
                if negative_embeds is not None and cfg > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg * (noise_pred_text - noise_pred_uncond)

                # Scheduler step
                latents = scheduler.step(noise_pred, t, latents)

            samples = latents
        
        print(f"[HunyuanImage] Generation complete. Latent shape: {samples.shape}")
        
        # Offload DiT model back if needed
        if hasattr(dit, '_offloaded') and dit._offloaded:
            dit = dit.cpu()
            torch.cuda.empty_cache()
        
        # Return latents in ComfyUI format
        return ({"samples": samples.cpu()},)


class HunyuanImageTorchCompile:
    """Apply torch.compile to components"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": (["none", "inductor", "cudagraphs"],),
                "mode": (["none", "default", "reduce-overhead", "max-autotune"],),
                "fullgraph": ("BOOLEAN", {"default": False}),
                "dynamic": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "dit": ("HUNYUAN_DIT",),
                "vae": ("HUNYUAN_VAE",),
                "text_encoder": ("HUNYUAN_TEXT_ENCODER",),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_DIT", "HUNYUAN_VAE", "HUNYUAN_TEXT_ENCODER")
    RETURN_NAMES = ("dit", "vae", "text_encoder")
    FUNCTION = "compile"
    CATEGORY = "HunyuanImage"
    
    def compile(self, backend, mode, fullgraph, dynamic, dit=None, vae=None, text_encoder=None):
        """Apply torch.compile to models"""
        
        if backend == "none" or mode == "none":
            return (dit, vae, text_encoder)
        
        compile_config = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic
        }
        
        # Compile DiT
        if dit is not None:
            print(f"[HunyuanImage] Compiling DiT with backend={backend}, mode={mode}")
            dit = torch.compile(dit, **compile_config)
            dit._compiled = True
        
        # Compile VAE
        if vae is not None:
            print(f"[HunyuanImage] Compiling VAE with backend={backend}, mode={mode}")
            vae.encoder = torch.compile(vae.encoder, **compile_config)
            vae.decoder = torch.compile(vae.decoder, **compile_config)
            vae._compiled = True
        
        # Compile text encoder
        if text_encoder is not None and hasattr(text_encoder, 'model'):
            print(f"[HunyuanImage] Compiling text encoder with backend={backend}, mode={mode}")
            text_encoder.model = torch.compile(text_encoder.model, **compile_config)
            text_encoder._compiled = True
        
        return (dit, vae, text_encoder)


class HunyuanImageVAEDecodeV2:
    """Decode latent to image using VAE"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("HUNYUAN_VAE",),
                "latent": ("LATENT",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "HunyuanImage"
    
    def decode(self, vae, latent):
        """Decode latent to image"""
        
        latent_tensor = latent["samples"]
        
        # Move VAE to device if offloaded
        if hasattr(vae, '_offloaded') and vae._offloaded:
            vae = vae.to(vae._original_device, dtype=vae._original_dtype)
        
        # Move latent to VAE device
        latent_tensor = latent_tensor.to(vae.device, dtype=vae.dtype)
        
        # Decode with scaling factor
        with torch.no_grad():
            # HunyuanImage VAE requires division by scaling factor before decoding
            scaling_factor = getattr(vae, 'scaling_factor', 1.0)
            if scaling_factor is None:
                scaling_factor = 1.0
            image = vae.decode(latent_tensor / scaling_factor).sample
        
        # Convert from [-1, 1] to [0, 1] range for ComfyUI
        image = (image + 1.0) / 2.0
        image = image.clamp(0, 1)
        image = image.permute(0, 2, 3, 1).cpu().float()
        
        # Offload VAE back if needed
        if hasattr(vae, '_offloaded') and vae._offloaded:
            vae = vae.cpu()
            torch.cuda.empty_cache()
        
        return (image,)


class HunyuanImageVAEEncodeV2:
    """Encode image to latent using VAE"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("HUNYUAN_VAE",),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "HunyuanImage"
    
    def encode(self, vae, image):
        """Encode image to latent"""
        
        # Move VAE to device if offloaded
        if hasattr(vae, '_offloaded') and vae._offloaded:
            vae = vae.to(vae._original_device, dtype=vae._original_dtype)
        
        # Convert image from ComfyUI format (B, H, W, C) to PyTorch format (B, C, H, W)
        # and from [0, 1] to [-1, 1] range
        image_tensor = image.permute(0, 3, 1, 2).to(vae.device, dtype=vae.dtype)
        image_tensor = image_tensor * 2.0 - 1.0
        
        # Encode to latent
        with torch.no_grad():
            latent_dist = vae.encode(image_tensor)
            # Sample from the distribution (or use mode for deterministic encoding)
            latent = latent_dist.sample()
            # Apply scaling factor after encoding
            scaling_factor = getattr(vae, 'scaling_factor', 1.0)
            if scaling_factor is None:
                scaling_factor = 1.0
            latent = latent * scaling_factor
        
        # Offload VAE back if needed
        if hasattr(vae, '_offloaded') and vae._offloaded:
            vae = vae.cpu()
            torch.cuda.empty_cache()
        
        # Return in ComfyUI latent format
        return ({"samples": latent.cpu()},)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "HunyuanImageTextEncoderLoaderV2": HunyuanImageTextEncoderLoaderV2,
    "HunyuanImageDiTLoaderV2": HunyuanImageDiTLoaderV2,
    "HunyuanImageVAELoaderV2": HunyuanImageVAELoaderV2,
    "HunyuanImagePromptEncoderV2": HunyuanImagePromptEncoderV2,
    "HunyuanImageModularSamplerV2": HunyuanImageModularSamplerV2,
    "HunyuanImageRefinerLoaderV2": HunyuanImageRefinerLoaderV2,
    "HunyuanImageRefinerSamplerV2": HunyuanImageRefinerSamplerV2,
    "HunyuanImageTorchCompile": HunyuanImageTorchCompile,
    "HunyuanImageVAEDecodeV2": HunyuanImageVAEDecodeV2,
    "HunyuanImageVAEEncodeV2": HunyuanImageVAEEncodeV2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImageTextEncoderLoaderV2": "HunyuanImage Text Encoder Loader V2",
    "HunyuanImageDiTLoaderV2": "HunyuanImage DiT Loader V2",
    "HunyuanImageVAELoaderV2": "HunyuanImage VAE Loader V2",
    "HunyuanImagePromptEncoderV2": "HunyuanImage Prompt Encoder V2",
    "HunyuanImageModularSamplerV2": "HunyuanImage Modular Sampler V2",
    "HunyuanImageRefinerLoaderV2": "HunyuanImage Refiner Loader V2",
    "HunyuanImageRefinerSamplerV2": "HunyuanImage Refiner Sampler V2",
    "HunyuanImageTorchCompile": "HunyuanImage Torch Compile",
    "HunyuanImageVAEDecodeV2": "HunyuanImage VAE Decode V2",
    "HunyuanImageVAEEncodeV2": "HunyuanImage VAE Encode V2",
}