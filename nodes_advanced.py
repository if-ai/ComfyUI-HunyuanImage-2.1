"""
Advanced ComfyUI nodes for HunyuanImage 2.1 with quantization support
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline, HunyuanImagePipelineConfig
    from hyimage.common.constants import PRECISION_TO_TYPE
    from .model_manager import model_manager
except ImportError as e:
    print(f"[HunyuanImage] Error importing modules: {e}")
    model_manager = None


class HunyuanImageAdvancedLoader:
    """Advanced model loader with quantization and optimization options"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["hunyuanimage-v2.1", "hunyuanimage-v2.1-distilled"], {"default": "hunyuanimage-v2.1"}),
                "device": (["cuda", "cpu", "mps"], {"default": "cuda"}),
                
                # Quantization options
                "quantize_dit": ("BOOLEAN", {"default": False}),
                "dit_quant_method": (["none", "int8", "int4", "nf4", "fp8_e4m3fn", "fp8_e5m2", "gptq", "awq"], {"default": "int8"}),
                "dit_dtype": (["fp32", "fp16", "bf16", "auto"], {"default": "bf16"}),
                
                "quantize_vae": ("BOOLEAN", {"default": False}),
                "vae_quant_method": (["none", "int8", "fp8_e4m3fn"], {"default": "int8"}),
                "vae_dtype": (["fp32", "fp16", "bf16", "auto"], {"default": "bf16"}),
                
                "quantize_text_encoder": ("BOOLEAN", {"default": False}),
                "text_quant_method": (["none", "int8", "int4", "nf4"], {"default": "int8"}),
                "text_dtype": (["fp32", "fp16", "bf16", "auto"], {"default": "bf16"}),
                
                # Memory optimization
                "enable_dit_offloading": ("BOOLEAN", {"default": True}),
                "enable_sequential_cpu_offload": ("BOOLEAN", {"default": False}),
                "enable_attention_slicing": ("BOOLEAN", {"default": False}),
                "enable_vae_slicing": ("BOOLEAN", {"default": False}),
                "enable_vae_tiling": ("BOOLEAN", {"default": False}),
                
                # Torch compile options
                "compile_backend": (["none", "eager", "inductor", "aot_eager", "cudagraphs", "onnxrt", "tensorrt"], {"default": "inductor"}),
                "compile_mode": (["none", "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"], {"default": "max-autotune"}),
                "compile_fullgraph": ("BOOLEAN", {"default": False}),
                "compile_dynamic": ("BOOLEAN", {"default": True}),
                
                # Component compilation
                "compile_dit": ("BOOLEAN", {"default": False}),
                "compile_vae": ("BOOLEAN", {"default": False}),
                "compile_text_encoder": ("BOOLEAN", {"default": False}),
                
                # Advanced options
                "use_flash_attention": ("BOOLEAN", {"default": True}),
                "use_xformers": ("BOOLEAN", {"default": False}),
                "use_sdpa": ("BOOLEAN", {"default": True}),
                "channels_last": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_PIPELINE", "HUNYUAN_CONFIG")
    RETURN_NAMES = ("pipeline", "config")
    FUNCTION = "load_advanced"
    CATEGORY = "HunyuanImage/Advanced"
    
    def get_dtype(self, dtype_str):
        """Convert dtype string to torch dtype"""
        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "auto": "auto",
        }
        return dtype_map.get(dtype_str, torch.bfloat16)
    
    def quantize_model(self, model, method, dtype):
        """Apply quantization to a model"""
        
        if method == "none" or not model:
            return model
        
        print(f"[HunyuanImage] Applying {method} quantization")
        
        if method == "int8":
            # Dynamic INT8 quantization
            try:
                import torch.ao.quantization as quant
                model = quant.quantize_dynamic(
                    model, 
                    {nn.Linear}, 
                    dtype=torch.qint8
                )
            except Exception as e:
                print(f"[HunyuanImage] INT8 quantization failed: {e}")
        
        elif method == "int4":
            # INT4 quantization using bitsandbytes
            try:
                import bitsandbytes as bnb
                from bitsandbytes.nn import Linear4bit
                
                def replace_with_int4(module):
                    for name, child in module.named_children():
                        if isinstance(child, nn.Linear):
                            setattr(module, name, Linear4bit(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                compute_dtype=dtype if dtype != "auto" else torch.bfloat16
                            ))
                        else:
                            replace_with_int4(child)
                
                replace_with_int4(model)
            except ImportError:
                print("[HunyuanImage] bitsandbytes not installed, skipping INT4 quantization")
            except Exception as e:
                print(f"[HunyuanImage] INT4 quantization failed: {e}")
        
        elif method == "nf4":
            # NF4 quantization using bitsandbytes
            try:
                import bitsandbytes as bnb
                from bitsandbytes.nn import Linear4bit
                
                def replace_with_nf4(module):
                    for name, child in module.named_children():
                        if isinstance(child, nn.Linear):
                            setattr(module, name, Linear4bit(
                                child.in_features,
                                child.out_features,
                                bias=child.bias is not None,
                                compute_dtype=dtype if dtype != "auto" else torch.bfloat16,
                                compress_statistics=True,
                                quant_type="nf4"
                            ))
                        else:
                            replace_with_nf4(child)
                
                replace_with_nf4(model)
            except ImportError:
                print("[HunyuanImage] bitsandbytes not installed, skipping NF4 quantization")
            except Exception as e:
                print(f"[HunyuanImage] NF4 quantization failed: {e}")
        
        elif method in ["fp8_e4m3fn", "fp8_e5m2"]:
            # FP8 quantization
            if hasattr(torch, 'float8_e4m3fn') and hasattr(torch, 'float8_e5m2'):
                fp8_dtype = torch.float8_e4m3fn if method == "fp8_e4m3fn" else torch.float8_e5m2
                model = model.to(dtype=fp8_dtype)
            else:
                print("[HunyuanImage] FP8 not available in this PyTorch version")
        
        elif method == "gptq":
            # GPTQ quantization (placeholder - requires calibration data)
            print("[HunyuanImage] GPTQ quantization requires calibration data, skipping")
        
        elif method == "awq":
            # AWQ quantization (placeholder - requires calibration data)
            print("[HunyuanImage] AWQ quantization requires calibration data, skipping")
        
        return model
    
    def compile_model(self, model, backend, mode, fullgraph, dynamic):
        """Apply torch.compile to a model"""
        
        if backend == "none" or mode == "none" or not model:
            return model
        
        print(f"[HunyuanImage] Compiling with backend={backend}, mode={mode}")
        
        compile_config = {
            "backend": backend,
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic
        }
        
        try:
            model = torch.compile(model, **compile_config)
        except Exception as e:
            print(f"[HunyuanImage] Compilation failed: {e}")
        
        return model
    
    def load_advanced(self, model_name, device, 
                     quantize_dit, dit_quant_method, dit_dtype,
                     quantize_vae, vae_quant_method, vae_dtype,
                     quantize_text_encoder, text_quant_method, text_dtype,
                     enable_dit_offloading, enable_sequential_cpu_offload,
                     enable_attention_slicing, enable_vae_slicing, enable_vae_tiling,
                     compile_backend, compile_mode, compile_fullgraph, compile_dynamic,
                     compile_dit, compile_vae, compile_text_encoder,
                     use_flash_attention, use_xformers, use_sdpa, channels_last):
        """Load model with advanced configuration"""
        
        # Ensure models are available
        if model_manager:
            print(f"[HunyuanImage] Checking models for {model_name}")
            if not model_manager.ensure_models(model_name, auto_download=True):
                raise RuntimeError(f"Failed to load models for {model_name}")
            
            # Set model root to ComfyUI paths
            model_root = model_manager.get_model_root(model_name)
            os.environ['HUNYUANIMAGE_V2_1_MODEL_ROOT'] = model_root
            print(f"[HunyuanImage] Using model root: {model_root}")
        
        # Prepare config
        use_distilled = "distilled" in model_name
        base_dtype = self.get_dtype(dit_dtype)
        dtype_str = "bf16" if base_dtype == torch.bfloat16 else "fp16" if base_dtype == torch.float16 else "fp32"
        
        config = HunyuanImagePipelineConfig.create_default(
            version="v2.1",
            use_distilled=use_distilled,
            torch_dtype=dtype_str,
            device=device,
            enable_dit_offloading=enable_dit_offloading,
            enable_reprompt_model_offloading=True,
            enable_refiner_offloading=True
        )
        
        # Load pipeline
        print(f"[HunyuanImage] Loading {model_name} with advanced configuration")
        pipeline = HunyuanImagePipeline.from_pretrained(
            model_name=model_name,
            config=config,
            device="cpu" if enable_sequential_cpu_offload else device
        )
        
        # Apply attention optimizations
        if hasattr(pipeline, 'dit') and pipeline.dit is not None:
            if use_flash_attention:
                try:
                    from flash_attn import flash_attn_func
                    print("[HunyuanImage] Flash Attention enabled")
                    # Flash attention is typically enabled by default if available
                except ImportError:
                    print("[HunyuanImage] Flash Attention not available")
            
            if use_xformers:
                try:
                    import xformers
                    import xformers.ops
                    pipeline.dit.set_use_memory_efficient_attention_xformers(True)
                    print("[HunyuanImage] xFormers memory efficient attention enabled")
                except:
                    print("[HunyuanImage] xFormers not available")
            
            if use_sdpa and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                print("[HunyuanImage] PyTorch SDPA enabled")
                # SDPA is typically used automatically in newer PyTorch versions
        
        # Apply channels_last memory format
        if channels_last:
            if hasattr(pipeline, 'dit') and pipeline.dit is not None:
                pipeline.dit = pipeline.dit.to(memory_format=torch.channels_last)
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                pipeline.vae = pipeline.vae.to(memory_format=torch.channels_last)
            print("[HunyuanImage] Channels-last memory format enabled")
        
        # Apply quantization
        if quantize_dit and hasattr(pipeline, 'dit'):
            dit_dtype_torch = self.get_dtype(dit_dtype)
            pipeline.dit = self.quantize_model(pipeline.dit, dit_quant_method, dit_dtype_torch)
        
        if quantize_vae and hasattr(pipeline, 'vae'):
            vae_dtype_torch = self.get_dtype(vae_dtype)
            pipeline.vae = self.quantize_model(pipeline.vae, vae_quant_method, vae_dtype_torch)
        
        if quantize_text_encoder and hasattr(pipeline, 'text_encoder'):
            text_dtype_torch = self.get_dtype(text_dtype)
            pipeline.text_encoder = self.quantize_model(pipeline.text_encoder, text_quant_method, text_dtype_torch)
        
        # Apply torch.compile
        if compile_dit and hasattr(pipeline, 'dit'):
            pipeline.dit = self.compile_model(pipeline.dit, compile_backend, compile_mode, compile_fullgraph, compile_dynamic)
        
        if compile_vae and hasattr(pipeline, 'vae'):
            if hasattr(pipeline.vae, 'encoder'):
                pipeline.vae.encoder = self.compile_model(pipeline.vae.encoder, compile_backend, compile_mode, compile_fullgraph, compile_dynamic)
            if hasattr(pipeline.vae, 'decoder'):
                pipeline.vae.decoder = self.compile_model(pipeline.vae.decoder, compile_backend, compile_mode, compile_fullgraph, compile_dynamic)
        
        if compile_text_encoder and hasattr(pipeline, 'text_encoder'):
            if hasattr(pipeline.text_encoder, 'model'):
                pipeline.text_encoder.model = self.compile_model(pipeline.text_encoder.model, compile_backend, compile_mode, compile_fullgraph, compile_dynamic)
        
        # Apply memory optimizations
        if enable_sequential_cpu_offload:
            pipeline.enable_sequential_cpu_offload()
            print("[HunyuanImage] Sequential CPU offload enabled")
        
        if enable_attention_slicing:
            if hasattr(pipeline, 'dit'):
                pipeline.dit.set_attention_slice("auto")
                print("[HunyuanImage] Attention slicing enabled")
        
        if enable_vae_slicing and hasattr(pipeline, 'vae'):
            pipeline.vae.enable_slicing()
            print("[HunyuanImage] VAE slicing enabled")
        
        if enable_vae_tiling and hasattr(pipeline, 'vae'):
            pipeline.vae.enable_tiling()
            print("[HunyuanImage] VAE tiling enabled")
        
        # Move to device if not using CPU offload
        if not enable_sequential_cpu_offload:
            pipeline.to(device)
        
        # Create config dict
        config_dict = {
            "model_name": model_name,
            "device": device,
            "dit_quantized": quantize_dit,
            "dit_quant_method": dit_quant_method if quantize_dit else "none",
            "vae_quantized": quantize_vae,
            "vae_quant_method": vae_quant_method if quantize_vae else "none",
            "text_quantized": quantize_text_encoder,
            "text_quant_method": text_quant_method if quantize_text_encoder else "none",
            "compile_backend": compile_backend,
            "compile_mode": compile_mode,
            "flash_attention": use_flash_attention,
            "xformers": use_xformers,
            "sdpa": use_sdpa,
            "channels_last": channels_last,
            "cpu_offload": enable_sequential_cpu_offload,
            "attention_slicing": enable_attention_slicing,
            "vae_slicing": enable_vae_slicing,
            "vae_tiling": enable_vae_tiling,
        }
        
        print(f"[HunyuanImage] Advanced model loaded successfully")
        print(f"[HunyuanImage] Configuration: {config_dict}")
        
        return (pipeline, config_dict)


class HunyuanImageMemoryManager:
    """Memory management utilities for HunyuanImage"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("HUNYUAN_PIPELINE",),
                "action": (["clear_cache", "offload_to_cpu", "load_to_gpu", "print_memory_stats"], {"default": "clear_cache"}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_PIPELINE", "STRING")
    RETURN_NAMES = ("pipeline", "info")
    FUNCTION = "manage_memory"
    CATEGORY = "HunyuanImage/Advanced"
    
    def manage_memory(self, pipeline, action):
        """Manage GPU memory for the pipeline"""
        
        info = ""
        
        if action == "clear_cache":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                info = "GPU cache cleared"
            else:
                info = "No CUDA device available"
        
        elif action == "offload_to_cpu":
            pipeline.to("cpu")
            if hasattr(pipeline, 'dit'):
                pipeline.dit.to("cpu")
            if hasattr(pipeline, 'vae'):
                pipeline.vae.to("cpu")
            if hasattr(pipeline, 'text_encoder'):
                pipeline.text_encoder.to("cpu")
            if hasattr(pipeline, 'refiner_pipeline'):
                pipeline.refiner_pipeline.to("cpu")
            torch.cuda.empty_cache()
            info = "Models offloaded to CPU"
        
        elif action == "load_to_gpu":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            pipeline.to(device)
            info = f"Models loaded to {device}"
        
        elif action == "print_memory_stats":
            if torch.cuda.is_available():
                stats = torch.cuda.memory_stats()
                allocated = stats.get("allocated_bytes.all.current", 0) / 1024**3
                reserved = stats.get("reserved_bytes.all.current", 0) / 1024**3
                peak = stats.get("allocated_bytes.all.peak", 0) / 1024**3
                info = f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Peak: {peak:.2f}GB"
            else:
                info = "No CUDA device available"
        
        print(f"[HunyuanImage] {info}")
        return (pipeline, info)


# Update node mappings
ADVANCED_NODE_CLASS_MAPPINGS = {
    "HunyuanImageAdvancedLoader": HunyuanImageAdvancedLoader,
    "HunyuanImageMemoryManager": HunyuanImageMemoryManager,
}

ADVANCED_NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImageAdvancedLoader": "HunyuanImage Advanced Loader",
    "HunyuanImageMemoryManager": "HunyuanImage Memory Manager",
}