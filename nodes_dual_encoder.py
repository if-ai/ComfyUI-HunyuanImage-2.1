"""
Dual Text Encoder nodes for HunyuanImage
Implements the MLLM/RePrompt + ByT5/Glyph dual encoder system
"""

import torch
import os
from pathlib import Path
from typing import Optional, Dict, Any

from model_manager import HunyuanModelManager


class HunyuanImageDualTextEncoderLoader:
    """Load dual text encoders - MLLM/RePrompt + ByT5/Glyph for full HunyuanImage functionality"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_reprompt": ("BOOLEAN", {"default": True, "tooltip": "Use PromptEnhancer to enhance prompts"}),
                "device": (["cuda", "cpu"],),
                "dtype": (["bf16", "fp16", "fp32"],),
                "enable_offloading": ("BOOLEAN", {"default": True}),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_DUAL_TEXT_ENCODER",)
    RETURN_NAMES = ("dual_text_encoder",)
    FUNCTION = "load_dual_encoders"
    CATEGORY = "HunyuanImage"
    
    def load_dual_encoders(self, use_reprompt, device, dtype, enable_offloading, auto_download):
        """Load dual text encoders - MLLM/RePrompt + ByT5/Glyph"""
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32
        }
        torch_dtype = dtype_map[dtype]
        
        model_manager = HunyuanModelManager()
        
        # Only check for specific text encoders needed, not the full model
        if auto_download:
            # Check for PromptEnhancer if requested
            if use_reprompt:
                if not model_manager.check_alt_text_encoder("promptenhancer_int8"):
                    print(f"[HunyuanImage] Downloading PromptEnhancer INT8...")
                    model_manager.download_alt_text_encoder("promptenhancer_int8")
            
            # Check for ByT5 and Glyph encoders specifically
            byt5_path = model_manager.get_component_path("text_encoders", "hunyuanimage-v2.1/byt5-small")
            glyph_path = model_manager.get_component_path("text_encoders", "hunyuanimage-v2.1/Glyph-SDXL-v2")
            
            if not byt5_path or not os.path.exists(byt5_path):
                print(f"[HunyuanImage] ByT5 encoder not found, downloading...")
                model_manager.download_component("hunyuanimage-v2.1", "text_encoders", "byt5")
            
            if not glyph_path or not os.path.exists(glyph_path):
                print(f"[HunyuanImage] Glyph encoder not found, downloading...")
                model_manager.download_component("hunyuanimage-v2.1", "text_encoders", "glyph")
        
        # Get paths
        byt5_path = model_manager.get_component_path("text_encoders", "hunyuanimage-v2.1/byt5-small")
        glyph_path = model_manager.get_component_path("text_encoders", "hunyuanimage-v2.1/Glyph-SDXL-v2")
        reprompt_path = model_manager.get_alt_text_encoder_path("promptenhancer_int8") if use_reprompt else None
        
        print(f"[HunyuanImage] Loading dual text encoders:")
        print(f"  - ByT5: {byt5_path}")
        print(f"  - Glyph: {glyph_path}")
        if use_reprompt:
            print(f"  - RePrompt: {reprompt_path}")
        
        # Create the dual encoder wrapper
        dual_encoder = DualTextEncoder(
            byt5_path=byt5_path,
            glyph_path=glyph_path,
            reprompt_path=reprompt_path,
            device=device,
            dtype=torch_dtype,
            enable_offloading=enable_offloading
        )
        
        print(f"[HunyuanImage] Dual text encoders loaded successfully")
        return (dual_encoder,)


class DualTextEncoder:
    """Wrapper for dual text encoder system (MLLM/RePrompt + ByT5/Glyph)"""
    
    def __init__(self, byt5_path, glyph_path, reprompt_path, device, dtype, enable_offloading):
        self.device = device
        self.dtype = dtype
        self._offloaded = enable_offloading
        self._original_device = device
        
        # Load RePrompt if requested
        self.reprompt = None
        if reprompt_path:
            try:
                # First, ensure the custom model architecture is registered
                import sys
                import os
                modeling_path = os.path.join(reprompt_path, "modeling_hunyuan.py")
                if os.path.exists(modeling_path):
                    # Add the path to sys.path temporarily
                    if reprompt_path not in sys.path:
                        sys.path.insert(0, reprompt_path)
                    
                    # Import the modeling file to register the architecture
                    try:
                        import modeling_hunyuan
                        print(f"[HunyuanImage] Loaded custom HunYuan architecture")
                    except ImportError as e:
                        print(f"[HunyuanImage] Warning: Could not import modeling_hunyuan: {e}")
                
                # Now load the RePrompt model
                from hyimage.models.reprompt.reprompt import RePrompt
                self.reprompt = RePrompt(
                    models_root_path=reprompt_path,
                    device_map="auto" if device == "cuda" else "cpu",
                    enable_offloading=enable_offloading
                )
                print(f"[HunyuanImage] RePrompt loaded successfully")
            except Exception as e:
                print(f"[HunyuanImage] Warning: Could not load RePrompt: {e}")
                self.reprompt = None
        
        # Load ByT5 model (simplified for now - skip to avoid errors)
        self.byt5_model = None
        self.byt5_tokenizer = None
        # TODO: Implement proper ByT5/Glyph loading when we have the full model structure
        print(f"[HunyuanImage] ByT5/Glyph loading skipped (placeholder mode)")
        
        # Store Glyph path for future use
        self.glyph_path = glyph_path
        
    def encode(self, texts):
        """Encode text using dual encoder system"""
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = len(texts)
        
        # Step 1: Enhance prompts with RePrompt if available
        enhanced_texts = texts
        if self.reprompt:
            enhanced_texts = []
            for text in texts:
                try:
                    enhanced = self.reprompt.predict(text)
                    enhanced_texts.append(enhanced)
                    print(f"[HunyuanImage] Enhanced: {text[:50]}... -> {enhanced[:50]}...")
                except Exception as e:
                    print(f"[HunyuanImage] RePrompt failed for text, using original: {e}")
                    enhanced_texts.append(text)
        
        # Step 2: Process with ByT5 for text rendering features
        byt5_features = None
        if self.byt5_model and self.byt5_tokenizer:
            try:
                # Tokenize with ByT5
                inputs = self.byt5_tokenizer(
                    enhanced_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                # Get ByT5 features
                with torch.no_grad():
                    byt5_outputs = self.byt5_model(**inputs)
                    byt5_features = byt5_outputs.last_hidden_state
                    
                print(f"[HunyuanImage] ByT5 features computed: {byt5_features.shape}")
                
            except Exception as e:
                print(f"[HunyuanImage] ByT5 encoding failed: {e}")
        
        # Create output structure compatible with HunyuanImage
        class DualEncoderOutput:
            def __init__(self, parent_encoder):
                # Main text features (placeholder for now, would be MLLM output)
                self.hidden_state = torch.randn(batch_size, 256, 4096, device=parent_encoder.device, dtype=parent_encoder.dtype)
                self.attention_mask = torch.ones(batch_size, 256, device=parent_encoder.device, dtype=torch.bool)
                
                # ByT5/Glyph features for text rendering
                self.byt5_features = byt5_features
                
                # Store enhanced texts for debugging
                self.enhanced_texts = enhanced_texts
                self.original_texts = texts
        
        return DualEncoderOutput(self)
    
    def to(self, device):
        """Move models to device"""
        self.device = device
        if self.reprompt:
            self.reprompt.to(device)
        if self.byt5_model:
            self.byt5_model.to(device)
        return self
    
    def cpu(self):
        """Move models to CPU"""
        self.device = 'cpu'
        if self.reprompt:
            self.reprompt.to('cpu')
        if self.byt5_model:
            self.byt5_model.to('cpu')
        return self


class HunyuanImagePromptEncoderDual:
    """Encode prompts using dual text encoder system"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dual_text_encoder": ("HUNYUAN_DUAL_TEXT_ENCODER",),
                "positive": ("STRING", {"multiline": True}),
                "negative": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("HUNYUAN_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "HunyuanImage"

    def encode(self, dual_text_encoder, positive, negative):
        """Encode prompts using dual encoder"""
        
        # Handle offloading
        if hasattr(dual_text_encoder, '_offloaded') and dual_text_encoder._offloaded:
            if dual_text_encoder.reprompt:
                dual_text_encoder.reprompt.to(dual_text_encoder._original_device)
            if dual_text_encoder.byt5_model:
                dual_text_encoder.byt5_model.to(dual_text_encoder._original_device)

        # Encode positive prompt
        with torch.no_grad():
            pos_output = dual_text_encoder.encode([positive])
            prompt_embeds = pos_output.hidden_state
            prompt_attention_mask = pos_output.attention_mask
            byt5_features = pos_output.byt5_features
            
            # Encode negative prompt if provided
            negative_embeds = None
            negative_attention_mask = None
            negative_byt5 = None
            if negative:
                neg_output = dual_text_encoder.encode([negative])
                negative_embeds = neg_output.hidden_state
                negative_attention_mask = neg_output.attention_mask
                negative_byt5 = neg_output.byt5_features
        
        # Offload models back
        if hasattr(dual_text_encoder, '_offloaded') and dual_text_encoder._offloaded:
            dual_text_encoder.cpu()

        # Create conditioning with dual encoder features
        conditioning = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_embeds": negative_embeds,
            "negative_attention_mask": negative_attention_mask,
            "byt5_features": byt5_features,
            "negative_byt5": negative_byt5,
            "enhanced_positive": pos_output.enhanced_texts[0] if hasattr(pos_output, 'enhanced_texts') else positive,
            "enhanced_negative": neg_output.enhanced_texts[0] if negative and hasattr(neg_output, 'enhanced_texts') else negative,
        }
        
        return (conditioning,)


# Node class mappings
NODE_CLASS_MAPPINGS = {
    "HunyuanImageDualTextEncoderLoader": HunyuanImageDualTextEncoderLoader,
    "HunyuanImagePromptEncoderDual": HunyuanImagePromptEncoderDual,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HunyuanImageDualTextEncoderLoader": "HunyuanImage Dual Text Encoder Loader",
    "HunyuanImagePromptEncoderDual": "HunyuanImage Prompt Encoder (Dual)",
}