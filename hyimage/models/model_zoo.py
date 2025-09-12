import os
import copy

from hyimage.common.config import LazyCall as L
from hyimage.models.hunyuan.configs.hunyuanimage_config import (
    hunyuanimage_v2_1_cfg,
    hunyuanimage_v2_1_distilled_cfg,
    hunyuanimage_refiner_cfg,
)
from hyimage.models.vae import load_refiner_vae, load_vae
from hyimage.common.config.base_config import (
    DiTConfig,
    RepromptConfig,
    TextEncoderConfig,
    VAEConfig,
)
from hyimage.models.text_encoder import TextEncoder
from model_manager import model_manager

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# =============================================================================
# V2.1 MODELS
# =============================================================================

def HUNYUANIMAGE_V2_1_TEXT_ENCODER(**kwargs):
    model_name = "hunyuanimage-v2.1"
    text_encoder_path = model_manager.get_model_path(model_name, "text_encoders", "mllm")
    
    # Debug logging
    import loguru
    loguru.logger.info(f"[HUNYUANIMAGE_V2_1_TEXT_ENCODER] text_encoder_path: {text_encoder_path}")
    
    return TextEncoderConfig(
        model=L(TextEncoder)(
            text_encoder_type="llm",
            max_length=1000,
            text_encoder_precision='fp16',
            tokenizer_type="llm",
            text_encoder_path=None,
            prompt_template=None,
            prompt_template_video=None,
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=None,
        ),
        prompt_template="dit-llm-encode-v2",
        load_from=text_encoder_path,
        text_len=1000,
    )


def HUNYUANIMAGE_V2_1_VAE_32x(**kwargs):
    model_name = "hunyuanimage-v2.1"
    vae_path = model_manager.get_model_path(model_name, "vae")
    
    return VAEConfig(
        model=L(load_vae)(
            vae_path=None,
            device="cuda",
        ),
        load_from=vae_path,
        cpu_offload=False,
    )


def HUNYUANIMAGE_V2_1_DIT(**kwargs):
    model_name = "hunyuanimage-v2.1"
    dit_path = model_manager.get_model_path(model_name, "dit")
    
    return DiTConfig(
        model=copy.deepcopy(hunyuanimage_v2_1_cfg),
        use_lora=False,
        use_cpu_offload=False,
        gradient_checkpointing=True,
        load_from=os.path.join(dit_path, "hunyuanimage2.1.safetensors") if dit_path else None,
        use_compile=True,
    )


def HUNYUANIMAGE_V2_1_DIT_CFG_DISTILL(**kwargs):
    model_name = "hunyuanimage-v2.1-distilled"
    dit_path = model_manager.get_model_path(model_name, "dit")
    
    return DiTConfig(
        model=copy.deepcopy(hunyuanimage_v2_1_distilled_cfg),
        use_lora=False,
        use_cpu_offload=False,
        gradient_checkpointing=True,
        load_from=os.path.join(dit_path, "hunyuanimage2.1-distilled.safetensors") if dit_path else None,
        use_compile=True,
    )

# =============================================================================
# REFINER MODELS
# =============================================================================

def HUNYUANIMAGE_REFINER_DIT(**kwargs):
    model_name = "hunyuanimage-v2.1"
    refiner_path = model_manager.get_model_path(model_name, "refiner")
    
    return DiTConfig(
        model=copy.deepcopy(hunyuanimage_refiner_cfg),
        use_lora=False,
        use_cpu_offload=False,
        gradient_checkpointing=True,
        load_from=os.path.join(refiner_path, "hunyuanimage-refiner.safetensors") if refiner_path else None,
        use_compile=True,
    )

def HUNYUANIMAGE_REFINER_VAE_16x(**kwargs):
    model_name = "hunyuanimage-v2.1"
    refiner_path = model_manager.get_model_path(model_name, "refiner")
    
    return VAEConfig(
        model=L(load_refiner_vae)(
            vae_path=None,
            device="cuda",
        ),
        load_from=os.path.join(refiner_path, "vae_refiner") if refiner_path else None,
        cpu_offload=False,
    )


def HUNYUANIMAGE_REFINER_TEXT_ENCODER(**kwargs):
    model_name = "hunyuanimage-v2.1"
    text_encoder_path = model_manager.get_model_path(model_name, "text_encoders", "mllm")
    
    return TextEncoderConfig(
        model=L(TextEncoder)(
            text_encoder_type="llm",
            max_length=1000,
            text_encoder_precision='fp16',
            tokenizer_type="llm",
            text_encoder_path=None,
            prompt_template=None,
            prompt_template_video=None,
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=None,
        ),
        prompt_template="dit-llm-encode",
        load_from=text_encoder_path,
        text_len=256,
    )


# =============================================================================
# SPECIALIZED MODELS
# =============================================================================

def HUNYUANIMAGE_REPROMPT(**kwargs):
    from hyimage.models.reprompt import RePrompt
    model_name = "hunyuanimage-v2.1"
    reprompt_path = model_manager.get_model_path(model_name, "text_encoders", "mllm")
    
    return RepromptConfig(
        model=L(RePrompt)(
            models_root_path=None,
            device_map="auto",
        ),
        load_from=reprompt_path,
    )