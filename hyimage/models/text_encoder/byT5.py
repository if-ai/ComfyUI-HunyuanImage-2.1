from transformers import T5Config, T5EncoderModel, T5Tokenizer
import torch
import comfy.utils

def load_glyph_byT5_v2(args, device="cuda"):
    """
    Load byT5 model for glyph processing.
    """
    byt5_tokenizer = T5Tokenizer.from_pretrained(args["byT5_google_path"])
    config = T5Config.from_pretrained(args["byT5_google_path"])
    config.use_cache = False
    
    with torch.no_grad():
        byt5_model = T5EncoderModel(config)
        
        # Determine file type and load accordingly
        ckpt_path = args["byT5_ckpt_path"]
        state_dict = comfy.utils.load_torch_file(ckpt_path)

        # Adapt key names if necessary
        if "model" in state_dict:
            state_dict = state_dict["model"]
        
        # Remove unexpected keys
        expected_keys = set(byt5_model.state_dict().keys())
        unexpected_keys = [k for k in state_dict.keys() if k not in expected_keys]
        if unexpected_keys:
            print(f"[HunyuanImage] Warning: Removing unexpected keys from byT5 state_dict: {unexpected_keys}")
            for k in unexpected_keys:
                del state_dict[k]
        
        byt5_model.load_state_dict(state_dict, strict=False)

    byt5_model.eval()
    byt5_model.to(device=device)

    return {
        "byt5_tokenizer": byt5_tokenizer,
        "byt5_model": byt5_model,
        "byt5_max_length": args["byt5_max_length"],
    }
