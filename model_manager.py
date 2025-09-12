"""
Model management for HunyuanImage with ComfyUI integration
Handles model paths and automatic downloading
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HUGGINGFACE_HUB_AVAILABLE = True
except ImportError:
    HUGGINGFACE_HUB_AVAILABLE = False

# Defer folder_paths import
folder_paths = None

def get_folder_paths():
    global folder_paths
    if folder_paths is None:
        try:
            import folder_paths as comfy_folder_paths
            folder_paths = comfy_folder_paths
        except ImportError:
            print("[HunyuanImage] Warning: folder_paths not available, using fallback paths")
            # Fallback for when running outside ComfyUI
            class FallbackFolderPaths:
                folder_names_and_paths = {
                    "diffusion_models": ([os.path.join(os.getcwd(), "models", "diffusion_models")], {".safetensors", ".bin", ".pt"}),
                    "vae": ([os.path.join(os.getcwd(), "models", "vae")], {".safetensors", ".bin", ".pt"}),
                    "text_encoders": ([os.path.join(os.getcwd(), "models", "text_encoders")], {".safetensors", ".bin", ".pt"}),
                }
                
                @staticmethod
                def get_temp_directory():
                    return os.path.join(os.getcwd(), "temp")
            
            folder_paths = FallbackFolderPaths()
    return folder_paths

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "hunyuanimage-v2.1": {
        "dit": {
            "repo": "tencent/HunyuanImage-2.1",
            "files": [
                "dit/hunyuanimage2.1.safetensors"
            ],
            "dest_filename": "hunyuanimage2.1.safetensors",
            "folder": "diffusion_models",
            "subfolder": "hunyuanimage-v2.1"
        },
        "vae": {
            "repo": "tencent/HunyuanImage-2.1", 
            "files": [
                "vae/vae_2_1/pytorch_model.ckpt",
                "vae/vae_2_1/config.json"
            ],
            "folder": "vae",
            "subfolder": "hunyuanimage-v2.1"
        },
        "text_encoders": {
            "mllm": {
                "repo": "tencent/HunyuanImage-2.1",
                "files": [
                    "reprompt/chat_template.jinja",
                    "reprompt/config.json",
                    "reprompt/generation_config.json",
                    "reprompt/hy.tiktoken",
                    "reprompt/model-00001-of-00004.safetensors",
                    "reprompt/model-00002-of-00004.safetensors",
                    "reprompt/model-00003-of-00004.safetensors",
                    "reprompt/model-00004-of-00004.safetensors",
                    "reprompt/model.safetensors.index.json",
                    "reprompt/special_tokens_map.json",
                    "reprompt/tokenization_hy.py",
                    "reprompt/tokenizer_config.json"
                ],
                "folder": "text_encoders",
                "subfolder": "hunyuanimage-v2.1/llm"
            },
            "byt5": {
                "repo": "google/byt5-small",
                "files": ["*"],
                "folder": "text_encoders", 
                "subfolder": "hunyuanimage-v2.1/byt5-small"
            },
            "glyph": {
                "repo": "AI-ModelScope/Glyph-SDXL-v2",
                "files": ["*"],
                "folder": "text_encoders",
                "subfolder": "hunyuanimage-v2.1/Glyph-SDXL-v2",
                "use_modelscope": True
            }
        },
        "refiner": {
            "repo": "tencent/HunyuanImage-2.1",
            "files": [
                "vae/vae_refiner/pytorch_model.pt",
                "vae/vae_refiner/config.json"
            ],
            "folder": "diffusion_models",
            "subfolder": "hunyuanimage-refiner"
        }
    },
    "hunyuanimage-v2.1-distilled": {
        "dit": {
            "repo": "tencent/HunyuanImage-2.1",
            "files": [
                "dit/hunyuanimage2.1-distilled.safetensors"
            ],
            "dest_filename": "hunyuanimage2.1-distilled.safetensors",
            "folder": "diffusion_models",
            "subfolder": "hunyuanimage-v2.1-distilled"
        },
        # VAE and text encoders are shared with base model
        "vae": "hunyuanimage-v2.1",
        "text_encoders": "hunyuanimage-v2.1",
        "refiner": "hunyuanimage-v2.1"
    },
    "hunyuanimage-v2.1-fp8": {
        "dit": {
            "repo": "drbaph/HunyuanImage-2.1_fp8",
            "files": [
                "hunyuanimage2.1_fp8_e4m3fn.safetensors"
            ],
            "dest_filename": "hunyuanimage2.1_fp8_e4m3fn.safetensors",
            "folder": "diffusion_models",
            "subfolder": "hunyuanimage-v2.1-fp8"
        },
        "vae": "hunyuanimage-v2.1",
        "text_encoders": "hunyuanimage-v2.1",
        "refiner": "hunyuanimage-v2.1"
    },
    "hunyuanimage-v2.1-distilled-fp8": {
        "dit": {
            "repo": "drbaph/HunyuanImage-2.1_fp8",
            "files": [
                "hunyuanimage2.1-distilled_fp8_e4m3fn.safetensors"
            ],
            "dest_filename": "hunyuanimage2.1-distilled_fp8_e4m3fn.safetensors",
            "folder": "diffusion_models",
            "subfolder": "hunyuanimage-v2.1-distilled-fp8"
        },
        "vae": "hunyuanimage-v2.1",
        "text_encoders": "hunyuanimage-v2.1",
        "refiner": "hunyuanimage-v2.1"
    },
    "hunyuanimage-refiner-fp8": {
        "refiner": {
            "repo": "drbaph/HunyuanImage-2.1_fp8",
            "files": [
                "hunyuanimage-refiner_fp8_e4m3fn.safetensors"
            ],
            "dest_filename": "hunyuanimage-refiner_fp8_e4m3fn.safetensors",
            "folder": "diffusion_models",
            "subfolder": "hunyuanimage-refiner-fp8"
        },
        "vae": "hunyuanimage-v2.1",
        "text_encoders": "hunyuanimage-v2.1"
    }
}

# Alternative single-file text encoder models
ALT_TEXT_ENCODER_MODELS = {
    # PromptEnhancer models (reprompt) - These work well
    "promptenhancer_int8": {
        "repo": "leeooo001/Hunyuan-PromptEnhancer-INT8",
        "files": [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
            "generation_config.json",
            "special_tokens_map.json",
            "tokenization_hy.py",
            "tokenizer_config.json",
            "hy.tiktoken"
        ],
        "folder": "text_encoders",
        "subfolder": "promptenhancer-int8",
        "type": "reprompt"
    }
}


class HunyuanModelManager:
    """Manages HunyuanImage model paths and downloading"""
    
    def __init__(self):
        self.model_paths = {}
        self._setup_folders()
        self.alt_text_encoders = {}
        
    def _setup_folders(self):
        """Add HunyuanImage specific folders to ComfyUI paths"""
        # Ensure folders exist in ComfyUI's model directory structure
        paths = get_folder_paths()
        for folder_type in ["diffusion_models", "vae", "text_encoders"]:
            if folder_type in paths.folder_names_and_paths:
                base_paths = paths.folder_names_and_paths[folder_type][0]
                for path in base_paths:
                    os.makedirs(path, exist_ok=True)
    
    def get_full_path(self, model_name: str, component: str) -> Optional[str]:
        """Gets the full, absolute path to a model file."""
        component_config = self.get_config(model_name, component)
        if not component_config:
            return None

        model_path = self.get_model_path(model_name, component)
        dest_filename = component_config.get("dest_filename")

        if model_path and dest_filename:
            return os.path.join(model_path, dest_filename)
        
        return model_path # Fallback for components without a single file

    def get_config(self, model_name: str, component: str) -> Optional[Dict]:
        """Get the configuration for a specific model component, resolving references."""
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        config = MODEL_CONFIGS[model_name]
        component_config = config.get(component)
        
        if isinstance(component_config, str):
            # It's a reference to another model's component
            return self.get_config(component_config, component)
        
        return component_config

    def get_model_path(self, model_name: str, component: str, sub_component: Optional[str] = None) -> Optional[str]:
        """Get the path for a specific model component"""
        
        component_config = self.get_config(model_name, component)
        
        if not component_config:
            logger.error(f"Component {component} not found for {model_name}")
            return None
        
        if component == "text_encoders":
            if sub_component:
                if sub_component not in component_config:
                    logger.error(f"Sub-component {sub_component} not found for {component}")
                    return None
                
                encoder_config = component_config[sub_component]
                folder_type = encoder_config["folder"]
                subfolder = encoder_config["subfolder"]
                return self._get_component_path(folder_type, subfolder)
            else:
                paths = {}
                for encoder_name, encoder_config in component_config.items():
                    folder_type = encoder_config["folder"]
                    subfolder = encoder_config["subfolder"]
                    paths[encoder_name] = self._get_component_path(folder_type, subfolder)
                return paths
        else:
            folder_type = component_config["folder"]
            subfolder = component_config["subfolder"]
            return self._get_component_path(folder_type, subfolder)
    
    def _get_component_path(self, folder_type: str, subfolder: str) -> str:
        """Get the full path for a component"""
        paths = get_folder_paths()
        if folder_type in paths.folder_names_and_paths:
            base_paths = paths.folder_names_and_paths[folder_type][0]
            if base_paths:
                # Normalize subfolder to use OS-appropriate separators
                # Convert forward slashes to the OS separator
                subfolder_parts = subfolder.replace('\\', '/').split('/')
                
                # For diffusion_models, prefer the actual diffusion_models folder over unet
                if folder_type == "diffusion_models" and len(base_paths) > 1:
                    # Check if one of the paths contains 'diffusion_models' in its name
                    for path in base_paths:
                        if 'diffusion_models' in path:
                            full_path = os.path.join(path, *subfolder_parts)
                            logger.info(f"[_get_component_path] Returning path for {folder_type}/{subfolder}: {full_path}")
                            return full_path
                # Fall back to first path
                full_path = os.path.join(base_paths[0], *subfolder_parts)
                logger.info(f"[_get_component_path] Returning path for {folder_type}/{subfolder}: {full_path}")
                return full_path
        logger.warning(f"[_get_component_path] Could not find path for {folder_type}/{subfolder}")
        return None
    
    def check_models_exist(self, model_name: str) -> Dict[str, bool]:
        """Check which model components exist"""
        results = {}
        
        if model_name not in MODEL_CONFIGS:
            return results
        
        config = MODEL_CONFIGS[model_name]
        
        for component in config.keys():
            component_config = self.get_config(model_name, component)

            if not component_config:
                results[component] = False
                continue

            if component == "text_encoders":
                encoder_results = {}
                for encoder_name in component_config.keys():
                    encoder_path = self.get_model_path(model_name, "text_encoders", encoder_name)
                    
                    if not encoder_path or not os.path.exists(encoder_path):
                        encoder_results[encoder_name] = False
                    elif encoder_name == "mllm":
                        # For mllm, check if reprompt model weight files exist
                        required_files = [
                            "model-00001-of-00004.safetensors",
                            "model-00002-of-00004.safetensors",
                            "model-00003-of-00004.safetensors",
                            "model-00004-of-00004.safetensors",
                            "model.safetensors.index.json"
                        ]
                        all_exist = all(os.path.exists(os.path.join(encoder_path, f)) for f in required_files)
                        encoder_results[encoder_name] = all_exist
                    elif encoder_name == "byt5":
                        # For byt5, check if model files exist
                        required_files = ["pytorch_model.bin", "config.json"]
                        all_exist = all(os.path.exists(os.path.join(encoder_path, f)) for f in required_files)
                        encoder_results[encoder_name] = all_exist
                    elif encoder_name == "glyph":
                        # For glyph, check if checkpoint exists
                        checkpoint_path = os.path.join(encoder_path, "checkpoints", "byt5_model.pt")
                        encoder_results[encoder_name] = os.path.exists(checkpoint_path)
                    else:
                        # Default check for other encoders
                        encoder_results[encoder_name] = bool(os.listdir(encoder_path))
                results["text_encoders"] = encoder_results
            else:
                full_path = self.get_full_path(model_name, component)
                if full_path and os.path.exists(full_path):
                    results[component] = True
                # Handle folder-based components like VAE
                elif full_path and os.path.isdir(self.get_model_path(model_name, component)):
                    results[component] = True
                else:
                    # More thorough check for VAE/Refiner which are folders
                    path = self.get_model_path(model_name, component)
                    if path and os.path.exists(path) and os.path.isdir(path) and os.listdir(path):
                         results[component] = True
                    else:
                        results[component] = False
        
        return results
    
    def download_component(self, model_name: str, component: str, encoder_name: Optional[str] = None) -> bool:
        """Download a specific model component"""
        
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        config = MODEL_CONFIGS[model_name]
        
        # Handle references
        if isinstance(config.get(component), str):
            return self.download_component(config[component], component, encoder_name)
        
        if component == "text_encoders" and encoder_name:
            if encoder_name not in config["text_encoders"]:
                logger.error(f"Unknown encoder: {encoder_name}")
                return False
            
            encoder_config = config["text_encoders"][encoder_name]
            return self._download_from_config(encoder_config)
        elif component in config:
            return self._download_from_config(config[component])
        
        return False
    
    def _download_from_config(self, config: Dict) -> bool:
        """Download model based on configuration"""
        
        folder_type = config["folder"]
        subfolder = config["subfolder"]
        repo = config["repo"]
        files = config.get("files", ["*"])
        use_modelscope = config.get("use_modelscope", False)
        
        # Get target directory
        paths = get_folder_paths()
        if folder_type in paths.folder_names_and_paths:
            base_paths = paths.folder_names_and_paths[folder_type][0]
            if base_paths:
                # Normalize subfolder to use OS-appropriate separators
                subfolder_parts = subfolder.replace('\\', '/').split('/')
                target_dir = os.path.join(base_paths[0], *subfolder_parts)
            else:
                logger.error(f"No base path for folder type: {folder_type}")
                return False
        else:
            logger.error(f"Unknown folder type: {folder_type}")
            return False
        
        os.makedirs(target_dir, exist_ok=True)
        
        if use_modelscope:
            return self._download_modelscope(repo, target_dir)
        else:
            return self._download_huggingface(repo, target_dir, files)
    
    def _download_huggingface(self, repo: str, target_dir: str, files: List[str]) -> bool:
        """Download from HuggingFace using huggingface_hub library with retries."""
        if not HUGGINGFACE_HUB_AVAILABLE:
            logger.error("huggingface_hub library not found. Please install it with: pip install huggingface_hub")
            print("[HunyuanImage] huggingface_hub library not found.")
            return False

        retries = 3
        delay = 10  # seconds between retries

        try:
            print(f"[HunyuanImage] Downloading {repo} to {target_dir} using huggingface_hub")

            if files == ["*"]:
                for attempt in range(retries):
                    try:
                        snapshot_download(
                            repo,
                            local_dir=target_dir,
                            local_dir_use_symlinks=False,
                            resume_download=True,
                        )
                        print(f"[HunyuanImage] Successfully downloaded {repo}")
                        return True
                    except Exception as e:
                        if attempt < retries - 1:
                            print(f"[HunyuanImage] Download attempt {attempt + 1}/{retries} failed. Retrying in {delay}s...")
                            time.sleep(delay)
                        else:
                            print(f"[HunyuanImage] Final download attempt failed for {repo}.")
                            raise e
            else:
                for file in files:
                    for attempt in range(retries):
                        try:
                            downloaded_path = hf_hub_download(
                                repo,
                                filename=file,
                                local_dir=target_dir,
                                local_dir_use_symlinks=False,
                                resume_download=True,
                            )
                            
                            # Handle reprompt files - move them to the correct location
                            if "reprompt/" in file:
                                source_path = os.path.join(target_dir, file)
                                dest_path = os.path.join(target_dir, os.path.basename(file))
                                if source_path != dest_path and os.path.exists(source_path):
                                    # Check if destination exists before moving
                                    if not os.path.exists(dest_path):
                                        os.rename(source_path, dest_path)
                                    # Clean up empty reprompt directory
                                    reprompt_dir = os.path.join(target_dir, "reprompt")
                                    if os.path.exists(reprompt_dir) and not os.listdir(reprompt_dir):
                                        os.rmdir(reprompt_dir)

                            print(f"[HunyuanImage] Successfully downloaded {file}")
                            break  # Move to the next file
                        except Exception as e:
                            print(f"[HunyuanImage]! Exception during download of {file}. See console for details.")
                            import traceback
                            traceback.print_exc()
                            if attempt < retries - 1:
                                print(f"[HunyuanImage] Download attempt {attempt + 1}/{retries} for {file} failed. Retrying in {delay}s...")
                                time.sleep(delay)
                            else:
                                print(f"[HunyuanImage] Final download attempt failed for {file}.")
                                raise e # Propagate error for the specific file
                return True

        except Exception as e:
            logger.error(f"Download error using huggingface_hub for {repo}: {e}")
            print(f"[HunyuanImage] Download error for {repo}. See console for details.")
            import traceback
            traceback.print_exc()
            return False
    
    def _download_modelscope(self, repo: str, target_dir: str) -> bool:
        """Download from ModelScope with retries."""
        retries = 3
        delay = 10  # seconds

        for attempt in range(retries):
            try:
                print(f"[HunyuanImage] Downloading {repo} from ModelScope to {target_dir} (Attempt {attempt + 1}/{retries})")
                
                cmd = ["modelscope", "download", "--model", repo, "--local_dir", target_dir]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore'
                )
                
                if result.returncode == 0:
                    print(f"[HunyuanImage] Successfully downloaded {repo}")
                    return True
                else:
                    logger.error(f"Attempt {attempt + 1} failed to download {repo}: {result.stderr}")
                    if attempt < retries - 1:
                        print(f"[HunyuanImage] Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        print(f"[HunyuanImage] Final attempt failed for {repo}.")
                        return False
            
            except FileNotFoundError:
                logger.error("modelscope CLI not found. Install with: pip install modelscope")
                return False
            except Exception as e:
                logger.error(f"Download error on attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    print(f"[HunyuanImage] An unexpected error occurred. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"[HunyuanImage] Final attempt failed due to an unexpected error.")
                    return False
        return False
    
    def ensure_models(self, model_name: str, auto_download: bool = True) -> bool:
        """Ensure all model components are available"""
        
        print(f"[HunyuanImage] Checking models for {model_name}")
        
        if model_name not in MODEL_CONFIGS:
            print(f"[HunyuanImage] Unknown model: {model_name}")
            return False
            
        config = MODEL_CONFIGS[model_name]
        missing = []
        
        # Check each component
        for component in config.keys():
            component_config = config[component]
            
            # Skip if it's a reference to another model's component
            if isinstance(component_config, str):
                # This is a reference, don't check or download it here
                continue
                
            # Check if this component exists
            if component == "text_encoders":
                # Check each encoder
                for encoder_name in component_config.keys():
                    encoder_path = self.get_model_path(model_name, "text_encoders", encoder_name)
                    if not encoder_path or not os.path.exists(encoder_path):
                        missing.append(f"text_encoders:{encoder_name}")
            else:
                # Check single component
                component_path = self.get_full_path(model_name, component)
                if not component_path or not os.path.exists(component_path):
                    missing.append(component)
        
        if not missing:
            print(f"[HunyuanImage] All required components found for {model_name}")
            return True
        
        print(f"[HunyuanImage] Missing components: {missing}")
        
        if not auto_download:
            return False
        
        # Download missing components
        for component_str in missing:
            if ":" in component_str:
                component, encoder_name = component_str.split(":")
                print(f"[HunyuanImage] Downloading {component}/{encoder_name}")
                if not self.download_component(model_name, component, encoder_name):
                    print(f"[HunyuanImage]! Failed to download {component}/{encoder_name}")
                    logger.error(f"Failed to download {component}/{encoder_name}")
                    return False
            else:
                print(f"[HunyuanImage] Downloading {component_str}")
                if not self.download_component(model_name, component_str):
                    print(f"[HunyuanImage]! Failed to download {component_str}")
                    logger.error(f"Failed to download {component_str}")
                    return False
        
        return True
    
    def check_alt_text_encoder(self, encoder_name: str) -> bool:
        """Check if an alternative text encoder exists"""
        if encoder_name not in ALT_TEXT_ENCODER_MODELS:
            return False
            
        config = ALT_TEXT_ENCODER_MODELS[encoder_name]
        folder_type = config["folder"]
        
        paths = get_folder_paths()
        if folder_type in paths.folder_names_and_paths:
            base_paths = paths.folder_names_and_paths[folder_type][0]
            if base_paths:
                # Handle multi-file models
                if "files" in config:
                    subfolder = config.get("subfolder", encoder_name)
                    model_dir = os.path.join(base_paths[0], subfolder)
                    if not os.path.exists(model_dir):
                        return False
                    # Check if key files exist
                    if config["type"] == "reprompt":
                        required_files = ["model.safetensors.index.json"]
                    elif config["type"] == "mllm":
                        required_files = ["config.json"]
                    else:
                        required_files = []
                    return all(os.path.exists(os.path.join(model_dir, f)) for f in required_files)
                # Handle single-file models
                else:
                    file_path = os.path.join(base_paths[0], f"{encoder_name}.safetensors")
                    return os.path.exists(file_path)
        return False
    
    def download_alt_text_encoder(self, encoder_name: str) -> bool:
        """Download an alternative text encoder model"""
        if encoder_name not in ALT_TEXT_ENCODER_MODELS:
            print(f"[HunyuanImage] Unknown alternative text encoder: {encoder_name}")
            return False
            
        config = ALT_TEXT_ENCODER_MODELS[encoder_name]
        folder_type = config["folder"]
        
        paths = get_folder_paths()
        if folder_type not in paths.folder_names_and_paths:
            print(f"[HunyuanImage] Unknown folder type: {folder_type}")
            return False
            
        base_paths = paths.folder_names_and_paths[folder_type][0]
        if not base_paths:
            print(f"[HunyuanImage] No base path for folder type: {folder_type}")
            return False
        
        # Handle multi-file models
        if "files" in config:
            subfolder = config.get("subfolder", encoder_name)
            target_dir = os.path.join(base_paths[0], subfolder)
            
            # Check if already exists
            if self.check_alt_text_encoder(encoder_name):
                print(f"[HunyuanImage] {encoder_name} already exists")
                return True
            
            os.makedirs(target_dir, exist_ok=True)
            
            try:
                print(f"[HunyuanImage] Downloading {encoder_name} from {config['repo']}")
                from huggingface_hub import hf_hub_download, snapshot_download
                
                retries = 3
                delay = 10  # seconds

                if config["files"] == ["*"]:
                    # Download entire repository
                    for attempt in range(retries):
                        try:
                            snapshot_download(
                                repo_id=config['repo'],
                                local_dir=target_dir,
                                local_dir_use_symlinks=False,
                                resume_download=True
                            )
                            break # Success
                        except Exception as e:
                            if attempt < retries - 1:
                                print(f"[HunyuanImage] Download attempt {attempt + 1}/{retries} for {encoder_name} failed. Retrying in {delay}s...")
                                time.sleep(delay)
                            else:
                                print(f"[HunyuanImage] Final download attempt failed for {encoder_name}.")
                                raise e
                else:
                    # Download specific files
                    for file in config["files"]:
                        for attempt in range(retries):
                            try:
                                hf_hub_download(
                                    repo_id=config['repo'],
                                    filename=file,
                                    local_dir=target_dir,
                                    local_dir_use_symlinks=False,
                                    resume_download=True
                                )
                                break # Move to next file
                            except Exception as e:
                                if attempt < retries - 1:
                                    print(f"[HunyuanImage] Download attempt {attempt + 1}/{retries} for {file} failed. Retrying in {delay}s...")
                                    time.sleep(delay)
                                else:
                                    print(f"[HunyuanImage] Final download attempt for {file} failed.")
                                    raise e
                
                print(f"[HunyuanImage] Successfully downloaded {encoder_name}")
                return True
                
            except Exception as e:
                print(f"[HunyuanImage] Failed to download {encoder_name}: {e}")
                return False
        
        # Handle single-file models
        else:
            target_file = os.path.join(base_paths[0], f"{encoder_name}.safetensors")
            
            # Check if already exists
            if os.path.exists(target_file):
                print(f"[HunyuanImage] {encoder_name} already exists")
                return True
                
            try:
                print(f"[HunyuanImage] Downloading {encoder_name} from {config['repo']}")
                from huggingface_hub import hf_hub_download
                
                retries = 3
                delay = 10 # seconds

                for attempt in range(retries):
                    try:
                        downloaded_file = hf_hub_download(
                            repo_id=config['repo'],
                            filename=config['filename'],
                            local_dir=base_paths[0],
                            local_dir_use_symlinks=False,
                            resume_download=True
                        )
                        break # Success
                    except Exception as e:
                        if attempt < retries - 1:
                            print(f"[HunyuanImage] Download attempt {attempt + 1}/{retries} for {encoder_name} failed. Retrying in {delay}s...")
                            time.sleep(delay)
                        else:
                            print(f"[HunyuanImage] Final download attempt failed for {encoder_name}.")
                            raise e

                # Move to the correct name
                if downloaded_file != target_file:
                    os.rename(downloaded_file, target_file)
                    # Clean up directory structure
                    split_dir = os.path.join(base_paths[0], "split_files")
                    if os.path.exists(split_dir) and not os.listdir(split_dir):
                        import shutil
                        shutil.rmtree(split_dir)
                
                print(f"[HunyuanImage] Successfully downloaded {encoder_name}")
                return True
                
            except Exception as e:
                print(f"[HunyuanImage] Failed to download {encoder_name}: {e}")
                return False
    
    def get_alt_text_encoder_path(self, encoder_name: str) -> Optional[str]:
        """Get the path to an alternative text encoder"""
        if encoder_name not in ALT_TEXT_ENCODER_MODELS:
            return None
            
        config = ALT_TEXT_ENCODER_MODELS[encoder_name]
        folder_type = config["folder"]
        
        paths = get_folder_paths()
        if folder_type in paths.folder_names_and_paths:
            base_paths = paths.folder_names_and_paths[folder_type][0]
            if base_paths:
                # Handle multi-file models - return directory path
                if "files" in config:
                    subfolder = config.get("subfolder", encoder_name)
                    return os.path.join(base_paths[0], subfolder)
                # Handle single-file models
                else:
                    return os.path.join(base_paths[0], f"{encoder_name}.safetensors")
        return None
    
    def get_alt_text_encoder_list(self) -> List[str]:
        """Get list of available alternative text encoders"""
        return list(ALT_TEXT_ENCODER_MODELS.keys())
    
    def get_available_models(self) -> List[str]:
        """Get list of available model configurations"""
        return list(MODEL_CONFIGS.keys())
    
    def check_and_download_models(self, model_name: str, auto_download: bool = True) -> bool:
        """Check if models exist and download if needed"""
        return self.ensure_models(model_name, auto_download)
    
    def get_component_path(self, folder_type: str, subfolder: str) -> str:
        """Get the path for a component"""
        return self._get_component_path(folder_type, subfolder)


# Global instance
model_manager = HunyuanModelManager()