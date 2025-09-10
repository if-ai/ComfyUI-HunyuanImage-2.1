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

try:
    import folder_paths
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

logger = logging.getLogger(__name__)

# Model configurations
MODEL_CONFIGS = {
    "hunyuanimage-v2.1": {
        "dit": {
            "repo": "tencent/HunyuanImage-2.1",
            "files": [
                "dit/hunyuanimage2.1.safetensors"
            ],
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
            "folder": "diffusion_models",
            "subfolder": "hunyuanimage-v2.1-distilled"
        },
        # VAE and text encoders are shared with base model
        "vae": "hunyuanimage-v2.1",
        "text_encoders": "hunyuanimage-v2.1",
        "refiner": "hunyuanimage-v2.1"
    }
}


class HunyuanModelManager:
    """Manages HunyuanImage model paths and downloading"""
    
    def __init__(self):
        self.model_paths = {}
        self._setup_folders()
        
    def _setup_folders(self):
        """Add HunyuanImage specific folders to ComfyUI paths"""
        # Ensure folders exist in ComfyUI's model directory structure
        for folder_type in ["diffusion_models", "vae", "text_encoders"]:
            if folder_type in folder_paths.folder_names_and_paths:
                paths = folder_paths.folder_names_and_paths[folder_type][0]
                for path in paths:
                    os.makedirs(path, exist_ok=True)
    
    def get_model_path(self, model_name: str, component: str) -> Optional[str]:
        """Get the path for a specific model component"""
        
        if model_name not in MODEL_CONFIGS:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        config = MODEL_CONFIGS[model_name]
        
        # Handle references to other models
        if isinstance(config.get(component), str):
            return self.get_model_path(config[component], component)
        
        if component == "text_encoders":
            # Return paths for all text encoders
            paths = {}
            for encoder_name, encoder_config in config[component].items():
                folder_type = encoder_config["folder"]
                subfolder = encoder_config["subfolder"]
                paths[encoder_name] = self._get_component_path(folder_type, subfolder)
            return paths
        else:
            if component not in config:
                logger.error(f"Component {component} not found for {model_name}")
                return None
            
            comp_config = config[component]
            folder_type = comp_config["folder"]
            subfolder = comp_config["subfolder"]
            return self._get_component_path(folder_type, subfolder)
    
    def _get_component_path(self, folder_type: str, subfolder: str) -> str:
        """Get the full path for a component"""
        if folder_type in folder_paths.folder_names_and_paths:
            base_paths = folder_paths.folder_names_and_paths[folder_type][0]
            if base_paths:
                return os.path.join(base_paths[0], subfolder)
        return None
    
    def check_models_exist(self, model_name: str) -> Dict[str, bool]:
        """Check which model components exist"""
        results = {}
        
        if model_name not in MODEL_CONFIGS:
            return results
        
        config = MODEL_CONFIGS[model_name]
        
        for component in ["dit", "vae", "refiner"]:
            if component in config:
                if isinstance(config[component], str):
                    # Reference to another model
                    ref_results = self.check_models_exist(config[component])
                    results[component] = ref_results.get(component, False)
                else:
                    path = self.get_model_path(model_name, component)
                    if path:
                        # Check if model files exist based on the config
                        component_config = config.get(component, {})
                        files_to_check = component_config.get("files", [])
                        
                        if files_to_check:
                            exists = all(os.path.exists(os.path.join(path, os.path.basename(f))) for f in files_to_check)
                        else:
                            # Fallback for older configs or components without explicit file lists
                            model_file = os.path.join(path, "pytorch_model.bin")
                            safetensors_file = os.path.join(path, "model.safetensors")
                            exists = os.path.exists(model_file) or os.path.exists(safetensors_file)
                        
                        results[component] = exists
                    else:
                        results[component] = False
        
        # Check text encoders
        if "text_encoders" in config:
            if isinstance(config["text_encoders"], str):
                ref_results = self.check_models_exist(config["text_encoders"])
                results["text_encoders"] = ref_results.get("text_encoders", {})
            else:
                encoder_results = {}
                for encoder_name, encoder_config in config["text_encoders"].items():
                    paths = self.get_model_path(model_name, "text_encoders")
                    if paths and encoder_name in paths:
                        encoder_path = paths[encoder_name]
                        
                        # More robust check for text encoders
                        if encoder_name == "mllm":
                            # Specific check for the reprompt model
                            has_model = os.path.exists(os.path.join(encoder_path, "config.json")) and \
                                        os.path.exists(os.path.join(encoder_path, "model.safetensors.index.json"))
                        else:
                            # Fallback for other text encoders
                            has_model = any(
                                os.path.exists(os.path.join(encoder_path, f))
                                for f in ["pytorch_model.bin", "model.safetensors", "config.json"]
                            )
                        encoder_results[encoder_name] = has_model
                    else:
                        encoder_results[encoder_name] = False
                results["text_encoders"] = encoder_results
        
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
        if folder_type in folder_paths.folder_names_and_paths:
            base_paths = folder_paths.folder_names_and_paths[folder_type][0]
            if base_paths:
                target_dir = os.path.join(base_paths[0], subfolder)
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
                            hf_hub_download(
                                repo,
                                filename=file,
                                local_dir=target_dir,
                                local_dir_use_symlinks=False,
                                resume_download=True,
                            )
                            # Manually move file to the correct location if it's a reprompt file
                            source_path = os.path.join(target_dir, file)
                            dest_path = os.path.join(target_dir, os.path.basename(file))
                            if source_path != dest_path and os.path.exists(source_path):
                                os.rename(source_path, dest_path)
                                # Clean up empty directories
                                dir_to_clean = os.path.dirname(source_path)
                                if not os.listdir(dir_to_clean):
                                    os.rmdir(dir_to_clean)

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
        """Download from ModelScope"""
        try:
            print(f"[HunyuanImage] Downloading {repo} from ModelScope to {target_dir}")
            
            cmd = ["modelscope", "download", "--model", repo, "--local_dir", target_dir]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='ignore'
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to download {repo}: {result.stderr}")
                return False
            
            print(f"[HunyuanImage] Successfully downloaded {repo}")
            return True
            
        except FileNotFoundError:
            logger.error("modelscope CLI not found. Install with: pip install modelscope")
            return False
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def ensure_models(self, model_name: str, auto_download: bool = True) -> bool:
        """Ensure all model components are available"""
        
        print(f"[HunyuanImage] Checking models for {model_name}")
        exists = self.check_models_exist(model_name)
        
        missing = []
        for component, status in exists.items():
            if component == "text_encoders":
                if isinstance(status, dict):
                    for encoder_name, encoder_exists in status.items():
                        if not encoder_exists:
                            missing.append(f"text_encoders:{encoder_name}")
            elif not status:
                missing.append(component)
        
        if not missing:
            print(f"[HunyuanImage] All models found for {model_name}")
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
    
    def get_model_root(self, model_name: str) -> str:
        """Get the root directory for model components"""
        paths = {
            "dit": self.get_model_path(model_name, "dit"),
            "vae": self.get_model_path(model_name, "vae"),
            "text_encoders": self.get_model_path(model_name, "text_encoders"),
            "refiner": self.get_model_path(model_name, "refiner")
        }
        
        # Create a temporary root that contains symlinks to actual paths
        temp_root = os.path.join(folder_paths.get_temp_directory(), f"hunyuan_{model_name}")
        os.makedirs(temp_root, exist_ok=True)
        
        # Create structure expected by HunyuanImage
        for component, path in paths.items():
            if path:
                if component == "text_encoders" and isinstance(path, dict):
                    text_encoder_dir = os.path.join(temp_root, "text_encoder")
                    os.makedirs(text_encoder_dir, exist_ok=True)
                    for encoder_name, encoder_path in path.items():
                        link_path = os.path.join(text_encoder_dir, encoder_name)
                        if not os.path.exists(link_path) and os.path.exists(encoder_path):
                            try:
                                os.symlink(encoder_path, link_path)
                            except:
                                # Fallback to copying on Windows
                                import shutil
                                shutil.copytree(encoder_path, link_path, dirs_exist_ok=True)
                else:
                    link_path = os.path.join(temp_root, component)
                    if not os.path.exists(link_path) and os.path.exists(path):
                        try:
                            os.symlink(path, link_path)
                        except:
                            # Fallback to copying on Windows
                            import shutil
                            shutil.copytree(path, link_path, dirs_exist_ok=True)
        
        return temp_root


# Global instance
model_manager = HunyuanModelManager()