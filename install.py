#!/usr/bin/env python3
"""
Installation script for ComfyUI-HunyuanImage-2.1
Ensures all dependencies are properly installed
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n[HunyuanImage] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[HunyuanImage] Warning: {description} failed")
            print(f"Error: {result.stderr}")
            return False
        print(f"[HunyuanImage] {description} completed successfully")
        return True
    except Exception as e:
        print(f"[HunyuanImage] Error during {description}: {e}")
        return False

def main():
    print("=" * 60)
    print("ComfyUI-HunyuanImage-2.1 Installation Script")
    print("=" * 60)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check Python version
    print(f"\n[HunyuanImage] Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("[HunyuanImage] Error: Python 3.8+ is required")
        sys.exit(1)
    
    # Install basic requirements
    print("\n[HunyuanImage] Installing requirements...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip")
    run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements")
    
    # Try to install flash-attn (may fail on some systems)
    print("\n[HunyuanImage] Attempting to install flash-attn...")
    flash_result = run_command(
        f"{sys.executable} -m pip install 'flash-attn>=2.7.3' --no-build-isolation",
        "Installing flash-attn"
    )
    if not flash_result:
        print("[HunyuanImage] Note: flash-attn installation failed. This is optional but recommended for better performance.")
        print("[HunyuanImage] The node will still work without flash-attn.")
    
    # Install additional dependencies for model downloading
    print("\n[HunyuanImage] Installing model download tools...")
    run_command(f"{sys.executable} -m pip install huggingface-hub[cli]", "Installing huggingface-hub")
    run_command(f"{sys.executable} -m pip install modelscope", "Installing modelscope")
    
    # Test imports
    print("\n[HunyuanImage] Testing imports...")
    try:
        # Add current directory to path
        sys.path.insert(0, script_dir)
        
        # Try importing core modules
        from hyimage.diffusion.pipelines.hunyuanimage_pipeline import HunyuanImagePipeline, HunyuanImagePipelineConfig
        print("[HunyuanImage] ✓ Core HunyuanImage modules imported successfully")
        
        # Try importing model manager
        from model_manager import model_manager
        print("[HunyuanImage] ✓ Model manager imported successfully")
        
        # Try importing nodes
        from nodes import NODE_CLASS_MAPPINGS
        print("[HunyuanImage] ✓ Node classes imported successfully")
        
    except ImportError as e:
        print(f"[HunyuanImage] ✗ Import test failed: {e}")
        print("\n[HunyuanImage] Troubleshooting steps:")
        print("1. Ensure you're in the ComfyUI Python environment")
        print("2. Try running: pip install -e . (to install hyimage package)")
        print("3. Restart ComfyUI after installation")
        return False
    
    print("\n" + "=" * 60)
    print("[HunyuanImage] Installation completed!")
    print("\nNext steps:")
    print("1. Restart ComfyUI")
    print("2. Load one of the example workflows from examples_workflows/")
    print("3. Models will be automatically downloaded on first use")
    print("\nIf you encounter issues:")
    print("- Check that all dependencies are installed")
    print("- Ensure you have enough disk space for models (~50GB)")
    print("- See README_COMFYUI.md for troubleshooting")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)