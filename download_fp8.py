#!/usr/bin/env python3
"""Manual download script for FP8 models"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from model_manager import HunyuanModelManager

def download_fp8_models():
    """Download FP8 models manually"""
    
    manager = HunyuanModelManager()
    
    models_to_download = [
        ("hunyuanimage-v2.1-distilled-fp8", "dit"),
        ("hunyuanimage-refiner-fp8", "refiner"),
    ]
    
    for model_name, component in models_to_download:
        print(f"\n{'='*60}")
        print(f"Downloading {model_name} - {component}")
        print(f"{'='*60}")
        
        try:
            success = manager.download_component(model_name, component)
            if success:
                print(f"✓ Successfully downloaded {model_name} - {component}")
            else:
                print(f"✗ Failed to download {model_name} - {component}")
        except Exception as e:
            print(f"✗ Error downloading {model_name} - {component}: {e}")
    
    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    download_fp8_models()