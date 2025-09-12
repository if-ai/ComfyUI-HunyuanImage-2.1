"""
ComfyUI-HunyuanImage-2.1: Advanced Text-to-Image Generation Node for ComfyUI
Supports HunyuanImage 2.1 with modular nodes for better resource management
"""

import os
import sys

# Add the current directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import nodes using absolute import
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from nodes_modular import NODE_CLASS_MAPPINGS as MODULAR_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as MODULAR_DISPLAY
    NODE_CLASS_MAPPINGS.update(MODULAR_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(MODULAR_DISPLAY)
except ImportError as e:
    print(f"[HunyuanImage] Error importing modular nodes: {e}")

try:
    from nodes_dual_encoder import NODE_CLASS_MAPPINGS as DUAL_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as DUAL_DISPLAY
    NODE_CLASS_MAPPINGS.update(DUAL_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(DUAL_DISPLAY)
except ImportError as e:
    print(f"[HunyuanImage] Error importing dual encoder nodes: {e}")

# Export nodes
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web"