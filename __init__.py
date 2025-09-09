"""
ComfyUI-HunyuanImage-2.1: Advanced Text-to-Image Generation Node for ComfyUI
Supports HunyuanImage 2.1 with dtype selection, torch compile, and prompt enhancement
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

WEB_DIRECTORY = "./web"