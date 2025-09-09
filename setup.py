"""
Setup script for HunyuanImage package
"""

from setuptools import setup, find_packages

setup(
    name="hyimage",
    version="2.1.0",
    description="HunyuanImage 2.1 - High-resolution text-to-image diffusion model",
    packages=find_packages(include=["hyimage", "hyimage.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.35.0",
        "diffusers>=0.24.0",
        "accelerate>=0.24.0",
        "sentencepiece",
        "einops",
        "loguru",
        "omegaconf",
        "peft",
        "protobuf",
        "tqdm",
        "Pillow",
        "numpy",
    ],
    extras_require={
        "flash": ["flash-attn>=2.7.3"],
        "xformers": ["xformers"],
        "bitsandbytes": ["bitsandbytes>=0.41.0"],
    },
)