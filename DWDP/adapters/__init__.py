"""Model adapter interfaces for DWDP runtime integration."""

from .base import BaseModelAdapter
from .huggingface import HuggingFaceAdapter
from .qwen15_moe import DWDPMoEBlock, Qwen15MoEAdapter
from .registry import build_adapter, detect_adapter_class, get_adapter_class, register_adapter, register_model_adapter

__all__ = [
    "BaseModelAdapter",
    "DWDPMoEBlock",
    "HuggingFaceAdapter",
    "Qwen15MoEAdapter",
    "build_adapter",
    "detect_adapter_class",
    "get_adapter_class",
    "register_adapter",
    "register_model_adapter",
]
