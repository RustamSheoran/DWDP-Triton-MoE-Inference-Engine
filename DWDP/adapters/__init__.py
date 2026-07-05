"""Model adapter interfaces for DWDP runtime integration."""

from .base import BaseModelAdapter
from .huggingface import HuggingFaceAdapter
from .registry import build_adapter, get_adapter_class, register_adapter

__all__ = [
    "BaseModelAdapter",
    "HuggingFaceAdapter",
    "build_adapter",
    "get_adapter_class",
    "register_adapter",
]
