"""Model-specific expert weight extractors."""

from .qwen import extract_qwen_swiglu_weight_provider, is_qwen_swiglu_expert

__all__ = ["extract_qwen_swiglu_weight_provider", "is_qwen_swiglu_expert"]
