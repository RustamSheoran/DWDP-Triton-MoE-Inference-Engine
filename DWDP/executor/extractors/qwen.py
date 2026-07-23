"""Qwen MoE expert extraction for optimized execution backends."""

from __future__ import annotations

from torch import nn

from ..experts import ExpertRegistry
from ..weights import QwenSwiGLUWeightProvider, build_qwen_swiglu_weight_provider


def is_qwen_swiglu_expert(expert: nn.Module) -> bool:
    """Return whether a module exposes the Qwen gate/up/down projection contract."""

    return all(
        hasattr(getattr(expert, name, None), "weight")
        for name in ("gate_proj", "up_proj", "down_proj")
    )


def extract_qwen_swiglu_weight_provider(experts: ExpertRegistry) -> QwenSwiGLUWeightProvider:
    """Create a storage-preserving provider from a Qwen-style expert registry."""

    items = tuple((expert_id, experts.get(expert_id)) for expert_id in experts.expert_ids)
    if not all(is_qwen_swiglu_expert(expert) for _, expert in items):
        raise ValueError("Triton expert execution requires Qwen-style gate_proj, up_proj, and down_proj experts")
    return build_qwen_swiglu_weight_provider(items)
