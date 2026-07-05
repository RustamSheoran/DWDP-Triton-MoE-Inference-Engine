from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(slots=True)
class MoELayerSpec:
    """Extracted metadata for one Hugging Face MoE layer."""

    name: str
    module: nn.Module
    parent: nn.Module
    child_name: str
    gate: nn.Linear
    experts: nn.ModuleList | list[nn.Module]
    hidden_size: int
    num_experts: int
    top_k: int
    has_shared_expert: bool
    returns_router_logits: bool = True


def get_parent_module(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    """Return parent module and final child name for a qualified module name."""

    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _config_top_k(config: Any) -> int | None:
    for attr in ("num_experts_per_tok", "num_experts_per_token", "moe_top_k", "top_k"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    return None


def _looks_like_qwen_moe(module: nn.Module) -> bool:
    return hasattr(module, "gate") and hasattr(module, "experts") and isinstance(getattr(module, "gate"), nn.Linear)


def discover_qwen_moe_layers(model: nn.Module) -> tuple[MoELayerSpec, ...]:
    """Discover Qwen1.5/Qwen2-style sparse MoE blocks in a HF model."""

    config = getattr(model, "config", None)
    specs: list[MoELayerSpec] = []
    for name, module in model.named_modules():
        if not name or not _looks_like_qwen_moe(module):
            continue
        gate = getattr(module, "gate")
        experts = getattr(module, "experts")
        if not isinstance(experts, (nn.ModuleList, list, tuple)) or len(experts) == 0:
            continue
        parent, child_name = get_parent_module(model, name)
        top_k = getattr(module, "top_k", None) or _config_top_k(config)
        if top_k is None:
            continue
        specs.append(
            MoELayerSpec(
                name=name,
                module=module,
                parent=parent,
                child_name=child_name,
                gate=gate,
                experts=experts,
                hidden_size=int(gate.in_features),
                num_experts=int(gate.out_features),
                top_k=int(top_k),
                has_shared_expert=hasattr(module, "shared_expert"),
                returns_router_logits=True,
            )
        )
    return tuple(specs)
