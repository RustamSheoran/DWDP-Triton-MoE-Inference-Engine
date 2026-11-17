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


@dataclass(frozen=True, slots=True)
class QwenMoEConfig:
    """Standardized MoE architectural hyperparameters read from model config."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    num_experts: int
    top_k: int
    hidden_size: int
    moe_intermediate_size: int


def parse_qwen_moe_config(config: Any) -> QwenMoEConfig:
    """Extract standard MoE architectural hyperparameters from a HF config or dict.

    Reads: num_layers, num_kv_heads, head_dim, num_experts, top_k, hidden_size,
    and moe_intermediate_size.
    """

    def _get(keys: tuple[str, ...], default: Any = None) -> Any:
        if isinstance(config, dict):
            for k in keys:
                if k in config and config[k] is not None:
                    return config[k]
        else:
            for k in keys:
                val = getattr(config, k, None)
                if val is not None:
                    return val
        return default

    hidden_size = _get(("hidden_size",))
    if hidden_size is None:
        raise ValueError("Config missing required attribute 'hidden_size'")
    hidden_size = int(hidden_size)

    num_attention_heads = _get(("num_attention_heads", "n_head", "num_heads"), default=1)
    num_kv_heads = _get(
        ("num_key_value_heads", "num_kv_heads", "n_head_kv"),
        default=num_attention_heads,
    )
    num_kv_heads = int(num_kv_heads)

    head_dim = _get(("head_dim", "kv_channels"))
    if head_dim is None:
        head_dim = hidden_size // int(num_attention_heads)
    head_dim = int(head_dim)

    num_layers = _get(("num_hidden_layers", "num_layers", "n_layer"))
    if num_layers is None:
        raise ValueError("Config missing required attribute 'num_layers' / 'num_hidden_layers'")
    num_layers = int(num_layers)

    num_experts = _get(("num_experts", "num_local_experts", "n_routed_experts"))
    if num_experts is None:
        raise ValueError("Config missing required attribute 'num_experts'")
    num_experts = int(num_experts)

    top_k = _get(("num_experts_per_tok", "num_experts_per_token", "top_k", "moe_top_k"))
    if top_k is None:
        raise ValueError("Config missing required attribute 'top_k' / 'num_experts_per_tok'")
    top_k = int(top_k)

    moe_intermediate_size = _get(("moe_intermediate_size", "intermediate_size"))
    if moe_intermediate_size is None:
        raise ValueError("Config missing required attribute 'moe_intermediate_size'")
    moe_intermediate_size = int(moe_intermediate_size)

    return QwenMoEConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
    )


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
    return (
        hasattr(module, "gate")
        and hasattr(module, "experts")
        and isinstance(getattr(module, "gate"), nn.Linear)
    )


def discover_qwen_moe_layers(model: nn.Module) -> tuple[MoELayerSpec, ...]:
    """Discover Qwen1.5/Qwen2-style sparse MoE blocks in a HF model."""

    config = getattr(model, "config", None)
    specs: list[MoELayerSpec] = []
    for name, module in model.named_modules():
        if not name or not any(k in name for k in ("mlp", "moe", "block", "layer")):
            continue
        if not _looks_like_qwen_moe(module):
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
