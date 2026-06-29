from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class MetadataLevel(str, Enum):
    """Controls how much routing metadata is materialized."""

    NONE = "none"
    COUNTS = "counts"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Configuration for a production MoE router.

    The router is inference-first, but the settings remain valid for training
    and offline analysis.
    """

    hidden_size: int
    num_experts: int
    top_k: int
    bias: bool = False
    router_type: str = "linear_topk"
    softmax_dtype: torch.dtype | None = None
    probability_dtype: torch.dtype | None = None
    topk_sorted: bool = False
    renormalize: bool = True
    metadata_level: MetadataLevel = MetadataLevel.FULL
    score_scale: float = 1.0
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.top_k > self.num_experts:
            raise ValueError("top_k must be <= num_experts")
        if self.score_scale <= 0.0:
            raise ValueError("score_scale must be > 0")
        if self.eps <= 0.0:
            raise ValueError("eps must be > 0")
