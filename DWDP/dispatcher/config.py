from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DispatcherConfig:
    """Configuration for expert-major MoE dispatch planning."""

    num_experts: int
    dispatcher_type: str = "expert_major"
    algorithm: str = "counting_scatter"
    stable_order: bool = True
    reuse_router_metadata: bool = True
    validate_inputs: bool = True
    index_dtype: torch.dtype = torch.int64

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if self.index_dtype is not torch.int64:
            raise ValueError("The reference dispatcher currently requires torch.int64 indices")
        if self.algorithm not in ("counting_scatter", "stable_sort"):
            raise ValueError("algorithm must be 'counting_scatter' or 'stable_sort'")
        if self.algorithm == "counting_scatter" and not self.stable_order:
            raise ValueError("counting_scatter requires stable_order=True")
