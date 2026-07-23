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
        if self.algorithm not in ("counting_scatter", "stable_sort", "triton_counting_scatter"):
            raise ValueError("algorithm must be 'counting_scatter', 'stable_sort', or 'triton_counting_scatter'")
        if self.algorithm in ("counting_scatter", "triton_counting_scatter") and not self.stable_order:
            raise ValueError(f"{self.algorithm} requires stable_order=True")
