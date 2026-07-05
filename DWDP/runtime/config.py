from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Immutable configuration for the DWDP orchestration layer."""

    backend: str = "dwdp"
    device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None
    torch_compile: bool = False
    enable_workspace: bool = True
    enable_profiling: bool = False
    enable_statistics: bool = True
    deterministic: bool = True
    adapter: str = "huggingface"
    router_type: str = "linear_topk"
    dispatcher_type: str = "expert_major"
    scheduling_policy: str = "round_robin"
    communication_policy: str = "static"
    executor_backend: str = "pytorch"
    merger_backend: str = "pytorch"
    future_distributed: bool = False
    world_size: int = 1
    local_rank: int = 0

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.local_rank < 0 or self.local_rank >= self.world_size:
            raise ValueError("local_rank must satisfy 0 <= local_rank < world_size")
