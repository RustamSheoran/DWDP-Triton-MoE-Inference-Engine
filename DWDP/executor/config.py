from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    """Immutable configuration for expert execution."""

    backend: str = "pytorch"
    dtype: torch.dtype | None = None
    enable_workspace: bool = True
    enable_statistics: bool = True
    enable_profiling: bool = False
    deterministic: bool = True
    max_tokens_per_expert: int | None = None
    allow_empty_experts: bool = True
    enable_distributed_placeholders: bool = True
    enable_async_placeholders: bool = True
    enable_weight_prefetch_placeholders: bool = True

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.max_tokens_per_expert is not None and self.max_tokens_per_expert <= 0:
            raise ValueError("max_tokens_per_expert must be > 0 when provided")
