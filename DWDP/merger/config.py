from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MergerConfig:
    """Immutable configuration for MoE output reconstruction."""

    backend: str = "pytorch"
    enable_workspace: bool = True
    enable_statistics: bool = True
    enable_profiling: bool = False
    deterministic: bool = True
    apply_routing_weights: bool = False
    validate_shapes: bool = True
    enable_distributed_placeholders: bool = True
    enable_async_placeholders: bool = True

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
