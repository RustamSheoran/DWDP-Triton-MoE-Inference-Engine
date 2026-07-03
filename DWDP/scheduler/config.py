from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SchedulerMetadataLevel(str, Enum):
    """Controls scheduler metadata materialization."""

    MINIMAL = "minimal"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Immutable configuration for MoE execution scheduling."""

    scheduling_policy: str = "round_robin"
    deterministic: bool = True
    enable_workspace: bool = True
    metadata_level: SchedulerMetadataLevel = SchedulerMetadataLevel.FULL
    stream_count: int = 1
    enable_dependency_metadata: bool = True
    enable_barrier_metadata: bool = True
    enable_prefetch_metadata: bool = False
    enable_communication_metadata: bool = False
    max_execution_batch_size: int | None = None

    def __post_init__(self) -> None:
        if not self.scheduling_policy:
            raise ValueError("scheduling_policy must be non-empty")
        if self.stream_count <= 0:
            raise ValueError("stream_count must be > 0")
        if self.max_execution_batch_size is not None and self.max_execution_batch_size <= 0:
            raise ValueError("max_execution_batch_size must be > 0 when provided")
