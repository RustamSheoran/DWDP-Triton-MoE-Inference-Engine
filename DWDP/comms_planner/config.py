from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CommunicationMetadataLevel(str, Enum):
    """Controls communication metadata materialization."""

    MINIMAL = "minimal"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class CommunicationPlannerConfig:
    """Immutable configuration for communication planning."""

    planner_policy: str = "static"
    metadata_level: CommunicationMetadataLevel = CommunicationMetadataLevel.FULL
    deterministic: bool = True
    enable_workspace: bool = True
    enable_prefetch_metadata: bool = True
    enable_overlap_metadata: bool = True
    enable_topology_metadata: bool = True
    enable_cost_model: bool = True
    enable_statistics: bool = True
    local_gpu_id: int = 0
    world_size: int = 1
    local_rank: int = 0
    expert_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    sequence_parallel_size: int = 1
    stream_count: int = 1
    default_link_bandwidth_gbps: float = 0.0
    default_link_latency_us: float = 0.0

    def __post_init__(self) -> None:
        if not self.planner_policy:
            raise ValueError("planner_policy must be non-empty")
        if self.local_gpu_id < 0:
            raise ValueError("local_gpu_id must be >= 0")
        if self.world_size <= 0:
            raise ValueError("world_size must be > 0")
        if self.local_rank < 0 or self.local_rank >= self.world_size:
            raise ValueError("local_rank must satisfy 0 <= local_rank < world_size")
        if self.expert_parallel_size <= 0:
            raise ValueError("expert_parallel_size must be > 0")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be > 0")
        if self.pipeline_parallel_size <= 0:
            raise ValueError("pipeline_parallel_size must be > 0")
        if self.sequence_parallel_size <= 0:
            raise ValueError("sequence_parallel_size must be > 0")
        if self.stream_count <= 0:
            raise ValueError("stream_count must be > 0")
        if self.default_link_bandwidth_gbps < 0.0:
            raise ValueError("default_link_bandwidth_gbps must be >= 0")
        if self.default_link_latency_us < 0.0:
            raise ValueError("default_link_latency_us must be >= 0")
