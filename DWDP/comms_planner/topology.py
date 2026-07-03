from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CommunicationDomain:
    """Logical communication domain for future distributed planning."""

    domain_id: int
    domain_type: str
    gpu_ids: tuple[int, ...]
    bandwidth_gbps: float
    latency_us: float


@dataclass(slots=True)
class TopologyMetadata:
    """Hardware-independent topology description."""

    local_gpu_id: int
    world_size: int
    local_rank: int
    gpu_ids: torch.Tensor
    numa_domains: torch.Tensor
    nvlink_connectivity: torch.Tensor | None
    nvswitch_domains: torch.Tensor | None
    pcie_hierarchy: torch.Tensor | None
    communication_domains: tuple[CommunicationDomain, ...]
    locality_groups: tuple[tuple[int, ...], ...]
    fabric: str
    default_link_bandwidth_gbps: float
    default_link_latency_us: float
