from __future__ import annotations

from dataclasses import dataclass

import torch

from .cost_model import CommunicationCostModel
from .graph import CommunicationGraph
from .topology import TopologyMetadata


@dataclass(slots=True)
class CommunicationDescriptor:
    """Descriptor for a planned communication operation."""

    descriptor_id: int
    op_type: str
    source_gpu: int
    destination_gpu: int
    source_expert_id: int
    destination_expert_id: int
    start: int
    end: int
    count: int
    priority: int
    stream_id: int


@dataclass(slots=True)
class TransferDescriptor:
    """Descriptor for a future tensor transfer."""

    transfer_id: int
    descriptor_id: int
    source_gpu: int
    destination_gpu: int
    source_expert_id: int
    destination_expert_id: int
    estimated_bytes: int
    priority: int


@dataclass(slots=True)
class CommunicationGroup:
    """Logical group of planned communication descriptors."""

    group_id: int
    descriptor_ids: torch.Tensor
    source_gpu: int
    destination_gpu: int
    domain_id: int
    priority: int


@dataclass(slots=True)
class SynchronizationMetadata:
    """Synchronization placeholders for future communication execution."""

    barrier_node_ids: torch.Tensor
    cuda_event_ids: torch.Tensor
    stream_wait_edges: torch.Tensor
    synchronization_points: torch.Tensor


@dataclass(slots=True)
class DependencyMetadata:
    """Dependency graph tensors for communication planning."""

    dependency_src: torch.Tensor
    dependency_dst: torch.Tensor
    dependency_type: torch.Tensor


@dataclass(slots=True)
class PrefetchPlan:
    """Asynchronous expert-weight prefetch metadata placeholder."""

    prefetch_expert_ids: torch.Tensor
    prefetch_priorities: torch.Tensor
    prefetch_windows_us: torch.Tensor


@dataclass(slots=True)
class OverlapPlan:
    """Communication/computation overlap metadata placeholder."""

    communication_node_ids: torch.Tensor
    compute_batch_ids: torch.Tensor
    overlap_windows_us: torch.Tensor


@dataclass(slots=True)
class CommunicationStatistics:
    """Summary statistics for a communication plan."""

    num_local_experts: int
    num_remote_experts: int
    num_communication_nodes: int
    num_communication_edges: int
    num_transfer_descriptors: int
    num_communication_groups: int
    total_estimated_bytes: int
    total_estimated_latency_us: float
    planner_policy: str


@dataclass(slots=True)
class CommunicationPlan:
    """Reusable communication blueprint consumed by future executor stages."""

    local_expert_ids: torch.Tensor
    remote_expert_ids: torch.Tensor
    graph: CommunicationGraph
    communication_descriptors: tuple[CommunicationDescriptor, ...]
    transfer_descriptors: tuple[TransferDescriptor, ...]
    communication_groups: tuple[CommunicationGroup, ...]
    topology: TopologyMetadata
    synchronization: SynchronizationMetadata
    dependencies: DependencyMetadata
    prefetch: PrefetchPlan
    overlap: OverlapPlan
    cost_model: CommunicationCostModel
    statistics: CommunicationStatistics
    planner_policy: str
    deterministic: bool
