"""Expert-Parallel Communication Planner for Multi-GPU PCIe / NVLink Clusters.

Partitions MoE experts evenly across visible CUDA GPUs (e.g. E0..E3 on GPU 0, E4..E7 on GPU 1)
and schedules token-only transfers across GPUs over PCIe/NVLink instead of streaming weights.
"""

from __future__ import annotations

import torch

from DWDP.scheduler.execution import ExecutionPlan
from .base import BaseCommunicationPlanner
from .config import CommunicationPlannerConfig
from .metadata import (
    CommunicationPlan,
    CommunicationStatistics,
    DependencyMetadata,
    OverlapPlan,
    PrefetchPlan,
    SynchronizationMetadata,
)
from .registry import register_communication_planner
from .topology import CommunicationDomain, TopologyMetadata
from .workspace import CommunicationPlannerWorkspace


@register_communication_planner("expert_parallel")
class ExpertParallelPlanner(BaseCommunicationPlanner):
    """Expert-Parallel communication planner for multi-GPU inference."""

    def __init__(self, config: CommunicationPlannerConfig) -> None:
        super().__init__(config)
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    def forward(
        self,
        execution_plan: ExecutionPlan,
        workspace: CommunicationPlannerWorkspace | None = None,
    ) -> CommunicationPlan:
        """Partition expert assignments across available GPU ranks."""
        device = execution_plan.expert_queue.device
        queue = execution_plan.expert_queue
        num_experts = queue.numel()

        # Partition expert range across available GPU ranks
        experts_per_gpu = max(1, (num_experts + self.num_gpus - 1) // self.num_gpus)

        # Local vs Remote Expert IDs
        rank = device.index if device.index is not None else 0
        local_start = rank * experts_per_gpu
        local_end = min(num_experts, (rank + 1) * experts_per_gpu)

        local_expert_ids = torch.arange(local_start, local_end, device=device, dtype=torch.int64)
        remote_mask = torch.ones(num_experts, dtype=torch.bool, device=device)
        remote_mask[local_start:local_end] = False
        remote_expert_ids = torch.where(remote_mask)[0].to(torch.int64)

        topology = TopologyMetadata(
            rank=rank,
            world_size=self.num_gpus,
            local_device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu",
            communication_domain=CommunicationDomain.INTRA_NODE,
            node_ids=torch.arange(self.num_gpus, device=device, dtype=torch.int64),
            edge_src=torch.tensor([], device=device, dtype=torch.int64),
            edge_dst=torch.tensor([], device=device, dtype=torch.int64),
        )

        return CommunicationPlan(
            local_expert_ids=local_expert_ids,
            remote_expert_ids=remote_expert_ids,
            topology=topology,
            prefetch=PrefetchPlan(enabled=True),
            overlap=OverlapPlan(enabled=True),
            sync=SynchronizationMetadata(events=[]),
            deps=DependencyMetadata(stage_dependencies=[]),
            stats=CommunicationStatistics(total_bytes_planned=0),
        )
