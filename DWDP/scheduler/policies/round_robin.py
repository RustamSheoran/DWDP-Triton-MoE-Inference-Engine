from __future__ import annotations

import torch

from DWDP.dispatcher.plan import DispatchPlan

from ..base import BaseScheduler
from ..config import SchedulerConfig, SchedulerMetadataLevel
from ..execution import ExecutionPlan
from ..kernels import reference_round_robin_schedule
from ..metadata import DependencyMetadata, SchedulerStatistics, SynchronizationMetadata
from ..registry import register_scheduler
from ..utils import make_execution_batches, validate_dispatch_plan
from ..workspace import SchedulerWorkspace


class RoundRobinScheduler(BaseScheduler):
    """Deterministic ascending-expert scheduler."""

    def __init__(self, config: SchedulerConfig) -> None:
        super().__init__(config)

    def forward(
        self,
        dispatch_plan: DispatchPlan,
        workspace: SchedulerWorkspace | None = None,
    ) -> ExecutionPlan:
        """Build an ExecutionPlan from expert-major dispatch metadata."""

        validate_dispatch_plan(dispatch_plan)
        metadata = dispatch_plan.metadata
        active_workspace = workspace if self.config.enable_workspace else None

        (
            execution_order,
            expert_queue,
            expert_starts,
            expert_ends,
            expert_counts,
            execution_priority,
            stream_assignments,
        ) = reference_round_robin_schedule(
            metadata.expert_counts,
            metadata.expert_offsets,
            stream_count=self.config.stream_count,
            workspace=active_workspace,
        )

        active_count = expert_queue.numel()
        sync_metadata = self._build_synchronization_metadata(
            active_count,
            expert_queue.device,
            active_workspace,
        )
        dependency_metadata = self._build_dependency_metadata(
            expert_queue.device,
            active_workspace,
        )
        statistics = self._build_statistics(
            dispatch_plan,
            active_count=active_count,
            active_counts=expert_counts,
        )
        batches = ()
        if self.config.metadata_level == SchedulerMetadataLevel.FULL:
            batches = make_execution_batches(
                expert_queue,
                expert_starts,
                expert_ends,
                expert_counts,
                execution_priority,
                stream_assignments,
            )

        return ExecutionPlan(
            execution_order=execution_order,
            expert_queue=expert_queue,
            expert_starts=expert_starts,
            expert_ends=expert_ends,
            expert_counts=expert_counts,
            execution_priority=execution_priority,
            stream_assignments=stream_assignments,
            batches=batches,
            synchronization=sync_metadata,
            dependencies=dependency_metadata,
            statistics=statistics,
            scheduling_policy=self.config.scheduling_policy,
            deterministic=self.config.deterministic,
        )

    def _build_synchronization_metadata(
        self,
        active_count: int,
        device: torch.device,
        workspace: SchedulerWorkspace | None,
    ) -> SynchronizationMetadata:
        if workspace is None:
            barrier_after_batch = torch.zeros(active_count, dtype=torch.bool, device=device)
        else:
            barrier_after_batch = workspace.get_barrier_buffer(active_count, device=device)
            barrier_after_batch.zero_()

        if not self.config.enable_barrier_metadata:
            barrier_after_batch = barrier_after_batch[:0]

        return SynchronizationMetadata(
            barrier_after_batch=barrier_after_batch,
            cuda_event_ids=None,
            stream_waits=None,
        )

    def _build_dependency_metadata(
        self,
        device: torch.device,
        workspace: SchedulerWorkspace | None,
    ) -> DependencyMetadata:
        dependency_count = 0
        if workspace is None:
            dependency_src = torch.empty(dependency_count, dtype=torch.int64, device=device)
            dependency_dst = torch.empty(dependency_count, dtype=torch.int64, device=device)
        else:
            dependency_src, dependency_dst = workspace.get_dependency_buffers(
                dependency_count,
                device=device,
            )

        if not self.config.enable_dependency_metadata:
            dependency_src = dependency_src[:0]
            dependency_dst = dependency_dst[:0]

        return DependencyMetadata(
            dependency_src=dependency_src,
            dependency_dst=dependency_dst,
            prefetch_expert_ids=None,
            communication_groups=None,
        )

    def _build_statistics(
        self,
        dispatch_plan: DispatchPlan,
        *,
        active_count: int,
        active_counts: torch.Tensor,
    ) -> SchedulerStatistics:
        metadata = dispatch_plan.metadata
        num_empty = metadata.num_experts - active_count
        if active_count == 0:
            max_count = 0
            min_count = 0
        else:
            max_count = int(active_counts.max().item())
            min_count = int(active_counts.min().item())

        return SchedulerStatistics(
            num_experts=metadata.num_experts,
            num_active_experts=active_count,
            num_empty_experts=num_empty,
            num_execution_batches=active_count,
            num_assignments=metadata.num_assignments,
            max_tokens_per_expert=max_count,
            min_tokens_per_active_expert=min_count,
            scheduling_policy=self.config.scheduling_policy,
        )


register_scheduler("round_robin", RoundRobinScheduler)
