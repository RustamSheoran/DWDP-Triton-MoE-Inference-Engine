from __future__ import annotations

from dataclasses import dataclass

import torch

from .metadata import DependencyMetadata, SchedulerStatistics, SynchronizationMetadata


@dataclass(slots=True)
class ExecutionBatch:
    """A contiguous expert-major work item for the Executor."""

    expert_id: int
    start: int
    end: int
    count: int
    priority: int
    stream_id: int


@dataclass(slots=True)
class ExecutionPlan:
    """Structured execution plan consumed by the Executor."""

    execution_order: torch.Tensor
    expert_queue: torch.Tensor
    expert_starts: torch.Tensor
    expert_ends: torch.Tensor
    expert_counts: torch.Tensor
    execution_priority: torch.Tensor
    stream_assignments: torch.Tensor
    batches: tuple[ExecutionBatch, ...]
    synchronization: SynchronizationMetadata
    dependencies: DependencyMetadata
    statistics: SchedulerStatistics
    scheduling_policy: str
    deterministic: bool
