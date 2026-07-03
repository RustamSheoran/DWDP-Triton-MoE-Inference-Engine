from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SynchronizationMetadata:
    """Metadata placeholders for future stream and barrier scheduling."""

    barrier_after_batch: torch.Tensor
    cuda_event_ids: torch.Tensor | None = None
    stream_waits: torch.Tensor | None = None


@dataclass(slots=True)
class DependencyMetadata:
    """Metadata placeholders for future dependency-aware scheduling."""

    dependency_src: torch.Tensor
    dependency_dst: torch.Tensor
    prefetch_expert_ids: torch.Tensor | None = None
    communication_groups: torch.Tensor | None = None


@dataclass(slots=True)
class SchedulerStatistics:
    """Small summary of scheduling decisions."""

    num_experts: int
    num_active_experts: int
    num_empty_experts: int
    num_execution_batches: int
    num_assignments: int
    max_tokens_per_expert: int
    min_tokens_per_active_expert: int
    scheduling_policy: str
