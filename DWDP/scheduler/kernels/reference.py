from __future__ import annotations

import torch

from ..ops import build_round_robin_schedule
from ..workspace import SchedulerWorkspace


def reference_round_robin_schedule(
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    stream_count: int,
    workspace: SchedulerWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference scheduling boundary for Round Robin metadata generation."""

    return build_round_robin_schedule(
        expert_counts,
        expert_offsets,
        stream_count=stream_count,
        workspace=workspace,
    )
