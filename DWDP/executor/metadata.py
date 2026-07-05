from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TimingMetadata:
    """Optional timing metadata placeholder."""

    expert_durations_us: torch.Tensor | None = None
    total_duration_us: float | None = None


@dataclass(slots=True)
class WorkspaceMetadata:
    """Workspace usage metadata."""

    used_workspace: bool
    workspace_bytes: int
