from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class MergeStatistics:
    """Summary of merge work performed."""

    num_tokens: int
    num_assignments: int
    top_k: int
    output_size: int
    used_weighted_executor_outputs: bool
    backend: str


@dataclass(slots=True)
class MergeMetadata:
    """Metadata useful for profiling and future distributed merging."""

    token_shape: tuple[int, ...]
    assignment_shape: tuple[int, int]
    inverse_permutation: torch.Tensor
    top_k: int
    deterministic: bool


@dataclass(slots=True)
class TimingMetadata:
    """Optional timing metadata placeholder."""

    total_duration_us: float | None = None
    reduction_duration_us: float | None = None


@dataclass(slots=True)
class WorkspaceMetadata:
    """Workspace usage metadata."""

    used_workspace: bool
    workspace_bytes: int
