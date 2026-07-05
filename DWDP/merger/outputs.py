from __future__ import annotations

from dataclasses import dataclass

import torch

from .metadata import MergeMetadata, MergeStatistics, TimingMetadata, WorkspaceMetadata


@dataclass(slots=True)
class MergerOutput:
    """Final reconstructed hidden states and merge metadata."""

    hidden_states: torch.Tensor
    metadata: MergeMetadata
    statistics: MergeStatistics
    timing: TimingMetadata
    workspace: WorkspaceMetadata
    backend: str
    deterministic: bool
