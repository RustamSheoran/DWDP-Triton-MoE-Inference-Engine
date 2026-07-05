from __future__ import annotations

from dataclasses import dataclass

import torch

from .metadata import TimingMetadata, WorkspaceMetadata


@dataclass(slots=True)
class ExpertOutput:
    """Output metadata for a single executed expert."""

    expert_id: int
    start: int
    end: int
    count: int
    priority: int
    stream_id: int


@dataclass(slots=True)
class ExecutionStatistics:
    """Summary of executor work performed."""

    num_executed_experts: int
    num_skipped_experts: int
    num_input_tokens: int
    num_assignments: int
    hidden_size: int
    output_size: int
    backend: str


@dataclass(slots=True)
class OutputMetadata:
    """Packed output metadata required by the future Merger."""

    packed_token_indices: torch.Tensor
    packed_expert_ids: torch.Tensor
    packed_routing_weights: torch.Tensor
    token_permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    token_shape: tuple[int, ...]
    top_k: int


@dataclass(slots=True)
class ExecutionMetadata:
    """Executor metadata preserving scheduler and communication context."""

    execution_order: torch.Tensor
    expert_queue: torch.Tensor
    expert_starts: torch.Tensor
    expert_ends: torch.Tensor
    stream_assignments: torch.Tensor
    communication_remote_expert_ids: torch.Tensor
    communication_policy: str
    scheduling_policy: str


@dataclass(slots=True)
class ExecutorOutput:
    """Packed expert outputs produced by the Executor."""

    packed_expert_outputs: torch.Tensor
    weighted_expert_outputs: torch.Tensor
    expert_outputs: tuple[ExpertOutput, ...]
    output_metadata: OutputMetadata
    execution_metadata: ExecutionMetadata
    statistics: ExecutionStatistics
    timing: TimingMetadata
    workspace: WorkspaceMetadata
    backend: str
    deterministic: bool
