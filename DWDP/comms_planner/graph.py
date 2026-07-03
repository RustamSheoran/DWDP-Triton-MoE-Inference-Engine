from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CommunicationNode:
    """A future communication operation represented as a graph node."""

    node_id: int
    descriptor_id: int
    op_type: str
    source_gpu: int
    destination_gpu: int
    expert_id: int
    priority: int
    stream_id: int


@dataclass(slots=True)
class CommunicationEdge:
    """A dependency edge between communication graph nodes."""

    source_node_id: int
    destination_node_id: int
    dependency_type: str


@dataclass(slots=True)
class CommunicationGraph:
    """Canonical graph representation of planned communication."""

    nodes: tuple[CommunicationNode, ...]
    edges: tuple[CommunicationEdge, ...]
    node_ids: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor

    @property
    def is_empty(self) -> bool:
        return len(self.nodes) == 0 and len(self.edges) == 0
