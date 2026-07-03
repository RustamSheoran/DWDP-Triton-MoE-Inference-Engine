from __future__ import annotations

import torch

from ..ops import classify_single_gpu_experts, empty_graph_tensors
from ..workspace import CommunicationPlannerWorkspace


def reference_static_communication_plan(
    expert_queue: torch.Tensor,
    *,
    workspace: CommunicationPlannerWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference metadata boundary for static single-GPU communication planning."""

    local_expert_ids, remote_expert_ids = classify_single_gpu_experts(expert_queue)
    node_ids, edge_src, edge_dst = empty_graph_tensors(
        device=expert_queue.device,
        workspace=workspace,
    )
    return local_expert_ids, remote_expert_ids, node_ids, edge_src, edge_dst
