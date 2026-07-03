from __future__ import annotations

import torch

from ..workspace import CommunicationPlannerWorkspace


def classify_single_gpu_experts(
    expert_queue: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Classify all scheduled experts as local for single-GPU execution."""

    remote_expert_ids = torch.empty(0, dtype=torch.int64, device=expert_queue.device)
    return expert_queue, remote_expert_ids


def empty_graph_tensors(
    *,
    device: torch.device,
    workspace: CommunicationPlannerWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empty graph tensors for single-GPU communication planning."""

    if workspace is None:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, empty
    empty = workspace.get_empty_int64(device=device)
    return empty, empty, empty


def build_single_gpu_topology_tensors(
    *,
    world_size: int,
    device: torch.device,
    workspace: CommunicationPlannerWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build local topology tensors without encoding hardware-specific policy."""

    if workspace is None:
        gpu_ids = torch.arange(world_size, dtype=torch.int64, device=device)
        numa_domains = torch.zeros(world_size, dtype=torch.int64, device=device)
        return gpu_ids, numa_domains

    gpu_ids, numa_domains = workspace.get_topology_buffers(world_size, device=device)
    torch.arange(world_size, dtype=torch.int64, device=device, out=gpu_ids)
    numa_domains.zero_()
    return gpu_ids, numa_domains
