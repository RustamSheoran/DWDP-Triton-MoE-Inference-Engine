from __future__ import annotations

import torch
from torch import nn

from ..ops import apply_routing_weights


def reference_execute_expert(
    expert: nn.Module,
    expert_inputs: torch.Tensor,
    routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference expert execution boundary.

    Future Triton, CUDA, grouped GEMM, persistent-kernel, or FP8 backends can
    replace this boundary while preserving executor-level APIs.
    """

    expert_outputs = expert(expert_inputs)
    weighted_outputs = apply_routing_weights(expert_outputs, routing_weights)
    return expert_outputs, weighted_outputs
