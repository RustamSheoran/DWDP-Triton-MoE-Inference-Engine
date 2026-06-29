from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ExpertAssignments:
    """Expert-major packed routing assignments.

    All tensors have shape `[num_assignments]`, where `num_assignments = T * K`.
    """

    expert_ids: torch.Tensor
    packed_token_indices: torch.Tensor
    packed_routing_weights: torch.Tensor
