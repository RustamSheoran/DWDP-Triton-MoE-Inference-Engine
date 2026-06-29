from __future__ import annotations

from dataclasses import dataclass

import torch

from .metadata import RoutingMetadata


@dataclass(slots=True)
class RouterOutput:
    """Structured router outputs for downstream runtime stages."""

    router_logits: torch.Tensor
    routing_probabilities: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    metadata: RoutingMetadata | None = None
