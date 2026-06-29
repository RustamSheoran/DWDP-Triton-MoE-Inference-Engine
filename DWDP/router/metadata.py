from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import MetadataLevel


@dataclass(slots=True)
class RoutingMetadata:
    """Auxiliary routing tensors consumed by later runtime stages.

    The flattened tensors stay in token-major order. A downstream dispatcher can
    either consume them directly or sort/group them by expert as needed.
    """

    num_tokens: int
    num_experts: int
    top_k: int
    tokens_per_expert: torch.Tensor | None = None
    expert_offsets: torch.Tensor | None = None
    flattened_token_indices: torch.Tensor | None = None
    flattened_expert_indices: torch.Tensor | None = None
    flattened_weights: torch.Tensor | None = None


def build_routing_metadata(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    level: MetadataLevel,
) -> RoutingMetadata | None:
    """Build downstream-facing routing metadata.

    Args:
        topk_indices: Flattened top-k expert ids with shape [num_tokens, top_k].
        topk_weights: Flattened top-k routing weights with shape [num_tokens, top_k].
        num_experts: Global expert count.
        level: Metadata materialization level.
    """

    if level == MetadataLevel.NONE:
        return None

    num_tokens, top_k = topk_indices.shape
    flat_expert_indices = topk_indices.reshape(-1)
    tokens_per_expert = torch.bincount(flat_expert_indices, minlength=num_experts)
    expert_offsets = torch.cat(
        (tokens_per_expert.new_zeros(1), tokens_per_expert.cumsum(dim=0)),
        dim=0,
    )

    metadata = RoutingMetadata(
        num_tokens=num_tokens,
        num_experts=num_experts,
        top_k=top_k,
        tokens_per_expert=tokens_per_expert,
        expert_offsets=expert_offsets,
    )

    if level == MetadataLevel.COUNTS:
        return metadata

    token_indices = torch.arange(
        num_tokens,
        device=topk_indices.device,
        dtype=torch.int64,
    ).repeat_interleave(top_k)
    metadata.flattened_token_indices = token_indices
    metadata.flattened_expert_indices = flat_expert_indices
    metadata.flattened_weights = topk_weights.reshape(-1)
    return metadata
