from __future__ import annotations

import torch

from DWDP.router.types import RouterOutput


def validate_router_output(router_output: RouterOutput, num_experts: int) -> None:
    """Validate the router output contract expected by the dispatcher."""

    topk_indices = router_output.topk_indices
    topk_weights = router_output.topk_weights

    if topk_indices.ndim < 1:
        raise ValueError("router_output.topk_indices must have at least 1 dimension")
    if topk_indices.shape != topk_weights.shape:
        raise ValueError("topk_indices and topk_weights must have identical shapes")
    if topk_indices.shape[-1] <= 0:
        raise ValueError("topk_indices must have a positive top-k dimension")
    if topk_indices.dtype.is_floating_point or topk_indices.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise ValueError("topk_indices must use an integer dtype")
    if not topk_weights.dtype.is_floating_point:
        raise ValueError("topk_weights must use a floating-point dtype")

    if topk_indices.numel() == 0:
        return

    min_expert = int(topk_indices.min().item())
    max_expert = int(topk_indices.max().item())
    if min_expert < 0 or max_expert >= num_experts:
        raise ValueError(
            f"topk_indices contain out-of-range expert ids: [{min_expert}, {max_expert}]"
        )


def flatten_router_output(
    router_output: RouterOutput,
) -> tuple[torch.Tensor, torch.Tensor, int, int, tuple[int, ...]]:
    """Flatten token-major router outputs into dispatch-friendly 1D tensors."""

    topk_indices = router_output.topk_indices
    topk_weights = router_output.topk_weights
    top_k = topk_indices.shape[-1]
    token_shape = tuple(topk_indices.shape[:-1])
    num_tokens = topk_indices.numel() // top_k

    flat_expert_indices = topk_indices.reshape(-1).to(dtype=torch.int64)
    flat_routing_weights = topk_weights.reshape(-1)
    return flat_expert_indices, flat_routing_weights, num_tokens, top_k, token_shape


def maybe_reuse_router_metadata(
    router_output: RouterOutput,
    *,
    num_experts: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Reuse router-provided counts and offsets when compatible."""

    metadata = router_output.metadata
    if metadata is None:
        return None, None
    if metadata.tokens_per_expert is None or metadata.expert_offsets is None:
        return None, None
    if metadata.tokens_per_expert.numel() != num_experts:
        return None, None
    if metadata.expert_offsets.numel() != num_experts + 1:
        return None, None
    if metadata.tokens_per_expert.dtype != torch.int64:
        return None, None
    if metadata.expert_offsets.dtype != torch.int64:
        return None, None
    return metadata.tokens_per_expert, metadata.expert_offsets


def estimate_tensor_bytes(tensor: torch.Tensor | None) -> int:
    """Estimate storage size in bytes for a tensor."""

    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()
