from __future__ import annotations

from math import prod

import torch

from DWDP.executor.outputs import ExecutorOutput


def num_tokens_from_shape(token_shape: tuple[int, ...]) -> int:
    """Compute flattened token count from token shape."""

    return int(prod(token_shape)) if token_shape else 1


def validate_executor_output(executor_output: ExecutorOutput) -> None:
    """Validate the ExecutorOutput contract required by the Merger."""

    metadata = executor_output.output_metadata
    packed = executor_output.packed_expert_outputs
    weighted = executor_output.weighted_expert_outputs
    merged = executor_output.merged_expert_outputs
    # ``packed`` is optional: an executor configured with
    # materialize_packed_outputs=False skips it because the Merger reads
    # weighted outputs. Validate whichever tensors are present.
    if merged is not None:
        if merged.ndim != 2 or merged.shape[0] != num_tokens_from_shape(metadata.token_shape):
            raise ValueError("merged expert output must be rank-2 and token-major")
    elif weighted is None or weighted.ndim != 2:
        raise ValueError("executor output tensors must be rank-2")
    if packed is not None:
        if packed.ndim != 2:
            raise ValueError("executor output tensors must be rank-2")
        if weighted is not None and packed.shape != weighted.shape:
            raise ValueError(
                "packed and weighted expert outputs must have identical shapes"
            )
    num_assignments = metadata.packed_token_indices.numel()
    if metadata.inverse_permutation.numel() != num_assignments:
        raise ValueError("inverse_permutation length must match num assignments")
    if metadata.packed_routing_weights.numel() != num_assignments:
        raise ValueError("packed_routing_weights length must match num assignments")
    if metadata.top_k <= 0:
        raise ValueError("top_k must be positive")
    num_tokens = num_tokens_from_shape(metadata.token_shape)
    if num_tokens * metadata.top_k != num_assignments:
        raise ValueError("num assignments must equal num tokens * top_k")


def estimate_tensor_bytes(tensor: torch.Tensor | None) -> int:
    """Estimate tensor storage size in bytes."""

    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()
