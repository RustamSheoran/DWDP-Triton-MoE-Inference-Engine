from __future__ import annotations

from collections.abc import Sequence

import torch


def validate_hidden_states(hidden_states: torch.Tensor, hidden_size: int) -> None:
    """Validate the hidden-state tensor contract."""

    if hidden_states.ndim < 2:
        raise ValueError("hidden_states must have at least 2 dimensions")
    if hidden_states.shape[-1] != hidden_size:
        raise ValueError(
            f"Expected hidden size {hidden_size}, got {hidden_states.shape[-1]}"
        )


def flatten_token_dims(hidden_states: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten all token dimensions into a single leading dimension."""

    token_shape = tuple(hidden_states.shape[:-1])
    flat_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    return flat_hidden_states, token_shape


def restore_token_dims(
    tensor: torch.Tensor,
    token_shape: Sequence[int],
) -> torch.Tensor:
    """Restore the original token dimensions after flat routing ops."""

    if not token_shape:
        return tensor
    return tensor.reshape(*token_shape, *tensor.shape[1:])


def default_softmax_dtype(dtype: torch.dtype) -> torch.dtype | None:
    """Use FP32 accumulation for reduced-precision routing by default."""

    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return None
