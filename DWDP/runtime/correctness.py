from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TensorComparison:
    """Numerical comparison result for two tensors."""

    max_abs_error: float
    mean_abs_error: float
    allclose: bool
    shape: tuple[int, ...]


@dataclass(slots=True)
class CorrectnessReport:
    """Correctness report for native and DWDP outputs."""

    tensor: TensorComparison | None = None
    generated_token_parity: bool | None = None


def compare_tensors(
    reference: torch.Tensor,
    actual: torch.Tensor,
    *,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> TensorComparison:
    """Compare tensors with max/mean absolute error and `torch.allclose`."""

    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: reference={tuple(reference.shape)} actual={tuple(actual.shape)}")
    diff = (reference - actual).abs()
    return TensorComparison(
        max_abs_error=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs_error=float(diff.mean().item()) if diff.numel() else 0.0,
        allclose=bool(torch.allclose(reference, actual, rtol=rtol, atol=atol)),
        shape=tuple(reference.shape),
    )
