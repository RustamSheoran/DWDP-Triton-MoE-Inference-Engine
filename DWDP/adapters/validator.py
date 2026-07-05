from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class AdapterTensorComparison:
    """Numerical comparison for adapter validation."""

    max_abs_error: float
    mean_abs_error: float
    max_relative_error: float
    allclose: bool


@dataclass(slots=True)
class AdapterParityReport:
    """Adapter parity report for native and DWDP execution."""

    output: AdapterTensorComparison | None = None
    router: AdapterTensorComparison | None = None
    generated_token_parity: bool | None = None


def compare_outputs(reference: torch.Tensor, actual: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-4) -> AdapterTensorComparison:
    """Compare tensors including relative error."""

    if reference.shape != actual.shape:
        raise ValueError(f"shape mismatch: reference={tuple(reference.shape)} actual={tuple(actual.shape)}")
    diff = (reference - actual).abs()
    denom = reference.abs().clamp_min(1e-12)
    rel = diff / denom
    return AdapterTensorComparison(
        max_abs_error=float(diff.max().item()) if diff.numel() else 0.0,
        mean_abs_error=float(diff.mean().item()) if diff.numel() else 0.0,
        max_relative_error=float(rel.max().item()) if rel.numel() else 0.0,
        allclose=bool(torch.allclose(reference, actual, rtol=rtol, atol=atol)),
    )


def generated_token_parity(reference: Any, actual: Any) -> bool:
    """Return token parity for tensor or Python generation outputs."""

    if isinstance(reference, torch.Tensor) and isinstance(actual, torch.Tensor):
        return bool(torch.equal(reference, actual))
    return bool(reference == actual)
