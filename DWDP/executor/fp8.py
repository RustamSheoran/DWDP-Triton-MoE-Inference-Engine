"""FP8 storage preparation for the native persistent executor.

Conversion is intentionally performed at storage boundaries only.  Expert
parameters are converted in place once, preserving their tensor identities so
TensorList continues to reference the model's original allocation objects.
Input activations are quantized once per forward into workspace-owned FP8
storage; persistent kernels never cast operands inside their work loop.
"""

from __future__ import annotations

import torch

from .weights import QwenSwiGLUWeightProvider


def convert_qwen_weights_to_fp8_once(
    provider: QwenSwiGLUWeightProvider,
    dtype: torch.dtype,
    scale_map: dict[int, torch.Tensor] | None = None,
) -> None:
    """Convert original Qwen parameter storage to one selected FP8 format.

    Reassigning ``Tensor.data`` preserves the tensor object retained by weight
    views and therefore keeps TensorList's pointer-array layout unchanged. An
    already converted provider is a no-op; requesting a different format after
    conversion is rejected rather than silently quantizing FP8 a second time.
    If ``scale_map`` is provided, fine-grained inverse scaling factors per expert
    are calculated and stored into the map.
    """

    weights = (
        *provider.gate_up_weights.gate_weights.expert_weights,
        *provider.gate_up_weights.up_weights.expert_weights,
        *provider.down_weights.expert_weights,
    )
    current = {weight.dtype for weight in weights}
    if current == {dtype}:
        return
    if any(_is_fp8(weight.dtype) for weight in weights):
        raise ValueError("Qwen weights are already FP8 in a different execution format")
    with torch.no_grad():
        fp8_max = 448.0 if "e4m3" in str(dtype) else 57344.0
        for i, weight in enumerate(weights):
            if scale_map is not None:
                max_val = weight.abs().max().clamp(min=1e-5)
                scale = max_val / fp8_max
                scale_map[i] = scale
                weight.data = (weight / scale).to(dtype=dtype)
            else:
                weight.data = weight.data.to(dtype=dtype)


def quantize_activations_once(
    source: torch.Tensor,
    destination: torch.Tensor,
    scale_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize an activation matrix once into preallocated FP8 workspace.

    Optionally computes fine-grained scaling factors if ``scale_out`` is provided.
    """
    if scale_out is not None:
        fp8_max = 448.0 if "e4m3" in str(destination.dtype) else 57344.0
        max_val = source.abs().max().clamp(min=1e-5)
        scale = max_val / fp8_max
        scale_out.copy_(scale)
        destination.copy_(source / scale)
    else:
        # copy_ performs conversion directly into reusable destination storage
        destination.copy_(source)
    return destination


def _is_fp8(dtype: torch.dtype) -> bool:
    return "float8" in str(dtype)
