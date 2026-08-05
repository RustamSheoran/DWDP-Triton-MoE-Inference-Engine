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


MICRO_SCALE_BLOCK = 128


def _fp8_max(dtype: torch.dtype) -> float:
    return 448.0 if "e4m3" in str(dtype) else 57344.0


def quantize_activations_blockwise(
    source: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations to FP8 with one scale per (token, 128-channel block).

    This is the activation half of the fine-grained scheme the Hopper TMA
    kernel expects. A single scalar per tensor throws away far too much range
    when one channel block is much hotter than its neighbours; scoping the
    scale to 128 channels keeps each block's values near the top of the FP8
    range without touching the others.

    Returns ``(fp8_values [M, K], scales [M, ceil(K/128)] float32)``.
    """

    if source.ndim != 2:
        raise ValueError("activation quantization expects a 2D [tokens, channels] tensor")

    rows, channels = source.shape
    blocks = (channels + MICRO_SCALE_BLOCK - 1) // MICRO_SCALE_BLOCK
    padded = blocks * MICRO_SCALE_BLOCK

    work = source.float()
    if padded != channels:
        work = torch.nn.functional.pad(work, (0, padded - channels))

    tiled = work.view(rows, blocks, MICRO_SCALE_BLOCK)
    scales = tiled.abs().amax(dim=-1).clamp(min=1e-5) / _fp8_max(dtype)
    quantized = (tiled / scales.unsqueeze(-1)).view(rows, padded)[:, :channels]

    return quantized.to(dtype), scales.contiguous()


def quantize_weights_blockwise(
    weight: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a [N, K] weight matrix to FP8 with one scale per 128x128 block.

    Returns ``(fp8_values [N, K], scales [ceil(N/128), ceil(K/128)] float32)``.
    """

    if weight.ndim != 2:
        raise ValueError("weight quantization expects a 2D [out, in] tensor")

    out_features, in_features = weight.shape
    n_blocks = (out_features + MICRO_SCALE_BLOCK - 1) // MICRO_SCALE_BLOCK
    k_blocks = (in_features + MICRO_SCALE_BLOCK - 1) // MICRO_SCALE_BLOCK
    padded_n = n_blocks * MICRO_SCALE_BLOCK
    padded_k = k_blocks * MICRO_SCALE_BLOCK

    work = weight.float()
    if padded_n != out_features or padded_k != in_features:
        work = torch.nn.functional.pad(
            work, (0, padded_k - in_features, 0, padded_n - out_features)
        )

    tiled = work.view(n_blocks, MICRO_SCALE_BLOCK, k_blocks, MICRO_SCALE_BLOCK)
    scales = tiled.abs().amax(dim=(1, 3)).clamp(min=1e-5) / _fp8_max(dtype)
    quantized = tiled / scales[:, None, :, None]
    quantized = quantized.view(padded_n, padded_k)[:out_features, :in_features]

    return quantized.to(dtype), scales.contiguous()
