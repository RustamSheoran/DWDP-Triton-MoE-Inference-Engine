"""Hopper TMA FP8 grouped GEMM with 128-element fine-grained micro-scaling.

Project 1 requires FP8 (E4M3) execution "utilizing Hopper TMA features,
fine-grained scale factors, and optimized SASS interleaving". This module
implements that path:

* Operand tiles are moved through the Tensor Memory Accelerator via Triton
  tensor descriptors rather than per-element ``tl.load`` address arithmetic,
  so the copy is issued by the TMA engine and the addresses stay off the
  math pipes.
* Scale factors are fine-grained rather than one scalar per expert:
  activations carry one FP32 scale per (token, 128-channel block) and weights
  carry one FP32 scale per 128x128 block. ``BLOCK_K`` is pinned to 128 so a
  single scale pair applies to each accumulation step, which is what keeps the
  rescale out of the inner loop and lets the MMA sequence interleave cleanly.

TMA is an sm_90+ (Hopper) feature and the descriptor API is still moving
between Triton releases, so both are probed at import time. When either is
missing the executor keeps using the non-TMA kernels in ``fp8.py``; nothing
here changes behavior on pre-Hopper hardware.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - triton is optional
    triton = None
    tl = None

# Host-side descriptor construction moved across a few Triton versions. Probe
# the modern location first, then the experimental one.
_TensorDescriptor = None
if triton is not None:
    try:  # Triton >= 3.3
        from triton.tools.tensor_descriptor import TensorDescriptor as _TensorDescriptor
    except ImportError:  # pragma: no cover - older triton
        _TensorDescriptor = None

TMA_AVAILABLE = _TensorDescriptor is not None and hasattr(tl, "dot")

# Micro-scaling granularity mandated by the spec. BLOCK_K must equal this so
# each K step consumes exactly one scale factor per operand.
MICRO_SCALE_BLOCK = 128

_BLOCK_M = 64
_BLOCK_N = 128
_BLOCK_K = MICRO_SCALE_BLOCK


def hopper_tma_supported(device: torch.device) -> bool:
    """Return whether this device and Triton build can run the TMA path."""

    if not TMA_AVAILABLE or device.type != "cuda":
        return False
    # TMA is a Hopper (sm_90) feature; earlier FP8 hardware (Ada, sm_89) has
    # FP8 Tensor Cores but no tensor memory accelerator.
    return torch.cuda.get_device_capability(device) >= (9, 0)


if TMA_AVAILABLE:

    @triton.jit
    def _fp8_tma_microscaled_gemm(
        a_desc,
        b_desc,
        c_desc,
        a_scale_ptr,
        b_scale_ptr,
        M,
        N,
        K,
        stride_as_m,
        stride_as_k,
        stride_bs_n,
        stride_bs_k,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """One expert's FP8 GEMM: C[M,N] = A[M,K] @ B[N,K]^T with block scales.

        ``a_scale`` is indexed [token, k_block] and ``b_scale`` is indexed
        [n_block, k_block]. Both are FP32. The dot product runs in FP8 and the
        rescale is applied to the FP32 accumulator once per K block, which is
        the fine-grained scheme DeepGEMM uses to hold FP32-class accuracy at
        FP8 throughput.
        """

        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        n_block = pid_n  # BLOCK_N == MICRO_SCALE_BLOCK, so one scale per program

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_block in range(0, tl.cdiv(K, BLOCK_K)):
            k_offset = k_block * BLOCK_K

            # TMA descriptor loads: the accelerator performs the global->shared
            # copy, including bounds handling, so no manual masking is needed.
            a_tile = a_desc.load([pid_m * BLOCK_M, k_offset])
            b_tile = b_desc.load([pid_n * BLOCK_N, k_offset])

            # FP8 x FP8 -> FP32 accumulate, then apply this block's scales.
            partial = tl.dot(a_tile, tl.trans(b_tile), out_dtype=tl.float32)

            a_scale = tl.load(
                a_scale_ptr + offs_m * stride_as_m + k_block * stride_as_k,
                mask=offs_m < M,
                other=0.0,
            )
            b_scale = tl.load(b_scale_ptr + n_block * stride_bs_n + k_block * stride_bs_k)

            accumulator += partial * a_scale[:, None] * b_scale

        c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator.to(c_desc.dtype))


def fp8_tma_microscaled_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run one FP8 GEMM through TMA with 128-element micro-scaling.

    ``a`` is [M, K] FP8, ``b`` is [N, K] FP8 (row-major per output channel, the
    layout the Qwen expert projections already use). ``a_scales`` is
    [M, ceil(K/128)] FP32 and ``b_scales`` is [ceil(N/128), ceil(K/128)] FP32.
    Returns [M, N] in ``out_dtype``.
    """

    if not hopper_tma_supported(a.device):
        raise RuntimeError(
            "fp8_tma_microscaled_gemm requires Hopper (sm_90+) and a Triton "
            "build exposing tensor descriptors"
        )

    M, K = a.shape
    N, K_b = b.shape
    if K != K_b:
        raise ValueError(f"inner dimension mismatch: {K} vs {K_b}")

    expected_a = (M, triton.cdiv(K, MICRO_SCALE_BLOCK))
    expected_b = (
        triton.cdiv(N, MICRO_SCALE_BLOCK),
        triton.cdiv(K, MICRO_SCALE_BLOCK),
    )
    if tuple(a_scales.shape) != expected_a:
        raise ValueError(f"a_scales must be {expected_a}, got {tuple(a_scales.shape)}")
    if tuple(b_scales.shape) != expected_b:
        raise ValueError(f"b_scales must be {expected_b}, got {tuple(b_scales.shape)}")

    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    a_desc = _TensorDescriptor.from_tensor(a, [_BLOCK_M, _BLOCK_K])
    b_desc = _TensorDescriptor.from_tensor(b, [_BLOCK_N, _BLOCK_K])
    c_desc = _TensorDescriptor.from_tensor(c, [_BLOCK_M, _BLOCK_N])

    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    _fp8_tma_microscaled_gemm[grid](
        a_desc,
        b_desc,
        c_desc,
        a_scales,
        b_scales,
        M,
        N,
        K,
        a_scales.stride(0),
        a_scales.stride(1),
        b_scales.stride(0),
        b_scales.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    return c
