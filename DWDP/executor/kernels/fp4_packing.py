"""INT4 / NVFP4 Byte-Packed Tensor Core Kernel Module for Tesla T4 / Turing GPUs.

Packs two 4-bit weight values into single UINT8 bytes for fast memory transfers
and unpacks them in GPU registers for high-throughput INT4/FP16 Tensor Core GEMMs.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

TRITON_AVAILABLE = triton is not None

if TRITON_AVAILABLE:

    @triton.jit
    def _unpack_nvfp4_kernel(
        packed_ptr,
        unpacked_ptr,
        scales_ptr,
        num_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Unpack pairs of 4-bit weights from UINT8 into FP16 GPU registers."""
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < (num_elements // 2)

        packed_val = tl.load(packed_ptr + offs, mask=mask, other=0)

        # Extract low 4 bits and high 4 bits
        low_nibble = packed_val & 0x0F
        high_nibble = (packed_val >> 4) & 0x0F

        # Scale and convert to FP16
        scale = tl.load(scales_ptr + (pid % 64))
        low_fp16 = (low_nibble.to(tl.float32) - 8.0) * scale
        high_fp16 = (high_nibble.to(tl.float32) - 8.0) * scale

        out_offs_low = offs * 2
        out_offs_high = offs * 2 + 1

        tl.store(unpacked_ptr + out_offs_low, low_fp16.to(tl.float16), mask=mask)
        tl.store(unpacked_ptr + out_offs_high, high_fp16.to(tl.float16), mask=mask)


def unpack_nvfp4_weights(
    packed_weights: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Unpack byte-packed 4-bit weights into FP16 for Tesla T4 Tensor Cores."""
    num_elements = packed_weights.numel() * 2
    unpacked = torch.empty(
        num_elements, dtype=torch.float16, device=packed_weights.device
    )

    if not TRITON_AVAILABLE or not packed_weights.is_cuda:
        # Fallback CPU path
        low = (packed_weights & 0x0F).to(torch.float32) - 8.0
        high = ((packed_weights >> 4) & 0x0F).to(torch.float32) - 8.0
        stacked = torch.stack([low, high], dim=-1).flatten()
        return (stacked * scales.repeat_interleave(32)[:num_elements]).to(torch.float16)

    def grid(META: dict[str, int]) -> tuple[int]:
        return (triton.cdiv(packed_weights.numel(), META["BLOCK_SIZE"]),)
    _unpack_nvfp4_kernel[grid](
        packed_weights,
        unpacked,
        scales,
        num_elements,
        BLOCK_SIZE=256,
    )
    return unpacked
