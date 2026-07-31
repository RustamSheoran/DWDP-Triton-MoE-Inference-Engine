"""Fused Triton Token Merger Kernel.

Fuses routing-weight multiplication and token index scattering directly inside
GPU SRAM, restoring token-major hidden states in a single VRAM pass.
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
    def _fused_token_merger_kernel(
        expert_outputs_ptr,
        routing_weights_ptr,
        token_indices_ptr,
        out_ptr,
        hidden_size: tl.constexpr,
        BLOCK_SIZE_H: tl.constexpr,
    ):
        """Single-pass Triton token scatter-add kernel."""
        pid_item = tl.program_id(axis=0)
        pid_h = tl.program_id(axis=1)

        token_idx = tl.load(token_indices_ptr + pid_item)
        weight = tl.load(routing_weights_ptr + pid_item)

        h_offs = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        h_mask = h_offs < hidden_size

        src_ptrs = expert_outputs_ptr + pid_item * hidden_size + h_offs
        vals = tl.load(src_ptrs, mask=h_mask, other=0.0)

        weighted_vals = vals * weight

        dst_ptrs = out_ptr + token_idx * hidden_size + h_offs
        tl.atomic_add(dst_ptrs, weighted_vals, mask=h_mask, sem="relaxed")


def fused_triton_merge_tokens(
    expert_outputs: torch.Tensor,
    routing_weights: torch.Tensor,
    token_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    """Fuse weight multiplication and token index scatter-add into a single Triton pass."""
    num_items, hidden_size = expert_outputs.shape
    out = torch.zeros(
        (num_tokens, hidden_size),
        dtype=expert_outputs.dtype,
        device=expert_outputs.device,
    )

    if not TRITON_AVAILABLE or not expert_outputs.is_cuda:
        # Fallback PyTorch path
        out.index_add_(
            0,
            token_indices,
            expert_outputs * routing_weights.unsqueeze(-1),
        )
        return out

    def grid(META: dict[str, int]) -> tuple[int, int]:
        return (
            num_items,
            triton.cdiv(hidden_size, META["BLOCK_SIZE_H"]),
        )

    _fused_token_merger_kernel[grid](
        expert_outputs,
        routing_weights,
        token_indices,
        out,
        hidden_size,
        BLOCK_SIZE_H=64,
    )
    return out
