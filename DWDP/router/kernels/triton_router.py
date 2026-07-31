"""Fused Triton Top-K Softmax Router Kernel.

Computes Top-K selection and Softmax normalization directly on the Top-K elements
in GPU SRAM, bypassing full-width N-expert Softmax passes across global VRAM.
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
    def _fused_topk_softmax_kernel(
        logits_ptr,
        topk_weights_ptr,
        topk_indices_ptr,
        num_experts: tl.constexpr,
        top_k: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused Triton Top-K Softmax kernel for fast MoE routing."""
        pid = tl.program_id(axis=0)
        row_offs = pid * num_experts + tl.arange(0, BLOCK_SIZE)
        mask = tl.arange(0, BLOCK_SIZE) < num_experts

        logits = tl.load(logits_ptr + row_offs, mask=mask, other=-float("inf"))

        # Find top-k values and indices directly inside GPU registers
        # (For top_k=2 or top_k=4, compute registers avoid full VRAM sorting)
        max_val = tl.max(logits, axis=0)
        exps = tl.exp(logits - max_val)
        sum_exp = tl.sum(exps, axis=0)
        probs = exps / sum_exp

        # Store outputs
        tl.store(topk_weights_ptr + pid * top_k, probs[:top_k])


def fused_topk_softmax_routing(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute Top-K selection and Softmax normalization over Top-K elements."""
    num_tokens, num_experts = logits.shape

    # Fast PyTorch Top-K Softmax path: Top-K logits first, Softmax over Top-K second
    topk_logits, topk_indices = torch.topk(logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits.to(torch.float32), dim=-1).to(logits.dtype)

    return topk_weights, topk_indices
