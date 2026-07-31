"""Fused Triton Grouped-GEMM kernel for MoE inference (vLLM / TensorRT-LLM style).

This module implements a single-pass fused MoE Triton kernel that executes token
routing gather, Gate+Up projection GEMM, SwiGLU activation, Down projection GEMM,
and routing-weight scaling in a unified Triton launch.
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
    def _fused_moe_kernel(
        hidden_states_ptr,
        w1_ptr,
        w2_ptr,
        output_ptr,
        topk_ids_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_offsets_ptr,
        num_tokens,
        hidden_dim: tl.constexpr,
        intermediate_dim: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """Fused Triton Grouped-GEMM Kernel for MoE SwiGLU layers."""
        pid = tl.program_id(axis=0)
        expert_idx = tl.program_id(axis=1)

        start_offset = tl.load(expert_offsets_ptr + expert_idx)
        end_offset = tl.load(expert_offsets_ptr + expert_idx + 1)
        num_expert_tokens = end_offset - start_offset

        if pid * BLOCK_SIZE_M >= num_expert_tokens:
            return

        offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        token_indices_offs = start_offset + offs_m
        token_mask = token_indices_offs < end_offset

        token_ids = tl.load(sorted_token_ids_ptr + token_indices_offs, mask=token_mask, other=0)
        token_weights = tl.load(topk_weights_ptr + token_indices_offs, mask=token_mask, other=0.0)

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)

        # Gate + Up GEMM
        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, hidden_dim, BLOCK_SIZE_K):
            x = tl.load(
                hidden_states_ptr + token_ids[:, None] * hidden_dim + (k + offs_k)[None, :],
                mask=token_mask[:, None] & ((k + offs_k)[None, :] < hidden_dim),
                other=0.0,
            )
            w_gate = tl.load(
                w1_ptr + expert_idx * (2 * intermediate_dim * hidden_dim) + (k + offs_k)[:, None] * (2 * intermediate_dim) + offs_n[None, :],
                mask=((k + offs_k)[:, None] < hidden_dim) & (offs_n[None, :] < intermediate_dim),
                other=0.0,
            )
            w_up = tl.load(
                w1_ptr + expert_idx * (2 * intermediate_dim * hidden_dim) + (k + offs_k)[:, None] * (2 * intermediate_dim) + (intermediate_dim + offs_n)[None, :],
                mask=((k + offs_k)[:, None] < hidden_dim) & (offs_n[None, :] < intermediate_dim),
                other=0.0,
            )
            acc_gate += tl.dot(x, w_gate)
            acc_up += tl.dot(x, w_up)

        # Fused SwiGLU activation: gate * sigmoid(gate) * up
        gate_act = acc_gate * (1.0 / (1.0 + tl.exp(-acc_gate)))
        act_out = gate_act * acc_up

        # Down GEMM
        acc_down = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, intermediate_dim, BLOCK_SIZE_K):
            w_down = tl.load(
                w2_ptr + expert_idx * (hidden_dim * intermediate_dim) + (k + offs_k)[:, None] * hidden_dim + offs_n[None, :],
                mask=((k + offs_k)[:, None] < intermediate_dim) & (offs_n[None, :] < hidden_dim),
                other=0.0,
            )
            acc_down += tl.dot(act_out.to(tl.float16), w_down)

        acc_down = acc_down * token_weights[:, None]

        # Atomic add to output
        tl.atomic_add(
            output_ptr + token_ids[:, None] * hidden_dim + offs_n[None, :],
            acc_down,
            mask=token_mask[:, None] & (offs_n[None, :] < hidden_dim),
        )


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Execute fused MoE grouped-GEMM computation.

    Args:
        hidden_states: [num_tokens, hidden_dim]
        w1: [num_experts, 2 * intermediate_dim, hidden_dim] or [num_experts, hidden_dim, 2 * intermediate_dim]
        w2: [num_experts, hidden_dim, intermediate_dim] or [num_experts, intermediate_dim, hidden_dim]
        topk_ids: [num_tokens, top_k]
        topk_weights: [num_tokens, top_k]

    Returns:
        output: [num_tokens, hidden_dim]
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for fused_moe execution.")

    num_tokens, hidden_dim = hidden_states.shape
    num_experts = w1.shape[0]
    top_k = topk_ids.shape[1]

    output = torch.zeros_like(hidden_states)
    flat_topk_ids = topk_ids.view(-1)
    flat_topk_weights = topk_weights.view(-1)

    sorted_topk_ids, sorted_indices = torch.sort(flat_topk_ids)
    sorted_token_ids = sorted_indices // top_k
    sorted_weights = flat_topk_weights[sorted_indices]

    expert_counts = torch.bincount(sorted_topk_ids, minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=hidden_states.device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)

    intermediate_dim = w2.shape[1] if w2.ndim == 3 else w2.shape[0]
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    max_tokens_per_expert = int(expert_counts.max().item()) if expert_counts.numel() > 0 else 0
    if max_tokens_per_expert == 0:
        return output

    grid = (
        triton.cdiv(max_tokens_per_expert, BLOCK_SIZE_M),
        num_experts,
    )

    _fused_moe_kernel[grid](
        hidden_states,
        w1,
        w2,
        output,
        topk_ids,
        sorted_weights,
        sorted_token_ids,
        expert_offsets,
        num_tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )

    return output
