"""End-to-end Fused Triton MoE Layer.

Orchestrates token alignment, Kernel 1 (Fused SwiGLU GEMM), Kernel 2 (Down Projection
with routing-weight scaling), and deterministic slot-based top-k reduction.

Total GPU compute launch count:
1. Kernel 1: _fused_swiglu_kernel (1 Triton launch)
2. Kernel 2: _down_proj_kernel (1 Triton launch)
3. Reduction: torch.sum across top_k slots (1 PyTorch CUDA launch)
No atomic operations are used in the reduction, ensuring exact determinism.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .align import align_tokens_by_expert
from .kernels import launch_down_proj, launch_fused_swiglu


def triton_moe_layer(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    block_m: int = 32,
    block_n: int = 64,
    block_k: int = 32,
) -> torch.Tensor:
    """Execute fused MoE layer inference.

    Args:
        hidden_states: [num_tokens, hidden_dim] in float16.
        w1: Fused gate and up weights [num_experts, 2 * intermediate_dim, hidden_dim] in float16.
        w2: Down projection weights [num_experts, hidden_dim, intermediate_dim] in float16.
        topk_indices: [num_tokens, top_k] expert assignments (int).
        topk_weights: [num_tokens, top_k] routing weights (float16/float32).
        block_m: Token tile size (default 32).
        block_n: Intermediate/Hidden tile size (used for CPU/fallback, autotuned on GPU).
        block_k: Inner reduction tile size (used for CPU/fallback, autotuned on GPU).

    Returns:
        Tensor of shape [num_tokens, hidden_dim] matching hidden_states.dtype.
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_experts, double_intermediate_dim, hidden_dim_w1 = w1.shape
    intermediate_dim = double_intermediate_dim // 2
    top_k = topk_indices.shape[1]

    if num_tokens == 0:
        return torch.zeros((0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    # 1. Align & group tokens by expert with block_m padding (zero host sync)
    aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size=block_m)

    if aligned.num_blocks_ub == 0:
        return torch.zeros((num_tokens, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    device = hidden_states.device
    dtype = hidden_states.dtype

    # 2. Preallocate intermediate activations and slot output buffers using static upper bound
    inter_act = torch.zeros((aligned.max_padded_tokens, intermediate_dim), dtype=dtype, device=device)
    partial_outputs = torch.zeros((num_tokens * top_k, hidden_dim), dtype=dtype, device=device)

    # 3. Kernel 1: Fused Gate+Up GEMM with SwiGLU activation
    launch_fused_swiglu(
        hidden_states=hidden_states,
        w1=w1,
        inter_act=inter_act,
        aligned=aligned,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        block_n=block_n,
        block_k=block_k,
    )

    # 4. Kernel 2: Down GEMM with routing weight scaling into slot positions
    launch_down_proj(
        inter_act=inter_act,
        w2=w2,
        partial_outputs=partial_outputs,
        aligned=aligned,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        block_n=block_n,
        block_k=block_k,
    )

    # 5. Deterministic top-k reduction across slots (no atomic adds)
    return partial_outputs.view(num_tokens, top_k, hidden_dim).sum(dim=1)


class TritonMoELayer(nn.Module):
    """PyTorch Module wrapper for the fused Triton MoE layer."""

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
        block_m: int = 32,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.block_m = block_m

        # w1: [num_experts, 2 * intermediate_dim, hidden_dim]
        self.w1 = nn.Parameter(torch.empty(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.float16))
        # w2: [num_experts, hidden_dim, intermediate_dim]
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim, dtype=torch.float16))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.w1, a=5**0.5)
        nn.init.kaiming_uniform_(self.w2, a=5**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return triton_moe_layer(
            hidden_states=hidden_states,
            w1=self.w1,
            w2=self.w2,
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            block_m=self.block_m,
        )
