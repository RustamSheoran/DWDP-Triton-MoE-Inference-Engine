"""Golden PyTorch reference for SwiGLU MoE layers.

Implements the standard top-k MoE calculation:
    out[t] = sum_{k=0}^{top_k-1} w[t, k] * Down(SwiGLU(x[t]))
where SwiGLU(x) = silu(x @ W_gate^T) * (x @ W_up^T)
and Down(a) = a @ W_down^T.

Accumulates intermediate partials in float32 for numerical fidelity,
matching the fused Triton kernel's internal FP32 accumulation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    accumulate_fp32: bool = True,
) -> torch.Tensor:
    """Compute golden MoE reference outputs using pure PyTorch operations.

    Args:
        hidden_states: Tensor of shape [num_tokens, hidden_dim].
        w1: Fused gate and up projection weights of shape
            [num_experts, 2 * intermediate_dim, hidden_dim].
            w1[:, :intermediate_dim, :] is gate_proj,
            w1[:, intermediate_dim:, :] is up_proj.
        w2: Down projection weights of shape [num_experts, hidden_dim, intermediate_dim].
        topk_indices: Expert indices of shape [num_tokens, top_k], dtype integer.
        topk_weights: Routing weights of shape [num_tokens, top_k], dtype float.
        accumulate_fp32: Whether to accumulate token output in FP32 before
            casting back to activation dtype (default True).

    Returns:
        Tensor of shape [num_tokens, hidden_dim] matching hidden_states.dtype.
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_experts, double_intermediate_dim, hidden_dim_w1 = w1.shape
    intermediate_dim = double_intermediate_dim // 2
    top_k = topk_indices.shape[1]

    if hidden_dim != hidden_dim_w1:
        raise ValueError(
            f"Hidden dim mismatch: hidden_states has {hidden_dim}, w1 expects {hidden_dim_w1}"
        )
    if w2.shape != (num_experts, hidden_dim, intermediate_dim):
        raise ValueError(
            f"w2 shape mismatch: expected ({num_experts}, {hidden_dim}, {intermediate_dim}), "
            f"got {w2.shape}"
        )

    if num_tokens == 0:
        return torch.zeros((0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    acc_dtype = torch.float32 if accumulate_fp32 else hidden_states.dtype
    output = torch.zeros((num_tokens, hidden_dim), dtype=acc_dtype, device=hidden_states.device)

    # Token-wise golden loop: exact mathematical formulation
    for t in range(num_tokens):
        x_t = hidden_states[t : t + 1]  # [1, hidden_dim]
        for k in range(top_k):
            expert_id = int(topk_indices[t, k])
            weight = topk_weights[t, k]
            if expert_id < 0 or expert_id >= num_experts:
                continue

            w_gate = w1[expert_id, :intermediate_dim, :]  # [intermediate_dim, hidden_dim]
            w_up = w1[expert_id, intermediate_dim:, :]    # [intermediate_dim, hidden_dim]
            w_down = w2[expert_id, :, :]                  # [hidden_dim, intermediate_dim]

            gate = F.linear(x_t, w_gate)  # [1, intermediate_dim]
            up = F.linear(x_t, w_up)      # [1, intermediate_dim]
            act = F.silu(gate.float() if accumulate_fp32 else gate) * (up.float() if accumulate_fp32 else up)
            if not accumulate_fp32:
                act = act.to(hidden_states.dtype)

            down = F.linear(act.to(w_down.dtype), w_down)  # [1, hidden_dim]
            down_acc = down.float() if accumulate_fp32 else down
            weight_val = weight.float() if accumulate_fp32 else weight
            output[t] += (down_acc * weight_val).squeeze(0)

    return output.to(hidden_states.dtype)


def moe_reference_grouped(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Grouped per-expert PyTorch loop (faster equivalent reference).

    Used as an optimized PyTorch baseline for unit testing and benchmarking.
    """
    num_tokens, hidden_dim = hidden_states.shape
    num_experts, double_intermediate_dim, _ = w1.shape
    intermediate_dim = double_intermediate_dim // 2
    top_k = topk_indices.shape[1]

    if num_tokens == 0:
        return torch.zeros((0, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device)

    flat_topk_ids = topk_indices.reshape(-1)
    flat_topk_weights = topk_weights.reshape(-1).float()
    token_indices = torch.arange(num_tokens, device=hidden_states.device).repeat_interleave(top_k)

    output = torch.zeros((num_tokens, hidden_dim), dtype=torch.float32, device=hidden_states.device)

    for expert_id in range(num_experts):
        mask = flat_topk_ids == expert_id
        if not mask.any():
            continue

        selected_tokens = token_indices[mask]
        selected_weights = flat_topk_weights[mask].unsqueeze(1)
        x_expert = hidden_states[selected_tokens]

        w_gate = w1[expert_id, :intermediate_dim, :]
        w_up = w1[expert_id, intermediate_dim:, :]
        w_down = w2[expert_id, :, :]

        gate = F.linear(x_expert, w_gate)
        up = F.linear(x_expert, w_up)
        act = (F.silu(gate.float()) * up.float()).to(w_down.dtype)
        down = F.linear(act, w_down).float()

        weighted_down = down * selected_weights
        output.index_add_(0, selected_tokens, weighted_down)

    return output.to(hidden_states.dtype)
