"""Triton kernels for fused SwiGLU MoE execution on sm_75 (Tesla T4).

Features:
- Fused SwiGLU GEMM: silu(x @ W_gate^T) * (x @ W_up^T) with FP32 accumulation.
- Fused Down GEMM: (act @ W_down^T) * routing_weight, written directly to slot buffer.
- Autotuned over valid SM 75 configurations (FP16 Tensor Cores, 2-stage pipelines, no TMA/FP8).
- CPU-safe execution under TRITON_INTERPRET=1.
- Zero-synchronization grid execution using static upper bounds and dynamic early exit.
"""

from __future__ import annotations

import os
from typing import Callable, Sequence
import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

from .align import AlignedMoEData


def _is_interpret_or_cpu() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1" or not torch.cuda.is_available()


def sm75_autotune(configs: Sequence[triton.Config], key: Sequence[str]) -> Callable:
    """Autotune on GPU, bypass on CPU/TRITON_INTERPRET to avoid driver errors."""
    def decorator(fn: Callable) -> Callable:
        if _is_interpret_or_cpu():
            return fn
        return triton.autotune(configs=configs, key=key)(fn)
    return decorator


# SM 75 (Turing / Tesla T4) valid configurations:
# - FP16 Tensor Cores (tl.dot), FP32 accumulation
# - 2 stages (Turing hardware pipeline)
# - 4 or 8 warps (128 or 256 threads)
SWIGLU_SM75_CONFIGS = [
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2) if triton else None,
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2) if triton else None,
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2) if triton else None,
    triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2) if triton else None,
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2) if triton else None,
]
SWIGLU_SM75_CONFIGS = [c for c in SWIGLU_SM75_CONFIGS if c is not None]

DOWN_PROJ_SM75_CONFIGS = [
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2) if triton else None,
    triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=2) if triton else None,
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2) if triton else None,
    triton.Config({"BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2) if triton else None,
    triton.Config({"BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8, num_stages=2) if triton else None,
]
DOWN_PROJ_SM75_CONFIGS = [c for c in DOWN_PROJ_SM75_CONFIGS if c is not None]


if triton is not None:

    @sm75_autotune(
        configs=SWIGLU_SM75_CONFIGS,
        key=["hidden_dim", "intermediate_dim"],
    )
    @triton.jit
    def _fused_swiglu_kernel(
        x_ptr,
        w1_ptr,
        inter_ptr,
        sorted_token_ids_ptr,
        expert_ids_per_block_ptr,
        valid_mask_ptr,
        num_tokens_post_padded_ptr,
        hidden_dim: tl.constexpr,
        intermediate_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute silu(x @ W_gate^T) * (x @ W_up^T) for a block of tokens."""
        pid_m = tl.program_id(axis=0)
        # Early exit for blocks exceeding exact post-padded token count
        if pid_m * BLOCK_M >= tl.load(num_tokens_post_padded_ptr):
            return

        pid_n = tl.program_id(axis=1)

        # Cast expert_id to int64 for safe 64-bit pointer arithmetic
        expert_id = tl.load(expert_ids_per_block_ptr + pid_m).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        token_ids = tl.load(sorted_token_ids_ptr + offs_m)
        token_mask = tl.load(valid_mask_ptr + offs_m)

        acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        w1_expert_offset = expert_id * (2 * intermediate_dim * hidden_dim)

        for k in range(0, hidden_dim, BLOCK_K):
            k_offs = k + offs_k
            k_mask = k_offs < hidden_dim

            a_ptrs = x_ptr + token_ids[:, None] * hidden_dim + k_offs[None, :]
            a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)

            gate_ptrs = w1_ptr + w1_expert_offset + offs_n[None, :] * hidden_dim + k_offs[:, None]
            w_gate = tl.load(
                gate_ptrs,
                mask=(offs_n[None, :] < intermediate_dim) & k_mask[:, None],
                other=0.0,
            )

            up_ptrs = (
                w1_ptr
                + w1_expert_offset
                + (intermediate_dim + offs_n[None, :]) * hidden_dim
                + k_offs[:, None]
            )
            w_up = tl.load(
                up_ptrs,
                mask=(offs_n[None, :] < intermediate_dim) & k_mask[:, None],
                other=0.0,
            )

            acc_gate += tl.dot(a, w_gate)
            acc_up += tl.dot(a, w_up)

        act = acc_gate * tl.sigmoid(acc_gate) * acc_up
        out_ptrs = inter_ptr + offs_m[:, None] * intermediate_dim + offs_n[None, :]
        out_mask = token_mask[:, None] & (offs_n[None, :] < intermediate_dim)
        tl.store(out_ptrs, act.to(tl.float16), mask=out_mask)

    @sm75_autotune(
        configs=DOWN_PROJ_SM75_CONFIGS,
        key=["hidden_dim", "intermediate_dim"],
    )
    @triton.jit
    def _down_proj_kernel(
        inter_ptr,
        w2_ptr,
        partial_out_ptr,
        sorted_slot_ids_ptr,
        sorted_weights_ptr,
        expert_ids_per_block_ptr,
        valid_mask_ptr,
        num_tokens_post_padded_ptr,
        hidden_dim: tl.constexpr,
        intermediate_dim: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute (inter @ W_down^T) * routing_weight and write to slot buffer."""
        pid_m = tl.program_id(axis=0)
        # Early exit for blocks exceeding exact post-padded token count
        if pid_m * BLOCK_M >= tl.load(num_tokens_post_padded_ptr):
            return

        pid_n = tl.program_id(axis=1)

        # Cast expert_id to int64 for safe 64-bit pointer arithmetic
        expert_id = tl.load(expert_ids_per_block_ptr + pid_m).to(tl.int64)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        slot_ids = tl.load(sorted_slot_ids_ptr + offs_m)
        token_mask = tl.load(valid_mask_ptr + offs_m)
        weights = tl.load(sorted_weights_ptr + offs_m)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        w2_expert_offset = expert_id * (hidden_dim * intermediate_dim)

        for k in range(0, intermediate_dim, BLOCK_K):
            k_offs = k + offs_k
            k_mask = k_offs < intermediate_dim

            a_ptrs = inter_ptr + offs_m[:, None] * intermediate_dim + k_offs[None, :]
            a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = (
                w2_ptr + w2_expert_offset + offs_n[None, :] * intermediate_dim + k_offs[:, None]
            )
            w = tl.load(
                w_ptrs,
                mask=(offs_n[None, :] < hidden_dim) & k_mask[:, None],
                other=0.0,
            )

            acc += tl.dot(a, w)

        weighted = acc * weights[:, None]
        out_ptrs = partial_out_ptr + slot_ids[:, None] * hidden_dim + offs_n[None, :]
        out_mask = token_mask[:, None] & (offs_n[None, :] < hidden_dim)
        tl.store(out_ptrs, weighted.to(tl.float16), mask=out_mask)


def launch_fused_swiglu(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    inter_act: torch.Tensor,
    aligned: AlignedMoEData,
    hidden_dim: int,
    intermediate_dim: int,
    block_n: int = 64,
    block_k: int = 32,
) -> None:
    """Launch Kernel 1: Fused SwiGLU GEMM."""
    if aligned.num_blocks_ub == 0:
        return

    if _is_interpret_or_cpu():
        grid = (aligned.num_blocks_ub, triton.cdiv(intermediate_dim, block_n))
        _fused_swiglu_kernel[grid](
            hidden_states,
            w1,
            inter_act,
            aligned.sorted_token_ids,
            aligned.expert_ids_per_block,
            aligned.valid_mask,
            aligned.num_tokens_post_padded,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            BLOCK_M=aligned.block_size,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
    else:
        grid = lambda META: (aligned.num_blocks_ub, triton.cdiv(intermediate_dim, META["BLOCK_N"]))
        _fused_swiglu_kernel[grid](
            hidden_states,
            w1,
            inter_act,
            aligned.sorted_token_ids,
            aligned.expert_ids_per_block,
            aligned.valid_mask,
            aligned.num_tokens_post_padded,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            BLOCK_M=aligned.block_size,
        )


def launch_down_proj(
    inter_act: torch.Tensor,
    w2: torch.Tensor,
    partial_outputs: torch.Tensor,
    aligned: AlignedMoEData,
    hidden_dim: int,
    intermediate_dim: int,
    block_n: int = 64,
    block_k: int = 32,
) -> None:
    """Launch Kernel 2: Down projection with routing-weight scaling."""
    if aligned.num_blocks_ub == 0:
        return

    if _is_interpret_or_cpu():
        grid = (aligned.num_blocks_ub, triton.cdiv(hidden_dim, block_n))
        _down_proj_kernel[grid](
            inter_act,
            w2,
            partial_outputs,
            aligned.sorted_slot_ids,
            aligned.sorted_weights,
            aligned.expert_ids_per_block,
            aligned.valid_mask,
            aligned.num_tokens_post_padded,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            BLOCK_M=aligned.block_size,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
    else:
        grid = lambda META: (aligned.num_blocks_ub, triton.cdiv(hidden_dim, META["BLOCK_N"]))
        _down_proj_kernel[grid](
            inter_act,
            w2,
            partial_outputs,
            aligned.sorted_slot_ids,
            aligned.sorted_weights,
            aligned.expert_ids_per_block,
            aligned.valid_mask,
            aligned.num_tokens_post_padded,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            BLOCK_M=aligned.block_size,
        )
