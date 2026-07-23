"""Grouped expert-major matrix multiplication reference and Triton kernels.

The Triton kernel consumes an expert-major input layout ``[N, K]``, physical
weights ``[E, O, K]``, and dispatcher offsets ``[E + 1]``. Each output row is
written to the same expert-major position as its input row.
"""

from __future__ import annotations

import torch

from DWDP.dispatcher.plan import DispatchPlan

from ..weights import ExpertMajorMatrixView

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised without Triton installed.
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None


if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=4),
        ],
        key=["hidden_size", "output_size"],
    )
    @triton.jit
    def _grouped_matmul_kernel(
        expert_inputs,
        expert_weights,
        expert_offsets,
        outputs,
        hidden_size: tl.constexpr,
        output_size: tl.constexpr,
        input_stride_m: tl.constexpr,
        input_stride_k: tl.constexpr,
        weight_stride_e: tl.constexpr,
        weight_stride_o: tl.constexpr,
        weight_stride_k: tl.constexpr,
        output_stride_m: tl.constexpr,
        output_stride_n: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute one ``[BLOCK_M, BLOCK_N]`` tile for one expert."""

        expert_id = tl.program_id(0)
        token_tile = tl.program_id(1)
        output_tile = tl.program_id(2)

        expert_start = tl.load(expert_offsets + expert_id)
        expert_end = tl.load(expert_offsets + expert_id + 1)
        token_offsets = expert_start + token_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        output_offsets = output_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, hidden_size, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            input_mask = (token_offsets[:, None] < expert_end) & (k_offsets[None, :] < hidden_size)
            weight_mask = (output_offsets[None, :] < output_size) & (k_offsets[:, None] < hidden_size)
            inputs = tl.load(
                expert_inputs + token_offsets[:, None] * input_stride_m + k_offsets[None, :] * input_stride_k,
                mask=input_mask,
                other=0.0,
            )
            weights = tl.load(
                expert_weights
                + expert_id * weight_stride_e
                + output_offsets[None, :] * weight_stride_o
                + k_offsets[:, None] * weight_stride_k,
                mask=weight_mask,
                other=0.0,
            )
            accumulator += tl.dot(inputs, weights)

        output_mask = (token_offsets[:, None] < expert_end) & (output_offsets[None, :] < output_size)
        tl.store(
            outputs + token_offsets[:, None] * output_stride_m + output_offsets[None, :] * output_stride_n,
            accumulator,
            mask=output_mask,
        )


def materialize_expert_major_weights(weight_view: ExpertMajorMatrixView) -> torch.Tensor:
    """Create physical ``[E, O, K]`` storage from a logical provider view.

    This is an explicit benchmark/prototype boundary. It is not used during
    provider construction and it is not invoked by ``TritonExpertExecutor``.
    Future packed-weight loaders or CUDA pointer-array kernels can remove this
    materialization without changing the grouped GEMM API.
    """

    return weight_view.materialize().contiguous()


def reference_grouped_matmul(
    expert_inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for expert-major grouped matrix multiplication."""

    _validate_grouped_matmul_inputs(expert_inputs, expert_weights, expert_offsets)
    output_size = expert_weights.shape[1]
    outputs = torch.empty(expert_inputs.shape[0], output_size, dtype=expert_inputs.dtype, device=expert_inputs.device)
    for expert_id in range(expert_weights.shape[0]):
        start = int(expert_offsets[expert_id].item())
        end = int(expert_offsets[expert_id + 1].item())
        if end > start:
            outputs[start:end] = torch.matmul(expert_inputs[start:end], expert_weights[expert_id].transpose(0, 1))
    return outputs


def grouped_matmul(
    expert_inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    *,
    max_tokens_per_expert: int | None = None,
) -> torch.Tensor:
    """Run one deterministic Triton grouped expert projection.

    Args:
        expert_inputs: Expert-major activations with shape ``[N, K]``.
        expert_weights: Dense expert-major matrices with shape ``[E, O, K]``.
        expert_offsets: Dispatcher expert offsets with shape ``[E + 1]``.
        max_tokens_per_expert: Optional host scalar used to size the launch
            grid without synchronizing on ``expert_offsets``. The future
            Executor obtains this from ``ExecutionPlan.statistics``.
    """

    _validate_grouped_matmul_inputs(expert_inputs, expert_weights, expert_offsets)
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton grouped matmul requested but triton is not installed")
    if not expert_inputs.is_cuda:
        raise RuntimeError("Triton grouped matmul requires CUDA tensors")
    if expert_inputs.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Triton grouped matmul currently supports FP16 and BF16 inputs")
    if expert_weights.dtype != expert_inputs.dtype:
        raise ValueError("expert inputs and weights must use the same dtype")
    if not expert_inputs.is_contiguous() or not expert_weights.is_contiguous():
        raise ValueError("Triton grouped matmul requires contiguous inputs and weights")

    num_experts, output_size, hidden_size = expert_weights.shape
    if max_tokens_per_expert is None:
        counts = expert_offsets[1:] - expert_offsets[:-1]
        max_tokens_per_expert = int(counts.max().item()) if counts.numel() else 0
    if max_tokens_per_expert < 0:
        raise ValueError("max_tokens_per_expert must be non-negative")
    outputs = torch.empty(expert_inputs.shape[0], output_size, dtype=expert_inputs.dtype, device=expert_inputs.device)
    if max_tokens_per_expert == 0:
        return outputs

    grid = lambda meta: (
        num_experts,
        triton.cdiv(max_tokens_per_expert, meta["BLOCK_M"]),
        triton.cdiv(output_size, meta["BLOCK_N"]),
    )
    _grouped_matmul_kernel[grid](
        expert_inputs,
        expert_weights,
        expert_offsets,
        outputs,
        hidden_size=hidden_size,
        output_size=output_size,
        input_stride_m=expert_inputs.stride(0),
        input_stride_k=expert_inputs.stride(1),
        weight_stride_e=expert_weights.stride(0),
        weight_stride_o=expert_weights.stride(1),
        weight_stride_k=expert_weights.stride(2),
        output_stride_m=outputs.stride(0),
        output_stride_n=outputs.stride(1),
    )
    return outputs


def grouped_matmul_from_dispatch(
    expert_inputs: torch.Tensor,
    weight_view: ExpertMajorMatrixView,
    dispatch_plan: DispatchPlan,
    *,
    max_tokens_per_expert: int | None = None,
) -> torch.Tensor:
    """Run a grouped projection from DWDP expert-major metadata.

    ``expert_inputs`` must already be arranged in the dispatcher expert-major
    order. The wrapper materializes the logical view explicitly for this first
    Triton prototype; the Executor does not call it in this milestone.
    """

    if dispatch_plan.metadata.num_experts != weight_view.shape[0]:
        raise ValueError("DispatchPlan and expert weight view disagree on num_experts")
    if expert_inputs.shape[0] != dispatch_plan.metadata.num_assignments:
        raise ValueError("expert-major inputs must contain one row per dispatch assignment")
    return grouped_matmul(
        expert_inputs,
        materialize_expert_major_weights(weight_view),
        dispatch_plan.metadata.expert_offsets,
        max_tokens_per_expert=max_tokens_per_expert,
    )


def _validate_grouped_matmul_inputs(
    expert_inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
) -> None:
    if expert_inputs.ndim != 2:
        raise ValueError("expert_inputs must have shape [assignments, hidden_size]")
    if expert_weights.ndim != 3:
        raise ValueError("expert_weights must have shape [num_experts, output_size, hidden_size]")
    if expert_offsets.ndim != 1 or expert_offsets.dtype != torch.int64:
        raise ValueError("expert_offsets must be a rank-1 int64 tensor")
    if expert_offsets.numel() != expert_weights.shape[0] + 1:
        raise ValueError("expert_offsets must have num_experts + 1 entries")
    if expert_inputs.shape[1] != expert_weights.shape[2]:
        raise ValueError("expert input hidden size and weight K dimension must match")
    if expert_offsets.device != expert_inputs.device or expert_weights.device != expert_inputs.device:
        raise ValueError("inputs, weights, and offsets must share a device")
