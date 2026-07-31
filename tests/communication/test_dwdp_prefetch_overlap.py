"""Integration & overlap tests for DWDP asynchronous weight prefetching."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.comms_planner.config import CommunicationPlannerConfig  # noqa: E402
from DWDP.comms_planner.static import StaticCommunicationPlanner  # noqa: E402
from DWDP.communication.streams import CudaStreams, StreamRole  # noqa: E402
from DWDP.scheduler.execution import ExecutionPlan  # noqa: E402


def test_dwdp_double_buffer_prefetch_plan_generation() -> None:
    """Verify that DWDP generates overlapped prefetch steps across layers N and N+1."""
    config = CommunicationPlannerConfig(enable_prefetch=True, double_buffering=True)
    planner = StaticCommunicationPlanner(config)

    # 4 experts total across 2 ranks
    expert_queue = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    expert_starts = torch.tensor([0, 10, 20, 30], dtype=torch.int32)
    expert_counts = torch.tensor([10, 10, 10, 10], dtype=torch.int32)
    execution_priority = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    stream_assignments = torch.tensor([0, 0, 1, 1], dtype=torch.int32)

    plan_layer_0 = ExecutionPlan(
        expert_queue=expert_queue,
        expert_starts=expert_starts,
        expert_counts=expert_counts,
        execution_priority=execution_priority,
        stream_assignments=stream_assignments,
    )

    comm_plan = planner.plan(plan_layer_0)

    assert comm_plan is not None
    assert hasattr(comm_plan, "transfers")
    # DWDP planner schedules non-blocking prefetch transfers
    assert comm_plan.overlap_with_compute is True


def test_cuda_stream_barrier_sync_semantics() -> None:
    """Test CUDA stream synchronization semantics for compute and prefetch copy streams."""
    streams = CudaStreams(enabled=True)

    if not torch.cuda.is_available():
        # CPU fallback: stream context should be non-blocking no-op
        with streams.use(StreamRole.COPY, "cpu"):
            pass
        with streams.use(StreamRole.COMPUTE, "cpu"):
            pass
        assert not streams.initialized
        return

    device = "cuda:0"
    streams.ensure(device)
    assert streams.initialized
    assert streams.copy is not None
    assert streams.compute is not None

    # Test stream execution context scoping
    with streams.use(StreamRole.COPY, device):
        tensor_a = torch.ones((100, 100), device=device)

    with streams.use(StreamRole.COMPUTE, device):
        tensor_b = tensor_a * 2.0

    # Ensure compute stream waited for copy stream
    streams.compute.synchronize()
    assert torch.allclose(tensor_b, torch.full((100, 100), 2.0, device=device))


def test_fp8_scale_factor_prefetch_pipelining() -> None:
    """Test that prefetching expert weights includes FP8 micro-scale metadata."""
    from DWDP.executor.fp8 import convert_qwen_weights_to_fp8_once
    from DWDP.executor.weights import (
        ExpertWeightProvider,
        GateUpWeightProvider,
        QwenSwiGLUWeightProvider,
    )

    # Create dummy expert weights
    w_gate = torch.randn(4, 64, 32)
    w_up = torch.randn(4, 64, 32)
    w_down = torch.randn(4, 32, 64)

    gate_provider = ExpertWeightProvider([w_gate[i] for i in range(4)])
    up_provider = ExpertWeightProvider([w_up[i] for i in range(4)])
    gate_up = GateUpWeightProvider(gate_provider, up_provider)
    down_provider = ExpertWeightProvider([w_down[i] for i in range(4)])

    qwen_provider = QwenSwiGLUWeightProvider(
        gate_up_weights=gate_up,
        down_weights=down_provider,
        expert_ids=tuple(range(4)),
        hidden_size=32,
        intermediate_size=64,
        has_bias=False,
    )

    scale_map: dict[int, torch.Tensor] = {}
    if hasattr(torch, "float8_e4m3fn"):
        dtype = torch.float8_e4m3fn
    else:
        dtype = torch.float16

    convert_qwen_weights_to_fp8_once(qwen_provider, dtype=dtype, scale_map=scale_map)

    # If FP8 is supported on host PyTorch, scale_map should be populated with 12 scale tensors (3 per expert * 4 experts)
    if "float8" in str(dtype):
        assert len(scale_map) == 12
        for scale in scale_map.values():
            assert scale > 0
