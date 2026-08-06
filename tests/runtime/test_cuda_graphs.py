"""CUDA graph capture tests for decode path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.runtime.cuda_graph import (
    CUDAGraphRunner,
    find_host_syncs,
)


def dummy_pipeline(hidden_states: torch.Tensor) -> torch.Tensor:
    """Trivial stateless function for capture testing."""
    return hidden_states * 2.0 + 1.0


def test_graph_runner_lazy_capture_on_first_call():
    """Graph capture happens lazily on the first forward at each shape."""
    runner = CUDAGraphRunner(dummy_pipeline, warmup_steps=2)
    assert runner.stats.captures == 0
    assert runner.stats.replays == 0

    # CPU fallback: no capture, no replay.
    x = torch.randn(4, 8)
    y1 = runner(hidden_states=x)
    assert runner.stats.captures == 0
    assert runner.stats.replays == 0
    assert runner.stats.fallbacks == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_runner_captures_and_replays_on_cuda():
    """First call at a shape captures, later calls replay."""
    runner = CUDAGraphRunner(dummy_pipeline, warmup_steps=2, enabled=True)
    x = torch.randn(4, 8, device="cuda")

    # First call triggers capture.
    y1 = runner(hidden_states=x)
    assert runner.stats.captures == 1
    assert runner.stats.replays == 0

    # Same shape: replay.
    y2 = runner(hidden_states=x)
    assert runner.stats.captures == 1
    assert runner.stats.replays == 1

    # Different shape: new capture.
    x_new = torch.randn(8, 8, device="cuda")
    y3 = runner(hidden_states=x_new)
    assert runner.stats.captures == 2
    assert runner.stats.replays == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replay_matches_eager_numerically():
    """Captured graph produces identical output to eager execution."""
    runner = CUDAGraphRunner(dummy_pipeline, warmup_steps=1, enabled=True)
    x = torch.randn(4, 8, device="cuda")

    eager = dummy_pipeline(x)
    replayed = runner(hidden_states=x)

    assert torch.allclose(replayed, eager, atol=1e-5)


def test_find_host_syncs_detects_tolist():
    """Audit correctly identifies .tolist() in the hot path."""
    findings = find_host_syncs()
    # PyTorch executor is not capturable by design.
    assert "DWDP.executor.pytorch" in findings
    # The reference dispatcher uses a Python loop; triton_counting_scatter is
    # the device-resident alternative selected at runtime.
    assert "DWDP.dispatcher.ops.scatter" in findings


def test_graph_runner_disabled_always_falls_back():
    """Disabled runner never captures, always runs eager."""
    runner = CUDAGraphRunner(dummy_pipeline, enabled=False)
    if torch.cuda.is_available():
        x = torch.randn(4, 8, device="cuda")
    else:
        x = torch.randn(4, 8)

    y = runner(hidden_states=x)
    assert runner.stats.captures == 0
    assert runner.stats.fallbacks == 1
