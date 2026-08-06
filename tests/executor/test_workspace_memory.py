"""Workspace memory-scaling regression tests.

The MoE workspace allocates several buffers sized ``num_tokens * top_k``
rather than ``num_tokens``, so memory grows top_k times faster than a dense
model as concurrency increases. These tests pin the buffers that are actually
required so a regression that reintroduces a dead allocation is caught here
rather than as an OOM at high batch size.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from DWDP.runtime.config import RuntimeConfig  # noqa: E402
from DWDP.runtime.runtime import DWDPRuntime  # noqa: E402

HIDDEN = 64
INTERMEDIATE = 128
NUM_EXPERTS = 8
TOP_K = 4


class _Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        return self.down_proj(gate * self.up_proj(hidden_states))


def _build_runtime(materialize_packed: bool) -> DWDPRuntime:
    torch.manual_seed(0)
    runtime = DWDPRuntime.build_reference(
        hidden_size=HIDDEN,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        experts={index: _Expert() for index in range(NUM_EXPERTS)},
        config=RuntimeConfig(device="cpu"),
    )
    runtime.executor.config = replace(
        runtime.executor.config, materialize_packed_outputs=materialize_packed
    )
    runtime.eval()
    return runtime


def _workspace_bytes(workspace) -> int:
    total = 0
    for name in workspace.__slots__:
        value = getattr(workspace, name, None)
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def test_skipping_packed_outputs_is_numerically_identical():
    """Dropping the unread packed buffer must not change results at all."""

    torch.manual_seed(42)
    hidden_states = torch.randn(128, HIDDEN)

    with torch.inference_mode():
        with_packed = _build_runtime(True)(hidden_states).merger_output.hidden_states
        without_packed = _build_runtime(False)(hidden_states).merger_output.hidden_states

    assert torch.equal(with_packed, without_packed)


def test_skipping_packed_outputs_reduces_workspace():
    """The packed buffer is a full assignment-sized allocation."""

    torch.manual_seed(42)
    hidden_states = torch.randn(128, HIDDEN)

    runtime_on = _build_runtime(True)
    runtime_off = _build_runtime(False)
    with torch.inference_mode():
        runtime_on(hidden_states)
        runtime_off(hidden_states)

    bytes_on = _workspace_bytes(runtime_on.context.workspaces.executor)
    bytes_off = _workspace_bytes(runtime_off.context.workspaces.executor)

    saved = bytes_on - bytes_off
    expected = 128 * TOP_K * HIDDEN * torch.finfo(torch.float32).bits // 8
    assert saved == expected, f"expected to save {expected} bytes, saved {saved}"


def test_packed_outputs_are_not_materialized_when_disabled():
    """The buffer must be absent, not merely unused."""

    runtime = _build_runtime(False)
    with torch.inference_mode():
        output = runtime(torch.randn(32, HIDDEN))

    assert output.executor_output.packed_expert_outputs is None
    assert runtime.context.workspaces.executor.packed_expert_outputs is None


def test_workspace_scales_linearly_per_token():
    """Per-token workspace cost must stay flat as batch grows.

    A superlinear term here means something is allocating per-assignment
    squared or accumulating across iterations.
    """

    per_token: list[float] = []
    with torch.inference_mode():
        for num_tokens in (32, 64, 128, 256):
            runtime = _build_runtime(False)
            runtime(torch.randn(num_tokens, HIDDEN))
            total = _workspace_bytes(
                runtime.context.workspaces.executor
            ) + _workspace_bytes(runtime.context.workspaces.merger)
            per_token.append(total / num_tokens)

    # Allow a small constant-overhead effect at the smallest size.
    assert max(per_token) - min(per_token) < 0.05 * max(per_token)


def test_workspace_does_not_grow_across_repeated_forwards():
    """Repeated forwards at a fixed shape must reuse buffers, not accumulate."""

    runtime = _build_runtime(False)
    hidden_states = torch.randn(64, HIDDEN)

    with torch.inference_mode():
        runtime(hidden_states)
        after_first = _workspace_bytes(runtime.context.workspaces.executor)
        for _ in range(10):
            runtime(hidden_states)
        after_many = _workspace_bytes(runtime.context.workspaces.executor)

    assert after_first == after_many
