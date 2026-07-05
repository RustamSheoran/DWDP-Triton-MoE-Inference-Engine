from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from DWDP.adapters import HuggingFaceAdapter, build_adapter, get_adapter_class  # noqa: E402
from DWDP.runtime import DWDPRuntime, RuntimeConfig, compare_tensors  # noqa: E402


class ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


def make_runtime() -> DWDPRuntime:
    router = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        router.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                ]
            )
        )
    from DWDP.router import LinearTopKRouter, RouterConfig

    dwdp_router = LinearTopKRouter(RouterConfig(hidden_size=4, num_experts=2, top_k=1))
    with torch.no_grad():
        dwdp_router.weight.copy_(router.weight)
    return DWDPRuntime.build_reference(
        hidden_size=4,
        num_experts=2,
        top_k=1,
        experts=[ScaleExpert(2.0), ScaleExpert(3.0)],
        router=dwdp_router,
        config=RuntimeConfig(enable_workspace=True, enable_profiling=True),
    )


def test_runtime_executes_complete_pipeline() -> None:
    runtime = make_runtime()
    hidden_states = torch.tensor([[[2.0, 1.0, 0.0, 0.0], [1.0, 4.0, 0.0, 0.0]]])

    output = runtime(hidden_states)

    expected = torch.tensor([[[4.0, 2.0, 0.0, 0.0], [3.0, 12.0, 0.0, 0.0]]])
    assert torch.allclose(output.hidden_states, expected)
    assert output.dispatch_plan.metadata.num_assignments == 2
    assert output.execution_plan.statistics.num_active_experts == 2
    assert output.communication_plan.statistics.num_remote_experts == 0
    assert output.executor_output.statistics.num_executed_experts == 2
    assert output.merger_output.statistics.num_tokens == 2
    assert output.profile is not None


def test_runtime_workspace_reuse() -> None:
    runtime = make_runtime()
    hidden_states = torch.randn(1, 2, 4)

    runtime(hidden_states)
    first_bytes = runtime.context.workspaces.estimated_bytes()
    runtime(hidden_states)

    assert runtime.context.workspaces.estimated_bytes() == first_bytes
    assert first_bytes > 0


def test_runtime_config_validation() -> None:
    with pytest.raises(ValueError):
        RuntimeConfig(world_size=0)
    with pytest.raises(ValueError):
        RuntimeConfig(world_size=2, local_rank=2)


def test_adapter_registry_builds_huggingface_adapter() -> None:
    adapter = build_adapter("huggingface", model=nn.Identity(), config=RuntimeConfig())

    assert isinstance(adapter, HuggingFaceAdapter)
    assert get_adapter_class("huggingface") is HuggingFaceAdapter


def test_correctness_tensor_comparison() -> None:
    report = compare_tensors(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.001]), atol=1e-2)

    assert report.allclose
    assert report.max_abs_error > 0.0


def test_huggingface_adapter_can_bind_explicit_moe_layer() -> None:
    adapter = HuggingFaceAdapter(model=nn.Identity(), config=RuntimeConfig())
    adapter.bind_moe_layer(
        hidden_size=4,
        num_experts=2,
        top_k=1,
        experts=[ScaleExpert(1.0), ScaleExpert(1.0)],
    )

    runtime = adapter.create_runtime()

    assert isinstance(runtime, DWDPRuntime)
