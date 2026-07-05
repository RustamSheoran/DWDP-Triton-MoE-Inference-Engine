from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.executor import (  # noqa: E402
    ExecutionMetadata,
    ExecutionStatistics,
    ExecutorOutput,
    ExpertOutput,
    OutputMetadata,
)
from DWDP.executor.metadata import TimingMetadata as ExecutorTimingMetadata  # noqa: E402
from DWDP.executor.metadata import WorkspaceMetadata as ExecutorWorkspaceMetadata  # noqa: E402
from DWDP.merger import MergerConfig, MergerWorkspace, PyTorchMerger, build_merger  # noqa: E402


def make_executor_output() -> ExecutorOutput:
    packed_outputs = torch.tensor(
        [
            [10.0, 0.0],
            [30.0, 0.0],
            [1.0, 20.0],
            [1.0, 40.0],
        ]
    )
    weights = torch.tensor([0.25, 0.75, 0.5, 0.5], dtype=torch.float32)
    weighted = packed_outputs * weights.unsqueeze(-1)
    # Expert-major positions map to token-major assignment positions:
    # token0/k0 <- expert pos 0, token0/k1 <- expert pos 2
    # token1/k0 <- expert pos 1, token1/k1 <- expert pos 3
    inverse = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
    return ExecutorOutput(
        packed_expert_outputs=packed_outputs,
        weighted_expert_outputs=weighted,
        expert_outputs=(
            ExpertOutput(0, 0, 2, 2, 0, 0),
            ExpertOutput(1, 2, 4, 2, 1, 0),
        ),
        output_metadata=OutputMetadata(
            packed_token_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            packed_expert_ids=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
            packed_routing_weights=weights,
            token_permutation=torch.tensor([0, 2, 1, 3], dtype=torch.int64),
            inverse_permutation=inverse,
            token_shape=(2,),
            top_k=2,
        ),
        execution_metadata=ExecutionMetadata(
            execution_order=torch.tensor([0, 1], dtype=torch.int64),
            expert_queue=torch.tensor([0, 1], dtype=torch.int64),
            expert_starts=torch.tensor([0, 2], dtype=torch.int64),
            expert_ends=torch.tensor([2, 4], dtype=torch.int64),
            stream_assignments=torch.tensor([0, 0], dtype=torch.int64),
            communication_remote_expert_ids=torch.empty(0, dtype=torch.int64),
            communication_policy="static",
            scheduling_policy="round_robin",
        ),
        statistics=ExecutionStatistics(
            num_executed_experts=2,
            num_skipped_experts=0,
            num_input_tokens=2,
            num_assignments=4,
            hidden_size=2,
            output_size=2,
            backend="pytorch",
        ),
        timing=ExecutorTimingMetadata(),
        workspace=ExecutorWorkspaceMetadata(False, 0),
        backend="pytorch",
        deterministic=True,
    )


def test_merger_reconstructs_topk_weighted_outputs() -> None:
    merger = PyTorchMerger(MergerConfig())

    output = merger(make_executor_output())

    expected = torch.tensor(
        [
            [3.0, 10.0],
            [23.0, 20.0],
        ]
    )
    assert torch.allclose(output.hidden_states, expected)
    assert output.statistics.num_tokens == 2
    assert output.statistics.num_assignments == 4
    assert output.statistics.top_k == 2
    assert output.statistics.used_weighted_executor_outputs


def test_merger_can_apply_routing_weights_itself() -> None:
    merger = PyTorchMerger(MergerConfig(apply_routing_weights=True))

    output = merger(make_executor_output())

    expected = torch.tensor(
        [
            [3.0, 10.0],
            [23.0, 20.0],
        ]
    )
    assert torch.allclose(output.hidden_states, expected)
    assert not output.statistics.used_weighted_executor_outputs


def test_merger_restores_batch_dimensions() -> None:
    executor_output = make_executor_output()
    executor_output.output_metadata.token_shape = (1, 2)
    merger = PyTorchMerger(MergerConfig())

    output = merger(executor_output)

    assert output.hidden_states.shape == (1, 2, 2)
    assert output.metadata.token_shape == (1, 2)


def test_workspace_reuses_buffers() -> None:
    merger = PyTorchMerger(MergerConfig())
    workspace = MergerWorkspace()

    first = merger(make_executor_output(), workspace=workspace)
    first_ptr = first.hidden_states.data_ptr()
    second = merger(make_executor_output(), workspace=workspace)

    assert second.hidden_states.data_ptr() == first_ptr
    assert second.workspace.used_workspace
    assert workspace.estimated_bytes() > 0


def test_workspace_can_be_disabled() -> None:
    merger = PyTorchMerger(MergerConfig(enable_workspace=False))
    workspace = MergerWorkspace()

    output = merger(make_executor_output(), workspace=workspace)

    assert not output.workspace.used_workspace
    assert workspace.estimated_bytes() == 0


def test_registry_builds_merger() -> None:
    merger = build_merger(MergerConfig())

    assert isinstance(merger, PyTorchMerger)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        MergerConfig(backend="")


def test_shape_validation_rejects_bad_assignment_count() -> None:
    executor_output = make_executor_output()
    executor_output.output_metadata.top_k = 3
    merger = PyTorchMerger(MergerConfig())

    with pytest.raises(ValueError):
        merger(executor_output)
