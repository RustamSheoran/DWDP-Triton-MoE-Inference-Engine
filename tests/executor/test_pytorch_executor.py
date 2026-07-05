from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.comms_planner import (  # noqa: E402
    CommunicationCostModel,
    CommunicationGraph,
    CommunicationPlan,
    CommunicationStatistics,
    DependencyMetadata as CommunicationDependencyMetadata,
    OverlapPlan,
    PrefetchPlan,
    SynchronizationMetadata as CommunicationSynchronizationMetadata,
    TopologyMetadata,
)
from DWDP.dispatcher import DispatchMetadata, DispatchPlan, ExpertAssignments  # noqa: E402
from DWDP.executor import ExecutorConfig, ExecutorWorkspace, PyTorchExecutor, build_executor  # noqa: E402
from DWDP.executor.experts import ExpertRegistry  # noqa: E402
from DWDP.scheduler import (  # noqa: E402
    DependencyMetadata as SchedulerDependencyMetadata,
    ExecutionPlan,
    SchedulerStatistics,
    SynchronizationMetadata as SchedulerSynchronizationMetadata,
)


class ScaleExpert(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * self.scale


def make_dispatch_plan() -> DispatchPlan:
    expert_ids = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    token_indices = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
    routing_weights = torch.tensor([0.5, 1.0, 0.25, 0.75], dtype=torch.float32)
    expert_counts = torch.tensor([2, 2], dtype=torch.int64)
    expert_offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
    assignments = ExpertAssignments(
        expert_ids=expert_ids,
        packed_token_indices=token_indices,
        packed_routing_weights=routing_weights,
    )
    metadata = DispatchMetadata(
        num_tokens=4,
        num_assignments=4,
        num_experts=2,
        top_k=1,
        token_shape=(4,),
        expert_counts=expert_counts,
        expert_offsets=expert_offsets,
        token_permutation=torch.arange(4, dtype=torch.int64),
        inverse_permutation=torch.arange(4, dtype=torch.int64),
        destination_positions=torch.arange(4, dtype=torch.int64),
        stable_order=True,
        algorithm="test",
    )
    return DispatchPlan(assignments=assignments, metadata=metadata)


def make_execution_plan() -> ExecutionPlan:
    order = torch.tensor([0, 1], dtype=torch.int64)
    return ExecutionPlan(
        execution_order=order,
        expert_queue=torch.tensor([0, 1], dtype=torch.int64),
        expert_starts=torch.tensor([0, 2], dtype=torch.int64),
        expert_ends=torch.tensor([2, 4], dtype=torch.int64),
        expert_counts=torch.tensor([2, 2], dtype=torch.int64),
        execution_priority=order,
        stream_assignments=torch.tensor([0, 0], dtype=torch.int64),
        batches=(),
        synchronization=SchedulerSynchronizationMetadata(
            barrier_after_batch=torch.zeros(2, dtype=torch.bool),
        ),
        dependencies=SchedulerDependencyMetadata(
            dependency_src=torch.empty(0, dtype=torch.int64),
            dependency_dst=torch.empty(0, dtype=torch.int64),
        ),
        statistics=SchedulerStatistics(
            num_experts=2,
            num_active_experts=2,
            num_empty_experts=0,
            num_execution_batches=2,
            num_assignments=4,
            max_tokens_per_expert=2,
            min_tokens_per_active_expert=2,
            scheduling_policy="round_robin",
        ),
        scheduling_policy="round_robin",
        deterministic=True,
    )


def make_communication_plan() -> CommunicationPlan:
    empty_i64 = torch.empty(0, dtype=torch.int64)
    empty_f32 = torch.empty(0, dtype=torch.float32)
    return CommunicationPlan(
        local_expert_ids=torch.tensor([0, 1], dtype=torch.int64),
        remote_expert_ids=empty_i64,
        graph=CommunicationGraph(nodes=(), edges=(), node_ids=empty_i64, edge_src=empty_i64, edge_dst=empty_i64),
        communication_descriptors=(),
        transfer_descriptors=(),
        communication_groups=(),
        topology=TopologyMetadata(
            local_gpu_id=0,
            world_size=1,
            local_rank=0,
            gpu_ids=torch.tensor([0], dtype=torch.int64),
            numa_domains=torch.tensor([0], dtype=torch.int64),
            nvlink_connectivity=None,
            nvswitch_domains=None,
            pcie_hierarchy=None,
            communication_domains=(),
            locality_groups=((0,),),
            fabric="single_gpu",
            default_link_bandwidth_gbps=0.0,
            default_link_latency_us=0.0,
        ),
        synchronization=CommunicationSynchronizationMetadata(
            barrier_node_ids=empty_i64,
            cuda_event_ids=empty_i64,
            stream_wait_edges=empty_i64,
            synchronization_points=empty_i64,
        ),
        dependencies=CommunicationDependencyMetadata(
            dependency_src=empty_i64,
            dependency_dst=empty_i64,
            dependency_type=empty_i64,
        ),
        prefetch=PrefetchPlan(
            prefetch_expert_ids=empty_i64,
            prefetch_priorities=empty_i64,
            prefetch_windows_us=empty_f32,
        ),
        overlap=OverlapPlan(
            communication_node_ids=empty_i64,
            compute_batch_ids=empty_i64,
            overlap_windows_us=empty_f32,
        ),
        cost_model=CommunicationCostModel(
            estimates=(),
            total_estimated_bytes=0,
            total_estimated_latency_us=0.0,
            critical_path_us=0.0,
            estimated_bandwidth_gbps=0.0,
        ),
        statistics=CommunicationStatistics(
            num_local_experts=2,
            num_remote_experts=0,
            num_communication_nodes=0,
            num_communication_edges=0,
            num_transfer_descriptors=0,
            num_communication_groups=0,
            total_estimated_bytes=0,
            total_estimated_latency_us=0.0,
            planner_policy="static",
        ),
        planner_policy="static",
        deterministic=True,
    )


def make_executor() -> PyTorchExecutor:
    experts = ExpertRegistry({0: ScaleExpert(2.0), 1: ScaleExpert(3.0)})
    return PyTorchExecutor(ExecutorConfig(), experts)


def test_executor_outputs_and_routing_weights_are_correct() -> None:
    hidden_states = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )
    executor = make_executor()

    output = executor(
        hidden_states,
        make_dispatch_plan(),
        make_execution_plan(),
        make_communication_plan(),
    )

    expected_unweighted = torch.tensor(
        [
            [2.0, 4.0],
            [10.0, 12.0],
            [9.0, 12.0],
            [21.0, 24.0],
        ]
    )
    expected_weighted = torch.tensor(
        [
            [1.0, 2.0],
            [10.0, 12.0],
            [2.25, 3.0],
            [15.75, 18.0],
        ]
    )
    assert torch.allclose(output.packed_expert_outputs, expected_unweighted)
    assert torch.allclose(output.weighted_expert_outputs, expected_weighted)
    assert output.statistics.num_executed_experts == 2
    assert output.statistics.num_assignments == 4
    assert output.output_metadata.top_k == 1


def test_execution_order_is_preserved() -> None:
    executor = make_executor()
    output = executor(
        torch.randn(4, 2),
        make_dispatch_plan(),
        make_execution_plan(),
        make_communication_plan(),
    )

    assert [record.expert_id for record in output.expert_outputs] == [0, 1]
    assert [record.start for record in output.expert_outputs] == [0, 2]


def test_workspace_reuses_output_buffers() -> None:
    executor = make_executor()
    workspace = ExecutorWorkspace()
    hidden_states = torch.randn(4, 2)

    first = executor(
        hidden_states,
        make_dispatch_plan(),
        make_execution_plan(),
        make_communication_plan(),
        workspace=workspace,
    )
    first_ptr = first.weighted_expert_outputs.data_ptr()
    second = executor(
        hidden_states,
        make_dispatch_plan(),
        make_execution_plan(),
        make_communication_plan(),
        workspace=workspace,
    )

    assert second.weighted_expert_outputs.data_ptr() == first_ptr
    assert second.workspace.used_workspace
    assert workspace.estimated_bytes() > 0


def test_workspace_can_be_disabled() -> None:
    executor = PyTorchExecutor(
        ExecutorConfig(enable_workspace=False),
        ExpertRegistry({0: ScaleExpert(1.0), 1: ScaleExpert(1.0)}),
    )
    workspace = ExecutorWorkspace()

    output = executor(
        torch.randn(4, 2),
        make_dispatch_plan(),
        make_execution_plan(),
        make_communication_plan(),
        workspace=workspace,
    )

    assert not output.workspace.used_workspace
    assert workspace.estimated_bytes() == 0


def test_registry_builds_executor() -> None:
    executor = build_executor(ExecutorConfig(), [ScaleExpert(1.0), ScaleExpert(2.0)])

    assert isinstance(executor, PyTorchExecutor)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        ExecutorConfig(backend="")
    with pytest.raises(ValueError):
        ExecutorConfig(max_tokens_per_expert=0)


def test_remote_experts_are_rejected_by_reference_backend() -> None:
    comms = make_communication_plan()
    comms.remote_expert_ids = torch.tensor([1], dtype=torch.int64)

    with pytest.raises(NotImplementedError):
        make_executor()(torch.randn(4, 2), make_dispatch_plan(), make_execution_plan(), comms)


def test_missing_expert_is_rejected() -> None:
    executor = PyTorchExecutor(ExecutorConfig(), ExpertRegistry({0: ScaleExpert(1.0)}))

    with pytest.raises(KeyError):
        executor(
            torch.randn(4, 2),
            make_dispatch_plan(),
            make_execution_plan(),
            make_communication_plan(),
        )
