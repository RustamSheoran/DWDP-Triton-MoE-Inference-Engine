from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.comms_planner import (  # noqa: E402
    CommunicationPlannerConfig,
    CommunicationPlannerWorkspace,
    StaticCommunicationPlanner,
    build_communication_planner,
)
from DWDP.scheduler import (  # noqa: E402
    DependencyMetadata as SchedulerDependencyMetadata,
    ExecutionPlan,
    SchedulerStatistics,
    SynchronizationMetadata as SchedulerSynchronizationMetadata,
)


def make_execution_plan(expert_ids: torch.Tensor) -> ExecutionPlan:
    count = expert_ids.numel()
    starts = torch.arange(count, dtype=torch.int64) * 4
    ends = starts + 4
    order = torch.arange(count, dtype=torch.int64)
    return ExecutionPlan(
        execution_order=order,
        expert_queue=expert_ids,
        expert_starts=starts,
        expert_ends=ends,
        expert_counts=torch.full((count,), 4, dtype=torch.int64),
        execution_priority=order,
        stream_assignments=torch.remainder(order, 2),
        batches=(),
        synchronization=SchedulerSynchronizationMetadata(
            barrier_after_batch=torch.zeros(count, dtype=torch.bool),
        ),
        dependencies=SchedulerDependencyMetadata(
            dependency_src=torch.empty(0, dtype=torch.int64),
            dependency_dst=torch.empty(0, dtype=torch.int64),
        ),
        statistics=SchedulerStatistics(
            num_experts=8,
            num_active_experts=count,
            num_empty_experts=8 - count,
            num_execution_batches=count,
            num_assignments=int(count * 4),
            max_tokens_per_expert=4 if count else 0,
            min_tokens_per_active_expert=4 if count else 0,
            scheduling_policy="round_robin",
        ),
        scheduling_policy="round_robin",
        deterministic=True,
    )


def test_single_gpu_plan_has_no_remote_communication() -> None:
    execution_plan = make_execution_plan(torch.tensor([0, 2, 5], dtype=torch.int64))
    planner = StaticCommunicationPlanner(CommunicationPlannerConfig())

    plan = planner(execution_plan)

    assert torch.equal(plan.local_expert_ids, execution_plan.expert_queue)
    assert plan.remote_expert_ids.numel() == 0
    assert plan.graph.is_empty
    assert plan.communication_descriptors == ()
    assert plan.transfer_descriptors == ()
    assert plan.communication_groups == ()
    assert plan.cost_model.total_estimated_bytes == 0
    assert plan.cost_model.total_estimated_latency_us == 0.0


def test_topology_metadata_describes_local_gpu() -> None:
    execution_plan = make_execution_plan(torch.tensor([1], dtype=torch.int64))
    planner = StaticCommunicationPlanner(
        CommunicationPlannerConfig(local_gpu_id=0, world_size=1, local_rank=0)
    )

    plan = planner(execution_plan)

    assert plan.topology.local_gpu_id == 0
    assert plan.topology.world_size == 1
    assert plan.topology.local_rank == 0
    assert torch.equal(plan.topology.gpu_ids, torch.tensor([0]))
    assert torch.equal(plan.topology.numa_domains, torch.tensor([0]))
    assert plan.topology.fabric == "single_gpu"
    assert len(plan.topology.communication_domains) == 1
    assert plan.topology.communication_domains[0].domain_type == "local"


def test_dependency_and_synchronization_graphs_are_empty() -> None:
    execution_plan = make_execution_plan(torch.tensor([0, 1], dtype=torch.int64))
    planner = StaticCommunicationPlanner(CommunicationPlannerConfig())

    plan = planner(execution_plan)

    assert plan.dependencies.dependency_src.numel() == 0
    assert plan.dependencies.dependency_dst.numel() == 0
    assert plan.dependencies.dependency_type.numel() == 0
    assert plan.synchronization.barrier_node_ids.numel() == 0
    assert plan.synchronization.cuda_event_ids.numel() == 0
    assert plan.synchronization.stream_wait_edges.numel() == 0
    assert plan.synchronization.synchronization_points.numel() == 0


def test_prefetch_and_overlap_are_empty_for_single_gpu() -> None:
    execution_plan = make_execution_plan(torch.tensor([0, 3], dtype=torch.int64))
    planner = StaticCommunicationPlanner(CommunicationPlannerConfig())

    plan = planner(execution_plan)

    assert plan.prefetch.prefetch_expert_ids.numel() == 0
    assert plan.prefetch.prefetch_priorities.numel() == 0
    assert plan.prefetch.prefetch_windows_us.numel() == 0
    assert plan.overlap.communication_node_ids.numel() == 0
    assert plan.overlap.compute_batch_ids.numel() == 0
    assert plan.overlap.overlap_windows_us.numel() == 0


def test_statistics_are_correct() -> None:
    execution_plan = make_execution_plan(torch.tensor([0, 2, 4, 6], dtype=torch.int64))
    planner = StaticCommunicationPlanner(CommunicationPlannerConfig())

    plan = planner(execution_plan)

    assert plan.statistics.num_local_experts == 4
    assert plan.statistics.num_remote_experts == 0
    assert plan.statistics.num_communication_nodes == 0
    assert plan.statistics.num_communication_edges == 0
    assert plan.statistics.num_transfer_descriptors == 0
    assert plan.statistics.num_communication_groups == 0
    assert plan.statistics.total_estimated_bytes == 0
    assert plan.statistics.total_estimated_latency_us == 0.0


def test_empty_execution_plan_is_valid() -> None:
    execution_plan = make_execution_plan(torch.empty(0, dtype=torch.int64))
    planner = StaticCommunicationPlanner(CommunicationPlannerConfig())

    plan = planner(execution_plan)

    assert plan.local_expert_ids.numel() == 0
    assert plan.remote_expert_ids.numel() == 0
    assert plan.graph.is_empty
    assert plan.statistics.num_local_experts == 0


def test_workspace_reuses_topology_buffers() -> None:
    execution_plan = make_execution_plan(torch.tensor([1, 2], dtype=torch.int64))
    planner = StaticCommunicationPlanner(CommunicationPlannerConfig(world_size=1))
    workspace = CommunicationPlannerWorkspace()

    first_plan = planner(execution_plan, workspace=workspace)
    gpu_ids_ptr = first_plan.topology.gpu_ids.data_ptr()
    numa_ptr = first_plan.topology.numa_domains.data_ptr()

    second_plan = planner(execution_plan, workspace=workspace)

    assert second_plan.topology.gpu_ids.data_ptr() == gpu_ids_ptr
    assert second_plan.topology.numa_domains.data_ptr() == numa_ptr
    assert workspace.estimated_bytes() > 0


def test_workspace_can_be_disabled() -> None:
    execution_plan = make_execution_plan(torch.tensor([1, 2], dtype=torch.int64))
    planner = StaticCommunicationPlanner(
        CommunicationPlannerConfig(enable_workspace=False)
    )
    workspace = CommunicationPlannerWorkspace()

    planner(execution_plan, workspace=workspace)

    assert workspace.estimated_bytes() == 0


def test_registry_builds_static_planner() -> None:
    planner = build_communication_planner(CommunicationPlannerConfig())

    assert isinstance(planner, StaticCommunicationPlanner)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        CommunicationPlannerConfig(planner_policy="")
    with pytest.raises(ValueError):
        CommunicationPlannerConfig(world_size=0)
    with pytest.raises(ValueError):
        CommunicationPlannerConfig(local_rank=1, world_size=1)
    with pytest.raises(ValueError):
        CommunicationPlannerConfig(stream_count=0)


def test_invalid_execution_plan_rejected() -> None:
    execution_plan = make_execution_plan(torch.tensor([1, 2], dtype=torch.int64))
    execution_plan.expert_counts = torch.tensor([4], dtype=torch.int64)
    planner = StaticCommunicationPlanner(CommunicationPlannerConfig())

    with pytest.raises(ValueError):
        planner(execution_plan)
