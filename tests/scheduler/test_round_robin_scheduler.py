from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.dispatcher import DispatchMetadata, DispatchPlan, ExpertAssignments  # noqa: E402
from DWDP.scheduler import (  # noqa: E402
    RoundRobinScheduler,
    SchedulerConfig,
    SchedulerMetadataLevel,
    SchedulerWorkspace,
    build_scheduler,
)


def make_dispatch_plan(expert_counts: torch.Tensor) -> DispatchPlan:
    expert_offsets = torch.cat(
        (expert_counts.new_zeros(1), expert_counts.cumsum(dim=0)),
        dim=0,
    )
    num_assignments = int(expert_counts.sum().item())
    assignments = ExpertAssignments(
        expert_ids=torch.empty(num_assignments, dtype=torch.int64),
        packed_token_indices=torch.empty(num_assignments, dtype=torch.int64),
        packed_routing_weights=torch.empty(num_assignments, dtype=torch.float32),
    )
    metadata = DispatchMetadata(
        num_tokens=num_assignments,
        num_assignments=num_assignments,
        num_experts=expert_counts.numel(),
        top_k=1,
        token_shape=(num_assignments,),
        expert_counts=expert_counts,
        expert_offsets=expert_offsets,
        token_permutation=torch.arange(num_assignments, dtype=torch.int64),
        inverse_permutation=torch.arange(num_assignments, dtype=torch.int64),
        destination_positions=torch.arange(num_assignments, dtype=torch.int64),
        stable_order=True,
        algorithm="test",
    )
    return DispatchPlan(assignments=assignments, metadata=metadata)


def test_round_robin_skips_empty_experts_and_preserves_order() -> None:
    dispatch_plan = make_dispatch_plan(torch.tensor([0, 3, 0, 2, 1], dtype=torch.int64))
    scheduler = RoundRobinScheduler(SchedulerConfig(stream_count=2))

    plan = scheduler(dispatch_plan)

    assert torch.equal(plan.expert_queue, torch.tensor([1, 3, 4]))
    assert torch.equal(plan.execution_order, torch.tensor([0, 1, 2]))
    assert torch.equal(plan.expert_starts, torch.tensor([0, 3, 5]))
    assert torch.equal(plan.expert_ends, torch.tensor([3, 5, 6]))
    assert torch.equal(plan.expert_counts, torch.tensor([3, 2, 1]))
    assert torch.equal(plan.execution_priority, torch.tensor([0, 1, 2]))
    assert torch.equal(plan.stream_assignments, torch.tensor([0, 1, 0]))


def test_execution_batches_match_tensor_metadata() -> None:
    dispatch_plan = make_dispatch_plan(torch.tensor([2, 0, 4], dtype=torch.int64))
    scheduler = RoundRobinScheduler(SchedulerConfig(stream_count=4))

    plan = scheduler(dispatch_plan)

    assert len(plan.batches) == 2
    assert plan.batches[0].expert_id == 0
    assert plan.batches[0].start == 0
    assert plan.batches[0].end == 2
    assert plan.batches[0].count == 2
    assert plan.batches[1].expert_id == 2
    assert plan.batches[1].start == 2
    assert plan.batches[1].end == 6
    assert plan.batches[1].count == 4


def test_empty_dispatch_plan_is_valid() -> None:
    dispatch_plan = make_dispatch_plan(torch.tensor([0, 0, 0], dtype=torch.int64))
    scheduler = RoundRobinScheduler(SchedulerConfig())

    plan = scheduler(dispatch_plan)

    assert plan.expert_queue.numel() == 0
    assert plan.execution_order.numel() == 0
    assert plan.statistics.num_active_experts == 0
    assert plan.statistics.num_empty_experts == 3
    assert plan.statistics.max_tokens_per_expert == 0
    assert plan.statistics.min_tokens_per_active_expert == 0


def test_workspace_buffers_are_reused() -> None:
    dispatch_plan = make_dispatch_plan(torch.tensor([1, 2, 0, 3], dtype=torch.int64))
    scheduler = RoundRobinScheduler(SchedulerConfig(enable_workspace=True))
    workspace = SchedulerWorkspace()

    first_plan = scheduler(dispatch_plan, workspace=workspace)
    first_queue_ptr = first_plan.expert_queue.data_ptr()
    first_stream_ptr = first_plan.stream_assignments.data_ptr()

    second_plan = scheduler(dispatch_plan, workspace=workspace)

    assert second_plan.expert_queue.data_ptr() == first_queue_ptr
    assert second_plan.stream_assignments.data_ptr() == first_stream_ptr
    assert workspace.estimated_bytes() > 0


def test_workspace_can_be_disabled() -> None:
    dispatch_plan = make_dispatch_plan(torch.tensor([1, 2, 0, 3], dtype=torch.int64))
    scheduler = RoundRobinScheduler(SchedulerConfig(enable_workspace=False))
    workspace = SchedulerWorkspace()

    scheduler(dispatch_plan, workspace=workspace)

    assert workspace.estimated_bytes() == 0


def test_minimal_metadata_skips_python_batches() -> None:
    dispatch_plan = make_dispatch_plan(torch.tensor([1, 0, 2], dtype=torch.int64))
    scheduler = RoundRobinScheduler(
        SchedulerConfig(metadata_level=SchedulerMetadataLevel.MINIMAL)
    )

    plan = scheduler(dispatch_plan)

    assert plan.batches == ()
    assert torch.equal(plan.expert_queue, torch.tensor([0, 2]))


def test_scheduler_statistics_are_correct() -> None:
    dispatch_plan = make_dispatch_plan(torch.tensor([5, 1, 0, 7], dtype=torch.int64))
    scheduler = RoundRobinScheduler(SchedulerConfig())

    plan = scheduler(dispatch_plan)

    assert plan.statistics.num_experts == 4
    assert plan.statistics.num_active_experts == 3
    assert plan.statistics.num_empty_experts == 1
    assert plan.statistics.num_execution_batches == 3
    assert plan.statistics.num_assignments == 13
    assert plan.statistics.max_tokens_per_expert == 7
    assert plan.statistics.min_tokens_per_active_expert == 1


def test_registry_builds_round_robin_scheduler() -> None:
    scheduler = build_scheduler(SchedulerConfig())

    assert isinstance(scheduler, RoundRobinScheduler)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(scheduling_policy="")
    with pytest.raises(ValueError):
        SchedulerConfig(stream_count=0)
    with pytest.raises(ValueError):
        SchedulerConfig(max_execution_batch_size=0)


def test_invalid_dispatch_metadata_rejected() -> None:
    dispatch_plan = make_dispatch_plan(torch.tensor([1, 2], dtype=torch.int64))
    dispatch_plan.metadata.expert_offsets = torch.tensor([0, 1], dtype=torch.int64)
    scheduler = RoundRobinScheduler(SchedulerConfig())

    with pytest.raises(ValueError):
        scheduler(dispatch_plan)
