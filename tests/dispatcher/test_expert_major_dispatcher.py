from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.dispatcher import (  # noqa: E402
    DispatchWorkspace,
    DispatcherConfig,
    ExpertMajorDispatcher,
    build_dispatcher,
)
from DWDP.router import RouterOutput, RoutingMetadata  # noqa: E402


def make_router_output(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    with_metadata: bool = False,
) -> RouterOutput:
    metadata = None
    if with_metadata:
        flat_experts = topk_indices.reshape(-1)
        counts = torch.bincount(flat_experts, minlength=num_experts)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)), dim=0)
        metadata = RoutingMetadata(
            num_tokens=topk_indices.numel() // topk_indices.shape[-1],
            num_experts=num_experts,
            top_k=topk_indices.shape[-1],
            tokens_per_expert=counts,
            expert_offsets=offsets,
        )

    return RouterOutput(
        router_logits=torch.empty(0),
        routing_probabilities=torch.empty(0),
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        metadata=metadata,
    )


def test_dispatch_groups_tokens_by_expert_in_stable_order() -> None:
    topk_indices = torch.tensor([[4], [1], [4], [2]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.8], [0.6], [0.7], [0.9]], dtype=torch.float32)
    router_output = make_router_output(topk_indices, topk_weights, num_experts=5)
    dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=5))

    plan = dispatcher(router_output)

    assert torch.equal(plan.assignments.expert_ids, torch.tensor([1, 2, 4, 4]))
    assert torch.equal(plan.assignments.packed_token_indices, torch.tensor([1, 3, 0, 2]))
    assert torch.allclose(
        plan.assignments.packed_routing_weights,
        torch.tensor([0.6, 0.9, 0.8, 0.7]),
    )
    assert torch.equal(plan.metadata.expert_counts, torch.tensor([0, 1, 1, 0, 2]))
    assert torch.equal(plan.metadata.expert_offsets, torch.tensor([0, 0, 1, 2, 2, 4]))
    assert plan.metadata.algorithm == "counting_scatter"


def test_dispatch_handles_topk_greater_than_one() -> None:
    topk_indices = torch.tensor(
        [
            [[2, 0], [1, 2]],
            [[0, 1], [2, 1]],
        ],
        dtype=torch.int64,
    )
    topk_weights = torch.tensor(
        [
            [[0.7, 0.3], [0.4, 0.6]],
            [[0.8, 0.2], [0.9, 0.1]],
        ],
        dtype=torch.float32,
    )
    router_output = make_router_output(topk_indices, topk_weights, num_experts=3)
    dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=3))

    plan = dispatcher(router_output)

    assert plan.metadata.num_tokens == 4
    assert plan.metadata.num_assignments == 8
    assert plan.metadata.top_k == 2
    assert plan.metadata.token_shape == (2, 2)
    assert torch.equal(plan.metadata.expert_counts, torch.tensor([2, 3, 3]))
    assert torch.equal(plan.metadata.expert_offsets, torch.tensor([0, 2, 5, 8]))
    assert torch.equal(
        plan.assignments.expert_ids,
        torch.tensor([0, 0, 1, 1, 1, 2, 2, 2]),
    )


def test_counting_scatter_matches_stable_sort_reference() -> None:
    topk_indices = torch.tensor(
        [
            [4, 1],
            [1, 4],
            [2, 1],
            [4, 2],
        ],
        dtype=torch.int64,
    )
    topk_weights = torch.tensor(
        [
            [0.6, 0.4],
            [0.7, 0.3],
            [0.2, 0.8],
            [0.9, 0.1],
        ],
        dtype=torch.float32,
    )
    router_output = make_router_output(topk_indices, topk_weights, num_experts=5)
    counting_dispatcher = ExpertMajorDispatcher(
        DispatcherConfig(num_experts=5, algorithm="counting_scatter")
    )
    sort_dispatcher = ExpertMajorDispatcher(
        DispatcherConfig(num_experts=5, algorithm="stable_sort")
    )

    counting_plan = counting_dispatcher(router_output)
    sort_plan = sort_dispatcher(router_output)

    assert torch.equal(
        counting_plan.assignments.expert_ids,
        sort_plan.assignments.expert_ids,
    )
    assert torch.equal(
        counting_plan.assignments.packed_token_indices,
        sort_plan.assignments.packed_token_indices,
    )
    assert torch.allclose(
        counting_plan.assignments.packed_routing_weights,
        sort_plan.assignments.packed_routing_weights,
    )
    assert torch.equal(
        counting_plan.metadata.token_permutation,
        sort_plan.metadata.token_permutation,
    )
    assert torch.equal(
        counting_plan.metadata.inverse_permutation,
        sort_plan.metadata.inverse_permutation,
    )


def test_inverse_permutation_round_trips() -> None:
    topk_indices = torch.tensor([[2, 1], [0, 2], [1, 0]], dtype=torch.int64)
    topk_weights = torch.ones_like(topk_indices, dtype=torch.float32)
    router_output = make_router_output(topk_indices, topk_weights, num_experts=3)
    dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=3))

    plan = dispatcher(router_output)
    num_assignments = plan.metadata.num_assignments
    positions = torch.arange(num_assignments, dtype=torch.int64)

    assert torch.equal(
        plan.metadata.inverse_permutation.index_select(0, plan.metadata.token_permutation),
        positions,
    )
    assert torch.equal(plan.metadata.destination_positions, plan.metadata.inverse_permutation)


def test_workspace_buffers_are_reused() -> None:
    topk_indices = torch.tensor([[2, 1], [0, 2], [1, 0]], dtype=torch.int64)
    topk_weights = torch.ones_like(topk_indices, dtype=torch.float32)
    router_output = make_router_output(topk_indices, topk_weights, num_experts=3)
    dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=3))
    workspace = DispatchWorkspace()

    first_plan = dispatcher(router_output, workspace=workspace)
    first_perm_ptr = first_plan.metadata.token_permutation.data_ptr()
    first_weight_ptr = first_plan.assignments.packed_routing_weights.data_ptr()

    second_plan = dispatcher(router_output, workspace=workspace)

    assert second_plan.metadata.token_permutation.data_ptr() == first_perm_ptr
    assert second_plan.assignments.packed_routing_weights.data_ptr() == first_weight_ptr
    assert workspace.estimated_bytes() > 0


def test_dispatcher_can_reuse_router_counts_and_offsets() -> None:
    topk_indices = torch.tensor([[1], [3], [1], [0]], dtype=torch.int64)
    topk_weights = torch.ones_like(topk_indices, dtype=torch.float32)
    router_output = make_router_output(
        topk_indices,
        topk_weights,
        num_experts=4,
        with_metadata=True,
    )
    dispatcher = ExpertMajorDispatcher(
        DispatcherConfig(num_experts=4, reuse_router_metadata=True)
    )

    plan = dispatcher(router_output)

    assert plan.metadata.expert_counts is router_output.metadata.tokens_per_expert
    assert plan.metadata.expert_offsets is router_output.metadata.expert_offsets


def test_registry_builds_dispatcher() -> None:
    dispatcher = build_dispatcher(DispatcherConfig(num_experts=8))

    assert isinstance(dispatcher, ExpertMajorDispatcher)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        DispatcherConfig(num_experts=0)
    with pytest.raises(ValueError):
        DispatcherConfig(num_experts=4, algorithm="invalid")
    with pytest.raises(ValueError):
        DispatcherConfig(num_experts=4, algorithm="counting_scatter", stable_order=False)


def test_out_of_range_expert_indices_rejected() -> None:
    topk_indices = torch.tensor([[0], [4]], dtype=torch.int64)
    topk_weights = torch.ones_like(topk_indices, dtype=torch.float32)
    dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=4))

    with pytest.raises(ValueError):
        dispatcher(make_router_output(topk_indices, topk_weights, num_experts=4))
