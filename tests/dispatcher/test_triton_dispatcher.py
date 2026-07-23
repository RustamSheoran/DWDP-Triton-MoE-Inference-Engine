from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")

from DWDP.dispatcher import DispatchWorkspace, DispatcherConfig, ExpertMajorDispatcher  # noqa: E402
from DWDP.router import RouterOutput, RoutingMetadata  # noqa: E402


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton dispatcher requires CUDA")


def make_router_output(topk_indices: torch.Tensor, topk_weights: torch.Tensor) -> RouterOutput:
    return RouterOutput(
        router_logits=torch.empty(0, device=topk_indices.device),
        routing_probabilities=torch.empty(0, device=topk_indices.device),
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        metadata=None,
    )


def assert_dispatch_plans_equal(reference_plan, triton_plan) -> None:
    assert torch.equal(triton_plan.assignments.expert_ids, reference_plan.assignments.expert_ids)
    assert torch.equal(triton_plan.assignments.packed_token_indices, reference_plan.assignments.packed_token_indices)
    assert torch.allclose(
        triton_plan.assignments.packed_routing_weights,
        reference_plan.assignments.packed_routing_weights,
    )
    assert torch.equal(triton_plan.metadata.expert_counts, reference_plan.metadata.expert_counts)
    assert torch.equal(triton_plan.metadata.expert_offsets, reference_plan.metadata.expert_offsets)
    assert torch.equal(triton_plan.metadata.token_permutation, reference_plan.metadata.token_permutation)
    assert torch.equal(triton_plan.metadata.inverse_permutation, reference_plan.metadata.inverse_permutation)
    assert torch.equal(triton_plan.metadata.destination_positions, reference_plan.metadata.destination_positions)
    assert triton_plan.metadata.num_tokens == reference_plan.metadata.num_tokens
    assert triton_plan.metadata.num_assignments == reference_plan.metadata.num_assignments
    assert triton_plan.metadata.num_experts == reference_plan.metadata.num_experts
    assert triton_plan.metadata.top_k == reference_plan.metadata.top_k
    assert triton_plan.metadata.token_shape == reference_plan.metadata.token_shape


@pytest.mark.parametrize(
    "num_tokens,num_experts,top_k",
    [
        (1, 1, 1),
        (8, 4, 1),
        (33, 8, 2),
        (127, 16, 4),
        (513, 8, 2),
        (513, 16, 4),
        (1025, 64, 8),
        (2049, 128, 2),
    ],
)
def test_triton_dispatcher_matches_reference_random(num_tokens: int, num_experts: int, top_k: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234 + num_tokens + num_experts + top_k)
    topk_indices = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    topk_weights = torch.rand((num_tokens, top_k), dtype=torch.float32, device="cuda", generator=generator)
    router_output = make_router_output(topk_indices, topk_weights)
    reference_dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=num_experts, algorithm="counting_scatter"))
    triton_dispatcher = ExpertMajorDispatcher(
        DispatcherConfig(num_experts=num_experts, algorithm="triton_counting_scatter")
    )

    reference_plan = reference_dispatcher(router_output)
    triton_plan = triton_dispatcher(router_output)

    assert_dispatch_plans_equal(reference_plan, triton_plan)


def test_triton_dispatcher_preserves_stable_order_edge_case() -> None:
    topk_indices = torch.tensor(
        [[3, 1], [3, 2], [1, 3], [0, 3], [2, 1]],
        dtype=torch.int64,
        device="cuda",
    )
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.5, 0.5], [0.9, 0.1]],
        dtype=torch.float32,
        device="cuda",
    )
    router_output = make_router_output(topk_indices, topk_weights)
    reference_dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=4, algorithm="counting_scatter"))
    triton_dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=4, algorithm="triton_counting_scatter"))

    reference_plan = reference_dispatcher(router_output)
    triton_plan = triton_dispatcher(router_output)

    assert_dispatch_plans_equal(reference_plan, triton_plan)


def test_triton_dispatcher_reuses_workspace() -> None:
    topk_indices = torch.tensor([[2, 1], [0, 2], [1, 0]], dtype=torch.int64, device="cuda")
    topk_weights = torch.ones_like(topk_indices, dtype=torch.float32)
    router_output = make_router_output(topk_indices, topk_weights)
    dispatcher = ExpertMajorDispatcher(DispatcherConfig(num_experts=3, algorithm="triton_counting_scatter"))
    workspace = DispatchWorkspace()

    first = dispatcher(router_output, workspace=workspace)
    second = dispatcher(router_output, workspace=workspace)

    assert second.metadata.token_permutation.data_ptr() == first.metadata.token_permutation.data_ptr()
    assert second.assignments.packed_routing_weights.data_ptr() == first.assignments.packed_routing_weights.data_ptr()
    assert workspace.triton_tile_counts is not None
    assert workspace.triton_tile_offsets is not None
    assert workspace.estimated_bytes() > 0


def test_triton_dispatcher_reuses_router_histogram_metadata() -> None:
    topk_indices = torch.tensor([[3, 1], [3, 2], [1, 0], [0, 3]], dtype=torch.int64, device="cuda")
    topk_weights = torch.rand((4, 2), dtype=torch.float32, device="cuda")
    router_output = make_router_output(topk_indices, topk_weights)
    counts = torch.bincount(topk_indices.reshape(-1), minlength=4)
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)), dim=0)
    router_output.metadata = RoutingMetadata(
        num_tokens=4,
        num_experts=4,
        top_k=2,
        tokens_per_expert=counts,
        expert_offsets=offsets,
    )

    dispatcher = ExpertMajorDispatcher(
        DispatcherConfig(num_experts=4, algorithm="triton_counting_scatter", reuse_router_metadata=True)
    )
    plan = dispatcher(router_output)

    assert plan.metadata.expert_counts is counts
    assert plan.metadata.expert_offsets is offsets


def test_triton_dispatcher_rejects_unstable_order() -> None:
    with pytest.raises(ValueError):
        DispatcherConfig(num_experts=4, algorithm="triton_counting_scatter", stable_order=False)
