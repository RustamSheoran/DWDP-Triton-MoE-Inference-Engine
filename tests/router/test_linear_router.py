from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.router import (  # noqa: E402
    LinearTopKRouter,
    MetadataLevel,
    RouterConfig,
    build_router,
)


def test_linear_router_output_shapes() -> None:
    config = RouterConfig(
        hidden_size=8,
        num_experts=6,
        top_k=2,
        metadata_level=MetadataLevel.FULL,
    )
    router = LinearTopKRouter(config)
    hidden_states = torch.randn(2, 4, 8)

    output = router(hidden_states)

    assert output.router_logits.shape == (2, 4, 6)
    assert output.routing_probabilities.shape == (2, 4, 6)
    assert output.topk_indices.shape == (2, 4, 2)
    assert output.topk_weights.shape == (2, 4, 2)
    assert output.metadata is not None
    assert output.metadata.flattened_token_indices is not None


def test_topk_weights_are_normalized() -> None:
    config = RouterConfig(hidden_size=16, num_experts=8, top_k=3)
    router = LinearTopKRouter(config)
    hidden_states = torch.randn(3, 5, 16)

    output = router(hidden_states)

    sums = output.topk_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6, rtol=0.0)


def test_topk1_weights_are_ones() -> None:
    config = RouterConfig(hidden_size=16, num_experts=8, top_k=1)
    router = LinearTopKRouter(config)
    hidden_states = torch.randn(2, 7, 16)

    output = router(hidden_states)

    assert torch.equal(output.topk_weights, torch.ones_like(output.topk_weights))


def test_metadata_counts_match_total_assignments() -> None:
    config = RouterConfig(
        hidden_size=12,
        num_experts=5,
        top_k=2,
        metadata_level=MetadataLevel.FULL,
    )
    router = LinearTopKRouter(config)
    hidden_states = torch.randn(4, 3, 12)

    output = router(hidden_states)
    metadata = output.metadata

    assert metadata is not None
    assert metadata.tokens_per_expert is not None
    assert metadata.expert_offsets is not None
    assert int(metadata.tokens_per_expert.sum().item()) == 4 * 3 * 2
    assert int(metadata.expert_offsets[-1].item()) == 4 * 3 * 2
    assert metadata.flattened_expert_indices is not None
    assert metadata.flattened_weights is not None
    assert metadata.flattened_token_indices is not None
    assert metadata.flattened_expert_indices.numel() == 4 * 3 * 2


def test_no_metadata_path() -> None:
    config = RouterConfig(
        hidden_size=10,
        num_experts=7,
        top_k=2,
        metadata_level=MetadataLevel.NONE,
    )
    router = LinearTopKRouter(config)
    hidden_states = torch.randn(2, 2, 10)

    output = router(hidden_states)

    assert output.metadata is None


def test_router_logits_match_manual_linear() -> None:
    config = RouterConfig(hidden_size=6, num_experts=4, top_k=2, bias=True)
    router = LinearTopKRouter(config)
    hidden_states = torch.randn(2, 3, 6)

    output = router(hidden_states)
    manual = torch.nn.functional.linear(hidden_states.reshape(-1, 6), router.weight, router.bias)
    manual = manual.reshape(2, 3, 4)

    assert torch.allclose(output.router_logits, manual)


def test_registry_builds_router() -> None:
    config = RouterConfig(hidden_size=8, num_experts=4, top_k=2)
    router = build_router(config)

    assert isinstance(router, LinearTopKRouter)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        RouterConfig(hidden_size=8, num_experts=4, top_k=5)
