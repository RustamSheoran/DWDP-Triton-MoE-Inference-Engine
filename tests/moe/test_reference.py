"""Unit tests for the PyTorch MoE reference implementation."""

from __future__ import annotations

import pytest
import torch

from DWDP.moe.reference import moe_reference, moe_reference_grouped


@pytest.fixture
def moe_params():
    torch.manual_seed(42)
    hidden_dim = 64
    intermediate_dim = 32
    num_experts = 8
    top_k = 2

    w1 = torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.float32) * 0.02
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.float32) * 0.02

    return {
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_experts": num_experts,
        "top_k": top_k,
        "w1": w1,
        "w2": w2,
    }


def test_reference_matches_grouped(moe_params):
    num_tokens = 17
    hidden_dim = moe_params["hidden_dim"]
    num_experts = moe_params["num_experts"]
    top_k = moe_params["top_k"]
    w1 = moe_params["w1"]
    w2 = moe_params["w2"]

    torch.manual_seed(101)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32)
    topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    raw_weights = torch.rand(num_tokens, top_k, dtype=torch.float32)
    topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)

    golden = moe_reference(x, w1, w2, topk_indices, topk_weights)
    grouped = moe_reference_grouped(x, w1, w2, topk_indices, topk_weights)

    assert torch.allclose(golden, grouped, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_tokens", [0, 1, 3, 17, 64])
def test_reference_token_counts(moe_params, num_tokens):
    hidden_dim = moe_params["hidden_dim"]
    num_experts = moe_params["num_experts"]
    top_k = moe_params["top_k"]
    w1 = moe_params["w1"]
    w2 = moe_params["w2"]

    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32)
    topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float32) / top_k

    out = moe_reference(x, w1, w2, topk_indices, topk_weights)
    assert out.shape == (num_tokens, hidden_dim)


def test_reference_skewed_all_tokens_one_expert(moe_params):
    num_tokens = 16
    hidden_dim = moe_params["hidden_dim"]
    w1 = moe_params["w1"]
    w2 = moe_params["w2"]

    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float32)
    topk_indices = torch.zeros((num_tokens, 2), dtype=torch.int64)  # All to expert 0
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32).repeat(num_tokens, 1)

    golden = moe_reference(x, w1, w2, topk_indices, topk_weights)
    grouped = moe_reference_grouped(x, w1, w2, topk_indices, topk_weights)

    assert torch.allclose(golden, grouped, atol=1e-5, rtol=1e-5)


def test_reference_fp16(moe_params):
    num_tokens = 8
    hidden_dim = moe_params["hidden_dim"]
    num_experts = moe_params["num_experts"]
    top_k = moe_params["top_k"]

    w1_fp16 = moe_params["w1"].half()
    w2_fp16 = moe_params["w2"].half()
    x = torch.randn(num_tokens, hidden_dim).half()
    topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.ones(num_tokens, top_k, dtype=torch.float16) / top_k

    out = moe_reference(x, w1_fp16, w2_fp16, topk_indices, topk_weights)
    assert out.dtype == torch.float16
    assert out.shape == (num_tokens, hidden_dim)
