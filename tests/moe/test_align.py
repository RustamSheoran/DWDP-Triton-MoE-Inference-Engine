"""Unit tests for token sorting and block alignment."""

from __future__ import annotations

import pytest
import torch

from DWDP.moe.align import align_tokens_by_expert


def test_align_basic_properties():
    num_tokens = 5
    top_k = 2
    num_experts = 4
    block_size = 4

    topk_indices = torch.tensor([
        [0, 1],
        [1, 2],
        [0, 3],
        [2, 3],
        [0, 2],
    ], dtype=torch.int64)

    topk_weights = torch.tensor([
        [0.6, 0.4],
        [0.7, 0.3],
        [0.5, 0.5],
        [0.8, 0.2],
        [0.9, 0.1],
    ], dtype=torch.float32)

    aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size)

    # Expert counts:
    # 0: [t=0, t=2, t=4] -> 3 tokens -> 1 block (4 tokens padded)
    # 1: [t=0, t=1] -> 2 tokens -> 1 block (4 tokens padded)
    # 2: [t=1, t=3, t=4] -> 3 tokens -> 1 block (4 tokens padded)
    # 3: [t=2, t=3] -> 2 tokens -> 1 block (4 tokens padded)
    # Total blocks = 4, Total padded tokens = 16

    assert aligned.num_blocks == 4
    assert aligned.num_tokens_post_padded == 16
    assert aligned.valid_mask.sum().item() == 10  # 5 tokens * top_k 2 = 10 valid assignments

    # Verify expert IDs per block
    assert aligned.expert_ids_per_block.tolist() == [0, 1, 2, 3]

    # Verify that valid tokens retrieve correct weights and slots
    valid_indices = torch.where(aligned.valid_mask)[0]
    for idx in valid_indices:
        token_id = aligned.sorted_token_ids[idx].item()
        slot_id = aligned.sorted_slot_ids[idx].item()
        weight = aligned.sorted_weights[idx].item()

        t = slot_id // top_k
        k = slot_id % top_k

        assert t == token_id
        assert abs(weight - topk_weights[t, k].item()) < 1e-6


def test_align_empty_tokens():
    topk_indices = torch.empty((0, 2), dtype=torch.int64)
    topk_weights = torch.empty((0, 2), dtype=torch.float32)

    aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts=8, block_size=16)

    assert aligned.num_blocks == 0
    assert aligned.num_tokens_post_padded == 0
    assert aligned.sorted_token_ids.numel() == 0


def test_align_skewed_all_one_expert():
    num_tokens = 9
    top_k = 1
    num_experts = 4
    block_size = 4

    topk_indices = torch.zeros((num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.ones((num_tokens, top_k), dtype=torch.float32)

    aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size)

    # 9 tokens with block_size 4 requires ceil(9/4) = 3 blocks = 12 padded tokens
    assert aligned.num_blocks == 3
    assert aligned.num_tokens_post_padded == 12
    assert aligned.valid_mask.sum().item() == 9
    assert aligned.expert_ids_per_block.tolist() == [0, 0, 0]


@pytest.mark.parametrize("top_k", [1, 2, 4])
@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_align_various_configurations(top_k, block_size):
    num_tokens = 35
    num_experts = 8

    torch.manual_seed(123)
    topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32)

    aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size)

    assert aligned.valid_mask.sum().item() == num_tokens * top_k
    assert aligned.num_tokens_post_padded == aligned.num_blocks * block_size
