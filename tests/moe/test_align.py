"""Unit tests for token sorting and block alignment.

Tests equivalence of zero-sync vectorized alignment against golden reference,
and asserts zero device-to-host synchronization on GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from DWDP.moe.align import align_tokens_by_expert

# Ensure tests/moe directory is on sys.path for local helper imports
_moe_dir = str(Path(__file__).parent.resolve())
if _moe_dir not in sys.path:
    sys.path.insert(0, _moe_dir)

from align_reference import align_reference


def _check_alignment_equivalence(aligned, ref):
    """Helper to verify that vectorized aligned data matches reference exactly."""
    ref_padded_tokens = ref.num_tokens_post_padded
    ref_blocks = ref.num_blocks

    # Exact token count must match
    assert int(aligned.num_tokens_post_padded.item()) == ref_padded_tokens

    if ref_padded_tokens == 0:
        assert aligned.valid_mask.numel() == 0 or not aligned.valid_mask.any()
        return

    # Slices up to exact padded tokens must be identical
    assert torch.equal(aligned.sorted_token_ids[:ref_padded_tokens], ref.sorted_token_ids)
    assert torch.equal(aligned.sorted_weights[:ref_padded_tokens], ref.sorted_weights)
    assert torch.equal(aligned.sorted_slot_ids[:ref_padded_tokens], ref.sorted_slot_ids)
    assert torch.equal(aligned.valid_mask[:ref_padded_tokens], ref.valid_mask)
    assert torch.equal(aligned.expert_ids_per_block[:ref_blocks], ref.expert_ids_per_block)

    # All padded tokens beyond ref_padded_tokens must be marked invalid
    if aligned.max_padded_tokens > ref_padded_tokens:
        assert not aligned.valid_mask[ref_padded_tokens:].any()


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 64])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_align_equivalence_various_tokens(num_tokens, top_k):
    """Test equivalence against golden align_reference across token counts and top_k."""
    num_experts = 8
    block_size = 16

    if num_tokens == 0:
        topk_indices = torch.empty((0, top_k), dtype=torch.int64)
        topk_weights = torch.empty((0, top_k), dtype=torch.float32)
    else:
        torch.manual_seed(42 + num_tokens * 10 + top_k)
        topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
        topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1)

    aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size)
    ref = align_reference(topk_indices, topk_weights, num_experts, block_size)

    _check_alignment_equivalence(aligned, ref)


def test_align_equivalence_skewed_all_one_expert():
    """Test extreme load imbalance: all tokens routed to Expert 0."""
    num_tokens = 33
    top_k = 2
    num_experts = 8
    block_size = 16

    topk_indices = torch.zeros((num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32).repeat(num_tokens, 1)

    aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size)
    ref = align_reference(topk_indices, topk_weights, num_experts, block_size)

    _check_alignment_equivalence(aligned, ref)


def test_align_equivalence_empty_experts():
    """Test sparse routing where several experts receive 0 tokens."""
    num_tokens = 10
    top_k = 1
    num_experts = 16
    block_size = 8

    # Only route to experts 2, 5, 14
    target_experts = torch.tensor([2, 5, 14], dtype=torch.int64)
    torch.manual_seed(99)
    choices = torch.randint(0, len(target_experts), (num_tokens, 1))
    topk_indices = target_experts[choices]
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32)

    aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size)
    ref = align_reference(topk_indices, topk_weights, num_experts, block_size)

    _check_alignment_equivalence(aligned, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_align_zero_host_sync_cuda():
    """Verify align_tokens_by_expert performs ZERO device-to-host synchronizations on GPU.

    Uses torch.cuda.set_sync_debug_mode('error') which immediately raises an exception
    if any CUDA synchronization (e.g. .item(), .tolist(), or implicit sync) occurs.
    """
    device = torch.device("cuda")
    num_tokens = 512
    top_k = 4
    num_experts = 64
    block_size = 32

    topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64, device=device)
    topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float16, device=device)

    # Warmup
    _ = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size)
    torch.cuda.synchronize()

    # Enable strict sync debug mode: error on ANY device-to-host sync
    prev_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size)
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)

    assert aligned.num_blocks_ub > 0
    assert aligned.valid_mask.shape[0] == aligned.max_padded_tokens
    assert aligned.valid_mask.sum().item() == num_tokens * top_k
