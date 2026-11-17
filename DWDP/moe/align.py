"""Token sorting and block alignment for MoE execution.

Groups tokens assigned to each expert and pads expert assignments to multiples
of BLOCK_M so that Triton GEMM kernels operate on uniform tiles with zero branching
and optimal cache reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True, slots=True)
class AlignedMoEData:
    """Structure holding token alignments and block metadata."""

    sorted_token_ids: torch.Tensor       # [num_tokens_post_padded] int32
    sorted_weights: torch.Tensor         # [num_tokens_post_padded] float
    sorted_slot_ids: torch.Tensor        # [num_tokens_post_padded] int32 (token_idx * top_k + k)
    expert_ids_per_block: torch.Tensor   # [num_blocks] int32
    valid_mask: torch.Tensor             # [num_tokens_post_padded] bool
    num_tokens_post_padded: int
    num_blocks: int
    block_size: int
    num_tokens: int
    top_k: int


def align_tokens_by_expert(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> AlignedMoEData:
    """Align and sort token assignments by expert with BLOCK_M padding.

    Written using straightforward PyTorch operations (correctness-first).

    Args:
        topk_indices: Tensor of shape [num_tokens, top_k], expert assignments.
        topk_weights: Tensor of shape [num_tokens, top_k], routing weights.
        num_experts: Total number of experts in the model.
        block_size: Tile size along token dimension (BLOCK_M).

    Returns:
        AlignedMoEData containing sorted indices, weights, slots, and block mapping.
    """
    device = topk_indices.device
    num_tokens, top_k = topk_indices.shape

    if num_tokens == 0:
        empty_int = torch.empty(0, dtype=torch.int32, device=device)
        empty_weight = torch.empty(0, dtype=topk_weights.dtype, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        return AlignedMoEData(
            sorted_token_ids=empty_int,
            sorted_weights=empty_weight,
            sorted_slot_ids=empty_int,
            expert_ids_per_block=empty_int,
            valid_mask=empty_bool,
            num_tokens_post_padded=0,
            num_blocks=0,
            block_size=block_size,
            num_tokens=0,
            top_k=top_k,
        )

    # Flatten assignments: each pair is (token_idx, k_idx)
    token_indices = torch.arange(num_tokens, device=device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    k_indices = torch.arange(top_k, device=device).unsqueeze(0).expand(num_tokens, top_k).reshape(-1)
    slot_ids = (token_indices * top_k + k_indices).to(torch.int32)

    flat_expert_ids = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    # Count tokens per expert
    counts = torch.bincount(flat_expert_ids, minlength=num_experts)

    # Calculate blocks needed per expert
    blocks_per_expert = (counts + block_size - 1) // block_size
    total_blocks = int(blocks_per_expert.sum().item())
    total_padded_tokens = total_blocks * block_size

    if total_blocks == 0:
        empty_int = torch.empty(0, dtype=torch.int32, device=device)
        empty_weight = torch.empty(0, dtype=topk_weights.dtype, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        return AlignedMoEData(
            sorted_token_ids=empty_int,
            sorted_weights=empty_weight,
            sorted_slot_ids=empty_int,
            expert_ids_per_block=empty_int,
            valid_mask=empty_bool,
            num_tokens_post_padded=0,
            num_blocks=0,
            block_size=block_size,
            num_tokens=num_tokens,
            top_k=top_k,
        )

    # Preallocate output buffers
    sorted_token_ids = torch.zeros(total_padded_tokens, dtype=torch.int32, device=device)
    sorted_weights = torch.zeros(total_padded_tokens, dtype=topk_weights.dtype, device=device)
    sorted_slot_ids = torch.zeros(total_padded_tokens, dtype=torch.int32, device=device)
    valid_mask = torch.zeros(total_padded_tokens, dtype=torch.bool, device=device)
    expert_ids_per_block = torch.zeros(total_blocks, dtype=torch.int32, device=device)

    # Sort assignments into expert blocks
    current_token_offset = 0
    current_block_offset = 0

    for expert_id in range(num_experts):
        cnt = int(counts[expert_id].item())
        if cnt == 0:
            continue

        n_blocks = int(blocks_per_expert[expert_id].item())
        padded_cnt = n_blocks * block_size

        # Find matching items for this expert
        mask = (flat_expert_ids == expert_id)
        selected_tokens = token_indices[mask].to(torch.int32)
        selected_weights = flat_weights[mask]
        selected_slots = slot_ids[mask]

        # Populate valid tokens
        valid_slice = slice(current_token_offset, current_token_offset + cnt)
        sorted_token_ids[valid_slice] = selected_tokens
        sorted_weights[valid_slice] = selected_weights
        sorted_slot_ids[valid_slice] = selected_slots
        valid_mask[valid_slice] = True

        # Padding slots remain 0 with valid_mask=False

        # Record expert ID for all blocks of this expert
        block_slice = slice(current_block_offset, current_block_offset + n_blocks)
        expert_ids_per_block[block_slice] = expert_id

        current_token_offset += padded_cnt
        current_block_offset += n_blocks

    return AlignedMoEData(
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_slot_ids=sorted_slot_ids,
        expert_ids_per_block=expert_ids_per_block,
        valid_mask=valid_mask,
        num_tokens_post_padded=total_padded_tokens,
        num_blocks=total_blocks,
        block_size=block_size,
        num_tokens=num_tokens,
        top_k=top_k,
    )
