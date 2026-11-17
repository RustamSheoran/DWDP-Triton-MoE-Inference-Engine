"""Token sorting and block alignment for MoE execution.

Vectorized zero-synchronization token alignment and expert grouping.
Computes token-to-block layout using pure GPU vector primitives:
- Zero device-to-host synchronization calls (0 .item(), 0 .tolist()).
- Zero Python loops over experts.
- Sized using static analytical upper bounds to avoid dynamic reallocation.
- Padding to multiples of BLOCK_M ensuring branchless Triton GEMM execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True, slots=True)
class AlignedMoEData:
    """Structure holding token alignments and block metadata."""

    sorted_token_ids: torch.Tensor       # [max_padded_tokens] int32
    sorted_weights: torch.Tensor         # [max_padded_tokens] float
    sorted_slot_ids: torch.Tensor        # [max_padded_tokens] int32 (token_idx * top_k + k)
    expert_ids_per_block: torch.Tensor   # [num_blocks_ub] int32
    valid_mask: torch.Tensor             # [max_padded_tokens] bool
    num_tokens_post_padded: torch.Tensor # [1] int64 device tensor (exact padded token count)
    num_blocks_ub: int                   # static upper bound on number of blocks
    max_padded_tokens: int               # num_blocks_ub * block_size
    block_size: int
    num_tokens: int
    top_k: int

    @property
    def num_blocks(self) -> int:
        """Alias for num_blocks_ub for backward compatibility."""
        return self.num_blocks_ub


def align_tokens_by_expert(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    block_size: int,
) -> AlignedMoEData:
    """Align and sort token assignments by expert with BLOCK_M padding (zero-sync).

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

    # Static upper bounds on blocks and padded tokens (known without GPU sync)
    if num_tokens == 0:
        empty_int = torch.empty(0, dtype=torch.int32, device=device)
        empty_weight = torch.empty(0, dtype=topk_weights.dtype, device=device)
        empty_bool = torch.empty(0, dtype=torch.bool, device=device)
        zero_tensor = torch.zeros(1, dtype=torch.int64, device=device)
        return AlignedMoEData(
            sorted_token_ids=empty_int,
            sorted_weights=empty_weight,
            sorted_slot_ids=empty_int,
            expert_ids_per_block=empty_int,
            valid_mask=empty_bool,
            num_tokens_post_padded=zero_tensor,
            num_blocks_ub=0,
            max_padded_tokens=0,
            block_size=block_size,
            num_tokens=0,
            top_k=top_k,
        )

    total_assignments = num_tokens * top_k
    num_blocks_ub = (total_assignments + num_experts * (block_size - 1) + block_size - 1) // block_size
    max_padded = num_blocks_ub * block_size

    # 1. Flatten expert assignments and weights
    flat_expert_ids = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    # Build token indices and slot IDs
    token_indices = torch.arange(num_tokens, device=device).unsqueeze(1).expand(num_tokens, top_k).reshape(-1)
    k_indices = torch.arange(top_k, device=device).unsqueeze(0).expand(num_tokens, top_k).reshape(-1)
    slot_ids = (token_indices * top_k + k_indices).to(torch.int32)

    # 2. Stable sort by expert id
    sorted_expert_ids, sort_perm = torch.sort(flat_expert_ids, stable=True)

    # 3. Expert token counts via scatter_add_ (zero sync; unlike bincount which calls .max().item())
    counts = torch.zeros(num_experts, dtype=torch.int64, device=device).scatter_add_(
        0, flat_expert_ids, torch.ones_like(flat_expert_ids)
    )
    padded_counts = ((counts + block_size - 1) // block_size) * block_size

    # 4. Expert start and end offsets in padded and unpadded space
    padded_end = torch.cumsum(padded_counts, dim=0)
    padded_start = padded_end - padded_counts
    unpadded_start = torch.cumsum(counts, dim=0) - counts

    # 5. Compute scatter destinations for each sorted token
    intra_expert_idx = torch.arange(total_assignments, device=device) - unpadded_start[sorted_expert_ids]
    dest_idx = padded_start[sorted_expert_ids] + intra_expert_idx

    # 6. Allocate static upper-bound output buffers (zero host sync)
    sorted_token_ids = torch.zeros(max_padded, dtype=torch.int32, device=device)
    sorted_weights = torch.zeros(max_padded, dtype=topk_weights.dtype, device=device)
    sorted_slot_ids = torch.zeros(max_padded, dtype=torch.int32, device=device)
    valid_mask = torch.zeros(max_padded, dtype=torch.bool, device=device)

    # 7. Scatter sorted assignments into destination positions
    sorted_token_ids[dest_idx] = token_indices[sort_perm].to(torch.int32)
    sorted_weights[dest_idx] = flat_weights[sort_perm]
    sorted_slot_ids[dest_idx] = slot_ids[sort_perm]
    valid_mask[dest_idx] = True

    # 8. Map each block to its corresponding expert using searchsorted
    block_start_positions = torch.arange(num_blocks_ub, device=device) * block_size
    expert_ids_per_block = torch.searchsorted(padded_end, block_start_positions, right=True).clamp(max=num_experts - 1).to(torch.int32)

    # Exact post-padded token count preserved as 1-element device tensor
    num_tokens_post_padded = padded_end[-1:].to(torch.int64)

    return AlignedMoEData(
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_slot_ids=sorted_slot_ids,
        expert_ids_per_block=expert_ids_per_block,
        valid_mask=valid_mask,
        num_tokens_post_padded=num_tokens_post_padded,
        num_blocks_ub=num_blocks_ub,
        max_padded_tokens=max_padded,
        block_size=block_size,
        num_tokens=num_tokens,
        top_k=top_k,
    )
