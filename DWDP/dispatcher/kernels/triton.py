"""Deterministic tiled Triton implementation of expert-major dispatch.

The kernel path deliberately avoids a global sort and global histogram
atomics.  Assignments are partitioned into fixed input-order tiles.  A tile
histogram plus exclusive prefixes determines each tile's reservation inside
every expert bucket.  A final fused kernel locally sorts only one tile using a
unique ``expert_id * tile_size + source_lane`` key, which preserves source
order for assignments routed to the same expert.
"""

from __future__ import annotations

import torch

from ..workspace import DispatchWorkspace

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised in environments without Triton.
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None
_DEFAULT_TILE_SIZE = 256


if TRITON_AVAILABLE:

    @triton.jit
    def _tile_histogram_kernel(
        expert_indices,
        tile_counts,
        num_assignments,
        num_experts,
        tile_count_stride,
        tile_size: tl.constexpr,
    ):
        """Accumulate one private histogram row per assignment tile.

        Atomics are restricted to a tile-local row.  Different programs never
        update the same histogram row, avoiding the global hot counters of a
        conventional expert histogram.
        """

        tile_id = tl.program_id(0)
        lanes = tl.arange(0, tile_size)
        source_positions = tile_id * tile_size + lanes
        mask = source_positions < num_assignments
        expert_ids = tl.load(expert_indices + source_positions, mask=mask, other=0)
        row = tile_counts + tile_id * tile_count_stride
        tl.atomic_add(row + expert_ids, 1, mask=mask, sem="relaxed")


    @triton.jit
    def _tile_local_start_kernel(
        tile_counts,
        num_experts: tl.constexpr,
        tile_count_stride: tl.constexpr,
        expert_block: tl.constexpr,
    ):
        """Overwrite each histogram row with expert-local exclusive starts."""

        tile_id = tl.program_id(0)
        experts = tl.arange(0, expert_block)
        mask = experts < num_experts
        row = tile_counts + tile_id * tile_count_stride
        counts = tl.load(row + experts, mask=mask, other=0)
        starts = tl.cumsum(counts, axis=0) - counts
        tl.store(row + experts, starts, mask=mask)


    @triton.jit
    def _stable_tile_pack_kernel(
        expert_indices,
        routing_weights,
        expert_offsets,
        tile_offsets,
        tile_local_starts,
        token_permutation,
        inverse_permutation,
        packed_expert_ids,
        packed_token_indices,
        packed_routing_weights,
        num_assignments,
        num_experts,
        tile_offset_stride,
        tile_local_start_stride,
        top_k,
        tile_size: tl.constexpr,
    ):
        """Locally stable-order and directly pack one assignment tile."""

        tile_id = tl.program_id(0)
        lanes = tl.arange(0, tile_size)
        source_base = tile_id * tile_size
        source_positions = source_base + lanes
        source_mask = source_positions < num_assignments
        expert_ids = tl.load(expert_indices + source_positions, mask=source_mask, other=0)

        # The lane suffix makes all valid keys unique. Sorting keys therefore
        # groups by expert and preserves source order inside each group.
        key_limit = num_experts * tile_size
        keys = tl.where(
            source_mask,
            expert_ids.to(tl.int32) * tile_size + lanes,
            key_limit + lanes,
        )
        sorted_keys = tl.sort(keys)
        valid = sorted_keys < key_limit
        sorted_experts = sorted_keys // tile_size
        source_lanes = sorted_keys - sorted_experts * tile_size
        sorted_positions = source_base + source_lanes

        tile_prefix = tl.load(
            tile_offsets + tile_id * tile_offset_stride + sorted_experts,
            mask=valid,
            other=0,
        )
        local_start = tl.load(
            tile_local_starts + tile_id * tile_local_start_stride + sorted_experts,
            mask=valid,
            other=0,
        )
        expert_start = tl.load(expert_offsets + sorted_experts, mask=valid, other=0)
        destination_positions = expert_start + tile_prefix + (lanes - local_start)

        weights = tl.load(routing_weights + sorted_positions, mask=valid, other=0.0)
        token_ids = sorted_positions // top_k

        tl.store(token_permutation + destination_positions, sorted_positions, mask=valid)
        tl.store(inverse_permutation + sorted_positions, destination_positions, mask=valid)
        tl.store(packed_expert_ids + destination_positions, sorted_experts, mask=valid)
        tl.store(packed_token_indices + destination_positions, token_ids, mask=valid)
        tl.store(packed_routing_weights + destination_positions, weights, mask=valid)


def _require_triton_cuda(flat_expert_indices: torch.Tensor) -> None:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton dispatcher requested but triton is not installed")
    if not flat_expert_indices.is_cuda:
        raise RuntimeError("Triton dispatcher requires CUDA tensors")
    if flat_expert_indices.dtype != torch.int64:
        raise RuntimeError("Triton dispatcher requires int64 expert indices")
    if not flat_expert_indices.is_contiguous():
        raise RuntimeError("Triton dispatcher requires contiguous flattened expert indices")


def _assignment_buffers(
    *,
    num_assignments: int,
    num_experts: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    workspace: DispatchWorkspace | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if workspace is None:
        return (
            torch.empty(num_experts, dtype=torch.int64, device=device),
            torch.empty(num_experts + 1, dtype=torch.int64, device=device),
            torch.empty(num_assignments, dtype=torch.int64, device=device),
            torch.empty(num_assignments, dtype=torch.int64, device=device),
            torch.empty(num_assignments, dtype=torch.int64, device=device),
            torch.empty(num_assignments, dtype=torch.int64, device=device),
            torch.empty(num_assignments, dtype=weight_dtype, device=device),
        )

    token_permutation, inverse_permutation, packed_expert_ids, packed_token_indices, packed_routing_weights = (
        workspace.get_assignment_buffers(
            num_assignments,
            weight_dtype=weight_dtype,
            device=device,
        )
    )
    expert_counts, expert_offsets = workspace.get_expert_buffers(num_experts, device=device)
    return (
        expert_counts,
        expert_offsets,
        token_permutation,
        inverse_permutation,
        packed_expert_ids,
        packed_token_indices,
        packed_routing_weights,
    )


def _tile_buffers(
    *,
    num_tiles: int,
    num_experts: int,
    device: torch.device,
    workspace: DispatchWorkspace | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if workspace is not None:
        return workspace.get_triton_tile_buffers(num_tiles, num_experts, device=device)
    return (
        torch.empty((num_tiles, num_experts), dtype=torch.int64, device=device),
        torch.empty((num_tiles, num_experts), dtype=torch.int64, device=device),
    )


def triton_counting_scatter_expert_major_dispatch(
    flat_expert_indices: torch.Tensor,
    flat_routing_weights: torch.Tensor,
    *,
    num_experts: int,
    top_k: int,
    stable_order: bool = True,
    workspace: DispatchWorkspace | None = None,
    router_counts: torch.Tensor | None = None,
    router_offsets: torch.Tensor | None = None,
    block_size: int = _DEFAULT_TILE_SIZE,
    scan_block_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a stable expert-major layout without a global sort.

    The implementation is O(N) in global memory traffic: each assignment is
    read once by the tile histogram and once by the fused pack.  The only
    intermediate storage is ``num_tiles * num_experts`` int64 metadata.  This
    trades a compact per-tile/per-expert scratch matrix for the absence of an
    assignment-sized rank buffer and global sorting workspace.

    ``scan_block_size`` is retained as a compatibility-only argument for the
    previous Triton implementation and is intentionally unused.
    """

    if not stable_order:
        raise ValueError("triton_counting_scatter requires stable_order=True")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("Triton tile size must be a positive power of two")
    if block_size > 1024:
        raise ValueError("Triton tile size must be <= 1024")
    _require_triton_cuda(flat_expert_indices)
    if not flat_routing_weights.is_contiguous():
        raise RuntimeError("Triton dispatcher requires contiguous flattened routing weights")

    num_assignments = flat_expert_indices.numel()
    device = flat_expert_indices.device
    (
        workspace_counts,
        workspace_offsets,
        token_permutation,
        inverse_permutation,
        packed_expert_ids,
        packed_token_indices,
        packed_routing_weights,
    ) = _assignment_buffers(
        num_assignments=num_assignments,
        num_experts=num_experts,
        device=device,
        weight_dtype=flat_routing_weights.dtype,
        workspace=workspace,
    )

    if num_assignments == 0:
        expert_counts = router_counts if router_counts is not None else workspace_counts.zero_()
        if router_offsets is None:
            workspace_offsets.zero_()
            expert_offsets = workspace_offsets
        else:
            expert_offsets = router_offsets
        return (
            expert_counts,
            expert_offsets,
            token_permutation,
            inverse_permutation,
            packed_expert_ids,
            packed_token_indices,
            packed_routing_weights,
        )

    num_tiles = triton.cdiv(num_assignments, block_size)
    tile_counts, tile_offsets = _tile_buffers(
        num_tiles=num_tiles,
        num_experts=num_experts,
        device=device,
        workspace=workspace,
    )
    tile_counts.zero_()
    _tile_histogram_kernel[(num_tiles,)](
        flat_expert_indices,
        tile_counts,
        num_assignments=num_assignments,
        num_experts=num_experts,
        tile_count_stride=tile_counts.stride(0),
        tile_size=block_size,
        num_warps=4,
    )

    if router_counts is None:
        torch.sum(tile_counts, dim=0, dtype=torch.int64, out=workspace_counts)
        expert_counts = workspace_counts
    else:
        expert_counts = router_counts

    if router_offsets is None:
        workspace_offsets[0].zero_()
        torch.cumsum(expert_counts, dim=0, out=workspace_offsets[1:])
        expert_offsets = workspace_offsets
    else:
        expert_offsets = router_offsets

    # Each tile reserves the sum of same-expert assignments in preceding
    # input-order tiles. This is the global stable segmented-prefix stage.
    torch.cumsum(tile_counts, dim=0, out=tile_offsets)
    tile_offsets.sub_(tile_counts)

    expert_block = triton.next_power_of_2(num_experts)
    _tile_local_start_kernel[(num_tiles,)](
        tile_counts,
        num_experts=num_experts,
        tile_count_stride=tile_counts.stride(0),
        expert_block=expert_block,
        num_warps=4,
    )
    _stable_tile_pack_kernel[(num_tiles,)](
        flat_expert_indices,
        flat_routing_weights,
        expert_offsets,
        tile_offsets,
        tile_counts,
        token_permutation,
        inverse_permutation,
        packed_expert_ids,
        packed_token_indices,
        packed_routing_weights,
        num_assignments=num_assignments,
        num_experts=num_experts,
        tile_offset_stride=tile_offsets.stride(0),
        tile_local_start_stride=tile_counts.stride(0),
        top_k=top_k,
        tile_size=block_size,
        num_warps=4,
    )

    return (
        expert_counts,
        expert_offsets,
        token_permutation,
        inverse_permutation,
        packed_expert_ids,
        packed_token_indices,
        packed_routing_weights,
    )
