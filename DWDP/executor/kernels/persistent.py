"""Persistent, pointer-array Triton execution for DWDP Qwen experts.

The queue is a device-resident structure-of-arrays of tile coordinates.  It
does not own activation, output, or weight storage; a claimed entry identifies
one TensorList descriptor whose pointer fields continue to reference the
original model allocations.  Queue counters are the sole inter-program
synchronization primitive.  A program atomically claims work until exhaustion,
which provides GPU-side work stealing for uneven expert token counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..tensor_list import TensorList
from ..workspace import ExecutorWorkspace

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


TRITON_AVAILABLE = triton is not None
_BLOCK_M = 16
_BLOCK_N = 64
_BLOCK_K = 32


@dataclass(frozen=True, slots=True)
class PersistentTileQueues:
    """Non-owning views of one forward's gather and down-projection queues.

    Both queues use the same stable entry format: descriptor index, M-tile
    coordinate, and N-tile coordinate.  Stage-specific pointer and dimension
    fields remain in TensorList, avoiding redundant per-tile metadata and
    leaving room for future prefetch/split-K flags without changing entries.
    """

    gather_descriptor_indices: torch.Tensor
    gather_tile_m: torch.Tensor
    gather_tile_n: torch.Tensor
    gather_counter: torch.Tensor
    gather_size: int
    down_descriptor_indices: torch.Tensor
    down_tile_m: torch.Tensor
    down_tile_n: torch.Tensor
    down_counter: torch.Tensor
    down_size: int


def build_persistent_tile_queues(
    tensors: TensorList,
    workspace: ExecutorWorkspace,
) -> PersistentTileQueues:
    """Convert TensorList descriptors into reusable device work queues.

    Construction is O(executable tiles), preserves scheduler descriptor order,
    and performs no sorting or per-tile allocation.  The host writes reusable
    pinned staging arrays, copies them to persistent device buffers, and resets
    counters before either kernel is enqueued.  No host synchronization occurs
    after the persistent kernels begin consuming the queues.
    """

    host_fields = workspace.tensorlist_host_fields
    if host_fields is None:
        raise ValueError(
            "TensorList host metadata must belong to the executor workspace"
        )
    gather_size = 0
    down_size = 0
    for descriptor in range(tensors.size):
        m = int(host_fields["m"][descriptor])
        gather_size += _ceil_div(m, _BLOCK_M) * _ceil_div(
            int(host_fields["intermediate_n"][descriptor]), _BLOCK_N
        )
        down_size += _ceil_div(m, _BLOCK_M) * _ceil_div(
            int(host_fields["n"][descriptor]), _BLOCK_N
        )
    workspace.ensure_persistent_queue_capacity(
        max(gather_size, down_size), device=tensors.input_ptrs.device
    )
    gather_host = workspace._persistent_gather_host
    down_host = workspace._persistent_down_host
    assert gather_host is not None and down_host is not None

    if workspace._host_copy_event is not None:
        workspace._host_copy_event.synchronize()

    gather_index = 0
    down_index = 0
    for descriptor in range(tensors.size):
        m_tiles = _ceil_div(int(host_fields["m"][descriptor]), _BLOCK_M)
        gather_n_tiles = _ceil_div(
            int(host_fields["intermediate_n"][descriptor]), _BLOCK_N
        )
        down_n_tiles = _ceil_div(int(host_fields["n"][descriptor]), _BLOCK_N)
        for tile_m in range(m_tiles):
            for tile_n in range(gather_n_tiles):
                gather_host[0][gather_index] = descriptor
                gather_host[1][gather_index] = tile_m
                gather_host[2][gather_index] = tile_n
                gather_index += 1
            for tile_n in range(down_n_tiles):
                down_host[0][down_index] = descriptor
                down_host[1][down_index] = tile_m
                down_host[2][down_index] = tile_n
                down_index += 1

    device_queue_fields = (
        workspace.persistent_gather_descriptor_indices,
        workspace.persistent_gather_tile_m,
        workspace.persistent_gather_tile_n,
        workspace.persistent_down_descriptor_indices,
        workspace.persistent_down_tile_m,
        workspace.persistent_down_tile_n,
        workspace.persistent_gather_counter,
        workspace.persistent_down_counter,
    )
    assert all(field is not None for field in device_queue_fields)
    gather_device = device_queue_fields[:3]
    down_device = device_queue_fields[3:6]
    if workspace._persistent_gather_device_buffer is not None and workspace._persistent_gather_host_buffer is not None:
        workspace._persistent_gather_device_buffer[:, :gather_size].copy_(
            workspace._persistent_gather_host_buffer[:, :gather_size], non_blocking=True
        )
    else:
        for destination, source in zip(gather_device, gather_host):
            destination[:gather_size].copy_(source[:gather_size], non_blocking=True)

    if workspace._persistent_down_device_buffer is not None and workspace._persistent_down_host_buffer is not None:
        workspace._persistent_down_device_buffer[:, :down_size].copy_(
            workspace._persistent_down_host_buffer[:, :down_size], non_blocking=True
        )
    else:
        for destination, source in zip(down_device, down_host):
            destination[:down_size].copy_(source[:down_size], non_blocking=True)
    # Counters are reset before enqueueing.  Atomic increments in the kernels
    # serialize claims; no global barrier or CPU participation is needed.
    workspace.persistent_gather_counter.zero_()
    workspace.persistent_down_counter.zero_()
    return PersistentTileQueues(
        gather_device[0][:gather_size],
        gather_device[1][:gather_size],
        gather_device[2][:gather_size],
        workspace.persistent_gather_counter,
        gather_size,
        down_device[0][:down_size],
        down_device[1][:down_size],
        down_device[2][:down_size],
        workspace.persistent_down_counter,
        down_size,
    )


def execute_persistent_qwen(tensors: TensorList, queues: PersistentTileQueues) -> None:
    """Enqueue the two dependent persistent kernels without host waiting.

    Qwen's down projection depends on all generated intermediate values, so it
    is a separate launch on the same CUDA stream. Stream order supplies the
    required stage dependency without a host synchronization; within either
    stage, programs dynamically steal tiles through the queue counter.
    """

    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "persistent DWDP execution requested but triton is not installed"
        )
    if tensors.size == 0:
        return
    if not tensors.input_ptrs.is_cuda:
        raise RuntimeError(
            "persistent DWDP execution requires CUDA TensorList metadata"
        )
    programs = _persistent_program_count(tensors.input_ptrs.device)
    if queues.gather_size:
        _persistent_gather_swiglu[(programs,)](
            queues.gather_descriptor_indices,
            queues.gather_tile_m,
            queues.gather_tile_n,
            queues.gather_counter,
            queues.gather_size,
            tensors.input_ptrs,
            tensors.token_index_ptrs,
            tensors.gate_weight_ptrs,
            tensors.up_weight_ptrs,
            tensors.intermediate_ptrs,
            tensors.token_offsets,
            tensors.m,
            tensors.k,
            tensors.intermediate_n,
            tensors.input_ld,
            tensors.gate_ld,
            tensors.up_ld,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=4,
        )
    if queues.down_size:
        _persistent_down_route_store[(programs,)](
            queues.down_descriptor_indices,
            queues.down_tile_m,
            queues.down_tile_n,
            queues.down_counter,
            queues.down_size,
            tensors.intermediate_ptrs,
            tensors.down_weight_ptrs,
            tensors.output_ptrs,
            tensors.weighted_output_ptrs,
            tensors.routing_weight_ptrs,
            tensors.token_offsets,
            tensors.m,
            tensors.n,
            tensors.intermediate_n,
            tensors.down_ld,
            tensors.output_ld,
            BLOCK_M=_BLOCK_M,
            BLOCK_N=_BLOCK_N,
            BLOCK_K=_BLOCK_K,
            num_warps=4,
        )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _persistent_program_count(device: torch.device) -> int:
    """Use one long-lived program per SM; no expert is statically assigned."""

    return torch.cuda.get_device_properties(device).multi_processor_count


if TRITON_AVAILABLE:

    @triton.jit
    def _persistent_gather_swiglu(
        queue_descriptors,
        queue_tile_m,
        queue_tile_n,
        queue_counter,
        queue_size,
        input_ptrs,
        token_ptrs,
        gate_ptrs,
        up_ptrs,
        intermediate_ptrs,
        token_offsets,
        m_values,
        k_values,
        intermediate_n_values,
        input_lds,
        gate_lds,
        up_lds,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        work = tl.atomic_add(queue_counter, 1)
        while work < queue_size:
            descriptor = tl.load(queue_descriptors + work)
            tile_m, tile_n = tl.load(queue_tile_m + work), tl.load(queue_tile_n + work)
            m, k, n = (
                tl.load(m_values + descriptor),
                tl.load(k_values + descriptor),
                tl.load(intermediate_n_values + descriptor),
            )
            offset = tl.load(token_offsets + descriptor)
            input_ptr, token_ptr = (
                tl.load(input_ptrs + descriptor),
                tl.load(token_ptrs + descriptor),
            )
            gate_ptr, up_ptr, output_ptr = (
                tl.load(gate_ptrs + descriptor),
                tl.load(up_ptrs + descriptor),
                tl.load(intermediate_ptrs + descriptor),
            )
            input_ld, gate_ld, up_ld = (
                tl.load(input_lds + descriptor),
                tl.load(gate_lds + descriptor),
                tl.load(up_lds + descriptor),
            )
            rows, cols = (
                tile_m * BLOCK_M + tl.arange(0, BLOCK_M),
                tile_n * BLOCK_N + tl.arange(0, BLOCK_N),
            )
            tokens = tl.load(token_ptr + offset + rows, mask=rows < m, other=0)
            gate_acc, up_acc, k_offset = (
                tl.zeros((BLOCK_M, BLOCK_N), tl.float32),
                tl.zeros((BLOCK_M, BLOCK_N), tl.float32),
                0,
            )
            while k_offset < k:
                ks = k_offset + tl.arange(0, BLOCK_K)
                mask_a, mask_b = (
                    (rows[:, None] < m) & (ks[None, :] < k),
                    (cols[None, :] < n) & (ks[:, None] < k),
                )
                values = tl.load(
                    input_ptr + tokens[:, None] * input_ld + ks[None, :],
                    mask=mask_a,
                    other=0.0,
                )
                gate = tl.load(
                    gate_ptr + cols[None, :] * gate_ld + ks[:, None],
                    mask=mask_b,
                    other=0.0,
                )
                up = tl.load(
                    up_ptr + cols[None, :] * up_ld + ks[:, None], mask=mask_b, other=0.0
                )
                gate_acc += tl.dot(values, gate)
                up_acc += tl.dot(values, up)
                k_offset += BLOCK_K
            tl.store(
                output_ptr + rows[:, None] * n + cols[None, :],
                gate_acc * tl.sigmoid(gate_acc) * up_acc,
                mask=(rows[:, None] < m) & (cols[None, :] < n),
            )
            work = tl.atomic_add(queue_counter, 1)

    @triton.jit
    def _persistent_down_route_store(
        queue_descriptors,
        queue_tile_m,
        queue_tile_n,
        queue_counter,
        queue_size,
        intermediate_ptrs,
        down_ptrs,
        output_ptrs,
        weighted_ptrs,
        routing_ptrs,
        token_offsets,
        m_values,
        n_values,
        intermediate_n_values,
        down_lds,
        output_lds,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        work = tl.atomic_add(queue_counter, 1)
        while work < queue_size:
            descriptor = tl.load(queue_descriptors + work)
            tile_m, tile_n = tl.load(queue_tile_m + work), tl.load(queue_tile_n + work)
            m, n, k = (
                tl.load(m_values + descriptor),
                tl.load(n_values + descriptor),
                tl.load(intermediate_n_values + descriptor),
            )
            offset, down_ld, output_ld = (
                tl.load(token_offsets + descriptor),
                tl.load(down_lds + descriptor),
                tl.load(output_lds + descriptor),
            )
            input_ptr, down_ptr = (
                tl.load(intermediate_ptrs + descriptor),
                tl.load(down_ptrs + descriptor),
            )
            output_ptr, weighted_ptr, routing_ptr = (
                tl.load(output_ptrs + descriptor),
                tl.load(weighted_ptrs + descriptor),
                tl.load(routing_ptrs + descriptor),
            )
            rows, cols = (
                tile_m * BLOCK_M + tl.arange(0, BLOCK_M),
                tile_n * BLOCK_N + tl.arange(0, BLOCK_N),
            )
            acc, k_offset = tl.zeros((BLOCK_M, BLOCK_N), tl.float32), 0
            while k_offset < k:
                ks = k_offset + tl.arange(0, BLOCK_K)
                values = tl.load(
                    input_ptr + rows[:, None] * k + ks[None, :],
                    mask=(rows[:, None] < m) & (ks[None, :] < k),
                    other=0.0,
                )
                weights = tl.load(
                    down_ptr + cols[None, :] * down_ld + ks[:, None],
                    mask=(cols[None, :] < n) & (ks[:, None] < k),
                    other=0.0,
                )
                acc += tl.dot(values, weights)
                k_offset += BLOCK_K
            mask = (rows[:, None] < m) & (cols[None, :] < n)
            routing = tl.load(routing_ptr + offset + rows, mask=rows < m, other=0.0)
            tl.store(
                output_ptr + rows[:, None] * output_ld + cols[None, :], acc, mask=mask
            )
            tl.store(
                weighted_ptr + rows[:, None] * output_ld + cols[None, :],
                acc * routing[:, None],
                mask=mask,
            )
            work = tl.atomic_add(queue_counter, 1)
