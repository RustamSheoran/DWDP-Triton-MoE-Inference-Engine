"""Native FP8 specializations of DWDP's persistent pointer-array engine.

The queue format and TensorList layout are intentionally shared with the
standard persistent engine.  Only operands and workspace storage differ: these
kernels load FP8 activations/weights directly, accumulate in FP32 as required
by Tensor Core dot operations, and store FP8 intermediates and outputs.
"""

from __future__ import annotations

import torch

from ..tensor_list import TensorList
from .persistent import PersistentTileQueues, _persistent_program_count

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


FP8_TRITON_AVAILABLE = triton is not None
_BLOCK_M = 16
_BLOCK_N = 64
_BLOCK_K = 32


def select_fp8_dtype(device: torch.device) -> torch.dtype | None:
    """Select the best native FP8 format available to this CUDA device.

    E4M3 is preferred whenever the PyTorch/Triton stack exposes it on FP8
    Tensor-Core hardware.  E5M2 is the compatibility choice when E4M3 is not
    exposed.  ``None`` means the dedicated backend must not be selected.
    """

    if not FP8_TRITON_AVAILABLE or device.type != "cuda":
        return None
    capability = torch.cuda.get_device_capability(device)
    # Native FP8 Tensor Cores begin with Ada (8.9); merely exposing a PyTorch
    # float8 dtype on an older device is not sufficient for this backend.
    if capability < (8, 9):
        return None
    candidates = (
        ("float8_e4m3fn", ("float8e4nv", "float8e4b8", "float8e4b15")),
        ("float8_e4m3fnuz", ("float8e4nv", "float8e4b8", "float8e4b15")),
        ("float8_e5m2", ("float8e5", "float8e5b16")),
        ("float8_e5m2fnuz", ("float8e5", "float8e5b16")),
    )
    for name, triton_types in candidates:
        dtype = getattr(torch, name, None)
        if isinstance(dtype, torch.dtype) and any(hasattr(tl, type_name) for type_name in triton_types):
            return dtype
    return None


def execute_persistent_qwen_fp8(tensors: TensorList, queues: PersistentTileQueues) -> None:
    """Execute FP8 Qwen tiles directly from TensorList pointer metadata with optional fine-grained scaling."""

    if not FP8_TRITON_AVAILABLE:
        raise RuntimeError("native FP8 execution requested but triton is not installed")
    if tensors.size == 0:
        return
    if not tensors.input_ptrs.is_cuda:
        raise RuntimeError("native FP8 execution requires CUDA TensorList metadata")
    programs = _persistent_program_count(tensors.input_ptrs.device)
    if queues.gather_size:
        _persistent_fp8_gather_swiglu[(programs,)](
            queues.gather_descriptor_indices, queues.gather_tile_m, queues.gather_tile_n,
            queues.gather_counter, queues.gather_size,
            tensors.input_ptrs, tensors.token_index_ptrs, tensors.gate_weight_ptrs,
            tensors.up_weight_ptrs, tensors.intermediate_ptrs, tensors.token_offsets,
            tensors.m, tensors.k, tensors.intermediate_n, tensors.input_ld,
            tensors.gate_ld, tensors.up_ld, tensors.quantization_ptrs,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K, num_warps=4,
        )
    if queues.down_size:
        _persistent_fp8_down_route_store[(programs,)](
            queues.down_descriptor_indices, queues.down_tile_m, queues.down_tile_n,
            queues.down_counter, queues.down_size,
            tensors.intermediate_ptrs, tensors.down_weight_ptrs, tensors.output_ptrs,
            tensors.weighted_output_ptrs, tensors.routing_weight_ptrs, tensors.token_offsets,
            tensors.m, tensors.n, tensors.intermediate_n, tensors.down_ld, tensors.output_ld,
            tensors.quantization_ptrs,
            BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K, num_warps=4,
        )


if FP8_TRITON_AVAILABLE:
    @triton.jit
    def _persistent_fp8_gather_swiglu(queue_descriptors, queue_tile_m, queue_tile_n, queue_counter, queue_size,
                                      input_ptrs, token_ptrs, gate_ptrs, up_ptrs, intermediate_ptrs, token_offsets,
                                      m_values, k_values, intermediate_n_values, input_lds, gate_lds, up_lds,
                                      quantization_ptrs,
                                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        work = tl.atomic_add(queue_counter, 1)
        while work < queue_size:
            descriptor = tl.load(queue_descriptors + work)
            tile_m, tile_n = tl.load(queue_tile_m + work), tl.load(queue_tile_n + work)
            m, k, n = tl.load(m_values + descriptor), tl.load(k_values + descriptor), tl.load(intermediate_n_values + descriptor)
            offset = tl.load(token_offsets + descriptor)
            input_ptr, token_ptr = tl.load(input_ptrs + descriptor), tl.load(token_ptrs + descriptor)
            gate_ptr, up_ptr, output_ptr = tl.load(gate_ptrs + descriptor), tl.load(up_ptrs + descriptor), tl.load(intermediate_ptrs + descriptor)
            input_ld, gate_ld, up_ld = tl.load(input_lds + descriptor), tl.load(gate_lds + descriptor), tl.load(up_lds + descriptor)
            scale_ptr = tl.load(quantization_ptrs + descriptor)
            rows, cols = tile_m * BLOCK_M + tl.arange(0, BLOCK_M), tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            tokens = tl.load(token_ptr + offset + rows, mask=rows < m, other=0)
            gate_acc, up_acc, k_offset = tl.zeros((BLOCK_M, BLOCK_N), tl.float32), tl.zeros((BLOCK_M, BLOCK_N), tl.float32), 0
            while k_offset < k:
                ks = k_offset + tl.arange(0, BLOCK_K)
                mask_a, mask_b = (rows[:, None] < m) & (ks[None, :] < k), (cols[None, :] < n) & (ks[:, None] < k)
                values = tl.load(input_ptr + tokens[:, None] * input_ld + ks[None, :], mask=mask_a, other=0.0)
                gate = tl.load(gate_ptr + cols[None, :] * gate_ld + ks[:, None], mask=mask_b, other=0.0)
                up = tl.load(up_ptr + cols[None, :] * up_ld + ks[:, None], mask=mask_b, other=0.0)
                gate_acc += tl.dot(values, gate)
                up_acc += tl.dot(values, up)
                k_offset += BLOCK_K
            if scale_ptr != 0:
                scale_val = tl.load(tl.cast(scale_ptr, tl.pointer_type(tl.float32)))
                gate_acc = gate_acc * scale_val
                up_acc = up_acc * scale_val
            tl.store(output_ptr + rows[:, None] * n + cols[None, :], gate_acc * tl.sigmoid(gate_acc) * up_acc, mask=(rows[:, None] < m) & (cols[None, :] < n))
            work = tl.atomic_add(queue_counter, 1)

    @triton.jit
    def _persistent_fp8_down_route_store(queue_descriptors, queue_tile_m, queue_tile_n, queue_counter, queue_size,
                                         intermediate_ptrs, down_ptrs, output_ptrs, weighted_ptrs, routing_ptrs, token_offsets,
                                         m_values, n_values, intermediate_n_values, down_lds, output_lds,
                                         quantization_ptrs,
                                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        work = tl.atomic_add(queue_counter, 1)
        while work < queue_size:
            descriptor = tl.load(queue_descriptors + work)
            tile_m, tile_n = tl.load(queue_tile_m + work), tl.load(queue_tile_n + work)
            m, n, k = tl.load(m_values + descriptor), tl.load(n_values + descriptor), tl.load(intermediate_n_values + descriptor)
            offset, down_ld, output_ld = tl.load(token_offsets + descriptor), tl.load(down_lds + descriptor), tl.load(output_lds + descriptor)
            input_ptr, down_ptr = tl.load(intermediate_ptrs + descriptor), tl.load(down_ptrs + descriptor)
            output_ptr, weighted_ptr, routing_ptr = tl.load(output_ptrs + descriptor), tl.load(weighted_ptrs + descriptor), tl.load(routing_ptrs + descriptor)
            scale_ptr = tl.load(quantization_ptrs + descriptor)
            rows, cols = tile_m * BLOCK_M + tl.arange(0, BLOCK_M), tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            acc, k_offset = tl.zeros((BLOCK_M, BLOCK_N), tl.float32), 0
            while k_offset < k:
                ks = k_offset + tl.arange(0, BLOCK_K)
                values = tl.load(input_ptr + rows[:, None] * k + ks[None, :], mask=(rows[:, None] < m) & (ks[None, :] < k), other=0.0)
                weights = tl.load(down_ptr + cols[None, :] * down_ld + ks[:, None], mask=(cols[None, :] < n) & (ks[:, None] < k), other=0.0)
                acc += tl.dot(values, weights)
                k_offset += BLOCK_K
            if scale_ptr != 0:
                scale_val = tl.load(tl.cast(scale_ptr, tl.pointer_type(tl.float32)))
                acc = acc * scale_val
            mask = (rows[:, None] < m) & (cols[None, :] < n)
            routing = tl.load(routing_ptr + offset + rows, mask=rows < m, other=0.0)
            tl.store(output_ptr + rows[:, None] * output_ld + cols[None, :], acc, mask=mask)
            tl.store(weighted_ptr + rows[:, None] * output_ld + cols[None, :], acc * routing[:, None], mask=mask)
            work = tl.atomic_add(queue_counter, 1)

