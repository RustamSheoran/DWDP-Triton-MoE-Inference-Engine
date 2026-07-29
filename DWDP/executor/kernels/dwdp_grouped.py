"""DWDP pointer-array grouped execution kernels.

Unlike dense grouped GEMM libraries, these kernels dereference independent
expert-weight addresses from TensorList.  This is the DWDP execution boundary:
future persistent work queues can consume the same SoA descriptor unchanged.
"""
from __future__ import annotations

from ..tensor_list import TensorList

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

TRITON_AVAILABLE = triton is not None

if TRITON_AVAILABLE:
    @triton.jit
    def _gather_swiglu(input_ptrs, token_ptrs, gate_ptrs, up_ptrs, intermediate_ptrs,
                       token_offsets, m_values, k_values, intermediate_n_values,
                       input_lds, gate_lds, up_lds,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        descriptor, tile_m, tile_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        m, k, n = tl.load(m_values + descriptor), tl.load(k_values + descriptor), tl.load(intermediate_n_values + descriptor)
        token_offset = tl.load(token_offsets + descriptor)
        input_ld, gate_ld, up_ld = tl.load(input_lds + descriptor), tl.load(gate_lds + descriptor), tl.load(up_lds + descriptor)
        input_ptr, token_ptr = tl.load(input_ptrs + descriptor), tl.load(token_ptrs + descriptor)
        gate_ptr, up_ptr, intermediate_ptr = tl.load(gate_ptrs + descriptor), tl.load(up_ptrs + descriptor), tl.load(intermediate_ptrs + descriptor)
        rows, cols = tile_m * BLOCK_M + tl.arange(0, BLOCK_M), tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        tokens = tl.load(token_ptr + token_offset + rows, mask=rows < m, other=0)
        gate_acc, up_acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32), tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        k_offset = 0
        while k_offset < k:
            ks = k_offset + tl.arange(0, BLOCK_K)
            mask_a = (rows[:, None] < m) & (ks[None, :] < k)
            mask_b = (cols[None, :] < n) & (ks[:, None] < k)
            activations = tl.load(input_ptr + tokens[:, None] * input_ld + ks[None, :], mask=mask_a, other=0.0)
            gate = tl.load(gate_ptr + cols[None, :] * gate_ld + ks[:, None], mask=mask_b, other=0.0)
            up = tl.load(up_ptr + cols[None, :] * up_ld + ks[:, None], mask=mask_b, other=0.0)
            gate_acc += tl.dot(activations, gate)
            up_acc += tl.dot(activations, up)
            k_offset += BLOCK_K
        tl.store(intermediate_ptr + rows[:, None] * n + cols[None, :], gate_acc * tl.sigmoid(gate_acc) * up_acc, mask=(rows[:, None] < m) & (cols[None, :] < n))

    @triton.jit
    def _down_route_store(intermediate_ptrs, down_ptrs, output_ptrs, weighted_ptrs,
                          routing_ptrs, token_offsets, m_values, n_values, intermediate_n_values,
                          down_lds, output_lds,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        descriptor, tile_m, tile_n = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        m, n, k = tl.load(m_values + descriptor), tl.load(n_values + descriptor), tl.load(intermediate_n_values + descriptor)
        token_offset, down_ld, output_ld = tl.load(token_offsets + descriptor), tl.load(down_lds + descriptor), tl.load(output_lds + descriptor)
        intermediate_ptr, down_ptr = tl.load(intermediate_ptrs + descriptor), tl.load(down_ptrs + descriptor)
        output_ptr, weighted_ptr, routing_ptr = tl.load(output_ptrs + descriptor), tl.load(weighted_ptrs + descriptor), tl.load(routing_ptrs + descriptor)
        rows, cols = tile_m * BLOCK_M + tl.arange(0, BLOCK_M), tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc, k_offset = tl.zeros((BLOCK_M, BLOCK_N), tl.float32), 0
        while k_offset < k:
            ks = k_offset + tl.arange(0, BLOCK_K)
            values = tl.load(intermediate_ptr + rows[:, None] * k + ks[None, :], mask=(rows[:, None] < m) & (ks[None, :] < k), other=0.0)
            weights = tl.load(down_ptr + cols[None, :] * down_ld + ks[:, None], mask=(cols[None, :] < n) & (ks[:, None] < k), other=0.0)
            acc += tl.dot(values, weights)
            k_offset += BLOCK_K
        mask = (rows[:, None] < m) & (cols[None, :] < n)
        routing = tl.load(routing_ptr + token_offset + rows, mask=rows < m, other=0.0)
        tl.store(output_ptr + rows[:, None] * output_ld + cols[None, :], acc, mask=mask)
        tl.store(weighted_ptr + rows[:, None] * output_ld + cols[None, :], acc * routing[:, None], mask=mask)


def grouped_qwen_swiglu(tensors: TensorList) -> None:
    """Launch DWDP tile work directly from one pointer-array descriptor."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("DWDP grouped execution requested but triton is not installed")
    if tensors.size == 0:
        return
    if not tensors.input_ptrs.is_cuda:
        raise RuntimeError("DWDP grouped execution requires CUDA TensorList metadata")
    max_m, max_n, max_intermediate = tensors.launch_dimensions
    gather_grid = lambda meta: (tensors.size, triton.cdiv(max_m, meta["BLOCK_M"]), triton.cdiv(max_intermediate, meta["BLOCK_N"]))
    _gather_swiglu[gather_grid](tensors.input_ptrs, tensors.token_index_ptrs, tensors.gate_weight_ptrs, tensors.up_weight_ptrs, tensors.intermediate_ptrs, tensors.token_offsets, tensors.m, tensors.k, tensors.intermediate_n, tensors.input_ld, tensors.gate_ld, tensors.up_ld, BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, num_warps=4)
    down_grid = lambda meta: (tensors.size, triton.cdiv(max_m, meta["BLOCK_M"]), triton.cdiv(max_n, meta["BLOCK_N"]))
    _down_route_store[down_grid](tensors.intermediate_ptrs, tensors.down_weight_ptrs, tensors.output_ptrs, tensors.weighted_output_ptrs, tensors.routing_weight_ptrs, tensors.token_offsets, tensors.m, tensors.n, tensors.intermediate_n, tensors.down_ld, tensors.output_ld, BLOCK_M=16, BLOCK_N=64, BLOCK_K=32, num_warps=4)
