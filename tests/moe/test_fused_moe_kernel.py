"""Rigorous correctness tests for the Fused Triton MoE Kernel.

Tests compare the fused Triton MoE layer against the golden PyTorch reference
across diverse token counts, routing distributions, non-divisible shapes, and edge cases.

TOLERANCE JUSTIFICATION FOR FP16:
IEEE 754 half-precision floating point (FP16) uses 10 mantissa bits, yielding an
epsilon of ~9.77e-4 (~0.001). In a chained MoE block:
    x -> [Gate & Up GEMM] -> SwiGLU activation -> [Down GEMM] -> scaled sum
the operations involve non-associative floating point additions in the GEMMs and
transcendental functions (sigmoid / exp). Although intermediate accumulation in
both our reference and Triton kernel is performed in FP32, the differing reduction
trees and final FP16 truncation bound the numerical error to a few multiples of
FP16 epsilon.
Hence, an absolute tolerance of atol=1e-2 and relative tolerance of rtol=1e-2
are mathematically sound, standard in MoE kernel testing (e.g., vLLM / Megablocks),
and strictly verified in this suite.
"""

from __future__ import annotations

import os

# Enable Triton CPU interpret mode for sandboxed / CPU-only testing
os.environ["TRITON_INTERPRET"] = "1"

import pytest
import torch

from DWDP.moe.layer import triton_moe_layer
from DWDP.moe.reference import moe_reference


@pytest.fixture
def moe_weights():
    """Create tiny FP16 weights suitable for CPU interpretation."""
    torch.manual_seed(42)
    hidden_dim = 64
    intermediate_dim = 32
    num_experts = 8

    w1 = torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.float16) * 0.05
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.float16) * 0.05

    return {
        "hidden_dim": hidden_dim,
        "intermediate_dim": intermediate_dim,
        "num_experts": num_experts,
        "w1": w1,
        "w2": w2,
    }


def _verify_parity(out: torch.Tensor, ref: torch.Tensor, test_name: str, atol: float = 1e-2, rtol: float = 1e-2):
    """Verify numerical parity and report max abs and rel errors."""
    if out.numel() == 0 and ref.numel() == 0:
        return

    diff = (out.float() - ref.float()).abs()
    max_abs_err = diff.max().item()
    denom = ref.float().abs() + 1e-6
    max_rel_err = (diff / denom).max().item()

    # Log error for audit traceability
    print(f"\n[{test_name}] max_abs_err={max_abs_err:.6e}, max_rel_err={max_rel_err:.6e}")

    assert torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol), (
        f"Parity failure in {test_name}: max_abs_err={max_abs_err:.6e} (atol={atol}), "
        f"max_rel_err={max_rel_err:.6e} (rtol={rtol})"
    )


@pytest.mark.parametrize("num_tokens", [0, 1, 3, 17, 64])
def test_triton_moe_token_counts(moe_weights, num_tokens):
    """Test token counts in {0, 1, 3, 17, 64} on CPU."""
    hidden_dim = moe_weights["hidden_dim"]
    num_experts = moe_weights["num_experts"]
    w1 = moe_weights["w1"]
    w2 = moe_weights["w2"]
    top_k = 2

    torch.manual_seed(100 + num_tokens)
    if num_tokens > 0:
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16)
        topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
        topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1).half()
    else:
        x = torch.empty(0, hidden_dim, dtype=torch.float16)
        topk_indices = torch.empty(0, top_k, dtype=torch.int64)
        topk_weights = torch.empty(0, top_k, dtype=torch.float16)

    out = triton_moe_layer(x, w1, w2, topk_indices, topk_weights, block_m=16, block_n=16, block_k=16)
    ref = moe_reference(x, w1, w2, topk_indices, topk_weights)

    assert out.shape == (num_tokens, hidden_dim)
    assert out.dtype == torch.float16
    _verify_parity(out, ref, f"CPU T={num_tokens}")


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_triton_moe_top_k(moe_weights, top_k):
    """Test varying top_k in {1, 2, 4}."""
    num_tokens = 19
    hidden_dim = moe_weights["hidden_dim"]
    num_experts = moe_weights["num_experts"]
    w1 = moe_weights["w1"]
    w2 = moe_weights["w2"]

    torch.manual_seed(200 + top_k)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16)
    topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1).half()

    out = triton_moe_layer(x, w1, w2, topk_indices, topk_weights, block_m=16, block_n=16, block_k=16)
    ref = moe_reference(x, w1, w2, topk_indices, topk_weights)

    _verify_parity(out, ref, f"CPU top_k={top_k}")


def test_triton_moe_zero_token_experts(moe_weights):
    """Test scenario where some experts receive 0 tokens."""
    num_tokens = 12
    hidden_dim = moe_weights["hidden_dim"]
    w1 = moe_weights["w1"]
    w2 = moe_weights["w2"]
    top_k = 2

    # Route only to experts 0, 1, 2 (leaving experts 3..7 empty)
    torch.manual_seed(300)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16)
    topk_indices = torch.randint(0, 3, (num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1).half()

    out = triton_moe_layer(x, w1, w2, topk_indices, topk_weights, block_m=16, block_n=16, block_k=16)
    ref = moe_reference(x, w1, w2, topk_indices, topk_weights)

    _verify_parity(out, ref, "CPU Zero Token Experts")


def test_triton_moe_skewed_all_one_expert(moe_weights):
    """Test extreme load imbalance where all tokens route to Expert 0."""
    num_tokens = 25
    hidden_dim = moe_weights["hidden_dim"]
    w1 = moe_weights["w1"]
    w2 = moe_weights["w2"]
    top_k = 2

    torch.manual_seed(400)
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16)
    topk_indices = torch.zeros((num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float16).repeat(num_tokens, 1)

    out = triton_moe_layer(x, w1, w2, topk_indices, topk_weights, block_m=16, block_n=16, block_k=16)
    ref = moe_reference(x, w1, w2, topk_indices, topk_weights)

    _verify_parity(out, ref, "CPU All Tokens to Expert 0")


def test_triton_moe_non_divisible_shapes():
    """Test dimensions not divisible by block sizes (H=50, I=25, block=16)."""
    torch.manual_seed(500)
    hidden_dim = 50
    intermediate_dim = 25
    num_experts = 5
    num_tokens = 11
    top_k = 2

    w1 = torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.float16) * 0.05
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.float16) * 0.05
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16)
    topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64)
    topk_weights = torch.softmax(torch.randn(num_tokens, top_k), dim=-1).half()

    out = triton_moe_layer(x, w1, w2, topk_indices, topk_weights, block_m=16, block_n=16, block_k=16)
    ref = moe_reference(x, w1, w2, topk_indices, topk_weights)

    _verify_parity(out, ref, "CPU Non-divisible Shapes (H=50, I=25)")


# ==============================================================================
# GPU-Gated Versions of the Tests (skipped on CPU-only machines)
# ==============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("num_tokens", [0, 1, 3, 17, 64])
def test_triton_moe_gpu_token_counts(num_tokens):
    """GPU execution: test token counts in {0, 1, 3, 17, 64}."""
    device = torch.device("cuda")
    hidden_dim = 64
    intermediate_dim = 32
    num_experts = 8
    top_k = 2

    w1 = torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.float16, device=device) * 0.05
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.float16, device=device) * 0.05

    if num_tokens > 0:
        x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=device)
        topk_indices = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int64, device=device)
        topk_weights = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1).half()
    else:
        x = torch.empty(0, hidden_dim, dtype=torch.float16, device=device)
        topk_indices = torch.empty(0, top_k, dtype=torch.int64, device=device)
        topk_weights = torch.empty(0, top_k, dtype=torch.float16, device=device)

    out = triton_moe_layer(x, w1, w2, topk_indices, topk_weights, block_m=16)
    ref = moe_reference(x, w1, w2, topk_indices, topk_weights)

    _verify_parity(out, ref, f"GPU T={num_tokens}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_moe_gpu_skewed():
    """GPU execution: extreme load imbalance (all tokens to Expert 0)."""
    device = torch.device("cuda")
    num_tokens = 32
    hidden_dim = 64
    intermediate_dim = 32
    num_experts = 8
    top_k = 2

    w1 = torch.randn(num_experts, 2 * intermediate_dim, hidden_dim, dtype=torch.float16, device=device) * 0.05
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=torch.float16, device=device) * 0.05
    x = torch.randn(num_tokens, hidden_dim, dtype=torch.float16, device=device)
    topk_indices = torch.zeros((num_tokens, top_k), dtype=torch.int64, device=device)
    topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float16, device=device).repeat(num_tokens, 1)

    out = triton_moe_layer(x, w1, w2, topk_indices, topk_weights, block_m=16)
    ref = moe_reference(x, w1, w2, topk_indices, topk_weights)

    _verify_parity(out, ref, "GPU Skewed Expert 0")
