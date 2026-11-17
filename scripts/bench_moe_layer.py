#!/usr/bin/env python3
"""Benchmark script for single MoE layer inference on Tesla T4 (sm_75).

Compares FOUR implementations:
  (a) Naive per-expert loop (un-sorted PyTorch loop with boolean indexing)
  (b) Sorted-token per-expert torch.matmul loop (the fair, optimized PyTorch baseline)
  (c) Fused Triton MoE layer (fused SwiGLU + down projection + deterministic sum)
  (d) Single torch.matmul of equal FLOPs (theoretical cuBLAS roofline)

Features:
- Shapes read from Qwen MoE config (hidden_size=2048, moe_intermediate_size=1408, E=64, top_k=4)
- FP16 only. Pre-flight check ensures weights + activations fit in 16 GB VRAM.
- Exact kernel launch counts measured using torch.profiler.
- 10 warmup + 50 timed iterations using torch.cuda.Event; records median and p95.
- Single GEMM Triton vs torch.matmul ratio measurement.
- Outputs both JSON and Markdown result summaries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from DWDP.adapters.extractor import QwenMoEConfig, parse_qwen_moe_config
from DWDP.moe.align import align_tokens_by_expert
from DWDP.moe.layer import triton_moe_layer


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def baseline_a_naive_loop(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """(a) Naive per-expert loop using boolean masking and dynamic non-zero indexing."""
    num_tokens, hidden_dim = x.shape
    num_experts, double_intermediate_dim, _ = w1.shape
    intermediate_dim = double_intermediate_dim // 2

    out = torch.zeros((num_tokens, hidden_dim), dtype=x.dtype, device=x.device)
    flat_indices = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1).unsqueeze(1)
    token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(topk_indices.shape[1])

    for e in range(num_experts):
        mask = flat_indices == e
        if not mask.any():
            continue
        selected_tokens = token_indices[mask]
        selected_weights = flat_weights[mask]
        x_e = x[selected_tokens]

        w_gate = w1[e, :intermediate_dim, :]
        w_up = w1[e, intermediate_dim:, :]
        w_down = w2[e, :, :]

        gate = F.linear(x_e, w_gate)
        up = F.linear(x_e, w_up)
        act = F.silu(gate) * up
        down = F.linear(act, w_down)

        out.index_add_(0, selected_tokens, down * selected_weights)

    return out


def baseline_b_sorted_matmul(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """(b) Sorted-token per-expert torch.matmul loop (the fair Megablocks-style baseline)."""
    num_tokens, hidden_dim = x.shape
    num_experts, double_intermediate_dim, _ = w1.shape
    intermediate_dim = double_intermediate_dim // 2
    top_k = topk_indices.shape[1]

    # Pre-sort all token-expert pairs upfront
    flat_expert_ids = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(top_k)

    sorted_expert_ids, sort_perm = torch.sort(flat_expert_ids)
    sorted_tokens = token_indices[sort_perm]
    sorted_weights = flat_weights[sort_perm].unsqueeze(1)

    x_sorted = x[sorted_tokens]
    expert_counts = torch.bincount(sorted_expert_ids, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=x.device)
    offsets[1:] = torch.cumsum(expert_counts, dim=0)

    out_sorted = torch.zeros_like(x_sorted)

    for e in range(num_experts):
        start = int(offsets[e].item())
        end = int(offsets[e + 1].item())
        if start == end:
            continue

        x_e = x_sorted[start:end]
        w_gate = w1[e, :intermediate_dim, :]
        w_up = w1[e, intermediate_dim:, :]
        w_down = w2[e, :, :]

        gate = F.linear(x_e, w_gate)
        up = F.linear(x_e, w_up)
        act = F.silu(gate) * up
        down = F.linear(act, w_down)

        out_sorted[start:end] = down * sorted_weights[start:end]

    # Invert permutation and sum top_k contributions
    out = torch.zeros((num_tokens, hidden_dim), dtype=x.dtype, device=x.device)
    out.index_add_(0, sorted_tokens, out_sorted)
    return out


def baseline_c_triton(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """(c) Fused Triton MoE layer."""
    return triton_moe_layer(x, w1, w2, topk_indices, topk_weights)


# ---------------------------------------------------------------------------
# Profiling & Timing Utilities
# ---------------------------------------------------------------------------

def measure_cuda_latency(fn: Callable[[], torch.Tensor], warmup: int = 10, timed: int = 50) -> Tuple[float, float]:
    """Measure median and p95 latency using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    latencies: List[float] = []

    for _ in range(timed):
        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    latencies.sort()
    median_ms = latencies[len(latencies) // 2]
    p95_ms = latencies[int(len(latencies) * 0.95)]
    return median_ms, p95_ms


def measure_launch_count(fn: Callable[[], torch.Tensor]) -> int:
    """Count real CUDA kernel launches using torch.profiler."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        fn()
    torch.cuda.synchronize()

    launches = 0
    for event in prof.events():
        if event.device_type == torch.profiler.DeviceType.CUDA:
            launches += 1
    return launches


def measure_single_gemm_ratio(m: int, k: int, n: int, device: torch.device) -> Tuple[float, float, float]:
    """Measure single GEMM latency: Triton vs torch.matmul (cuBLAS)."""
    import triton
    import triton.language as tl

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        ],
        key=["M", "N", "K"],
    )
    @triton.jit
    def _gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_idx in range(0, K, BLOCK_K):
            k_offs = k_idx + offs_k
            a = tl.load(a_ptr + offs_m[:, None] * K + k_offs[None, :], mask=(offs_m[:, None] < M) & (k_offs[None, :] < K), other=0.0)
            b = tl.load(b_ptr + k_offs[:, None] * N + offs_n[None, :], mask=(k_offs[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    a = torch.randn(m, k, dtype=torch.float16, device=device)
    b = torch.randn(k, n, dtype=torch.float16, device=device)
    c_triton = torch.zeros(m, n, dtype=torch.float16, device=device)

    def run_cublas():
        return torch.matmul(a, b)

    def run_triton():
        grid = (triton.cdiv(m, 64), triton.cdiv(n, 64))
        _gemm_kernel[grid](a, b, c_triton, m, n, k)
        return c_triton

    cublas_med, _ = measure_cuda_latency(run_cublas, warmup=5, timed=20)
    triton_med, _ = measure_cuda_latency(run_triton, warmup=5, timed=20)
    ratio = triton_med / max(cublas_med, 1e-6)
    return triton_med, cublas_med, ratio


# ---------------------------------------------------------------------------
# Main Benchmark Runner
# ---------------------------------------------------------------------------

def run_benchmark(output_dir: str = "benchmarks/results/t4_fused_moe") -> None:
    cuda_available = torch.cuda.is_available()

    # Read shapes from real Qwen1.5-MoE config
    raw_config = {
        "hidden_size": 2048,
        "moe_intermediate_size": 1408,
        "num_experts": 64,
        "top_k": 4,
        "num_layers": 24,
        "num_kv_heads": 16,
        "head_dim": 128,
    }
    cfg: QwenMoEConfig = parse_qwen_moe_config(raw_config)

    H = cfg.hidden_size
    I = cfg.moe_intermediate_size
    E = cfg.num_experts
    K = cfg.top_k
    dtype = torch.float16

    token_counts = [1, 8, 64, 512, 4096]

    print("=" * 70)
    print("DWDP FUSED TRITON MoE LAYER BENCHMARK (TESLA T4 / sm_75)")
    print("=" * 70)
    print(f"Model Shapes: hidden_dim={H}, moe_intermediate_dim={I}, num_experts={E}, top_k={K}")
    print(f"Data Type: {dtype}")

    if not cuda_available:
        print("\n[WARNING] CUDA is NOT available on this system.")
        print("Emitting placeholder report skeleton with 'TBD (run on T4)'.")
        print("To produce actual benchmarks, execute scripts/run_on_gpu.sh on a Tesla T4 GPU.")

        os.makedirs(output_dir, exist_ok=True)
        md_path = os.path.join(output_dir, "BENCHMARK_REPORT.md")
        json_path = os.path.join(output_dir, "benchmark_summary.json")

        skeleton_data = {
            "device": "TBD (run on T4)",
            "compute_capability": "7.5 (TBD)",
            "torch_version": torch.__version__,
            "triton_version": "TBD",
            "status": "AWAITING_GPU_EXECUTION",
            "token_counts": token_counts,
        }
        with open(json_path, "w") as f:
            json.dump(skeleton_data, f, indent=2)

        md_content = f"""# DWDP Fused MoE Layer Benchmark (Tesla T4)

*Status: Awaiting GPU Execution on Tesla T4*

| Tokens (T) | Naive Loop (a) | Sorted Matmul (b) | Triton Path (c) | Roofline cuBLAS (d) | Real Launches (c) | Speedup vs (b) |
|---|---|---|---|---|---|---|
"""
        for t in token_counts:
            md_content += f"| {t} | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) |\n"

        with open(md_path, "w") as f:
            f.write(md_content)

        print(f"\nPlaceholder generated: {md_path}")
        return

    # CUDA is present: verify device & memory
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    cc_str = f"{capability[0]}.{capability[1]}"
    total_mem_bytes = torch.cuda.get_device_properties(device).total_memory
    total_mem_gb = total_mem_bytes / (1024**3)

    print(f"GPU: {device_name} (Compute Capability {cc_str})")
    print(f"Total VRAM: {total_mem_gb:.2f} GB")

    # 1. Pre-flight VRAM estimation
    w1_bytes = E * (2 * I) * H * 2  # fp16 = 2 bytes
    w2_bytes = E * H * I * 2
    weights_total_bytes = w1_bytes + w2_bytes

    max_t = max(token_counts)
    max_act_bytes = (max_t * H * 2) + (max_t * K * I * 2) + (max_t * K * H * 2)
    total_estimated_bytes = weights_total_bytes + max_act_bytes
    total_estimated_gb = total_estimated_bytes / (1024**3)

    print(f"\n[VRAM PRE-FLIGHT CHECK]")
    print(f"  Layer Weights (FP16): {weights_total_bytes / (1024**2):.2f} MB")
    print(f"  Peak Activations (T={max_t}, top_k={K}): {max_act_bytes / (1024**2):.2f} MB")
    print(f"  Total Memory Needed: {total_estimated_gb:.2f} GB / {total_mem_gb:.2f} GB")

    if total_estimated_bytes > total_mem_bytes * 0.9:
        raise MemoryError(
            f"VRAM pre-flight check failed! Required {total_estimated_gb:.2f} GB exceeds safe threshold on {total_mem_gb:.2f} GB GPU."
        )
    print("  Pre-flight check PASSED! Memory comfortably fits on GPU.\n")

    # 2. Allocate layer weights
    torch.manual_seed(42)
    w1 = torch.randn(E, 2 * I, H, dtype=dtype, device=device) * 0.02
    w2 = torch.randn(E, H, I, dtype=dtype, device=device) * 0.02

    # 3. Single GEMM ratio measurement
    print("[Single GEMM Baseline Check: Triton vs cuBLAS]")
    gemm_m, gemm_k, gemm_n = 1024, H, I
    triton_gemm_ms, cublas_gemm_ms, gemm_ratio = measure_single_gemm_ratio(gemm_m, gemm_k, gemm_n, device)
    print(f"  Single GEMM [{gemm_m}x{gemm_k} @ {gemm_k}x{gemm_n}]:")
    print(f"  cuBLAS: {cublas_gemm_ms:.3f} ms | Triton: {triton_gemm_ms:.3f} ms | Ratio (Triton/cuBLAS): {gemm_ratio:.2f}x\n")

    results: List[Dict] = []

    print("-" * 70)
    print(f"{'T':<6} | {'Naive (a) ms':<12} | {'Sorted (b) ms':<13} | {'Triton (c) ms':<13} | {'cuBLAS (d) ms':<13} | {'Launches (c)':<12}")
    print("-" * 70)

    for T in token_counts:
        x = torch.randn(T, H, dtype=dtype, device=device)
        topk_indices = torch.randint(0, E, (T, K), dtype=torch.int64, device=device)
        topk_weights = torch.softmax(torch.randn(T, K, device=device), dim=-1).half()

        # (a) Naive per-expert loop
        naive_fn = lambda: baseline_a_naive_loop(x, w1, w2, topk_indices, topk_weights)
        naive_med, naive_p95 = measure_cuda_latency(naive_fn)

        # (b) Sorted-token per-expert matmul
        sorted_fn = lambda: baseline_b_sorted_matmul(x, w1, w2, topk_indices, topk_weights)
        sorted_med, sorted_p95 = measure_cuda_latency(sorted_fn)

        # (c) Triton fused MoE layer
        triton_fn = lambda: baseline_c_triton(x, w1, w2, topk_indices, topk_weights)
        triton_med, triton_p95 = measure_cuda_latency(triton_fn)
        triton_launches = measure_launch_count(triton_fn)

        # (d) Roofline equal-FLOPs single matmul
        # Total MoE FLOPs = 6 * T * K * H * I
        # Equivalent single GEMM: [T * K, H] @ [H, 3 * I] has 2 * (T * K) * H * (3 * I) = 6 * T * K * H * I FLOPs!
        roofline_a = torch.randn(T * K, H, dtype=dtype, device=device)
        roofline_b = torch.randn(H, 3 * I, dtype=dtype, device=device)
        roofline_fn = lambda: torch.matmul(roofline_a, roofline_b)
        roofline_med, roofline_p95 = measure_cuda_latency(roofline_fn)

        speedup_vs_sorted = sorted_med / max(triton_med, 1e-6)

        print(f"{T:<6} | {naive_med:<12.3f} | {sorted_med:<13.3f} | {triton_med:<13.3f} | {roofline_med:<13.3f} | {triton_launches:<12}")

        results.append({
            "num_tokens": T,
            "naive_loop_ms": naive_med,
            "naive_loop_p95_ms": naive_p95,
            "sorted_matmul_ms": sorted_med,
            "sorted_matmul_p95_ms": sorted_p95,
            "triton_ms": triton_med,
            "triton_p95_ms": triton_p95,
            "roofline_cublas_ms": roofline_med,
            "roofline_cublas_p95_ms": roofline_p95,
            "triton_cuda_launches": triton_launches,
            "speedup_vs_sorted": speedup_vs_sorted,
        })

    print("-" * 70)

    # 4. Save results to output_dir
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    report_md_path = os.path.join(output_dir, "BENCHMARK_REPORT.md")

    summary_payload = {
        "gpu_name": device_name,
        "compute_capability": cc_str,
        "torch_version": torch.__version__,
        "total_vram_gb": total_mem_gb,
        "shapes": {
            "hidden_dim": H,
            "moe_intermediate_dim": I,
            "num_experts": E,
            "top_k": K,
            "dtype": str(dtype),
        },
        "single_gemm_ratio_triton_vs_cublas": gemm_ratio,
        "results": results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# DWDP Fused MoE Layer Benchmark Results (Tesla T4)",
        "",
        f"- **GPU**: {device_name} (SM {cc_str})",
        f"- **PyTorch**: {torch.__version__}",
        f"- **Shapes**: Hidden={H}, Intermediate={I}, Experts={E}, Top-k={K}, Dtype=float16",
        f"- **Single GEMM Ratio (Triton/cuBLAS)**: {gemm_ratio:.2f}x",
        "",
        "| Tokens (T) | Naive Loop (a) | Sorted Matmul (b) | Triton Path (c) | Roofline cuBLAS (d) | Launches (c) | Speedup vs (b) |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in results:
        md_lines.append(
            f"| {r['num_tokens']} | {r['naive_loop_ms']:.3f} ms | {r['sorted_matmul_ms']:.3f} ms | "
            f"**{r['triton_ms']:.3f} ms** | {r['roofline_cublas_ms']:.3f} ms | {r['triton_cuda_launches']} | **{r['speedup_vs_sorted']:.2f}x** |"
        )

    with open(report_md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nBenchmark results saved to:")
    print(f"  JSON: {summary_path}")
    print(f"  Markdown: {report_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DWDP MoE layer benchmark.")
    parser.add_argument("--output-dir", type=str, default="benchmarks/results/t4_fused_moe", help="Output directory")
    args = parser.parse_args()
    run_benchmark(output_dir=args.output_dir)
