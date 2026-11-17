#!/usr/bin/env python3
"""Benchmark script for single MoE layer inference on Tesla T4 (sm_75).

Compares FIVE implementations:
  (a) Naive per-expert loop (un-sorted PyTorch loop with boolean indexing)
  (b) Sorted-token per-expert torch.matmul loop (Megablocks-style with per-expert .item() syncs)
  (b2) Sorted-token per-expert torch.matmul loop with single host sync (offsets.tolist() upfront)
  (c) Fused Triton MoE layer (fused SwiGLU + down projection + deterministic sum)
  (d) Single torch.matmul of equal FLOPs (theoretical cuBLAS roofline)

Features:
- Architectural shapes read from config JSON (defaults to configs/qwen1_5_moe_a2_7b.json).
  NOTE: The architecture parameters correspond to Qwen1.5-MoE-A2.7B. The user should verify
  these against the official Hugging Face config (Qwen/Qwen1.5-MoE-A2.7B).
- FP16 only. Pre-flight check ensures weights + activations fit in 16 GB VRAM.
- Accuracy check: verifies Triton layer output against FP32 PyTorch golden reference before timing.
  Raises RuntimeError and terminates if relative error > 1e-2.
- Exact CUDA kernel/memset event counts measured using torch.profiler.
- Per-stage latency breakdown of the Triton path (align, kernel1, kernel2, reduction).
- Single GEMM Triton vs torch.matmul (cuBLAS) ratio measurement.
- Outputs detailed JSON and Markdown summaries including driver and environment metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    triton_version_str = triton.__version__
except ImportError:
    triton = None
    triton_version_str = "N/A"

from DWDP.adapters.extractor import QwenMoEConfig, parse_qwen_moe_config
from DWDP.moe.align import align_tokens_by_expert
from DWDP.moe.kernels import launch_down_proj, launch_fused_swiglu
from DWDP.moe.layer import triton_moe_layer
from DWDP.moe.reference import moe_reference_grouped


def get_driver_version() -> str:
    """Retrieve NVIDIA driver version if available."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
        return res
    except Exception:
        return "N/A"


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
    """(b) Sorted-token per-expert torch.matmul loop (repeated .item() host syncs)."""
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
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=x.device).scatter_add_(
        0, sorted_expert_ids, torch.ones_like(sorted_expert_ids)
    )
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


def baseline_b2_sorted_matmul_single_sync(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """(b2) Sorted-token per-expert torch.matmul loop (single host sync).

    Copies offsets to CPU once via .tolist(), avoiding 2 * E device-to-host .item() syncs.
    """
    num_tokens, hidden_dim = x.shape
    num_experts, double_intermediate_dim, _ = w1.shape
    intermediate_dim = double_intermediate_dim // 2
    top_k = topk_indices.shape[1]

    flat_expert_ids = topk_indices.reshape(-1)
    flat_weights = topk_weights.reshape(-1)
    token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(top_k)

    sorted_expert_ids, sort_perm = torch.sort(flat_expert_ids)
    sorted_tokens = token_indices[sort_perm]
    sorted_weights = flat_weights[sort_perm].unsqueeze(1)
    x_sorted = x[sorted_tokens]
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=x.device).scatter_add_(
        0, sorted_expert_ids, torch.ones_like(sorted_expert_ids)
    )
    offsets = torch.zeros(num_experts + 1, dtype=torch.long, device=x.device)
    offsets[1:] = torch.cumsum(expert_counts, dim=0)

    # Single host-device sync
    offsets_cpu = offsets.tolist()

    out_sorted = torch.zeros_like(x_sorted)

    for e in range(num_experts):
        start = offsets_cpu[e]
        end = offsets_cpu[e + 1]
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


def measure_cuda_event_count(fn: Callable[[], torch.Tensor]) -> int:
    """Count real CUDA kernel/memset events using torch.profiler."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        fn()
    torch.cuda.synchronize()

    events = 0
    for event in prof.events():
        if event.device_type == torch.profiler.DeviceType.CUDA:
            events += 1
    return events


def measure_triton_breakdown(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    warmup: int = 5,
    timed: int = 20,
) -> Dict[str, float]:
    """Measure per-stage CUDA latency breakdown (ms) of the Triton path."""
    num_tokens, hidden_dim = x.shape
    num_experts, double_intermediate_dim, _ = w1.shape
    intermediate_dim = double_intermediate_dim // 2
    top_k = topk_indices.shape[1]
    dtype = x.dtype
    device = x.device

    for _ in range(warmup):
        triton_moe_layer(x, w1, w2, topk_indices, topk_weights)
    torch.cuda.synchronize()

    ev_start = torch.cuda.Event(enable_timing=True)
    ev_after_align = torch.cuda.Event(enable_timing=True)
    ev_after_k1 = torch.cuda.Event(enable_timing=True)
    ev_after_k2 = torch.cuda.Event(enable_timing=True)
    ev_after_red = torch.cuda.Event(enable_timing=True)

    align_times = []
    k1_times = []
    k2_times = []
    red_times = []

    for _ in range(timed):
        ev_start.record()
        aligned = align_tokens_by_expert(topk_indices, topk_weights, num_experts, block_size=32)
        ev_after_align.record()

        # Zero-sync buffer allocation using static upper bound
        inter_act = torch.zeros((aligned.max_padded_tokens, intermediate_dim), dtype=dtype, device=device)
        partial_outputs = torch.zeros((num_tokens * top_k, hidden_dim), dtype=dtype, device=device)

        launch_fused_swiglu(
            hidden_states=x,
            w1=w1,
            inter_act=inter_act,
            aligned=aligned,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        ev_after_k1.record()

        launch_down_proj(
            inter_act=inter_act,
            w2=w2,
            partial_outputs=partial_outputs,
            aligned=aligned,
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
        )
        ev_after_k2.record()

        _ = partial_outputs.view(num_tokens, top_k, hidden_dim).sum(dim=1)
        ev_after_red.record()
        torch.cuda.synchronize()

        align_times.append(ev_start.elapsed_time(ev_after_align))
        k1_times.append(ev_after_align.elapsed_time(ev_after_k1))
        k2_times.append(ev_after_k1.elapsed_time(ev_after_k2))
        red_times.append(ev_after_k2.elapsed_time(ev_after_red))

    align_times.sort()
    k1_times.sort()
    k2_times.sort()
    red_times.sort()

    return {
        "align_ms": align_times[len(align_times) // 2],
        "kernel1_ms": k1_times[len(k1_times) // 2],
        "kernel2_ms": k2_times[len(k2_times) // 2],
        "reduction_ms": red_times[len(red_times) // 2],
    }


def measure_single_gemm_ratio(m: int, k: int, n: int, device: torch.device) -> Tuple[float, float, float]:
    """Measure single GEMM latency: Triton vs torch.matmul (cuBLAS)."""
    if triton is None:
        return 0.0, 0.0, 1.0

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

def run_benchmark(
    config_path: str = "configs/qwen1_5_moe_a2_7b.json",
    small_model: bool = False,
    output_dir: str = "benchmarks/results/t4_fused_moe",
) -> None:
    cuda_available = torch.cuda.is_available()

    # NOTE: The architecture parameters here correspond to Qwen1.5-MoE-A2.7B.
    # The user should verify these against the official Hugging Face config (Qwen/Qwen1.5-MoE-A2.7B).
    if small_model:
        raw_config = {
            "hidden_size": 64,
            "moe_intermediate_size": 32,
            "num_experts": 8,
            "top_k": 2,
            "num_layers": 2,
            "num_kv_heads": 2,
            "head_dim": 32,
        }
        cfg: QwenMoEConfig = parse_qwen_moe_config(raw_config)
    else:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                raw_config = json.load(f)
        else:
            raw_config = {
                "hidden_size": 2048,
                "moe_intermediate_size": 1408,
                "num_experts": 64,
                "top_k": 4,
                "num_layers": 24,
                "num_kv_heads": 16,
                "head_dim": 128,
            }
        cfg = parse_qwen_moe_config(raw_config)

    H = cfg.hidden_size
    I = cfg.moe_intermediate_size
    E = cfg.num_experts
    K = cfg.top_k
    dtype = torch.float16

    token_counts = [1, 8, 64, 512, 4096] if not small_model else [1, 8, 16]

    driver_version = get_driver_version()
    cuda_version = getattr(torch.version, "cuda", "N/A")

    print("=" * 80)
    print("DWDP FUSED TRITON MoE LAYER BENCHMARK (TESLA T4 / sm_75)")
    print("=" * 80)
    print(f"Model Shapes: hidden_dim={H}, moe_intermediate_dim={I}, num_experts={E}, top_k={K}")
    print(f"Data Type: {dtype}")
    print(f"PyTorch Version: {torch.__version__} (CUDA {cuda_version})")
    print(f"Triton Version: {triton_version_str}")
    print(f"NVIDIA Driver Version: {driver_version}")

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
            "driver_version": "TBD (run on T4)",
            "torch_version": torch.__version__,
            "torch_cuda_version": cuda_version,
            "triton_version": triton_version_str,
            "status": "AWAITING_GPU_EXECUTION",
            "token_counts": token_counts,
        }
        with open(json_path, "w") as f:
            json.dump(skeleton_data, f, indent=2)

        md_content = f"""# DWDP Fused MoE Layer Benchmark (Tesla T4)

*Status: Awaiting GPU Execution on Tesla T4*

| Tokens (T) | Naive Loop (a) | Sorted Loop (b) | Sorted Single-Sync (b2) | Triton Path (c) | Roofline cuBLAS (d) | CUDA Events (c) | Rel Err vs Ref | Speedup vs (b2) |
|---|---|---|---|---|---|---|---|---|
"""
        for t in token_counts:
            md_content += f"| {t} | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) |\n"

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

    print("-" * 115)
    print(f"{'T':<6} | {'Naive (a)':<11} | {'Sorted (b)':<11} | {'Sync (b2)':<11} | {'Triton (c)':<11} | {'cuBLAS (d)':<11} | {'Events (c)':<11} | {'Rel Err':<9} | {'Speedup (b2)':<12}")
    print("-" * 115)

    for T in token_counts:
        torch.manual_seed(100 + T)
        x = torch.randn(T, H, dtype=dtype, device=device)
        topk_indices = torch.randint(0, E, (T, K), dtype=torch.int64, device=device)
        topk_weights = torch.softmax(torch.randn(T, K, device=device), dim=-1).half()

        # Accuracy verification before timing: compare Triton against FP32 golden reference
        triton_out = baseline_c_triton(x, w1, w2, topk_indices, topk_weights)
        ref_fp32 = moe_reference_grouped(
            x.float(), w1.float(), w2.float(), topk_indices, topk_weights.float()
        )
        diff = triton_out.float() - ref_fp32.float()
        ref_norm = torch.linalg.norm(ref_fp32.float()).item()
        rel_err = (torch.linalg.norm(diff).item() / ref_norm) if ref_norm > 0 else 0.0

        if rel_err > 1e-2:
            raise RuntimeError(
                f"Triton layer output relative error ({rel_err:.6e}) exceeds 1e-2 "
                f"against FP32 PyTorch reference at T={T}!"
            )

        # (a) Naive per-expert loop
        naive_fn = lambda: baseline_a_naive_loop(x, w1, w2, topk_indices, topk_weights)
        naive_med, naive_p95 = measure_cuda_latency(naive_fn)

        # (b) Sorted-token per-expert matmul (with per-expert item syncs)
        sorted_fn = lambda: baseline_b_sorted_matmul(x, w1, w2, topk_indices, topk_weights)
        sorted_med, sorted_p95 = measure_cuda_latency(sorted_fn)

        # (b2) Sorted-token per-expert matmul (single host sync)
        sorted_b2_fn = lambda: baseline_b2_sorted_matmul_single_sync(x, w1, w2, topk_indices, topk_weights)
        sorted_b2_med, sorted_b2_p95 = measure_cuda_latency(sorted_b2_fn)

        # (c) Triton fused MoE layer
        triton_fn = lambda: baseline_c_triton(x, w1, w2, topk_indices, topk_weights)
        triton_med, triton_p95 = measure_cuda_latency(triton_fn)
        triton_events = measure_cuda_event_count(triton_fn)

        # (d) Roofline equal-FLOPs single matmul
        roofline_a = torch.randn(T * K, H, dtype=dtype, device=device)
        roofline_b = torch.randn(H, 3 * I, dtype=dtype, device=device)
        roofline_fn = lambda: torch.matmul(roofline_a, roofline_b)
        roofline_med, roofline_p95 = measure_cuda_latency(roofline_fn)

        # Triton per-stage breakdown
        breakdown = measure_triton_breakdown(x, w1, w2, topk_indices, topk_weights)

        speedup_vs_b = sorted_med / max(triton_med, 1e-6)
        speedup_vs_b2 = sorted_b2_med / max(triton_med, 1e-6)

        print(
            f"{T:<6} | {naive_med:<11.3f} | {sorted_med:<11.3f} | {sorted_b2_med:<11.3f} | "
            f"{triton_med:<11.3f} | {roofline_med:<11.3f} | {triton_events:<11} | {rel_err:<9.2e} | {speedup_vs_b2:<12.2f}x"
        )

        results.append({
            "num_tokens": T,
            "reference_relative_error": rel_err,
            "naive_loop_ms": naive_med,
            "naive_loop_p95_ms": naive_p95,
            "sorted_matmul_ms": sorted_med,
            "sorted_matmul_p95_ms": sorted_p95,
            "sorted_matmul_b2_ms": sorted_b2_med,
            "sorted_matmul_b2_p95_ms": sorted_b2_p95,
            "triton_ms": triton_med,
            "triton_p95_ms": triton_p95,
            "roofline_cublas_ms": roofline_med,
            "roofline_cublas_p95_ms": roofline_p95,
            "cuda_kernel_memset_events": triton_events,
            "speedup_vs_sorted_b": speedup_vs_b,
            "speedup_vs_sorted_b2": speedup_vs_b2,
            "breakdown_ms": breakdown,
        })

    print("-" * 115)

    # 4. Save results to output_dir
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    report_md_path = os.path.join(output_dir, "BENCHMARK_REPORT.md")

    summary_payload = {
        "gpu_name": device_name,
        "compute_capability": cc_str,
        "driver_version": driver_version,
        "torch_version": torch.__version__,
        "torch_cuda_version": cuda_version,
        "triton_version": triton_version_str,
        "total_vram_gb": total_mem_gb,
        "config_path": config_path,
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
        f"- **Driver Version**: {driver_version}",
        f"- **PyTorch**: {torch.__version__} (CUDA {cuda_version})",
        f"- **Triton**: {triton_version_str}",
        f"- **Shapes**: Hidden={H}, Intermediate={I}, Experts={E}, Top-k={K}, Dtype=float16",
        f"- **Single GEMM Ratio (Triton/cuBLAS)**: {gemm_ratio:.2f}x",
        "",
        "## Overall Latency & Scaling",
        "",
        "| Tokens (T) | Naive Loop (a) | Sorted Loop (b) | Sorted Single-Sync (b2) | Triton Path (c) | Roofline cuBLAS (d) | CUDA Kernel/Memset Events (c) | Rel Err vs Ref | Speedup vs (b2) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        md_lines.append(
            f"| {r['num_tokens']} | {r['naive_loop_ms']:.3f} ms | {r['sorted_matmul_ms']:.3f} ms | "
            f"{r['sorted_matmul_b2_ms']:.3f} ms | **{r['triton_ms']:.3f} ms** | {r['roofline_cublas_ms']:.3f} ms | "
            f"{r['cuda_kernel_memset_events']} | {r['reference_relative_error']:.2e} | **{r['speedup_vs_sorted_b2']:.2f}x** |"
        )

    md_lines.extend([
        "",
        "## Triton Path Stage Breakdown (CUDA Events)",
        "",
        "| Tokens (T) | Alignment (ms) | Kernel 1 SwiGLU (ms) | Kernel 2 Down (ms) | Reduction (ms) | Total Triton (ms) |",
        "|---|---|---|---|---|---|",
    ])

    for r in results:
        b = r["breakdown_ms"]
        md_lines.append(
            f"| {r['num_tokens']} | {b['align_ms']:.3f} ms | {b['kernel1_ms']:.3f} ms | "
            f"{b['kernel2_ms']:.3f} ms | {b['reduction_ms']:.3f} ms | **{r['triton_ms']:.3f} ms** |"
        )

    with open(report_md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nBenchmark results saved to:")
    print(f"  JSON: {summary_path}")
    print(f"  Markdown: {report_md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DWDP MoE layer benchmark.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/t4_fused_moe",
        help="Output directory for benchmark reports",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen1_5_moe_a2_7b.json",
        help="Path to MoE architectural config JSON (defaults to configs/qwen1_5_moe_a2_7b.json)",
    )
    parser.add_argument(
        "--small-model",
        action="store_true",
        help="Use small model dimensions (H=64, I=32, E=8, K=2) for fast local/CPU verification",
    )
    args = parser.parse_args()
    run_benchmark(
        config_path=args.config,
        small_model=args.small_model,
        output_dir=args.output_dir,
    )
