#!/usr/bin/env python3
"""Diagnostic script for Triton GEMM performance and Tensor Core utilization on Tesla T4 (sm_75).

Investigates:
1. FP16 TFLOP/s of torch.matmul (cuBLAS) vs a standard Triton GEMM autotuned across
   various block sizes (64x64x32, 128x64x32, 128x128x32, 128x256x32), warps (4, 8),
   and pipeline stages (2, 3, 4).
2. PTX instruction analysis:
   - Dumps PTX of the winning Triton GEMM configuration.
   - Inspects .target architecture directive.
   - Counts hardware Tensor Core instructions (mma.sync) vs scalar ALU operations (fma.rn).
   - If mma.sync == 0, Tensor Cores are NOT being utilized by Triton on sm_75.
3. PTX instruction analysis for DWDP's _fused_swiglu_kernel.
4. Emits a detailed text report in the results directory.
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

from DWDP.moe.align import align_tokens_by_expert
from DWDP.moe.kernels import _fused_swiglu_kernel, launch_fused_swiglu


# ---------------------------------------------------------------------------
# Standard Autotuned Triton GEMM
# ---------------------------------------------------------------------------

if triton is not None:
    DIAG_GEMM_CONFIGS = [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 32}, num_warps=w, num_stages=s)
        for (bm, bn) in [(64, 64), (128, 64), (128, 128), (128, 256)]
        for w in [4, 8]
        for s in [2, 3, 4]
    ]

    @triton.autotune(
        configs=DIAG_GEMM_CONFIGS,
        key=["M", "N", "K"],
    )
    @triton.jit
    def _diag_gemm_kernel(
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

        for k in range(0, K, BLOCK_K):
            k_offs = k + offs_k
            a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

            a = tl.load(a_ptr + offs_m[:, None] * K + k_offs[None, :], mask=a_mask, other=0.0)
            b = tl.load(b_ptr + k_offs[:, None] * N + offs_n[None, :], mask=b_mask, other=0.0)

            acc += tl.dot(a, b)

        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=c_mask)


# ---------------------------------------------------------------------------
# Timing & PTX Helpers
# ---------------------------------------------------------------------------

def measure_cuda_latency(fn: Callable[[], Any], warmup: int = 10, timed: int = 50) -> float:
    """Measure median CUDA latency in milliseconds."""
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
    return latencies[len(latencies) // 2]


def extract_ptx_from_jit(fn: Any, *args: Any, grid: Any = (1, 1), **kwargs: Any) -> str:
    """Extract compiled PTX source code from a Triton JIT kernel."""
    target_fn = getattr(fn, "fn", fn)

    # 1. Warmup call
    try:
        compiled = target_fn.warmup(*args, grid=grid, **kwargs)
        if hasattr(compiled, "asm") and isinstance(compiled.asm, dict) and "ptx" in compiled.asm:
            return compiled.asm["ptx"]
    except Exception:
        pass

    # 2. Inspect device_caches
    if hasattr(target_fn, "device_caches"):
        for dev_cache in target_fn.device_caches.values():
            for compiled in dev_cache.values():
                if hasattr(compiled, "asm") and isinstance(compiled.asm, dict) and "ptx" in compiled.asm:
                    return compiled.asm["ptx"]

    # 3. Disk cache
    cache_files = glob.glob(os.path.expanduser("~/.triton/cache/*/*.ptx"))
    if cache_files:
        latest = max(cache_files, key=os.path.getmtime)
        try:
            with open(latest, "r") as f:
                return f.read()
        except Exception:
            pass

    return ""


def analyze_ptx(ptx_str: str) -> Dict[str, Any]:
    """Parse PTX string to identify target architecture and instruction distribution."""
    lines = ptx_str.splitlines()
    target = "UNKNOWN"
    mma_sync_count = 0
    fma_rn_count = 0
    total_inst = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith(".target"):
            target = stripped
        if "mma.sync" in stripped:
            mma_sync_count += 1
        if "fma.rn" in stripped:
            fma_rn_count += 1
        if stripped and not stripped.startswith("//") and not stripped.startswith(".") and not stripped.endswith(":"):
            total_inst += 1

    return {
        "target": target,
        "mma_sync_count": mma_sync_count,
        "fma_rn_count": fma_rn_count,
        "total_instructions": total_inst,
        "tensor_cores_active": mma_sync_count > 0,
    }


# ---------------------------------------------------------------------------
# Main Diagnostic Routine
# ---------------------------------------------------------------------------

def run_diagnostics(output_dir: str = "benchmarks/results/t4_fused_moe") -> None:
    lines: List[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log("=" * 80)
    log("DWDP TRITON GEMM & TENSOR CORE INSTRUCTION DIAGNOSTIC (TESLA T4)")
    log("=" * 80)

    cuda_available = torch.cuda.is_available()
    cuda_version = getattr(torch.version, "cuda", "N/A")
    triton_ver = getattr(triton, "__version__", "N/A")

    log(f"PyTorch Version: {torch.__version__} (CUDA {cuda_version})")
    log(f"Triton Version: {triton_ver}")

    if not cuda_available:
        log("\n[WARNING] CUDA is NOT available on this system.")
        log("Cannot perform compiled PTX inspection or GPU FLOPs measurement.")
        log("Run this script on a Tesla T4 GPU using: bash scripts/run_on_gpu.sh --diagnose")

        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "triton_gemm_diagnosis.txt")
        with open(out_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        return

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    cc_str = f"{cap[0]}.{cap[1]}"

    log(f"GPU Device: {device_name} (Compute Capability {cc_str})")

    # Retrieve driver version
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL
        ).strip().splitlines()[0]
    except Exception:
        driver = "N/A"
    log(f"NVIDIA Driver: {driver}")
    log("")

    # 1. Benchmark Shapes
    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (1024, 2048, 1408),  # Qwen MoE shape
    ]

    log("-" * 80)
    log("SECTION 1: FP16 TFLOP/s BENCHMARK (torch.matmul vs Autotuned Triton GEMM)")
    log("-" * 80)
    log(f"{'Shape (M x K @ K x N)':<28} | {'cuBLAS ms':<10} | {'cuBLAS TF/s':<12} | {'Triton ms':<10} | {'Triton TF/s':<12} | {'Ratio':<8}")
    log("-" * 80)

    gemm_ptx = ""
    last_m, last_k, last_n = 1024, 2048, 1408

    for M, K, N in shapes:
        a = torch.randn(M, K, dtype=torch.float16, device=device)
        b = torch.randn(K, N, dtype=torch.float16, device=device)
        c_triton = torch.zeros(M, N, dtype=torch.float16, device=device)

        total_flops = 2.0 * M * N * K

        # (a) cuBLAS
        cublas_fn = lambda: torch.matmul(a, b)
        cublas_ms = measure_cuda_latency(cublas_fn, warmup=10, timed=50)
        cublas_tflops = (total_flops / (cublas_ms * 1e-3)) / 1e12

        # (b) Triton GEMM (Autotuned)
        def run_triton():
            grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))
            _diag_gemm_kernel[grid](a, b, c_triton, M, N, K)
            return c_triton

        triton_ms = measure_cuda_latency(run_triton, warmup=10, timed=50)
        triton_tflops = (total_flops / (triton_ms * 1e-3)) / 1e12

        ratio = triton_ms / max(cublas_ms, 1e-6)
        shape_str = f"[{M}x{K} @ {K}x{N}]"
        log(f"{shape_str:<28} | {cublas_ms:<10.3f} | {cublas_tflops:<12.2f} | {triton_ms:<10.3f} | {triton_tflops:<12.2f} | {ratio:<8.2f}x")

    log("-" * 80)

    # 2. Extract and Analyze PTX from Standard Triton GEMM
    log("")
    log("-" * 80)
    log("SECTION 2: PTX INSTRUCTION ANALYSIS - STANDARD TRITON GEMM")
    log("-" * 80)

    a_sample = torch.randn(1024, 2048, dtype=torch.float16, device=device)
    b_sample = torch.randn(2048, 1408, dtype=torch.float16, device=device)
    c_sample = torch.zeros(1024, 1408, dtype=torch.float16, device=device)
    grid_sample = lambda META: (triton.cdiv(1024, META["BLOCK_M"]), triton.cdiv(1408, META["BLOCK_N"]))

    _diag_gemm_kernel[grid_sample](a_sample, b_sample, c_sample, 1024, 1408, 2048)
    torch.cuda.synchronize()

    gemm_ptx = extract_ptx_from_jit(_diag_gemm_kernel, a_sample, b_sample, c_sample, 1024, 1408, 2048, grid=(16, 22))
    gemm_diag = analyze_ptx(gemm_ptx)

    log(f"Architecture Target Directive : {gemm_diag['target']}")
    log(f"Tensor Core 'mma.sync' Count   : {gemm_diag['mma_sync_count']}")
    log(f"Scalar ALU 'fma.rn' Count     : {gemm_diag['fma_rn_count']}")
    log(f"Total Disassembled PTX Lines  : {gemm_diag['total_instructions']}")
    log(f"Hardware Tensor Cores Active  : {gemm_diag['tensor_cores_active']}")

    if gemm_diag["mma_sync_count"] == 0:
        log("\n>>> ROOT CAUSE IDENTIFIED:")
        log(">>> Triton generated ZERO 'mma.sync' instructions on sm_75!")
        log(">>> The kernel is compiling to scalar FMA loops instead of hardware Tensor Cores.")
    else:
        log("\n>>> Tensor Core instructions ('mma.sync') successfully generated by Triton.")

    # 3. Extract and Analyze PTX from Fused SwiGLU Kernel
    log("")
    log("-" * 80)
    log("SECTION 3: PTX INSTRUCTION ANALYSIS - DWDP _fused_swiglu_kernel")
    log("-" * 80)

    swiglu_H = 2048
    swiglu_I = 1408
    swiglu_E = 64
    swiglu_T = 16
    swiglu_K = 4

    swiglu_x = torch.randn(swiglu_T, swiglu_H, dtype=torch.float16, device=device)
    swiglu_w1 = torch.randn(swiglu_E, 2 * swiglu_I, swiglu_H, dtype=torch.float16, device=device) * 0.02
    swiglu_topk_idx = torch.randint(0, swiglu_E, (swiglu_T, swiglu_K), dtype=torch.int64, device=device)
    swiglu_topk_w = torch.rand(swiglu_T, swiglu_K, dtype=torch.float16, device=device)
    aligned_sample = align_tokens_by_expert(swiglu_topk_idx, swiglu_topk_w, swiglu_E, block_size=32)
    inter_sample = torch.zeros(aligned_sample.max_padded_tokens, swiglu_I, dtype=torch.float16, device=device)

    launch_fused_swiglu(swiglu_x, swiglu_w1, inter_sample, aligned_sample, swiglu_H, swiglu_I)
    torch.cuda.synchronize()

    swiglu_ptx = extract_ptx_from_jit(
        _fused_swiglu_kernel,
        swiglu_x,
        swiglu_w1,
        inter_sample,
        aligned_sample.sorted_token_ids,
        aligned_sample.expert_ids_per_block,
        aligned_sample.valid_mask,
        aligned_sample.num_tokens_post_padded,
        hidden_dim=swiglu_H,
        intermediate_dim=swiglu_I,
        BLOCK_M=32,
        grid=(aligned_sample.num_blocks_ub, triton.cdiv(swiglu_I, 32)),
    )
    swiglu_diag = analyze_ptx(swiglu_ptx)

    log(f"Architecture Target Directive : {swiglu_diag['target']}")
    log(f"Tensor Core 'mma.sync' Count   : {swiglu_diag['mma_sync_count']}")
    log(f"Scalar ALU 'fma.rn' Count     : {swiglu_diag['fma_rn_count']}")
    log(f"Total Disassembled PTX Lines  : {swiglu_diag['total_instructions']}")
    log(f"Hardware Tensor Cores Active  : {swiglu_diag['tensor_cores_active']}")

    if swiglu_diag["mma_sync_count"] == 0:
        log("\n>>> ROOT CAUSE CONFIRMED FOR DWDP FUSED KERNEL:")
        log(">>> _fused_swiglu_kernel contains ZERO 'mma.sync' instructions!")
        log(">>> tl.dot on sm_75 with this Triton version does not utilize Turing Tensor Cores.")
    else:
        log("\n>>> Hardware Tensor Cores actively utilized in _fused_swiglu_kernel.")

    log("-" * 80)

    # 4. Write full report to file
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, "triton_gemm_diagnosis.txt")
    with open(report_file, "w") as f:
        f.write("\n".join(lines) + "\n")

    log(f"\nDiagnostic report saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Triton GEMM PTX and TFLOP/s diagnostic.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/t4_fused_moe",
        help="Output directory for diagnostic report",
    )
    args = parser.parse_args()
    run_diagnostics(output_dir=args.output_dir)
