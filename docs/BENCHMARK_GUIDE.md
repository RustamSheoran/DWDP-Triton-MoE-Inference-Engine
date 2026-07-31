# DWDP FP8 Benchmarking & Profiling Guide

This guide describes how to run automated, end-to-end **FP8 benchmarking and profiling** for Mixture-of-Experts (MoE) models using the DWDP inference engine.

---

## Quick Start (Single Command)

To run the complete benchmark, profiling pass, and output packaging in one command:

```bash
bash scripts/run_all_benchmarks.sh
```

### What this script automatically does:
1. **Environment & Memory Pre-flight**: Inspects GPU VRAM (`nvidia-smi` / `torch.cuda`) and calculates total memory footprint:
   $$\text{Required VRAM} = \text{Model Parameters Memory} + \text{KV Cache Footprint} + \text{Workspace Buffer (1.5 GB)}$$
   where KV cache footprint is calculated as:
   $$\text{KV Cache} = 2 \times \text{layers} \times \text{heads} \times \text{head-dim} \times (\text{seq-len} + \text{max-new-tokens}) \times \text{batch-size} \times 2\text{ bytes}$$
2. **Dynamic Auto-Precision Selection**:
   - If FP8 fits in VRAM and CUDA capability $\ge 8.9$ (Ada/Hopper) with E4M3 exposed $\rightarrow$ runs **FP8 (E4M3)**.
   - If FP8 exceeds available VRAM or hardware capability $< 8.9$ $\rightarrow$ automatically switches to **4-bit (NF4 / NVFP4)** to guarantee zero Out-Of-Memory (OOM) failures.
3. **Editable Installation**: Installs `DWDP` in editable mode (`pip install -e .`).
4. **Comprehensive Profiling**: Captures:
   - **Prefill Latency** (prompt encoding time)
   - **Time to First Token (TTFT)**
   - **Decode Throughput** ($\text{tokens/sec}$)
   - **Peak VRAM Allocation**
   - **Torch Profiler Operator Breakdown** (categorizing router, dispatcher, scheduler, gather, and FP8 GEMM kernel times)
5. **Artifact Zip & Packaging**: Bundles all outputs into `DWDP_<e4m3|4bit|fp16>_Benchmark_Results_<timestamp>.zip` directly in the root directory reflecting the executed precision.


---

## Customizing for Different GPUs and Models

The master launcher script reads configuration from environment variables, allowing you to easily scale to different hardware and model architectures without modifying the script code.

### 1. Scaling to Different Models
To test a different MoE or dense model, set the `MODEL` variable:

```bash
# Benchmark Mixtral 8x7B
MODEL="mistralai/Mixtral-8x7B-v0.1" bash scripts/run_all_benchmarks.sh

# Benchmark Qwen2.5-1.5B
MODEL="Qwen/Qwen2.5-1.5B" bash scripts/run_all_benchmarks.sh
```

### 2. Customizing Batch Size, Sequence Length, & Token Counts
To test high-concurrency or long-context scenarios:

```bash
BATCH_SIZE=4 SEQ_LEN=512 MAX_NEW_TOKENS=256 bash scripts/run_all_benchmarks.sh
```

### 3. Hardware-Specific Scaling

#### Tesla T4 (Default / Colab Free Tier)
- **Optimal Settings**:
  ```bash
  MODEL="Qwen/Qwen1.5-MoE-A2.7B" QUANT="fp8" BATCH_SIZE=1 SEQ_LEN=128 MAX_NEW_TOKENS=128 bash scripts/run_all_benchmarks.sh
  ```

#### NVIDIA L4 / A10G (Single GPU Cloud)
- **Optimal Settings**:
  ```bash
  MODEL="Qwen/Qwen1.5-MoE-A2.7B" QUANT="fp8" BATCH_SIZE=4 SEQ_LEN=512 MAX_NEW_TOKENS=256 ITERS=50 bash scripts/run_all_benchmarks.sh
  ```

#### NVIDIA A100 / H100 / H200 (Frontier Hardware)
- **Optimal Settings**:
  ```bash
  MODEL="Qwen/Qwen1.5-MoE-A2.7B" QUANT="fp8" BATCH_SIZE=16 SEQ_LEN=2048 MAX_NEW_TOKENS=512 ITERS=100 bash scripts/run_all_benchmarks.sh
  ```

---

## Workflow for Downloading & Git Pushing Benchmark Results

After running `bash scripts/run_all_benchmarks.sh`:

1. **Locate Artifacts**:
   - Main Zip file: `DWDP_<e4m3|4bit|fp16>_Benchmark_Results_<timestamp>.zip` (in root directory).
   - Timestamped folder: `benchmark-results/<e4m3|4bit|fp16>_run_<timestamp>/`.
2. **Download Artifact**:
   - Download the generated `.zip` file from your cloud/Colab environment to your local machine.
3. **Commit & Push to Git**:
   ```bash
   git add benchmark-results/
   git commit -m "docs: add FP8 benchmark & profiling results for Qwen1.5-MoE"
   git push origin main
   ```


---

## Editing & Customizing `scripts/run_all_benchmarks.sh`

If you need to add custom metric logging or modify default options inside `scripts/run_all_benchmarks.sh`:

- **Lines 17–23**: Update default values for `MODEL`, `QUANT`, `BATCH_SIZE`, `SEQ_LEN`, etc.
- **Lines 75–88**: Customize command-line arguments passed to `scripts/benchmark_colab.py`.
- **Lines 98–108**: Customize zip archiving behavior or export paths.
