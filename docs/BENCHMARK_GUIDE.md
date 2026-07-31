# DWDP FP8 Benchmarking & Profiling Guide

This guide describes how to run automated, end-to-end **FP8 benchmarking and profiling** for Mixture-of-Experts (MoE) models using the DWDP inference engine.

---

## Quick Start (Single Command)

To run the complete benchmark, profiling pass, and output packaging in one command:

```bash
bash scripts/run_all_benchmarks.sh
```

### What this script automatically does:
1. **Environment Setup**: Validates CUDA GPU capabilities and installs required dependencies (`torch`, `triton`, `transformers`, `accelerate`, `bitsandbytes`, `safetensors`, `zip`).
2. **Editable Installation**: Installs `DWDP` in editable mode (`pip install -e .`).
3. **FP8 Model Execution**: Loads `Qwen/Qwen1.5-MoE-A2.7B` and runs native FP8 execution (`--quantization fp8`).
4. **Comprehensive Profiling**: Captures:
   - **Prefill Latency** (prompt encoding time)
   - **Time to First Token (TTFT)**
   - **Decode Throughput** ($\text{tokens/sec}$)
   - **Peak VRAM Allocation**
   - **Torch Profiler Operator Breakdown** (categorizing router, dispatcher, scheduler, gather, and GEMM kernel times)
5. **Artifact Zip & Packaging**: Creates a timestamped `.zip` archive (e.g., `DWDP_FP8_Benchmark_Results_2026-07-31_10-00-00.zip`) in the project root directory and copies the latest markdown report to `results/latest_fp8_report.md`.

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
   - Main Zip file: `DWDP_FP8_Benchmark_Results_<timestamp>.zip` (in root directory).
   - Markdown report: `results/latest_fp8_report.md`.
2. **Download Artifact**:
   - Download the generated `.zip` file from your cloud/Colab environment to your local machine.
3. **Commit & Push to Git**:
   ```bash
   git add results/
   git commit -m "docs: add FP8 benchmark & profiling results for Qwen1.5-MoE"
   git push origin main
   ```

---

## Editing & Customizing `scripts/run_all_benchmarks.sh`

If you need to add custom metric logging or modify default options inside `scripts/run_all_benchmarks.sh`:

- **Lines 17–23**: Update default values for `MODEL`, `QUANT`, `BATCH_SIZE`, `SEQ_LEN`, etc.
- **Lines 75–88**: Customize command-line arguments passed to `scripts/benchmark_colab.py`.
- **Lines 98–108**: Customize zip archiving behavior or export paths.
