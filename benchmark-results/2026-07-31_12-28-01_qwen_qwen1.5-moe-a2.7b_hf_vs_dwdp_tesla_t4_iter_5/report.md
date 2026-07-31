# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T12:28:01.941651+00:00`

# Environment

| Field | Value |
| --- | --- |
| GPU | Tesla T4 |
| GPU Memory | 15636037632 |
| CUDA | 12.8 |
| cuDNN | 91002 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.57.6 |
| Triton | 3.6.0 |
| NVIDIA Driver | 580.159.04 |
| Python | 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0] |
| OS | Linux-6.12.90+-x86_64-with-glibc2.35 |
| Git Commit | ea8946621c1cf528d2eacb8efe2445a2a057bdd7 |
| Git Branch | main |
| Runtime Backend | dwdp_reference |
| Precision | fp8 |
| Torch Compile | False |

# Configuration

| Field | Value |
| --- | --- |
| Prompt | `Explain the architecture of Mixture of Experts in deep learning.` |
| Batch Size | 2 |
| Sequence Length | 128 |
| Max New Tokens | 128 |
| Temperature | 1.0 |
| Top-k | N/A |
| Top-p | N/A |
| DType | float16_compute |
| Device | cuda |
| Random Seed | 0 |
| Workspace | yes |

# Performance Results

| Backend | TTFT ms | Prefill ms | Decode ms | Tokens/s | Total ms | Peak GPU bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hf | 1031.3312 | 1065.0189 | 17269.3096 | 6.9943 | 18300.6408 | 3681784832 |
| dwdp | 979.5490 | 1036.3295 | 21425.1926 | 5.7131 | 22404.7416 | 6796329472 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1031.3312 | 979.5490 | -5.02% |
| Prefill ms | 1065.0189 | 1036.3295 | -2.69% |
| Decode ms | 17269.3096 | 21425.1926 | +24.07% |
| Tokens/s | 6.9943 | 5.7131 | -18.32% |
| Total latency ms | 18300.6408 | 22404.7416 | +22.43% |
| Peak GPU memory bytes | 3681784832 | 6796329472 | +84.59% |

**Summary:** DWDP is 22.43% slower than native HF by end-to-end latency.
DWDP throughput is -18.32% versus native HF.

# Runtime Breakdown

| Module | Latency ms | Percentage |
| --- | ---: | ---: |
| Router | 0.0000 | N/A |
| Dispatcher | 0.0000 | N/A |
| Scheduler | 0.0000 | N/A |
| Comms Planner | 0.0000 | N/A |
| Executor | 0.0000 | N/A |
| Merger | 0.0000 | N/A |
| Total DWDP Overhead | N/A | N/A |

# Correctness Validation

| Metric | Value |
| --- | --- |
| Maximum Absolute Error | N/A |
| Mean Absolute Error | N/A |
| Relative Error | N/A |
| Cosine Similarity | N/A |
| torch.allclose | N/A |
| Generated Token Parity | no |
| Layer Output Parity | N/A |
| Router Output Parity | N/A |
| Executor Output Parity | N/A |
| Merger Output Parity | N/A |

# Memory Usage

| Backend | Peak GPU Bytes | Average GPU Bytes |
| --- | ---: | ---: |
| hf | 3681784832 | N/A |
| dwdp | 6796329472 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 337209.8384 |
| dwdp_load_time_ms | 159706.0980 |
| torch_profiler_enabled | True |

### HF operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 0.0000 | 0.0000 | N/A |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 206.5477 | 79.3942 | aten::index, aten::index_add_, aten::index_select |
| gemms | 58.4313 | 157.8506 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 146.6877 | 40.0332 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 641.9672 | 806.0507 | 10977 |
| cudaLaunchKernel | 586.0239 | 0.0000 | 64180 |
| aten::index | 169.8083 | 48.5096 | 7536 |
| aten::empty | 162.6583 | 0.0000 | 16361 |
| aten::mul | 132.4376 | 50.0693 | 8966 |
| aten::nonzero | 131.7827 | 60.3719 | 2891 |
| aten::copy_ | 75.0703 | 26.6953 | 6088 |
| cudaStreamSynchronize | 74.0756 | 0.0000 | 5452 |
| cudaMemcpyAsync | 73.2307 | 0.0000 | 6574 |
| aten::view | 72.6183 | 0.0000 | 26096 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1849.9627 | 79401.3218 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 39.7120 | 29.9095 | aten::index, aten::index_select |
| gemms | 56.1300 | 157.8994 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 167.3597 | 39.2418 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1849.9627 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 594.5653 | 799.7860 | 11013 |
| cudaLaunchKernel | 484.0103 | 0.0000 | 51948 |
| aten::empty | 147.1728 | 0.0000 | 17730 |
| cudaMemcpyAsync | 137.0045 | 0.0000 | 8543 |
| aten::mul | 120.7359 | 49.4007 | 8990 |
| aten::copy_ | 94.3156 | 31.4489 | 9796 |
| aten::view | 63.1949 | 0.0000 | 18121 |
| aten::add | 61.0300 | 17.6710 | 4423 |
| aten::silu | 51.0200 | 15.3584 | 2903 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 22.43% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 18.32% lower than native Hugging Face.
- DWDP peak GPU memory is +84.59% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
