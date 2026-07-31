# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T09:09:35.358767+00:00`

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
| Git Commit | d64426e8f3b81406ba7c6ba8fce7578688108b9a |
| Git Branch | main |
| Runtime Backend | dwdp_reference |
| Precision | fp8 |
| Torch Compile | False |

# Configuration

| Field | Value |
| --- | --- |
| Prompt | `Explain the architecture of Mixture of Experts in deep learning.` |
| Batch Size | 1 |
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
| hf | 883.1840 | 893.3955 | 15845.2612 | 7.6516 | 16728.4452 | 3632401408 |
| dwdp | 829.5837 | 839.8371 | 20948.1894 | 5.8776 | 21777.7731 | 6439725568 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 883.1840 | 829.5837 | -6.07% |
| Prefill ms | 893.3955 | 839.8371 | -5.99% |
| Decode ms | 15845.2612 | 20948.1894 | +32.20% |
| Tokens/s | 7.6516 | 5.8776 | -23.19% |
| Total latency ms | 16728.4452 | 21777.7731 | +30.18% |
| Peak GPU memory bytes | 3632401408 | 6439725568 | +77.29% |

**Summary:** DWDP is 30.18% slower than native HF by end-to-end latency.
DWDP throughput is -23.19% versus native HF.

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
| Generated Token Parity | yes |
| Layer Output Parity | N/A |
| Router Output Parity | N/A |
| Executor Output Parity | N/A |
| Merger Output Parity | N/A |

# Memory Usage

| Backend | Peak GPU Bytes | Average GPU Bytes |
| --- | ---: | ---: |
| hf | 3632401408 | N/A |
| dwdp | 6439725568 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 324348.5855 |
| dwdp_load_time_ms | 173372.9571 |
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
| gather | 181.3097 | 79.3713 | aten::index, aten::index_add_, aten::index_select |
| gemms | 34.1556 | 112.2660 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 112.6058 | 31.1996 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 556.4877 | 587.6192 | 10989 |
| cudaLaunchKernel | 525.4179 | 0.0000 | 61783 |
| aten::index | 147.8319 | 58.4271 | 7548 |
| aten::empty | 138.6630 | 0.0000 | 14937 |
| aten::mul | 118.8940 | 43.8584 | 8974 |
| aten::nonzero | 111.4188 | 57.9643 | 2895 |
| cudaStreamSynchronize | 77.8111 | 0.0000 | 5459 |
| cudaMemcpyAsync | 77.0365 | 0.0000 | 6948 |
| aten::view | 64.8139 | 0.0000 | 26145 |
| aten::copy_ | 51.3825 | 18.3881 | 4651 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1769.9259 | 8226.6966 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 38.5432 | 23.4624 | aten::index, aten::index_select |
| gemms | 36.5645 | 112.4012 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 160.4991 | 35.4309 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1769.9259 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 578.4074 | 589.8876 | 10980 |
| cudaLaunchKernel | 488.4505 | 0.0000 | 51184 |
| aten::empty | 149.8914 | 0.0000 | 18081 |
| cudaMemcpyAsync | 137.5018 | 0.0000 | 8514 |
| aten::mul | 121.6281 | 62.2724 | 8968 |
| aten::copy_ | 89.7665 | 27.4665 | 9400 |
| aten::view | 60.9676 | 0.0000 | 18093 |
| aten::add | 51.3075 | 14.6054 | 3961 |
| aten::silu | 50.1569 | 14.0782 | 2892 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 30.18% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 23.19% lower than native Hugging Face.
- DWDP peak GPU memory is +77.29% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
