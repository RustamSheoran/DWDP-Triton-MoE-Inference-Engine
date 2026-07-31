# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T08:12:34.750220+00:00`

# Environment

| Field | Value |
| --- | --- |
| GPU | Tesla T4 |
| GPU Memory | 15637086208 |
| CUDA | 12.8 |
| cuDNN | 91900 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 4.57.6 |
| Triton | 3.6.0 |
| NVIDIA Driver | 580.82.07 |
| Python | 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0] |
| OS | Linux-6.6.122+-x86_64-with-glibc2.35 |
| Git Commit | 52fba355273195e8af8804f55c12b4612ebde672 |
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
| hf | 858.8263 | 769.9872 | 13947.7836 | 8.6448 | 14806.6099 | 8442060288 |
| dwdp | 700.4419 | 858.0502 | 18235.7367 | 6.7595 | 18936.1786 | 8618040832 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 858.8263 | 700.4419 | -18.44% |
| Prefill ms | 769.9872 | 858.0502 | +11.44% |
| Decode ms | 13947.7836 | 18235.7367 | +30.74% |
| Tokens/s | 8.6448 | 6.7595 | -21.81% |
| Total latency ms | 14806.6099 | 18936.1786 | +27.89% |
| Peak GPU memory bytes | 8442060288 | 8618040832 | +2.08% |

**Summary:** DWDP is 27.89% slower than native HF by end-to-end latency.
DWDP throughput is -21.81% versus native HF.

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
| hf | 8442060288 | N/A |
| dwdp | 8618040832 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 379814.3553 |
| dwdp_load_time_ms | 122013.8026 |
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
| gather | 183.1287 | 74.7530 | aten::index, aten::index_add_, aten::index_select |
| gemms | 36.7676 | 103.1987 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 71.0162 | 27.5785 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 528.1065 | 550.0087 | 10989 |
| cudaLaunchKernel | 517.5192 | 0.0000 | 61783 |
| aten::index | 151.9651 | 55.0617 | 7548 |
| aten::nonzero | 117.1743 | 54.5927 | 2895 |
| aten::empty | 117.0419 | 0.0000 | 14937 |
| aten::mul | 111.4568 | 41.9931 | 8974 |
| cudaStreamSynchronize | 77.5239 | 0.0000 | 5459 |
| aten::view | 59.8476 | 0.0000 | 26145 |
| cudaMemcpyAsync | 59.7878 | 0.0000 | 5892 |
| aten::silu | 43.7848 | 13.2988 | 2895 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1354.7471 | 3475.7215 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 37.0017 | 22.7365 | aten::index, aten::index_select |
| gemms | 35.3992 | 109.6295 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 118.1465 | 32.9701 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1354.7471 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 527.5858 | 574.8900 | 10980 |
| cudaLaunchKernel | 449.7090 | 0.0000 | 51184 |
| aten::empty | 136.0661 | 0.0000 | 18081 |
| cudaMemcpyAsync | 115.5786 | 0.0000 | 7602 |
| aten::mul | 111.8251 | 61.9548 | 8968 |
| aten::copy_ | 73.5888 | 25.1319 | 8488 |
| aten::view | 55.4951 | 0.0000 | 18093 |
| cudaStreamSynchronize | 51.0790 | 0.0000 | 3125 |
| aten::add | 47.6157 | 14.3499 | 3961 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 27.89% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 21.81% lower than native Hugging Face.
- DWDP peak GPU memory is +2.08% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
