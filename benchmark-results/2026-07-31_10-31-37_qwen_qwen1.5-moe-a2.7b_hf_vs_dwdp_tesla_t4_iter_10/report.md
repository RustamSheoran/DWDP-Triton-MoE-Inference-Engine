# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T10:31:37.298583+00:00`

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
| Git Commit | 046ca3489eb95467a77c9de1dd79677fe21d7845 |
| Git Branch | main |
| Runtime Backend | dwdp_reference |
| Precision | fp8 |
| Torch Compile | False |

# Configuration

| Field | Value |
| --- | --- |
| Prompt | `Explain the architecture of Mixture of Experts in deep learning.` |
| Batch Size | 4 |
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
| hf | 1056.1292 | 1112.6129 | 17905.8360 | 6.7504 | 18961.9652 | 3780551680 |
| dwdp | 1013.7156 | 1069.3334 | 21878.5792 | 5.5914 | 22892.2948 | 7036295680 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1056.1292 | 1013.7156 | -4.02% |
| Prefill ms | 1112.6129 | 1069.3334 | -3.89% |
| Decode ms | 17905.8360 | 21878.5792 | +22.19% |
| Tokens/s | 6.7504 | 5.5914 | -17.17% |
| Total latency ms | 18961.9652 | 22892.2948 | +20.73% |
| Peak GPU memory bytes | 3780551680 | 7036295680 | +86.12% |

**Summary:** DWDP is 20.73% slower than native HF by end-to-end latency.
DWDP throughput is -17.17% versus native HF.

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
| hf | 3780551680 | N/A |
| dwdp | 7036295680 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 193082.5231 |
| dwdp_load_time_ms | 187718.2120 |
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
| gather | 211.9887 | 93.5529 | aten::index, aten::index_add_, aten::index_select |
| gemms | 61.1035 | 199.8957 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 154.6099 | 43.2091 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 673.7350 | 1137.5638 | 10983 |
| cudaLaunchKernel | 618.3860 | 0.0000 | 64300 |
| aten::index | 175.2536 | 49.2839 | 7542 |
| aten::empty | 168.2390 | 0.0000 | 16369 |
| aten::nonzero | 139.2935 | 59.3982 | 2893 |
| aten::mul | 136.2585 | 54.8164 | 8970 |
| cudaStreamSynchronize | 89.4307 | 0.0000 | 5456 |
| aten::copy_ | 80.7273 | 30.1140 | 6090 |
| cudaMemcpyAsync | 80.4880 | 0.0000 | 6578 |
| aten::view | 73.6328 | 0.0000 | 26112 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1952.6148 | 9074.1821 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 41.7109 | 37.1833 | aten::index, aten::index_select |
| gemms | 62.8253 | 196.7964 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 175.5923 | 41.4367 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1952.6148 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 641.4479 | 1109.8864 | 10983 |
| cudaLaunchKernel | 524.7496 | 0.0000 | 52292 |
| aten::empty | 156.3511 | 0.0000 | 18084 |
| cudaMemcpyAsync | 140.9198 | 0.0000 | 8149 |
| aten::mul | 128.7796 | 72.5725 | 8970 |
| aten::copy_ | 98.3116 | 33.6382 | 9402 |
| aten::view | 66.2501 | 0.0000 | 18205 |
| aten::add | 62.7635 | 18.8546 | 4438 |
| aten::silu | 54.1959 | 14.7137 | 2893 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 20.73% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 17.17% lower than native Hugging Face.
- DWDP peak GPU memory is +86.12% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
