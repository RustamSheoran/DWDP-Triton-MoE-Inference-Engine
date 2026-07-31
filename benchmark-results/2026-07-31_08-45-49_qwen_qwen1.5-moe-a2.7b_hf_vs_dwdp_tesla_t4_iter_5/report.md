# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T08:45:49.629615+00:00`

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
| Git Commit | dd129eb530128e4ce5ebcaf6476b841ef560406e |
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
| hf | 796.4126 | 825.0972 | 14696.1587 | 8.2620 | 15492.5713 | 8442060288 |
| dwdp | 729.8818 | 947.8301 | 18796.9862 | 6.5551 | 19526.8681 | 8618040832 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 796.4126 | 729.8818 | -8.35% |
| Prefill ms | 825.0972 | 947.8301 | +14.87% |
| Decode ms | 14696.1587 | 18796.9862 | +27.90% |
| Tokens/s | 8.2620 | 6.5551 | -20.66% |
| Total latency ms | 15492.5713 | 19526.8681 | +26.04% |
| Peak GPU memory bytes | 8442060288 | 8618040832 | +2.08% |

**Summary:** DWDP is 26.04% slower than native HF by end-to-end latency.
DWDP throughput is -20.66% versus native HF.

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
| hf_load_time_ms | 129403.8769 |
| dwdp_load_time_ms | 123033.0794 |
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
| gather | 207.6174 | 79.8721 | aten::index, aten::index_add_, aten::index_select |
| gemms | 47.7199 | 112.4210 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 76.4252 | 28.9335 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 597.7078 | 590.3559 | 10989 |
| cudaLaunchKernel | 578.3981 | 0.0000 | 61783 |
| aten::index | 173.1379 | 58.7247 | 7548 |
| aten::nonzero | 131.3929 | 58.0324 | 2895 |
| aten::empty | 131.1657 | 0.0000 | 14937 |
| aten::mul | 125.3711 | 44.2066 | 8974 |
| cudaStreamSynchronize | 81.6493 | 0.0000 | 5459 |
| cudaMemcpyAsync | 68.7659 | 0.0000 | 5892 |
| aten::view | 66.1515 | 0.0000 | 26145 |
| aten::silu | 49.6005 | 14.2260 | 2895 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1390.5594 | 3547.2870 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 38.3508 | 23.0247 | aten::index, aten::index_select |
| gemms | 36.6943 | 111.4655 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 120.0417 | 33.2880 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1390.5594 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 540.9538 | 583.9868 | 10980 |
| cudaLaunchKernel | 454.2214 | 0.0000 | 51184 |
| aten::empty | 136.7093 | 0.0000 | 18081 |
| cudaMemcpyAsync | 118.6055 | 0.0000 | 7602 |
| aten::mul | 115.5808 | 62.6762 | 8968 |
| aten::copy_ | 74.0900 | 25.3388 | 8488 |
| aten::view | 56.0375 | 0.0000 | 18093 |
| cudaStreamSynchronize | 51.7149 | 0.0000 | 3125 |
| aten::add | 48.3140 | 14.5203 | 3961 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 26.04% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 20.66% lower than native Hugging Face.
- DWDP peak GPU memory is +2.08% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
