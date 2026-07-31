# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T07:06:58.336386+00:00`

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
| Git Commit | 60ca852deba6363752ddbb527f8faebbcbf0be68 |
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
| hf | 784.0074 | 833.1865 | 14236.0452 | 8.5219 | 15020.0526 | 8442060288 |
| dwdp | 725.2747 | 837.9368 | 20273.1979 | 6.0957 | 20998.4726 | 8618040832 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 784.0074 | 725.2747 | -7.49% |
| Prefill ms | 833.1865 | 837.9368 | +0.57% |
| Decode ms | 14236.0452 | 20273.1979 | +42.41% |
| Tokens/s | 8.5219 | 6.0957 | -28.47% |
| Total latency ms | 15020.0526 | 20998.4726 | +39.80% |
| Peak GPU memory bytes | 8442060288 | 8618040832 | +2.08% |

**Summary:** DWDP is 39.80% slower than native HF by end-to-end latency.
DWDP throughput is -28.47% versus native HF.

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
| hf_load_time_ms | 132211.5656 |
| dwdp_load_time_ms | 122477.2804 |
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
| gather | 183.7603 | 77.3616 | aten::index, aten::index_add_, aten::index_select |
| gemms | 35.8694 | 107.5149 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 70.8153 | 28.4216 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 534.9983 | 570.8512 | 10989 |
| cudaLaunchKernel | 524.2745 | 0.0000 | 61783 |
| aten::index | 153.0718 | 57.0011 | 7548 |
| aten::empty | 120.7264 | 0.0000 | 14937 |
| aten::nonzero | 119.0378 | 56.4080 | 2895 |
| aten::mul | 113.6707 | 43.1623 | 8974 |
| cudaStreamSynchronize | 79.5291 | 0.0000 | 5459 |
| aten::view | 63.1733 | 0.0000 | 26145 |
| cudaMemcpyAsync | 62.0727 | 0.0000 | 5892 |
| aten::silu | 45.7600 | 13.6676 | 2895 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1489.0654 | 3814.9731 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 40.5931 | 23.1921 | aten::index, aten::index_select |
| gemms | 38.6553 | 112.3698 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 128.6013 | 33.6302 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1489.0654 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 573.7808 | 589.7336 | 10980 |
| cudaLaunchKernel | 493.0738 | 0.0000 | 51184 |
| aten::empty | 152.1141 | 0.0000 | 18081 |
| cudaMemcpyAsync | 130.4449 | 0.0000 | 7602 |
| aten::mul | 124.4842 | 63.1354 | 8968 |
| aten::copy_ | 79.8665 | 25.5836 | 8488 |
| aten::view | 61.8983 | 0.0000 | 18093 |
| aten::add | 52.7990 | 14.6444 | 3961 |
| cudaStreamSynchronize | 51.5520 | 0.0000 | 3125 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 39.80% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 28.47% lower than native Hugging Face.
- DWDP peak GPU memory is +2.08% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
