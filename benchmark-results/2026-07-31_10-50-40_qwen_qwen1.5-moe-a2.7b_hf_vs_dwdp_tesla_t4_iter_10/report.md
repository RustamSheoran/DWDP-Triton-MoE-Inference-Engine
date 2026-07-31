# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T10:50:40.375424+00:00`

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
| Git Commit | 386013369a99472f73a915b14e326295c595bbbf |
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
| hf | 1014.4120 | 1036.0796 | 18005.0797 | 6.7299 | 19019.4916 | 3632401408 |
| dwdp | 993.6841 | 1018.6203 | 23194.5438 | 5.2918 | 24188.2279 | 6439725568 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1014.4120 | 993.6841 | -2.04% |
| Prefill ms | 1036.0796 | 1018.6203 | -1.69% |
| Decode ms | 18005.0797 | 23194.5438 | +28.82% |
| Tokens/s | 6.7299 | 5.2918 | -21.37% |
| Total latency ms | 19019.4916 | 24188.2279 | +27.18% |
| Peak GPU memory bytes | 3632401408 | 6439725568 | +77.29% |

**Summary:** DWDP is 27.18% slower than native HF by end-to-end latency.
DWDP throughput is -21.37% versus native HF.

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
| hf_load_time_ms | 199299.5308 |
| dwdp_load_time_ms | 192307.4443 |
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
| gather | 206.3153 | 79.4456 | aten::index, aten::index_add_, aten::index_select |
| gemms | 43.4574 | 112.2685 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 130.3220 | 31.1998 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 660.1182 | 587.8625 | 10989 |
| cudaLaunchKernel | 605.0124 | 0.0000 | 61783 |
| aten::index | 169.4354 | 58.4653 | 7548 |
| aten::empty | 165.0790 | 0.0000 | 14937 |
| aten::mul | 135.9114 | 43.8966 | 8974 |
| aten::nonzero | 131.2232 | 57.9958 | 2895 |
| cudaMemcpyAsync | 85.9725 | 0.0000 | 6948 |
| cudaStreamSynchronize | 74.8829 | 0.0000 | 5459 |
| aten::view | 72.2839 | 0.0000 | 26145 |
| aten::copy_ | 60.7088 | 18.3858 | 4651 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 2016.3099 | 9286.0436 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 43.1486 | 23.4496 | aten::index, aten::index_select |
| gemms | 42.1481 | 112.4200 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 177.9939 | 35.3103 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 2016.3099 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 645.4384 | 589.6078 | 10980 |
| cudaLaunchKernel | 539.0855 | 0.0000 | 51184 |
| aten::empty | 168.5465 | 0.0000 | 18081 |
| cudaMemcpyAsync | 150.4554 | 0.0000 | 8514 |
| aten::mul | 131.0292 | 62.0067 | 8968 |
| aten::copy_ | 99.7849 | 27.3477 | 9400 |
| aten::view | 66.1881 | 0.0000 | 18093 |
| aten::silu | 55.7548 | 14.0602 | 2892 |
| aten::add | 55.2413 | 14.5871 | 3961 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 27.18% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 21.37% lower than native Hugging Face.
- DWDP peak GPU memory is +77.29% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
