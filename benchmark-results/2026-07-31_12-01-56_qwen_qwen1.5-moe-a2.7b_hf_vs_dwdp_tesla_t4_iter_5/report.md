# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T12:01:56.081030+00:00`

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
| Git Commit | 234d36d1f91af16745ac322dd949cb7353c94f6a |
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
| hf | 872.8611 | 896.5631 | 15752.1043 | 7.6993 | 16624.9654 | 3632401408 |
| dwdp | 844.0565 | 885.9788 | 19991.2191 | 6.1434 | 20835.2756 | 6601055744 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 872.8611 | 844.0565 | -3.30% |
| Prefill ms | 896.5631 | 885.9788 | -1.18% |
| Decode ms | 15752.1043 | 19991.2191 | +26.91% |
| Tokens/s | 7.6993 | 6.1434 | -20.21% |
| Total latency ms | 16624.9654 | 20835.2756 | +25.33% |
| Peak GPU memory bytes | 3632401408 | 6601055744 | +81.73% |

**Summary:** DWDP is 25.33% slower than native HF by end-to-end latency.
DWDP throughput is -20.21% versus native HF.

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
| hf | 3632401408 | N/A |
| dwdp | 6601055744 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 333395.4034 |
| dwdp_load_time_ms | 153386.3963 |
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
| gather | 182.3858 | 79.3187 | aten::index, aten::index_add_, aten::index_select |
| gemms | 34.5866 | 112.2957 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 119.0788 | 31.1452 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 579.6992 | 587.9595 | 10989 |
| cudaLaunchKernel | 536.4205 | 0.0000 | 61783 |
| aten::index | 149.0006 | 58.3597 | 7548 |
| aten::empty | 142.1982 | 0.0000 | 14937 |
| aten::mul | 120.3157 | 43.9446 | 8974 |
| aten::nonzero | 115.1775 | 57.8917 | 2895 |
| cudaStreamSynchronize | 78.9985 | 0.0000 | 5459 |
| cudaMemcpyAsync | 74.4112 | 0.0000 | 6948 |
| aten::view | 65.5145 | 0.0000 | 26145 |
| aten::copy_ | 53.0354 | 18.3391 | 4651 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1727.8721 | 73942.2802 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 37.1097 | 23.4875 | aten::index, aten::index_select |
| gemms | 34.2922 | 112.2971 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 159.3938 | 35.9573 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1727.8721 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 553.9529 | 590.2509 | 11013 |
| cudaLaunchKernel | 461.3939 | 0.0000 | 50859 |
| aten::empty | 140.5698 | 0.0000 | 17730 |
| cudaMemcpyAsync | 131.0974 | 0.0000 | 8909 |
| aten::mul | 111.6400 | 43.2141 | 8990 |
| aten::copy_ | 86.5713 | 28.5576 | 9795 |
| aten::view | 58.2527 | 0.0000 | 18126 |
| cudaStreamSynchronize | 50.5934 | 0.0000 | 3125 |
| aten::add | 49.7859 | 14.6345 | 3952 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 25.33% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 20.21% lower than native Hugging Face.
- DWDP peak GPU memory is +81.73% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
