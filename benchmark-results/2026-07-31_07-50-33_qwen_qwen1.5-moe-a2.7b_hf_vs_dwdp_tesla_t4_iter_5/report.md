# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T07:50:33.752085+00:00`

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
| Git Commit | 9bf86891a9f8d29ae6f185232c7cae11d15e9c3d |
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
| hf | 878.8400 | 745.6156 | 13890.3213 | 8.6667 | 14769.1613 | 8442060288 |
| dwdp | 689.5742 | 848.0674 | 18361.8308 | 6.7187 | 19051.4049 | 8618040832 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 878.8400 | 689.5742 | -21.54% |
| Prefill ms | 745.6156 | 848.0674 | +13.74% |
| Decode ms | 13890.3213 | 18361.8308 | +32.19% |
| Tokens/s | 8.6667 | 6.7187 | -22.48% |
| Total latency ms | 14769.1613 | 19051.4049 | +28.99% |
| Peak GPU memory bytes | 8442060288 | 8618040832 | +2.08% |

**Summary:** DWDP is 28.99% slower than native HF by end-to-end latency.
DWDP throughput is -22.48% versus native HF.

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
| hf_load_time_ms | 638012.6050 |
| dwdp_load_time_ms | 121597.4768 |
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
| gather | 178.9518 | 75.4922 | aten::index, aten::index_add_, aten::index_select |
| gemms | 35.1547 | 104.5672 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 71.6959 | 27.8718 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 512.5115 | 556.6953 | 10989 |
| cudaLaunchKernel | 508.4662 | 0.0000 | 61783 |
| aten::index | 148.1536 | 55.6506 | 7548 |
| aten::empty | 115.5269 | 0.0000 | 14937 |
| aten::nonzero | 115.0982 | 55.2331 | 2895 |
| aten::mul | 112.0350 | 42.3665 | 8974 |
| cudaStreamSynchronize | 79.8091 | 0.0000 | 5459 |
| cudaMemcpyAsync | 58.8518 | 0.0000 | 5892 |
| aten::view | 58.8036 | 0.0000 | 26145 |
| aten::silu | 44.2749 | 13.4520 | 2895 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1318.6661 | 3381.8012 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 36.3857 | 22.3413 | aten::index, aten::index_select |
| gemms | 35.5555 | 107.9239 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 116.8910 | 32.4581 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1318.6661 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 503.6138 | 565.2828 | 10980 |
| cudaLaunchKernel | 443.2269 | 0.0000 | 51184 |
| aten::empty | 134.0924 | 0.0000 | 18081 |
| cudaMemcpyAsync | 114.8139 | 0.0000 | 7602 |
| aten::mul | 112.7171 | 60.9150 | 8968 |
| aten::copy_ | 71.2873 | 24.7681 | 8488 |
| aten::view | 52.8854 | 0.0000 | 18093 |
| cudaStreamSynchronize | 50.2408 | 0.0000 | 3125 |
| aten::add | 46.8714 | 14.1403 | 3961 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 28.99% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 22.48% lower than native Hugging Face.
- DWDP peak GPU memory is +2.08% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
