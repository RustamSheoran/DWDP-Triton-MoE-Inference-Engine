# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T09:55:08.499939+00:00`

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
| hf | 941.8038 | 964.0411 | 16697.1992 | 7.2566 | 17639.0030 | 3632401408 |
| dwdp | 881.4918 | 901.0512 | 20782.2258 | 5.9085 | 21663.7176 | 6439725568 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 941.8038 | 881.4918 | -6.40% |
| Prefill ms | 964.0411 | 901.0512 | -6.53% |
| Decode ms | 16697.1992 | 20782.2258 | +24.47% |
| Tokens/s | 7.2566 | 5.9085 | -18.58% |
| Total latency ms | 17639.0030 | 21663.7176 | +22.82% |
| Peak GPU memory bytes | 3632401408 | 6439725568 | +77.29% |

**Summary:** DWDP is 22.82% slower than native HF by end-to-end latency.
DWDP throughput is -18.58% versus native HF.

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
| hf_load_time_ms | 183853.2014 |
| dwdp_load_time_ms | 177485.5007 |
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
| gather | 195.2296 | 79.4295 | aten::index, aten::index_add_, aten::index_select |
| gemms | 35.3337 | 112.2441 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 121.1830 | 31.2107 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 623.4687 | 587.8345 | 10989 |
| cudaLaunchKernel | 556.0507 | 0.0000 | 61783 |
| aten::index | 160.4863 | 58.4587 | 7548 |
| aten::empty | 154.2591 | 0.0000 | 14937 |
| aten::mul | 127.3926 | 43.8858 | 8974 |
| aten::nonzero | 122.0402 | 57.9758 | 2895 |
| cudaStreamSynchronize | 77.7573 | 0.0000 | 5459 |
| cudaMemcpyAsync | 77.3279 | 0.0000 | 6948 |
| aten::view | 70.5850 | 0.0000 | 26145 |
| aten::copy_ | 56.1915 | 18.3996 | 4651 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1814.8178 | 8352.1178 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 38.4522 | 23.4473 | aten::index, aten::index_select |
| gemms | 34.8898 | 112.4521 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 161.9473 | 35.4033 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1814.8178 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 597.7028 | 589.3862 | 10980 |
| cudaLaunchKernel | 479.8922 | 0.0000 | 51184 |
| aten::empty | 153.0645 | 0.0000 | 18081 |
| cudaMemcpyAsync | 131.9923 | 0.0000 | 8514 |
| aten::mul | 122.1009 | 62.2223 | 8968 |
| aten::copy_ | 90.8003 | 27.4479 | 9400 |
| aten::view | 62.6540 | 0.0000 | 18093 |
| aten::silu | 51.2822 | 14.0657 | 2892 |
| aten::add | 50.5737 | 14.5969 | 3961 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 22.82% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 18.58% lower than native Hugging Face.
- DWDP peak GPU memory is +77.29% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
