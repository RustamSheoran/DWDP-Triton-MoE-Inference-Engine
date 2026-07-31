# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T12:41:33.534159+00:00`

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
| hf | 1000.8989 | 1038.8531 | 17278.2466 | 7.0025 | 18279.1455 | 3780551680 |
| dwdp | 995.9520 | 1049.4407 | 24931.6330 | 4.9368 | 25927.5849 | 7197059584 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1000.8989 | 995.9520 | -0.49% |
| Prefill ms | 1038.8531 | 1049.4407 | +1.02% |
| Decode ms | 17278.2466 | 24931.6330 | +44.29% |
| Tokens/s | 7.0025 | 4.9368 | -29.50% |
| Total latency ms | 18279.1455 | 25927.5849 | +41.84% |
| Peak GPU memory bytes | 3780551680 | 7197059584 | +90.37% |

**Summary:** DWDP is 41.84% slower than native HF by end-to-end latency.
DWDP throughput is -29.50% versus native HF.

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
| dwdp | 7197059584 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 165900.7295 |
| dwdp_load_time_ms | 158095.6337 |
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
| gather | 198.7524 | 91.7105 | aten::index, aten::index_add_, aten::index_select |
| gemms | 57.1002 | 194.2841 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 140.0549 | 42.3791 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 627.3932 | 1106.1375 | 10983 |
| cudaLaunchKernel | 561.5524 | 0.0000 | 64300 |
| aten::index | 164.2358 | 48.3015 | 7542 |
| aten::empty | 156.8135 | 0.0000 | 16369 |
| aten::mul | 128.9168 | 53.9018 | 8970 |
| aten::nonzero | 126.1308 | 57.9893 | 2893 |
| cudaStreamSynchronize | 92.1313 | 0.0000 | 5456 |
| aten::copy_ | 73.4938 | 29.6172 | 6090 |
| cudaMemcpyAsync | 71.5522 | 0.0000 | 6578 |
| aten::view | 70.2529 | 0.0000 | 26112 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 2627.0527 | 110909.8505 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 58.6939 | 46.3989 | aten::index, aten::index_select |
| gemms | 70.3968 | 205.4030 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 213.3630 | 49.1842 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 2627.0527 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 900.9791 | 1275.6207 | 16677 |
| cudaLaunchKernel | 643.8521 | 0.0000 | 65946 |
| aten::empty | 214.9652 | 0.0000 | 23485 |
| cudaMemcpyAsync | 179.2790 | 0.0000 | 10431 |
| aten::mul | 178.1183 | 70.5174 | 12766 |
| aten::copy_ | 118.4076 | 40.6683 | 11684 |
| aten::view | 92.6852 | 0.0000 | 23986 |
| aten::silu | 82.4231 | 23.3225 | 4791 |
| aten::add | 67.9362 | 20.3933 | 4693 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 41.84% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 29.50% lower than native Hugging Face.
- DWDP peak GPU memory is +90.37% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
