# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T12:55:25.837954+00:00`

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
| Batch Size | 8 |
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
| hf | 1080.1877 | 1133.3729 | 21237.6265 | 5.7353 | 22317.8143 | 3978085376 |
| dwdp | 1003.1613 | 899.8387 | 25844.9142 | 4.7676 | 26848.0754 | 7990581760 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1080.1877 | 1003.1613 | -7.13% |
| Prefill ms | 1133.3729 | 899.8387 | -20.61% |
| Decode ms | 21237.6265 | 25844.9142 | +21.69% |
| Tokens/s | 5.7353 | 4.7676 | -16.87% |
| Total latency ms | 22317.8143 | 26848.0754 | +20.30% |
| Peak GPU memory bytes | 3978085376 | 7990581760 | +100.87% |

**Summary:** DWDP is 20.30% slower than native HF by end-to-end latency.
DWDP throughput is -16.87% versus native HF.

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
| hf | 3978085376 | N/A |
| dwdp | 7990581760 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 164787.5746 |
| dwdp_load_time_ms | 157557.8733 |
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
| gather | 198.9859 | 113.2903 | aten::index, aten::index_add_, aten::index_select |
| gemms | 253.1341 | 421.8116 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 143.0158 | 48.7385 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 779.1630 | 990.3951 | 10986 |
| cudaLaunchKernel | 630.2987 | 0.0040 | 73588 |
| aten::mm | 230.7355 | 391.5424 | 5830 |
| cudaStreamSynchronize | 175.3402 | 0.0000 | 5458 |
| aten::index | 164.4711 | 49.0334 | 7545 |
| aten::add | 135.4370 | 41.7490 | 8638 |
| aten::empty | 134.9266 | 0.0000 | 16373 |
| aten::nonzero | 126.3841 | 55.8167 | 2894 |
| aten::mul | 124.2089 | 64.1445 | 8972 |
| aten::empty_strided | 98.6217 | 0.0000 | 9026 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1654.9365 | 83858.6819 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 31.3328 | 22.0189 | aten::index, aten::index_select |
| gemms | 210.7832 | 396.3339 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 154.8461 | 46.9693 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1654.9365 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 614.7098 | 650.5125 | 8634 |
| cudaLaunchKernel | 472.0369 | 0.0000 | 54025 |
| cudaMemcpyAsync | 198.9834 | 0.0000 | 7750 |
| aten::mm | 192.5480 | 366.6165 | 4969 |
| aten::add | 124.7152 | 35.2398 | 8161 |
| aten::empty | 107.1545 | 0.0000 | 15351 |
| aten::mul | 96.3361 | 54.7028 | 7404 |
| aten::empty_strided | 87.3914 | 0.0000 | 9941 |
| aten::copy_ | 87.3514 | 40.0668 | 9003 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 20.30% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 16.87% lower than native Hugging Face.
- DWDP peak GPU memory is +100.87% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
