# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T10:13:49.005170+00:00`

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
| Batch Size | 2 |
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
| hf | 973.4193 | 1016.7023 | 17026.6555 | 7.1111 | 18000.0748 | 3681784832 |
| dwdp | 1008.0868 | 1031.7146 | 22316.8399 | 5.4877 | 23324.9267 | 6635510272 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 973.4193 | 1008.0868 | +3.56% |
| Prefill ms | 1016.7023 | 1031.7146 | +1.48% |
| Decode ms | 17026.6555 | 22316.8399 | +31.07% |
| Tokens/s | 7.1111 | 5.4877 | -22.83% |
| Total latency ms | 18000.0748 | 23324.9267 | +29.58% |
| Peak GPU memory bytes | 3681784832 | 6635510272 | +80.23% |

**Summary:** DWDP is 29.58% slower than native HF by end-to-end latency.
DWDP throughput is -22.83% versus native HF.

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
| hf | 3681784832 | N/A |
| dwdp | 6635510272 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 185311.7383 |
| dwdp_load_time_ms | 187304.9188 |
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
| gather | 196.6804 | 79.1190 | aten::index, aten::index_add_, aten::index_select |
| gemms | 53.2786 | 157.0047 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 140.7642 | 39.9374 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 636.4438 | 801.8010 | 10977 |
| cudaLaunchKernel | 571.8754 | 0.0000 | 64180 |
| aten::index | 162.5028 | 48.3452 | 7536 |
| aten::empty | 159.4385 | 0.0000 | 16361 |
| aten::mul | 126.5516 | 49.8933 | 8966 |
| aten::nonzero | 125.5634 | 60.2975 | 2891 |
| cudaStreamSynchronize | 76.8256 | 0.0000 | 5452 |
| aten::copy_ | 73.0443 | 26.6649 | 6088 |
| cudaMemcpyAsync | 71.6084 | 0.0000 | 6574 |
| aten::view | 69.2243 | 0.0000 | 26096 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1917.8280 | 8914.3684 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 39.7660 | 29.5917 | aten::index, aten::index_select |
| gemms | 59.2096 | 158.0485 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 172.9074 | 38.7485 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1917.8280 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 635.6046 | 805.9479 | 10980 |
| cudaLaunchKernel | 519.1847 | 0.0000 | 52201 |
| aten::empty | 160.5311 | 0.0000 | 18081 |
| cudaMemcpyAsync | 134.6594 | 0.0000 | 8148 |
| aten::mul | 126.4495 | 69.5238 | 8968 |
| aten::copy_ | 97.0757 | 30.6147 | 9401 |
| aten::view | 64.9070 | 0.0000 | 18095 |
| aten::add | 61.9098 | 17.6033 | 4396 |
| aten::silu | 52.7906 | 15.3558 | 2892 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 29.58% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 22.83% lower than native Hugging Face.
- DWDP peak GPU memory is +80.23% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
