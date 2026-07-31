# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T11:08:38.301646+00:00`

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
| hf | 1073.1162 | 1104.2644 | 18261.1990 | 6.6204 | 19334.3152 | 3681784832 |
| dwdp | 1040.5297 | 1073.7070 | 22199.4251 | 5.5078 | 23239.9548 | 6635510272 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1073.1162 | 1040.5297 | -3.04% |
| Prefill ms | 1104.2644 | 1073.7070 | -2.77% |
| Decode ms | 18261.1990 | 22199.4251 | +21.57% |
| Tokens/s | 6.6204 | 5.5078 | -16.81% |
| Total latency ms | 19334.3152 | 23239.9548 | +20.20% |
| Peak GPU memory bytes | 3681784832 | 6635510272 | +80.23% |

**Summary:** DWDP is 20.20% slower than native HF by end-to-end latency.
DWDP throughput is -16.81% versus native HF.

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
| hf_load_time_ms | 193185.8531 |
| dwdp_load_time_ms | 188484.6336 |
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
| gather | 196.8629 | 79.4371 | aten::index, aten::index_add_, aten::index_select |
| gemms | 55.6047 | 157.7173 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 138.9043 | 40.1077 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 639.8071 | 806.0107 | 10977 |
| cudaLaunchKernel | 577.4143 | 0.0000 | 64180 |
| aten::index | 162.5758 | 48.5457 | 7536 |
| aten::empty | 161.4628 | 0.0000 | 16361 |
| aten::mul | 125.0856 | 50.0665 | 8966 |
| aten::nonzero | 124.2510 | 60.5531 | 2891 |
| cudaStreamSynchronize | 76.8468 | 0.0000 | 5452 |
| cudaMemcpyAsync | 72.6024 | 0.0000 | 6574 |
| aten::copy_ | 72.1370 | 26.7672 | 6088 |
| aten::view | 68.4576 | 0.0000 | 26096 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1990.8398 | 9252.9599 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 42.4763 | 29.4018 | aten::index, aten::index_select |
| gemms | 60.7358 | 156.8620 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 177.4914 | 38.4645 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1990.8398 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 656.8531 | 799.7794 | 10980 |
| cudaLaunchKernel | 539.5228 | 0.0000 | 52201 |
| aten::empty | 165.3645 | 0.0000 | 18081 |
| cudaMemcpyAsync | 138.8786 | 0.0000 | 8148 |
| aten::mul | 128.6424 | 69.0963 | 8968 |
| aten::copy_ | 98.8677 | 30.3815 | 9401 |
| aten::view | 65.4534 | 0.0000 | 18095 |
| aten::add | 63.7582 | 17.4777 | 4396 |
| aten::silu | 54.7018 | 15.2473 | 2892 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 20.20% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 16.81% lower than native Hugging Face.
- DWDP peak GPU memory is +80.23% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
