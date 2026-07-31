# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T08:34:27.665713+00:00`

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
| Git Commit | 52fba355273195e8af8804f55c12b4612ebde672 |
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
| hf | 918.4404 | 794.3147 | 14285.7059 | 8.4188 | 15204.1462 | 8442060288 |
| dwdp | 710.2757 | 744.1797 | 18733.4458 | 6.5831 | 19443.7215 | 8618040832 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 918.4404 | 710.2757 | -22.67% |
| Prefill ms | 794.3147 | 744.1797 | -6.31% |
| Decode ms | 14285.7059 | 18733.4458 | +31.13% |
| Tokens/s | 8.4188 | 6.5831 | -21.80% |
| Total latency ms | 15204.1462 | 19443.7215 | +27.88% |
| Peak GPU memory bytes | 8442060288 | 8618040832 | +2.08% |

**Summary:** DWDP is 27.88% slower than native HF by end-to-end latency.
DWDP throughput is -21.80% versus native HF.

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
| hf_load_time_ms | 128324.2522 |
| dwdp_load_time_ms | 121727.8442 |
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
| gather | 187.9911 | 75.9876 | aten::index, aten::index_add_, aten::index_select |
| gemms | 34.8040 | 105.7451 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 71.9072 | 27.9813 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 538.9788 | 561.6312 | 10989 |
| cudaLaunchKernel | 516.9902 | 0.0000 | 61783 |
| aten::index | 156.3887 | 55.9672 | 7548 |
| aten::empty | 119.4656 | 0.0000 | 14937 |
| aten::nonzero | 119.1608 | 55.4889 | 2895 |
| aten::mul | 112.5924 | 42.5778 | 8974 |
| cudaStreamSynchronize | 79.3728 | 0.0000 | 5459 |
| aten::view | 61.9359 | 0.0000 | 26145 |
| cudaMemcpyAsync | 61.8335 | 0.0000 | 5892 |
| aten::silu | 44.8716 | 13.5429 | 2895 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1463.4071 | 3702.8968 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 38.7857 | 22.7979 | aten::index, aten::index_select |
| gemms | 37.5386 | 109.9307 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 126.2862 | 33.0123 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1463.4071 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 546.4017 | 576.5073 | 10980 |
| cudaLaunchKernel | 474.5362 | 0.0000 | 51184 |
| aten::empty | 142.5335 | 0.0000 | 18081 |
| cudaMemcpyAsync | 124.2891 | 0.0000 | 7602 |
| aten::mul | 118.9975 | 62.0401 | 8968 |
| aten::copy_ | 77.3410 | 25.1474 | 8488 |
| aten::view | 57.9411 | 0.0000 | 18093 |
| cudaStreamSynchronize | 50.7409 | 0.0000 | 3125 |
| aten::add | 49.3040 | 14.3726 | 3961 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 27.88% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 21.80% lower than native Hugging Face.
- DWDP peak GPU memory is +2.08% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
