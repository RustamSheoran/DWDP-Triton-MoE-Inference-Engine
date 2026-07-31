# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T11:26:44.284923+00:00`

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
| hf | 1057.8294 | 1126.0816 | 18243.5274 | 6.6317 | 19301.3568 | 3780551680 |
| dwdp | 1052.7732 | 1090.4088 | 22387.3423 | 5.4607 | 23440.1155 | 7036295680 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1057.8294 | 1052.7732 | -0.48% |
| Prefill ms | 1126.0816 | 1090.4088 | -3.17% |
| Decode ms | 18243.5274 | 22387.3423 | +22.71% |
| Tokens/s | 6.6317 | 5.4607 | -17.66% |
| Total latency ms | 19301.3568 | 23440.1155 | +21.44% |
| Peak GPU memory bytes | 3780551680 | 7036295680 | +86.12% |

**Summary:** DWDP is 21.44% slower than native HF by end-to-end latency.
DWDP throughput is -17.66% versus native HF.

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
| dwdp | 7036295680 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 194932.2670 |
| dwdp_load_time_ms | 188197.1424 |
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
| gather | 220.4564 | 95.4961 | aten::index, aten::index_add_, aten::index_select |
| gemms | 63.7105 | 205.8553 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 156.0733 | 43.8493 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 709.5092 | 1161.5590 | 10983 |
| cudaLaunchKernel | 635.0846 | 0.0000 | 64300 |
| aten::index | 182.3016 | 50.2787 | 7542 |
| aten::empty | 178.6328 | 0.0000 | 16369 |
| aten::nonzero | 141.8673 | 60.5867 | 2893 |
| aten::mul | 140.6901 | 55.7268 | 8970 |
| cudaStreamSynchronize | 89.0437 | 0.0000 | 5456 |
| aten::copy_ | 81.8784 | 30.5227 | 6090 |
| cudaMemcpyAsync | 81.1732 | 0.0000 | 6578 |
| aten::view | 75.8147 | 0.0000 | 26112 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 2025.7666 | 9395.4007 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 42.9254 | 37.2322 | aten::index, aten::index_select |
| gemms | 61.9679 | 193.8972 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 181.4314 | 41.8248 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 2025.7666 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 664.5799 | 1114.1013 | 10983 |
| cudaLaunchKernel | 535.7410 | 0.0000 | 52292 |
| aten::empty | 166.3872 | 0.0000 | 18084 |
| cudaMemcpyAsync | 142.5216 | 0.0000 | 8149 |
| aten::mul | 133.2112 | 72.4841 | 8970 |
| aten::copy_ | 102.7687 | 33.9418 | 9402 |
| aten::view | 67.0765 | 0.0000 | 18205 |
| aten::add | 64.9417 | 18.7748 | 4438 |
| aten::silu | 54.8758 | 14.6616 | 2893 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 21.44% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 17.66% lower than native Hugging Face.
- DWDP peak GPU memory is +86.12% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
