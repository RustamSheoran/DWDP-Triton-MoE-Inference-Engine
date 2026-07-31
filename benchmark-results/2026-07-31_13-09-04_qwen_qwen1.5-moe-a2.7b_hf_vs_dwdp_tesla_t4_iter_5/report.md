# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T13:09:04.799951+00:00`

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
| Batch Size | 16 |
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
| hf | 1124.1255 | 1240.9726 | 17174.8215 | 6.9949 | 18298.9470 | 4373152768 |
| dwdp | 772.3889 | 1092.7739 | 32461.6714 | 3.8515 | 33234.0603 | 9576405504 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1124.1255 | 772.3889 | -31.29% |
| Prefill ms | 1240.9726 | 1092.7739 | -11.94% |
| Decode ms | 17174.8215 | 32461.6714 | +89.01% |
| Tokens/s | 6.9949 | 3.8515 | -44.94% |
| Total latency ms | 18298.9470 | 33234.0603 | +81.62% |
| Peak GPU memory bytes | 4373152768 | 9576405504 | +118.98% |

**Summary:** DWDP is 81.62% slower than native HF by end-to-end latency.
DWDP throughput is -44.94% versus native HF.

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
| hf | 4373152768 | N/A |
| dwdp | 9576405504 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 164283.9933 |
| dwdp_load_time_ms | 163240.2824 |
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
| gather | 200.4891 | 125.6409 | aten::index, aten::index_add_, aten::index_select |
| gemms | 39.3374 | 332.5056 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 138.0812 | 57.7147 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 615.1551 | 986.7540 | 10989 |
| cudaLaunchKernel | 575.4789 | 0.0000 | 65138 |
| cudaStreamSynchronize | 318.4896 | 0.0000 | 5460 |
| aten::index | 165.7652 | 42.8677 | 7548 |
| aten::empty | 158.5963 | 0.0000 | 16377 |
| aten::nonzero | 133.5053 | 48.8613 | 2895 |
| aten::mul | 125.0677 | 80.4025 | 8974 |
| cudaMemcpyAsync | 72.4799 | 0.0000 | 6582 |
| aten::copy_ | 71.6416 | 45.9113 | 6092 |
| aten::view | 70.4573 | 0.0000 | 26128 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1657.1500 | 72862.6313 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 32.2868 | 27.3840 | aten::index, aten::index_select |
| gemms | 32.1768 | 360.5930 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 153.1207 | 60.8657 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1657.1500 | 0.0000 | 1 |
| bitsandbytes::gemm_4bit | 493.7543 | 726.6541 | 9342 |
| cudaLaunchKernel | 431.5032 | 0.0000 | 47254 |
| cudaMemcpyAsync | 346.2471 | 0.0000 | 7986 |
| aten::empty | 128.5207 | 0.0000 | 16059 |
| aten::mul | 101.8891 | 75.0061 | 7876 |
| aten::copy_ | 86.2011 | 54.7007 | 9239 |
| cudaStreamSynchronize | 56.2868 | 0.0000 | 3126 |
| aten::view | 54.3446 | 0.0000 | 17034 |
| aten::add | 50.0259 | 24.9206 | 3865 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 81.62% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 44.94% lower than native Hugging Face.
- DWDP peak GPU memory is +118.98% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
