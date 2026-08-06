# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-08-06T05:06:25.850434+00:00`

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
| Git Commit | 5d00dcb499979df291ef97d4456935796e62631a |
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
| hf | 885.0430 | 799.3804 | 14043.5492 | 8.5742 | 14928.5922 | 8442060288 |
| dwdp | 798.5812 | 975.0780 | 20213.0275 | 6.0919 | 21011.6087 | 8653429760 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 885.0430 | 798.5812 | -9.77% |
| Prefill ms | 799.3804 | 975.0780 | +21.98% |
| Decode ms | 14043.5492 | 20213.0275 | +43.93% |
| Tokens/s | 8.5742 | 6.0919 | -28.95% |
| Total latency ms | 14928.5922 | 21011.6087 | +40.75% |
| Peak GPU memory bytes | 8442060288 | 8653429760 | +2.50% |

**Summary:** DWDP is 40.75% slower than native HF by end-to-end latency.
DWDP throughput is -28.95% versus native HF.

# Runtime Breakdown

| Module | Latency ms | Percentage |
| --- | ---: | ---: |
| Router | 83.7703 | 13.7879% |
| Dispatcher | 116.0029 | 19.0931% |
| Scheduler | 76.0947 | 12.5246% |
| Comms Planner | 43.3572 | 7.1362% |
| Executor | 249.3265 | 41.0371% |
| Merger | 39.0124 | 6.4211% |
| Total DWDP Overhead | 607.5639 | N/A |

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
| hf | 8442060288 | N/A |
| dwdp | 8653429760 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 406088.5171 |
| dwdp_load_time_ms | 120665.2923 |
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
| gather | 205.5063 | 77.5061 | aten::index, aten::index_add_, aten::index_select |
| gemms | 34.9756 | 108.4527 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 81.8889 | 28.4123 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 611.6534 | 573.6796 | 10989 |
| cudaLaunchKernel | 559.4034 | 0.0000 | 61783 |
| aten::index | 172.0108 | 57.0830 | 7548 |
| aten::empty | 136.3569 | 0.0000 | 14937 |
| aten::nonzero | 129.1292 | 56.4636 | 2895 |
| aten::mul | 127.4824 | 43.1801 | 8974 |
| cudaStreamSynchronize | 76.7761 | 0.0000 | 5459 |
| aten::view | 68.7235 | 0.0000 | 26145 |
| cudaMemcpyAsync | 63.0247 | 0.0000 | 5892 |
| aten::silu | 51.0174 | 13.7213 | 2895 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 516.6706 | 85832.4661 | dwdp.python_orchestration |
| router | 83.7703 | 165.3112 | dwdp.router |
| dispatcher | 116.0029 | 93.0841 | dwdp.dispatcher |
| scheduler | 76.0947 | 151.3365 | dwdp.scheduler |
| comms_planner | 43.3572 | 0.0000 | dwdp.comms_planner |
| executor | 249.3265 | 1944.5486 | dwdp.executor |
| merger | 39.0124 | 32.0870 | dwdp.merger |
| gather | 118.4495 | 37.9235 | aten::index, aten::index_select, dwdp.gather |
| gemms | 654.3452 | 1209.1427 | aten::addmm, aten::bmm, aten::matmul, aten::mm, dwdp.expert_gemms |
| copies | 90.1160 | 23.4037 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.expert_gemms | 618.1437 | 0.0000 | 2519 |
| bitsandbytes::gemm_4bit | 594.4016 | 590.1948 | 11013 |
| dwdp.python_orchestration | 516.6706 | 0.0000 | 1 |
| cudaLaunchKernel | 481.9386 | 0.0000 | 48171 |
| dwdp.executor | 249.3265 | 0.0000 | 384 |
| aten::empty | 142.7773 | 0.0000 | 15426 |
| aten::mul | 134.2463 | 43.4724 | 8990 |
| dwdp.dispatcher | 116.0029 | 0.0000 | 384 |
| dwdp.router | 83.7703 | 0.0000 | 384 |
| dwdp.scheduler | 76.0947 | 0.0000 | 384 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 40.75% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 28.95% lower than native Hugging Face.
- DWDP peak GPU memory is +2.50% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
