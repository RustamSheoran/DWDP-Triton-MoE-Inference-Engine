# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-08-06T09:15:33.012758+00:00`

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
| Git Commit | b22dcac37c5b313d7f164f7ad2d485ec6f82920c |
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
| hf | 794.9216 | 945.6152 | 14978.1347 | 8.1151 | 15773.0564 | 8442060288 |
| dwdp | 896.2036 | 1056.6379 | 20915.2646 | 5.8685 | 21811.4682 | 8443640832 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 794.9216 | 896.2036 | +12.74% |
| Prefill ms | 945.6152 | 1056.6379 | +11.74% |
| Decode ms | 14978.1347 | 20915.2646 | +39.64% |
| Tokens/s | 8.1151 | 5.8685 | -27.68% |
| Total latency ms | 15773.0564 | 21811.4682 | +38.28% |
| Peak GPU memory bytes | 8442060288 | 8443640832 | +0.02% |

**Summary:** DWDP is 38.28% slower than native HF by end-to-end latency.
DWDP throughput is -27.68% versus native HF.

# Runtime Breakdown

| Module | Latency ms | Percentage |
| --- | ---: | ---: |
| Router | 86.2597 | 13.7148% |
| Dispatcher | 117.7250 | 18.7176% |
| Scheduler | 74.3163 | 11.8158% |
| Comms Planner | 44.2210 | 7.0309% |
| Executor | 290.7119 | 46.2215% |
| Merger | 15.7204 | 2.4994% |
| Total DWDP Overhead | 628.9542 | N/A |

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
| dwdp | 8443640832 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 436296.6591 |
| dwdp_load_time_ms | 122871.4978 |
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
| gather | 199.1579 | 77.5354 | aten::index, aten::index_add_, aten::index_select |
| gemms | 37.0152 | 108.5530 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 75.3904 | 28.1683 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| bitsandbytes::gemm_4bit | 581.9074 | 571.4624 | 10989 |
| cudaLaunchKernel | 537.7380 | 0.0000 | 61783 |
| aten::index | 165.4910 | 57.0195 | 7548 |
| aten::empty | 130.5976 | 0.0000 | 14937 |
| aten::nonzero | 123.3756 | 56.4678 | 2895 |
| aten::mul | 119.8633 | 43.0588 | 8974 |
| cudaStreamSynchronize | 76.5608 | 0.0000 | 5459 |
| aten::view | 68.9823 | 0.0000 | 26145 |
| cudaMemcpyAsync | 63.6849 | 0.0000 | 5892 |
| aten::silu | 48.8516 | 13.7615 | 2895 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 451.2444 | 3987.5186 | dwdp.python_orchestration |
| router | 86.2597 | 165.9768 | dwdp.router |
| dispatcher | 117.7250 | 91.8110 | dwdp.dispatcher |
| scheduler | 74.3163 | 147.4984 | dwdp.scheduler |
| comms_planner | 44.2210 | 0.0000 | dwdp.comms_planner |
| executor | 290.7119 | 2058.7135 | dwdp.executor |
| merger | 15.7204 | 0.0000 | dwdp.merger |
| gather | 147.3892 | 55.1390 | aten::index, aten::index_add_, aten::index_select, dwdp.gather |
| gemms | 674.1868 | 1209.4016 | aten::addmm, aten::bmm, aten::matmul, aten::mm, dwdp.expert_gemms |
| copies | 85.2231 | 21.9120 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.expert_gemms | 636.6398 | 0.0000 | 2518 |
| bitsandbytes::gemm_4bit | 581.2551 | 576.6977 | 11010 |
| cudaLaunchKernel | 488.4652 | 0.0000 | 50298 |
| dwdp.python_orchestration | 451.2444 | 0.0000 | 1 |
| dwdp.executor | 290.7119 | 0.0000 | 384 |
| aten::empty | 143.6338 | 0.0000 | 15807 |
| aten::mul | 124.6516 | 40.6234 | 8604 |
| dwdp.dispatcher | 117.7250 | 0.0000 | 384 |
| dwdp.router | 86.2597 | 0.0000 | 384 |
| dwdp.gather | 79.8572 | 0.0000 | 2518 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 38.28% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 27.68% lower than native Hugging Face.
- DWDP peak GPU memory is +0.02% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
