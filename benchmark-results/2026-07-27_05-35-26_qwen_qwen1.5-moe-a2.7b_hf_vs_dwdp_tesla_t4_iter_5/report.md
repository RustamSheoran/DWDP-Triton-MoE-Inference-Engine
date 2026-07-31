# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-27T05:35:26.487820+00:00`

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
| Git Commit | 4ae689022a5473be5a67b83f3efb8e04a922a445 |
| Git Branch | main |
| Runtime Backend | dwdp_reference |
| Precision | 4bit |
| Torch Compile | False |

# Configuration

| Field | Value |
| --- | --- |
| Prompt | `Who are you?` |
| Batch Size | 1 |
| Sequence Length | 22 |
| Max New Tokens | 32 |
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
| hf | 573.6584 | 570.2119 | 3447.3826 | 7.9581 | 4021.0410 | 8388371968 |
| dwdp | 886.7768 | 1004.3451 | 6050.7569 | 4.6126 | 6937.5338 | 8418305536 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 573.6584 | 886.7768 | +54.58% |
| Prefill ms | 570.2119 | 1004.3451 | +76.14% |
| Decode ms | 3447.3826 | 6050.7569 | +75.52% |
| Tokens/s | 7.9581 | 4.6126 | -42.04% |
| Total latency ms | 4021.0410 | 6937.5338 | +72.53% |
| Peak GPU memory bytes | 8388371968 | 8418305536 | +0.36% |

**Summary:** DWDP is 72.53% slower than native HF by end-to-end latency.
DWDP throughput is -42.04% versus native HF.

# Runtime Breakdown

| Module | Latency ms | Percentage |
| --- | ---: | ---: |
| Router | 179.8836 | 10.1817% |
| Dispatcher | 187.7742 | 10.6283% |
| Scheduler | 311.0494 | 17.6058% |
| Comms Planner | 107.0739 | 6.0605% |
| Executor | 891.5057 | 50.4604% |
| Merger | 89.4553 | 5.0633% |
| Total DWDP Overhead | 1766.7421 | N/A |

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
| hf | 8388371968 | N/A |
| dwdp | 8418305536 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 1413876.1146 |
| dwdp_load_time_ms | 123931.3243 |
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
| gather | 286.6452 | 107.2861 | aten::index, aten::index_add_, aten::index_select |
| gemms | 18.7167 | 107.2018 | aten::bmm, aten::matmul, aten::mm |
| copies | 120.4627 | 46.6556 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| cudaLaunchKernel | 876.0938 | 0.0000 | 99831 |
| bitsandbytes::gemm_4bit | 859.1067 | 915.9619 | 18273 |
| aten::index | 238.9608 | 81.5407 | 11392 |
| aten::empty | 196.4804 | 0.0000 | 23969 |
| aten::mul | 195.0639 | 68.7241 | 15446 |
| aten::nonzero | 183.2571 | 86.2710 | 4555 |
| cudaStreamSynchronize | 135.7421 | 0.0000 | 8443 |
| aten::view | 101.4107 | 0.0000 | 42041 |
| cudaMemcpyAsync | 85.7056 | 0.0000 | 8540 |
| aten::silu | 70.5118 | 20.8908 | 4555 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 904.5090 | 9144.6889 | dwdp.python_orchestration |
| router | 179.8836 | 411.1204 | dwdp.router |
| dispatcher | 187.7742 | 459.7617 | dwdp.dispatcher |
| scheduler | 311.0494 | 1024.0383 | dwdp.scheduler |
| comms_planner | 107.0739 | 20.9048 | dwdp.comms_planner |
| executor | 891.5057 | 4232.6088 | dwdp.executor |
| merger | 89.4553 | 63.3593 | dwdp.merger |
| gather | 255.2076 | 59.0849 | aten::index, aten::index_select, dwdp.gather |
| gemms | 962.4721 | 1700.1379 | aten::bmm, aten::matmul, aten::mm, dwdp.expert_gemms |
| copies | 288.4657 | 79.8886 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| cudaLaunchKernel | 1001.1097 | 0.0000 | 96266 |
| bitsandbytes::gemm_4bit | 969.1541 | 921.8357 | 18273 |
| dwdp.expert_gemms | 944.1343 | 0.0000 | 3787 |
| dwdp.python_orchestration | 904.5090 | 0.0000 | 1 |
| dwdp.executor | 891.5057 | 0.0000 | 768 |
| cudaMemcpyAsync | 468.4362 | 0.0000 | 63968 |
| cudaStreamSynchronize | 456.6274 | 0.0000 | 53225 |
| dwdp.scheduler | 311.0494 | 0.0000 | 768 |
| aten::empty | 297.0085 | 0.0000 | 33142 |
| aten::mul | 248.9462 | 95.7882 | 15446 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 72.53% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 42.04% lower than native Hugging Face.
- DWDP peak GPU memory is +0.36% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
