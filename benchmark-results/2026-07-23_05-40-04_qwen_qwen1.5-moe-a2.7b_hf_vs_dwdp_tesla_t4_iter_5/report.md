# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-23T05:40:04.212662+00:00`

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
| Git Commit | 0b4ea6a38dedc87e7417eabfcda3156c40b5e59c |
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

| Backend | TTFT ms | Prefill ms | Decode ms | Tokens/s | Total ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| hf | 892.2552 | 901.4676 | 4764.7108 | 5.6567 | 5656.9660 |
| dwdp | 1334.0914 | 1198.3724 | 7306.3080 | 3.7035 | 8640.3994 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 892.2552 | 1334.0914 | +49.52% |
| Prefill ms | 901.4676 | 1198.3724 | +32.94% |
| Decode ms | 4764.7108 | 7306.3080 | +53.34% |
| Tokens/s | 5.6567 | 3.7035 | -34.53% |
| Total latency ms | 5656.9660 | 8640.3994 | +52.74% |
| Peak GPU memory bytes | 8406358528 | 8436157952 | +0.35% |

**Summary:** DWDP is 52.74% slower than native HF by end-to-end latency.
DWDP throughput is -34.53% versus native HF.

# Runtime Breakdown

| Module | Latency ms | Percentage |
| --- | ---: | ---: |
| Router | 206.5342 | 13.5048% |
| Dispatcher | 188.8607 | 12.3491% |
| Scheduler | 315.0540 | 20.6006% |
| Comms Planner | 107.9181 | 7.0565% |
| Executor | 620.5208 | 40.5743% |
| Merger | 90.4554 | 5.9147% |
| Total DWDP Overhead | 1529.3432 | N/A |

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
| hf | 8406358528 | N/A |
| dwdp | 8436157952 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 1243927.3509 |
| dwdp_load_time_ms | 127314.0913 |
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
| gather | 302.9204 | 117.0301 | aten::index, aten::index_add_, aten::index_select |
| gemms | 79.0849 | 174.8615 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 154.0014 | 47.7293 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| cudaLaunchKernel | 1270.8654 | 0.0000 | 138066 |
| bitsandbytes::gemv_4bit | 888.9862 | 804.9136 | 16569 |
| bitsandbytes::dequantize_blockwise | 789.9937 | 106.7284 | 18264 |
| aten::add | 359.2644 | 104.9121 | 22041 |
| aten::index | 252.9886 | 90.9568 | 11383 |
| aten::empty_strided | 211.3566 | 0.0000 | 22347 |
| aten::mul | 206.3639 | 73.0984 | 15440 |
| aten::nonzero | 202.7368 | 88.5752 | 4552 |
| aten::empty | 180.8251 | 0.0000 | 23957 |
| MatMul4Bit | 139.5835 | 0.0000 | 1695 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1097.8686 | 11497.8773 | dwdp.python_orchestration |
| router | 206.5342 | 519.2920 | dwdp.router |
| dispatcher | 188.8607 | 457.1812 | dwdp.dispatcher |
| scheduler | 315.0540 | 1012.3650 | dwdp.scheduler |
| comms_planner | 107.9181 | 21.2671 | dwdp.comms_planner |
| executor | 620.5208 | 5554.8511 | dwdp.executor |
| merger | 90.4554 | 63.8394 | dwdp.merger |
| gather | 254.9424 | 64.3874 | aten::index, aten::index_select, dwdp.gather |
| gemms | 1317.9010 | 3334.2499 | aten::addmm, aten::bmm, aten::matmul, aten::mm, dwdp.expert_gemms |
| copies | 312.2294 | 81.6363 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| cudaLaunchKernel | 1327.8983 | 0.0000 | 134536 |
| dwdp.expert_gemms | 1238.9869 | 0.0000 | 3785 |
| dwdp.python_orchestration | 1097.8686 | 0.0000 | 1 |
| bitsandbytes::gemv_4bit | 924.1111 | 804.8412 | 16572 |
| bitsandbytes::dequantize_blockwise | 820.8832 | 106.8086 | 18267 |
| dwdp.executor | 620.5208 | 0.0000 | 768 |
| cudaMemcpyAsync | 455.2215 | 0.0000 | 63940 |
| cudaStreamSynchronize | 451.5410 | 0.0000 | 53201 |
| aten::add | 384.8890 | 107.3451 | 22812 |
| dwdp.scheduler | 315.0540 | 0.0000 | 768 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 52.74% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 34.53% lower than native Hugging Face.
- DWDP peak GPU memory is +0.35% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
