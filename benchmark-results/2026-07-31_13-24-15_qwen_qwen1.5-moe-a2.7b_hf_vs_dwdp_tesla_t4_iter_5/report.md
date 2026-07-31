# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-31T13:24:15.399706+00:00`

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
| Batch Size | 32 |
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
| hf | 1563.6816 | 1826.9634 | 17906.1487 | 6.5743 | 19469.8303 | 5163287552 |
| dwdp | 1212.3802 | 1527.7887 | 29874.1018 | 4.1175 | 31086.4820 | 12741278208 |

## DWDP vs Native Hugging Face

| Metric | Native HF | DWDP | DWDP change |
| --- | ---: | ---: | ---: |
| TTFT ms | 1563.6816 | 1212.3802 | -22.47% |
| Prefill ms | 1826.9634 | 1527.7887 | -16.38% |
| Decode ms | 17906.1487 | 29874.1018 | +66.84% |
| Tokens/s | 6.5743 | 4.1175 | -37.37% |
| Total latency ms | 19469.8303 | 31086.4820 | +59.66% |
| Peak GPU memory bytes | 5163287552 | 12741278208 | +146.77% |

**Summary:** DWDP is 59.66% slower than native HF by end-to-end latency.
DWDP throughput is -37.37% versus native HF.

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
| hf | 5163287552 | N/A |
| dwdp | 12741278208 | N/A |

# Profiling Summary

Load and profiler configuration:

| Field | Value |
| --- | ---: |
| hf_load_time_ms | 167140.5726 |
| dwdp_load_time_ms | 159977.4396 |
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
| gather | 199.6101 | 113.4135 | aten::index, aten::index_add_, aten::index_select |
| gemms | 66.8612 | 559.8279 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 137.9231 | 82.0426 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| cudaStreamSynchronize | 627.1355 | 0.0000 | 5456 |
| bitsandbytes::gemm_4bit | 624.3541 | 895.3827 | 10983 |
| cudaLaunchKernel | 586.4594 | 0.0000 | 67660 |
| aten::index | 165.8876 | 48.7471 | 7542 |
| aten::empty | 153.2256 | 0.0000 | 16369 |
| aten::nonzero | 133.5889 | 44.4631 | 2893 |
| aten::mul | 126.0749 | 119.2355 | 8970 |
| aten::copy_ | 72.8154 | 70.5126 | 6090 |
| cudaMemcpyAsync | 70.8762 | 0.0000 | 6578 |
| aten::view | 69.8718 | 0.0000 | 26127 |

### DWDP operator categories

| Category | CPU self ms | Device self ms | Operators |
| --- | ---: | ---: | --- |
| python_orchestration | 1589.5380 | 81750.0667 | dwdp.python_orchestration |
| router | 0.0000 | 0.0000 | N/A |
| dispatcher | 0.0000 | 0.0000 | N/A |
| scheduler | 0.0000 | 0.0000 | N/A |
| comms_planner | 0.0000 | 0.0000 | N/A |
| executor | 0.0000 | 0.0000 | N/A |
| merger | 0.0000 | 0.0000 | N/A |
| gather | 30.4003 | 4.2711 | aten::index, aten::index_select |
| gemms | 25.7034 | 653.3366 | aten::addmm, aten::bmm, aten::matmul, aten::mm |
| copies | 151.8732 | 98.1034 | aten::_to_copy, aten::copy_, aten::to, aten::topk |
| synchronization | 0.0000 | 0.0000 | N/A |

Top operators by CPU self time:

| Operator | CPU self ms | Device self ms | Calls |
| --- | ---: | ---: | ---: |
| dwdp.python_orchestration | 1589.5380 | 0.0000 | 1 |
| cudaMemcpyAsync | 736.7574 | 0.0000 | 7611 |
| bitsandbytes::gemm_4bit | 442.7821 | 654.5313 | 8217 |
| cudaLaunchKernel | 410.9621 | 0.0000 | 44049 |
| aten::empty | 123.3563 | 0.0000 | 14934 |
| cudaFree | 120.0371 | 0.0000 | 121 |
| aten::mul | 96.0075 | 125.5212 | 7126 |
| aten::copy_ | 86.0434 | 90.3194 | 8864 |
| cudaStreamSynchronize | 78.6221 | 0.0000 | 3126 |
| aten::view | 50.9256 | 0.0000 | 17110 |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
- DWDP is 59.66% slower than native Hugging Face by end-to-end latency.
- DWDP throughput is 37.37% lower than native Hugging Face.
- DWDP peak GPU memory is +146.77% versus native Hugging Face.
- Prefill is prompt-only forward latency; TTFT is one-token generation latency; decode is total latency minus TTFT.
