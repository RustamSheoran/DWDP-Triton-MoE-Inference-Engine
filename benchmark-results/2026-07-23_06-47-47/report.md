# Grouped Expert GEMM Benchmark

## Environment

| Field | Value |
| --- | --- |
| gpu_model | Tesla T4 |
| gpu_memory_bytes | 15637086208 |
| cuda_version | 12.8 |
| cudnn_version | 91900 |
| pytorch_version | 2.11.0+cu128 |
| triton_version | 3.6.0 |
| nvidia_driver_version | 580.82.07 |
| git_commit_hash | 0607c34cb25f774f6ed2226dcd15bbbfe0b743c3 |

## Kernel Configuration

```json
{
  "benchmark": "grouped_expert_gemm",
  "cases": [
    {
      "distribution": "balanced",
      "dtype": "fp16",
      "hidden_size": 2048,
      "name": "custom_e64_t4096_k4_balanced",
      "num_experts": 64,
      "output_size": 5632,
      "seed": 20260723,
      "tokens": 4096,
      "top_k": 4
    },
    {
      "distribution": "skewed",
      "dtype": "fp16",
      "hidden_size": 2048,
      "name": "custom_e64_t4096_k4_skewed",
      "num_experts": 64,
      "output_size": 5632,
      "seed": 20260724,
      "tokens": 4096,
      "top_k": 4
    }
  ],
  "iterations": 100,
  "optimized_backend": "triton_grouped_expert_gemm",
  "preset": "custom",
  "reference_backend": "pytorch_sequential_expert_gemm",
  "timing_excludes": [
    "allocation",
    "weight_materialization",
    "triton_compilation",
    "triton_autotuning",
    "python_setup"
  ],
  "warmup": 10
}
```

## Results

| Case | Distribution | Active/Experts | Max Count | Backend | Median ms | Speedup | FLOPs | Assignments/s | Tokens/s | TFLOPS | Utilization % | CUDA events | Peak bytes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| custom_e64_t4096_k4_balanced | balanced | 64/64 | 256 | pytorch_sequential_expert_gemm | 26.832 | 1.000 | 377957122048 | 610609.5 | 152652.4 | 14.086 | 21.627 | 192 | 1921123840 |
| custom_e64_t4096_k4_balanced | balanced | 64/64 | 256 | triton_grouped_expert_gemm | 440.189 | 0.061 | 377957122048 | 37220.4 | 9305.1 | 0.859 | 1.318 | 0 | 1921123840 |
| custom_e64_t4096_k4_skewed | skewed | 6/64 | 8390 | pytorch_sequential_expert_gemm | 24.822 | 1.000 | 377957122048 | 660054.5 | 165013.6 | 15.227 | 23.379 | 128 | 1921123840 |
| custom_e64_t4096_k4_skewed | skewed | 6/64 | 8390 | triton_grouped_expert_gemm | 11969.197 | 0.002 | 377957122048 | 1368.8 | 342.2 | 0.032 | 0.048 | 1 | 1921123840 |

## Correctness

| Case | torch.allclose | Max absolute error | Mean absolute error |
| --- | --- | ---: | ---: |
| custom_e64_t4096_k4_balanced | True | 0.125000 | 0.000063 |
| custom_e64_t4096_k4_skewed | True | 0.125000 | 0.000117 |

## Notes

- Timings use CUDA events and median steady-state latency.
- Inputs, physical packed weights, offsets, and output buffers are allocated before timing.
- Triton compilation/autotuning and cuBLAS initialization complete during warmup.
- Weight packing/materialization is excluded from timing.
- The utilization estimate is achieved TFLOPS divided by an approximate dense Tensor Core peak when known.
