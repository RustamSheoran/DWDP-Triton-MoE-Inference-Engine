# Benchmark Summary

- Experiment: `colab_hf_vs_dwdp`
- Model: `Qwen/Qwen1.5-MoE-A2.7B`
- Checkpoint: `Qwen/Qwen1.5-MoE-A2.7B`
- Backend comparison: `hf` vs `dwdp`
- Timestamp: `2026-07-19T05:52:18.572427+00:00`

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
| Git Commit | 70ce2bc0e6efac8bea1eb416b1986c430a609fca |
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
| hf | N/A | N/A | N/A | 5.6480 | 5665.6895 |
| dwdp | N/A | N/A | N/A | 3.7914 | 8440.0447 |

# Runtime Breakdown

| Module | Latency ms | Percentage |
| --- | ---: | ---: |
| Router | N/A | N/A |
| Dispatcher | N/A | N/A |
| Scheduler | N/A | N/A |
| Comms Planner | N/A | N/A |
| Executor | N/A | N/A |
| Merger | N/A | N/A |
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
| hf | 8406358528 | N/A |
| dwdp | 8436157952 | N/A |

# Profiling Summary

| Field | Value |
| --- | --- |
| dwdp | {'python_orchestration': {'cpu_ms': 2324.7609799995253, 'device_ms': 11479.701475, 'operators': ['dwdp.python_orchestration']}, 'dispatcher': {'cpu_ms': 201.44682800001394, 'device_ms': 469.2837610000112, 'operators': ['dwdp.dispatcher']}, 'gather': {'cpu_ms': 261.5821520000284, 'device_ms': 64.7940379999654, 'operators': ['aten::index', 'aten::index_select', 'dwdp.gather']}, 'gemms': {'cpu_ms': 1347.0677480000993, 'device_ms': 3368.4783940000375, 'operators': ['aten::addmm', 'aten::bmm', 'aten::matmul', 'aten::mm', 'dwdp.expert_gemms']}, 'copies': {'cpu_ms': 318.0392989999346, 'device_ms': 81.50643899988985, 'operators': ['aten::_to_copy', 'aten::copy_', 'aten::to', 'aten::topk']}, 'synchronization': {'cpu_ms': 0.0, 'device_ms': 0.0, 'operators': []}, 'top_operators': [{'operator': 'dwdp.python_orchestration', 'self_cpu_ms': 2324.7609799995253, 'self_device_ms': 0.0, 'calls': 1}, {'operator': 'cudaLaunchKernel', 'self_cpu_ms': 1359.1384909998544, 'self_device_ms': 0.0, 'calls': 134536}, {'operator': 'dwdp.expert_gemms', 'self_cpu_ms': 1265.5653650001016, 'self_device_ms': 0.0, 'calls': 3785}, {'operator': 'bitsandbytes::gemv_4bit', 'self_cpu_ms': 915.3019529998362, 'self_device_ms': 805.7867790000836, 'calls': 16572}, {'operator': 'bitsandbytes::dequantize_blockwise', 'self_cpu_ms': 811.2213709999616, 'self_device_ms': 106.73719000007513, 'calls': 18267}, {'operator': 'cudaMemcpyAsync', 'self_cpu_ms': 464.7355600001325, 'self_device_ms': 0.0, 'calls': 63940}, {'operator': 'cudaStreamSynchronize', 'self_cpu_ms': 456.16799099984064, 'self_device_ms': 0.0, 'calls': 53201}, {'operator': 'aten::add', 'self_cpu_ms': 391.31127900000837, 'self_device_ms': 106.55908200000478, 'calls': 22812}, {'operator': 'aten::empty_strided', 'self_cpu_ms': 263.04319499988856, 'self_device_ms': 0.0, 'calls': 28439}, {'operator': 'aten::empty', 'self_cpu_ms': 238.75105100005806, 'self_device_ms': 0.0, 'calls': 33136}, {'operator': 'aten::mul', 'self_cpu_ms': 237.20031900014513, 'self_device_ms': 108.97520900000126, 'calls': 15442}, {'operator': 'aten::_local_scalar_dense', 'self_cpu_ms': 232.96096300001787, 'self_device_ms': 87.26288999996522, 'calls': 49358}, {'operator': 'dwdp.dispatcher', 'self_cpu_ms': 201.44682800001394, 'self_device_ms': 0.0, 'calls': 768}, {'operator': 'dwdp.gather', 'self_cpu_ms': 186.39467299998455, 'self_device_ms': 0.0, 'calls': 3785}, {'operator': 'aten::copy_', 'self_cpu_ms': 172.91427199990144, 'self_device_ms': 65.99928599988823, 'calls': 21679}, {'operator': 'MatMul4Bit', 'self_cpu_ms': 139.51285100000368, 'self_device_ms': 0.0, 'calls': 1695}, {'operator': 'aten::as_strided', 'self_cpu_ms': 136.11664100040625, 'self_device_ms': 0.0, 'calls': 157279}, {'operator': 'aten::t', 'self_cpu_ms': 117.5054789999368, 'self_device_ms': 0.0, 'calls': 40724}, {'operator': 'aten::select', 'self_cpu_ms': 89.5177219999918, 'self_device_ms': 0.0, 'calls': 45516}, {'operator': 'aten::transpose', 'self_cpu_ms': 86.07692999966004, 'self_device_ms': 0.0, 'calls': 46900}, {'operator': 'aten::to', 'self_cpu_ms': 81.45262800002399, 'self_device_ms': 0.0, 'calls': 76086}, {'operator': 'aten::silu', 'self_cpu_ms': 79.45115000001186, 'self_device_ms': 22.48957100002096, 'calls': 4553}, {'operator': 'aten::slice', 'self_cpu_ms': 76.56724500000576, 'self_device_ms': 0.0, 'calls': 45776}, {'operator': 'aten::index_select', 'self_cpu_ms': 74.31481800005162, 'self_device_ms': 42.76904099996872, 'calls': 6889}, {'operator': 'bitsandbytes::dequantize_4bit', 'self_cpu_ms': 73.03140500000129, 'self_device_ms': 117.17765200000184, 'calls': 1695}, {'operator': 'aten::item', 'self_cpu_ms': 70.33041700024937, 'self_device_ms': 0.0, 'calls': 49358}, {'operator': 'aten::mm', 'self_cpu_ms': 70.00986200000165, 'self_device_ms': 169.7537649999949, 'calls': 1655}, {'operator': 'aten::add_', 'self_cpu_ms': 57.119593000060455, 'self_device_ms': 15.5990379999769, 'calls': 3927}, {'operator': 'aten::cat', 'self_cpu_ms': 54.120477000032466, 'self_device_ms': 23.333944000019677, 'calls': 3936}, {'operator': 'aten::empty_like', 'self_cpu_ms': 51.77695400011126, 'self_device_ms': 0.0, 'calls': 19099}]} |
| dwdp_load_time_ms | 123937.0619 |
| dwdp_output | I am a helpful assistant that can answer questions and provide information on a wide range of topics. I am here to assist you in any way I can. If |
| hf | {'python_orchestration': {'cpu_ms': 0.0, 'device_ms': 0.0, 'operators': []}, 'dispatcher': {'cpu_ms': 0.0, 'device_ms': 0.0, 'operators': []}, 'gather': {'cpu_ms': 278.1151410000752, 'device_ms': 120.28748300000132, 'operators': ['aten::index', 'aten::index_add_', 'aten::index_select']}, 'gemms': {'cpu_ms': 80.91840900000483, 'device_ms': 174.80646400000214, 'operators': ['aten::addmm', 'aten::bmm', 'aten::matmul', 'aten::mm']}, 'copies': {'cpu_ms': 140.6180199997806, 'device_ms': 47.889376000009726, 'operators': ['aten::_to_copy', 'aten::copy_', 'aten::to', 'aten::topk']}, 'synchronization': {'cpu_ms': 0.0, 'device_ms': 0.0, 'operators': []}, 'top_operators': [{'operator': 'cudaLaunchKernel', 'self_cpu_ms': 1157.999332999959, 'self_device_ms': 0.004736000000499189, 'calls': 138066}, {'operator': 'bitsandbytes::gemv_4bit', 'self_cpu_ms': 807.0645469999894, 'self_device_ms': 805.6785300000568, 'calls': 16569}, {'operator': 'bitsandbytes::dequantize_blockwise', 'self_cpu_ms': 716.1803240000336, 'self_device_ms': 106.56902799998203, 'calls': 18264}, {'operator': 'aten::add', 'self_cpu_ms': 327.16656799990625, 'self_device_ms': 104.02382299996016, 'calls': 22041}, {'operator': 'aten::index', 'self_cpu_ms': 231.8219510000264, 'self_device_ms': 92.56038200000599, 'calls': 11383}, {'operator': 'aten::empty_strided', 'self_cpu_ms': 186.6510579999634, 'self_device_ms': 0.0, 'calls': 22347}, {'operator': 'aten::mul', 'self_cpu_ms': 182.64856499997234, 'self_device_ms': 73.46223700008375, 'calls': 15440}, {'operator': 'aten::nonzero', 'self_cpu_ms': 180.4052330000011, 'self_device_ms': 91.00046400002279, 'calls': 4552}, {'operator': 'aten::empty', 'self_cpu_ms': 163.09471200000314, 'self_device_ms': 0.0, 'calls': 23957}, {'operator': 'MatMul4Bit', 'self_cpu_ms': 139.80584700000114, 'self_device_ms': 0.0, 'calls': 1695}, {'operator': 'cudaStreamSynchronize', 'self_cpu_ms': 137.37189599999064, 'self_device_ms': 0.0, 'calls': 8437}, {'operator': 'aten::t', 'self_cpu_ms': 113.53180499981401, 'self_device_ms': 0.0, 'calls': 44502}, {'operator': 'cudaMemcpyAsync', 'self_cpu_ms': 83.6752500000215, 'self_device_ms': 0.0, 'calls': 8534}, {'operator': 'aten::transpose', 'self_cpu_ms': 83.18999900023474, 'self_device_ms': 0.0, 'calls': 50678}, {'operator': 'aten::as_strided', 'self_cpu_ms': 76.94354899995952, 'self_device_ms': 0.0, 'calls': 93029}, {'operator': 'bitsandbytes::dequantize_4bit', 'self_cpu_ms': 71.316712999998, 'self_device_ms': 117.09748400000163, 'calls': 1695}, {'operator': 'aten::mm', 'self_cpu_ms': 68.93781900000556, 'self_device_ms': 169.45129300000286, 'calls': 1655}, {'operator': 'aten::silu', 'self_cpu_ms': 67.41194100001934, 'self_device_ms': 22.642586999984104, 'calls': 4552}, {'operator': 'aten::to', 'self_cpu_ms': 60.720880999770316, 'self_device_ms': 0.0, 'calls': 68451}, {'operator': 'aten::add_', 'self_cpu_ms': 52.60628300001501, 'self_device_ms': 15.620613000007753, 'calls': 3927}, {'operator': 'aten::empty_like', 'self_cpu_ms': 48.81997999997837, 'self_device_ms': 0.0, 'calls': 19136}, {'operator': 'aten::copy_', 'self_cpu_ms': 46.061597999978126, 'self_device_ms': 22.677621000020427, 'calls': 4988}, {'operator': 'aten::index_add_', 'self_cpu_ms': 45.59027200004454, 'self_device_ms': 27.565599999993996, 'calls': 3784}, {'operator': 'aten::cat', 'self_cpu_ms': 35.49922099998949, 'self_device_ms': 20.195627000027276, 'calls': 3168}, {'operator': 'aten::_local_scalar_dense', 'self_cpu_ms': 28.433803999986548, 'self_device_ms': 7.13332100001829, 'calls': 3882}, {'operator': 'aten::view', 'self_cpu_ms': 26.669167999997047, 'self_device_ms': 0.0, 'calls': 23921}, {'operator': 'aten::mean', 'self_cpu_ms': 23.825762999978053, 'self_device_ms': 15.770601000012613, 'calls': 1568}, {'operator': 'aten::pow', 'self_cpu_ms': 22.729358999984644, 'self_device_ms': 4.813870000011371, 'calls': 1568}, {'operator': 'aten::reshape', 'self_cpu_ms': 22.095136999953823, 'self_device_ms': 0.0, 'calls': 19871}, {'operator': 'aten::resize_', 'self_cpu_ms': 19.669442000000817, 'self_device_ms': 0.0, 'calls': 4584}]} |
| hf_load_time_ms | 706682.1234 |
| hf_output | I am a helpful assistant that can answer questions and provide information on a wide range of topics. I am here to assist you in any way I can. If |
| torch_profiler_enabled | True |

# Notes

- Native Transformers and DWDP used the same prompt and generation settings.
- DWDP is measured through the current Hugging Face adapter/reference PyTorch path.
