# DWDP: Distributed Weight Data Parallelism Triton-MoE Inference Engine

DWDP is an inference-oriented Mixture-of-Experts (MoE) engine targeting efficient execution on Tesla T4 GPUs. It routes tokens into a deterministic expert-major layout, executes experts using custom fused SwiGLU Triton grouped-GEMM kernels in FP16 precision, and merges results back into token-major order.

---

## 🚀 Key Features & Architecture

- **Fused Triton Grouped-GEMM MoE Layer** ([DWDP/moe/layer.py](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/DWDP/moe/layer.py)): Fuses token gather, Gate+Up projection, SwiGLU activation ($\text{gate} \times \text{sigmoid}(\text{gate}) \times \text{up}$), Down projection with routing-weight scaling, and deterministic slot reduction.
- **Token Alignment & Sorting** ([DWDP/moe/align.py](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/DWDP/moe/align.py)): Groups tokens by expert and pads to uniform tiles (`BLOCK_M`), eliminating irregular control divergence.
- **Golden Reference & Parity Testing** ([DWDP/moe/reference.py](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/DWDP/moe/reference.py)): Pure PyTorch golden reference for strict mathematical correctness validation and numerical parity checks.
- **sm_75 Tensor Core Autotuning** ([DWDP/moe/kernels.py](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/DWDP/moe/kernels.py)): FP16 Tensor Cores with FP32 accumulation and 2-stage hardware pipelines autotuned for Turing (Tesla T4).
- **Storage-Preserving MoE Engine**: Preserves the established pipeline (*Router → Dispatcher → Scheduler → Executor → Merger*) without duplicating model weight storage.

---

## 🔄 Runtime Pipeline

```mermaid
flowchart LR
  H[Hidden states] --> R[Router]
  R --> D[Dispatcher]
  D --> S[Scheduler]
  S --> E[Triton MoE Executor]
  D --> E
  E --> M[Merger]
  M --> O[Token-major output]
```

The dispatcher produces one contiguous expert-major assignment stream. The scheduler references ranges in that stream without reordering tensor data. The executor computes packed and routing-weighted outputs using fused Triton kernels, and the merger restores the original token layout.

---

## 🗂️ Repository & Documentation Map

| Module / Component | Implementation Code | Architectural Documentation |
| :--- | :--- | :--- |
| **Fused MoE Layer** | [DWDP/moe](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/DWDP/moe) | 📖 [DWDP/moe/layer.py](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/DWDP/moe/layer.py) |
| **Router** | [DWDP/router](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/DWDP/router) | 📖 [router.md](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/docs/router.md) |
| **Dispatcher** | [DWDP/dispatcher](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/DWDP/dispatcher) | 📖 [dispatcher.md](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/docs/dispatcher.md) |
| **Scheduler** | [DWDP/scheduler](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/DWDP/scheduler) | 📖 [scheduler.md](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/docs/scheduler.md) |
| **Executor** | [DWDP/executor](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/DWDP/executor) | 📖 [executor.md](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/docs/executor.md) |
| **Merger** | [DWDP/merger](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/DWDP/merger) | 📖 [merger.md](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/docs/merger.md) |
| **Runtime Orchestrator** | [DWDP/runtime](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/DWDP/runtime) | 📖 [runtime.md](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/docs/runtime.md) |
| **Hugging Face Adapters** | [DWDP/adapters](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/DWDP/adapters) | 📖 [adapters.md](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/docs/adapters.md) |
| **Benchmark Script** | [scripts/bench_moe_layer.py](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/scripts/bench_moe_layer.py) | ⚡ 4-Baseline MoE Layer Benchmark |
| **GPU Runner** | [scripts/run_on_gpu.sh](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/blob/main/scripts/run_on_gpu.sh) | 🚀 Automated T4 Benchmark Runner |
| **Historical Results** | [benchmark-results/](https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine/tree/main/benchmark-results) | 📊 Timestamped Benchmark Logs |

---

## Running

### Requirements
- Python 3.10+, PyTorch with CUDA, Triton
- An NVIDIA GPU for the benchmark (developed against a Tesla T4, compute capability 7.5, fp16 only). Tests run on CPU using the Triton interpreter.

### Tests (CPU, no GPU needed)
```bash
pip install -e .
TRITON_INTERPRET=1 pytest -q
```
GPU-gated tests are skipped automatically when CUDA is unavailable.

### GPU tests and benchmark (Google Colab T4)
In a Colab notebook with Runtime → Change runtime type → T4 GPU, run:
```python
!git clone -b cleanup https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine.git repo
%cd repo
!bash scripts/run_on_gpu.sh
import os
from google.colab import files
z = "moe_t4_results.zip"
assert os.path.exists(z), "No results zip was created; read the output above for the failure."
files.download(z)
```

### GPU tests and benchmark (local machine with CUDA)
```bash
bash scripts/run_on_gpu.sh
```
The script runs the GPU parity tests, then `scripts/bench_moe_layer.py`, and writes a zip with a markdown report and a JSON summary.

### What the benchmark measures
One fp16 SwiGLU MoE layer, with shapes taken from the model config and random weights, for token counts 1, 8, 64, 512 and 4096. It compares: (a) a naive per-expert loop, (b) a sorted-token per-expert matmul loop, (c) the fused Triton path, and (d) a single matmul of equal FLOPs as a reference. It reports median and p95 latency from CUDA events, plus the number of CUDA kernel launches measured with `torch.profiler`.

---

## Results (Tesla T4)

*Status: Awaiting execution on Tesla T4*

| Tokens (T) | Naive Loop (a) | Sorted Matmul (b) | Triton Path (c) | Roofline cuBLAS (d) | Real Launches (c) | Speedup vs (b) |
|---|---|---|---|---|---|---|
| 1 | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) |
| 8 | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) |
| 64 | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) |
| 512 | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) |
| 4096 | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) | TBD (run on T4) |

### Honest History Note
Earlier end-to-end benchmark logs stored in `benchmark-results/` (dated July–August 2026) were recorded using a sequential PyTorch fallback execution path (`runtime_backend: dwdp_reference`), where DWDP was 17% to 45% slower than standard Hugging Face on a T4. The new Triton kernel in `DWDP/moe` replaces this fallback with a true fused grouped-GEMM execution path.

---

## 🧪 Tests & Verification

Run the full test suite (CPU tests execute via Triton interpreter, GPU tests execute if CUDA is detected):
```bash
pytest -v
```

To run only the new fused MoE kernel test suite:
```bash
pytest tests/moe/ -v
```
