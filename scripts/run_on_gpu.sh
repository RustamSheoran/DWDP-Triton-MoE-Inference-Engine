#!/usr/bin/env bash
# ==============================================================================
# DWDP MoE Inference Engine: Tesla T4 Execution Runner
# ==============================================================================
# This script:
# 1. Enforces CUDA availability (fails loudly if running without a GPU).
# 2. Installs dependencies in the current environment.
# 3. Runs the test suite including all GPU-gated tests.
# 4. Executes the single MoE layer benchmark with real Qwen config shapes.
# 5. Packages benchmark JSON and Markdown results into moe_t4_results.zip.
#
# GOOGLE COLAB 1-CELL RUNNER:
# Paste the following into a Google Colab GPU (T4) notebook:
# ------------------------------------------------------------------------------
# !git clone -b cleanup <YOUR_REPO_URL> repo && cd repo && bash scripts/run_on_gpu.sh
# ------------------------------------------------------------------------------

set -euo pipefail

echo "======================================================================"
echo "DWDP MOE RUNNER: TESLA T4 VALIDATION & BENCHMARK"
echo "======================================================================"

# 1. Fail loudly if CUDA is not available
python3 -c "
import torch, sys
if not torch.cuda.is_available():
    print('\n[FATAL ERROR] CUDA is NOT available in this environment!', file=sys.stderr)
    print('This script requires a CUDA-enabled GPU (e.g., Tesla T4).', file=sys.stderr)
    print('If running in Colab, ensure Runtime -> Change runtime type -> T4 GPU is selected.\n', file=sys.stderr)
    sys.exit(1)

dev_name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f'CUDA Detected: {dev_name} (Compute Capability {cap[0]}.{cap[1]})')
"

# 2. Print environment details
echo ""
echo "=== Environment Info ==="
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv,noheader || true
python3 -c "import torch, triton; print(f'PyTorch {torch.__version__} | Triton {triton.__version__}')"

# 3. Install repo in editable mode
echo ""
echo "=== Installing DWDP and Dependencies ==="
pip install --quiet -e .
pip install --quiet pytest

# 4. Run test suite (both CPU and GPU-gated tests)
echo ""
echo "=== Running MoE Test Suite ==="
python3 -m pytest tests/moe/test_fused_moe_kernel.py tests/moe/test_align.py tests/moe/test_reference.py -v

# 5. Run single MoE layer benchmark
echo ""
echo "=== Running MoE Layer Benchmark ==="
python3 scripts/bench_moe_layer.py --output-dir benchmarks/results/t4_fused_moe

# 6. Package results
echo ""
echo "=== Packaging Benchmark Artifacts ==="
zip -r moe_t4_results.zip benchmarks/results/t4_fused_moe

echo ""
echo "======================================================================"
echo "DWDP RUN COMPLETE!"
echo "Results packaged into: moe_t4_results.zip"
echo "Summary markdown: benchmarks/results/t4_fused_moe/BENCHMARK_REPORT.md"
echo "Summary JSON: benchmarks/results/t4_fused_moe/benchmark_summary.json"
echo "======================================================================"
