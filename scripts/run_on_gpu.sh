#!/usr/bin/env bash
# ==============================================================================
# DWDP MoE Inference Engine: Tesla T4 Execution Runner
# ==============================================================================
# This script:
# 1. Enforces CUDA availability (fails loudly if running without a GPU).
# 2. Installs DWDP without modifying pre-installed torch/triton packages.
# 3. Runs the test suite including all GPU-gated tests.
# 4. Executes the single MoE layer benchmark with real Qwen config shapes.
# 5. Packages benchmark JSON and Markdown results into moe_t4_results.zip.
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo "DWDP MOE RUNNER: TESLA T4 VALIDATION & BENCHMARK"
echo "======================================================================"

# 1. Fail loudly if CUDA is not available
python3 -c "
import sys
try:
    import torch
except ImportError:
    print('\n[FATAL ERROR] PyTorch is not installed in this environment!', file=sys.stderr)
    sys.exit(1)

if not torch.cuda.is_available():
    print('\n[FATAL ERROR] CUDA is NOT available in this environment!', file=sys.stderr)
    print('This script requires an NVIDIA GPU with CUDA (e.g., Tesla T4).', file=sys.stderr)
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

# 3. Install repo in editable mode without reinstalling torch/triton
echo ""
echo "=== Installing DWDP (no-deps to preserve Colab torch/triton) ==="
python3 -m pip install --quiet --no-deps -e .
python3 -c "import pytest" 2>/dev/null || python3 -m pip install --quiet pytest

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
ZIP_NAME="moe_t4_results.zip"
ZIP_PATH="$(pwd)/${ZIP_NAME}"
if command -v zip >/dev/null 2>&1; then
    zip -r "${ZIP_NAME}" benchmarks/results/t4_fused_moe
else
    python3 -c 'import shutil; shutil.make_archive("moe_t4_results", "zip", "benchmarks/results/t4_fused_moe")'
fi

echo ""
echo "======================================================================"
echo "DWDP RUN COMPLETE!"
echo "Results packaged into: ${ZIP_PATH}"
echo "Summary markdown: $(pwd)/benchmarks/results/t4_fused_moe/BENCHMARK_REPORT.md"
echo "Summary JSON: $(pwd)/benchmarks/results/t4_fused_moe/benchmark_summary.json"
echo "======================================================================"
