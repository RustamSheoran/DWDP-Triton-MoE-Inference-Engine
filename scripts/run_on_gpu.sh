#!/usr/bin/env bash
# ==============================================================================
# DWDP MoE Inference Engine: Tesla T4 Execution Runner
# ==============================================================================
# This script:
# 1. Enforces CUDA availability (fails loudly if running without a GPU).
# 2. Logs full driver, CUDA, torch, and triton environment metadata.
# 3. Installs DWDP without modifying pre-installed torch/triton packages.
# 4. Runs the test suite (CPU and GPU-gated tests), saving full log to results.
# 5. Executes the single MoE layer benchmark with Qwen config shapes.
# 6. Optionally executes the Triton GEMM diagnostic (--diagnose flag).
# 7. Packages benchmark JSON, Markdown, and test/env logs into moe_t4_results.zip.
# ==============================================================================

set -euo pipefail

RESULTS_DIR="benchmarks/results/t4_fused_moe"
mkdir -p "${RESULTS_DIR}"

RUN_DIAGNOSE=false
for arg in "$@"; do
    if [[ "$arg" == "--diagnose" ]]; then
        RUN_DIAGNOSE=true
    fi
done

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

# 2. Save environment details to results folder
echo ""
echo "=== Environment Info ==="
ENV_FILE="${RESULTS_DIR}/environment_info.txt"
{
    echo "=== NVIDIA-SMI ==="
    nvidia-smi || true
    echo ""
    echo "=== GPU Details ==="
    nvidia-smi --query-gpu=gpu_name,memory.total,driver_version,pstate,compute_mode --format=csv || true
    echo ""
    echo "=== PyTorch & Triton Versions ==="
    python3 -c "import torch, triton; print(f'PyTorch: {torch.__version__} (CUDA {getattr(torch.version, \"cuda\", \"N/A\")}) | Triton: {triton.__version__}')"
    echo ""
    echo "=== Pip List (Torch / Triton / Accelerate / CUDA) ==="
    python3 -m pip list | grep -Ei "torch|triton|nvidia|cuda" || true
} > "${ENV_FILE}"
cat "${ENV_FILE}"

# 3. Install repo in editable mode without reinstalling torch/triton
echo ""
echo "=== Installing DWDP (no-deps to preserve Colab torch/triton) ==="
python3 -m pip install --quiet --no-deps -e .
python3 -c "import pytest" 2>/dev/null || python3 -m pip install --quiet pytest

# 4. Run test suite (both CPU and GPU-gated tests) and tee log to results
echo ""
echo "=== Running MoE Test Suite ==="
PYTEST_LOG="${RESULTS_DIR}/pytest.log"
python3 -m pytest tests/moe/test_fused_moe_kernel.py tests/moe/test_align.py tests/moe/test_reference.py -v 2>&1 | tee "${PYTEST_LOG}"

# 5. Run single MoE layer benchmark
echo ""
echo "=== Running MoE Layer Benchmark ==="
python3 scripts/bench_moe_layer.py --output-dir "${RESULTS_DIR}"

# 6. Optional: Run GEMM diagnostic if requested or available
if [[ "${RUN_DIAGNOSE}" == "true" ]]; then
    if [[ -f "scripts/diagnose_triton_gemm.py" ]]; then
        echo ""
        echo "=== Running Triton GEMM Diagnostic ==="
        python3 scripts/diagnose_triton_gemm.py --output-dir "${RESULTS_DIR}"
    else
        echo "[WARNING] --diagnose specified but scripts/diagnose_triton_gemm.py not found."
    fi
fi

# 7. Package results
echo ""
echo "=== Packaging Benchmark Artifacts ==="
ZIP_NAME="moe_t4_results.zip"
ZIP_PATH="$(pwd)/${ZIP_NAME}"
if command -v zip >/dev/null 2>&1; then
    zip -r "${ZIP_NAME}" "${RESULTS_DIR}"
else
    python3 -c 'import shutil, sys; shutil.make_archive("moe_t4_results", "zip", "benchmarks/results/t4_fused_moe")'
fi

echo ""
echo "======================================================================"
echo "DWDP RUN COMPLETE!"
echo "Results packaged into: ${ZIP_PATH}"
echo "Summary markdown: $(pwd)/${RESULTS_DIR}/BENCHMARK_REPORT.md"
echo "Summary JSON: $(pwd)/${RESULTS_DIR}/benchmark_summary.json"
echo "PyTest Log: $(pwd)/${PYTEST_LOG}"
echo "Environment Log: $(pwd)/${ENV_FILE}"
echo "======================================================================"
