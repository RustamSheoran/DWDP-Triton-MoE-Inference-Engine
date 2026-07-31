#!/usr/bin/env bash
# ==============================================================================
# DWDP Master FP8 Benchmark & Profiling Launcher
# ==============================================================================
# One-command automated script to install dependencies, run FP8 benchmarking
# and Torch profiling for MoE architectures (starting with Qwen1.5-MoE on T4),
# generate detailed breakdown reports, and bundle results into a ZIP archive
# in the main project directory.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ------------------------------------------------------------------------------
# Configurable Defaults (Can be overridden via Environment Variables)
# ------------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B}"
QUANT="${QUANT:-fp8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
PROMPT="${PROMPT:-Explain the architecture of Mixture of Experts in deep learning.}"

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RESULTS_DIR="${REPO_ROOT}/results/fp8_run_${TIMESTAMP}"
ZIP_NAME="DWDP_FP8_Benchmark_Results_${TIMESTAMP}.zip"
ZIP_PATH="${REPO_ROOT}/${ZIP_NAME}"
TEMP_LOG="$(mktemp -t dwdp_fp8_bench.XXXXXX.log)"

fail() {
  local status="$1"
  echo "[ERROR] Benchmark execution failed with status ${status}." >&2
  echo "[ERROR] Detailed log saved to: ${TEMP_LOG}" >&2
  exit "${status}"
}

echo "================================================================="
echo "        DWDP Master FP8 Benchmark & Profiling Launcher"
echo "================================================================="
echo "Model:            ${MODEL}"
echo "Quantization:     ${QUANT}"
echo "Batch Size:       ${BATCH_SIZE}"
echo "Sequence Length:  ${SEQ_LEN}"
echo "Max New Tokens:   ${MAX_NEW_TOKENS}"
echo "Warmup / Iters:   ${WARMUP} / ${ITERS}"
echo "Output Directory: ${RESULTS_DIR}"
echo "================================================================="

# ------------------------------------------------------------------------------
# Step 1: Pre-flight Environment Checks & Auto-installation
# ------------------------------------------------------------------------------
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERROR] Could not find ${PYTHON_BIN}. Set PYTHON_BIN environment variable." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] Detected GPU Hardware:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "[WARNING] nvidia-smi not found. Ensure CUDA environment is properly loaded."
fi

echo "[INFO] Verifying & installing Python dependencies..."
"${PYTHON_BIN}" -m pip install -q -U \
  'transformers>=4.40,<5' \
  'accelerate>=0.26' \
  'bitsandbytes>=0.43' \
  'safetensors' \
  'triton' \
  'sentencepiece'

# Install DWDP in editable mode if not already registered
if ! "${PYTHON_BIN}" -c "import DWDP" >/dev/null 2>&1; then
  echo "[INFO] Installing DWDP package in editable mode..."
  "${PYTHON_BIN}" -m pip install -q -e "${REPO_ROOT}"
fi

# Ensure zip tool is available
if ! command -v zip >/dev/null 2>&1; then
  echo "[INFO] Installing zip utility..."
  apt-get update -qq && apt-get install -y -qq zip || true
fi

# ------------------------------------------------------------------------------
# Step 2: Validate PyTorch, VRAM Capacity, & Hardware Capabilities
# ------------------------------------------------------------------------------
echo "[INFO] Validating PyTorch, VRAM Capacity, & Hardware Capabilities..."
"${PYTHON_BIN}" - <<'PY'
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    capability = torch.cuda.get_device_capability(0)
    has_e4m3 = hasattr(torch, "float8_e4m3fn")
    
    print(f"GPU Name:        {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM:      {vram_gb:.2f} GB")
    print(f"CUDA Capability: {capability[0]}.{capability[1]}")
    print(f"Native FP8 (E4M3) Exposed: {has_e4m3}")
    
    # Calculate estimated requirements for model weights + KV cache + CUDA overhead
    model_fp8_est_gb = 14.3
    kv_cache_est_gb = (2 * 32 * 32 * 128 * 256 * 1 * 2) / (1024 ** 3) # ~0.13 GB
    total_req_gb = model_fp8_est_gb + kv_cache_est_gb + 1.5
    
    print(f"[VRAM ESTIMATOR] Estimated FP8 requirements (Model + KV Cache + Workspace): ~{total_req_gb:.2f} GB")
    if vram_gb >= total_req_gb and capability >= (8, 9) and has_e4m3:
        print("[PRECISION SELECTION] FP8 (E4M3) execution fully supported and fits within VRAM.")
    else:
        print(f"[PRECISION SELECTION] VRAM ({vram_gb:.1f}GB) or Hardware Capability ({capability[0]}.{capability[1]}) is below FP8 threshold. Auto-fallback to 4bit (NF4/NVFP4) enabled.")
PY

# ------------------------------------------------------------------------------
# Step 3: Run FP8 Benchmark & Profiling Pass
# ------------------------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"
cd -- "${REPO_ROOT}"

echo "[INFO] Executing FP8 benchmark and collecting prefill/decode profile metrics..."
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_colab.py" \
  --model "${MODEL}" \
  --quantization "${QUANT}" \
  --batch-size "${BATCH_SIZE}" \
  --sequence-length "${SEQ_LEN}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --prompt "${PROMPT}" \
  --results-root "${RESULTS_DIR}" \
  --profile 2>&1 | tee "${TEMP_LOG}"
BENCH_STATUS="${PIPESTATUS[0]}"
set -e

if [[ "${BENCH_STATUS}" -ne 0 ]]; then
  fail "${BENCH_STATUS}"
fi

# Save log to results folder
cp "${TEMP_LOG}" "${RESULTS_DIR}/benchmark.log"

# ------------------------------------------------------------------------------
# Step 4: Archive and Package Results into Project Main Directory
# ------------------------------------------------------------------------------
echo "[INFO] Packaging benchmark results into ZIP archive..."
cd -- "${REPO_ROOT}"

if command -v zip >/dev/null 2>&1; then
  zip -r -q "${ZIP_NAME}" "results/$(basename "${RESULTS_DIR}")"
  echo "================================================================="
  echo " SUCCESS: FP8 Benchmark Completed & Packaged!"
  echo "================================================================="
  echo " Zip Archive Created: ${ZIP_PATH}"
  echo " Benchmark Folder:   ${RESULTS_DIR}"
  echo "================================================================="
else
  echo "[WARNING] zip utility not available. Results saved in ${RESULTS_DIR}"
fi
