#!/usr/bin/env bash
# ==============================================================================
# DWDP Master Benchmark & Profiling Launcher
# ==============================================================================
# One-command automated script to install dependencies, run benchmarking
# and Torch profiling for MoE architectures (Qwen1.5-MoE, Mixtral, DeepSeek),
# detect GPU VRAM & auto-select precision (e4m3 / 4bit / fp16), generate reports,
# and bundle results into a dynamic ZIP archive in the main project directory.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-python3}"

# ------------------------------------------------------------------------------
# Configurable Defaults (Can be overridden via Environment Variables)
# ------------------------------------------------------------------------------
MODEL="${MODEL:-Qwen/Qwen1.5-MoE-A2.7B}"
QUANT="${QUANT:-fp8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PROMPT="${PROMPT:-Explain the architecture of Mixture of Experts in deep learning.}"

# Detect if GPU is Tesla T4 or <= 16GB VRAM to set fast default iterations
IS_SLOW_GPU="$("${PYTHON_BIN}" -c '
import torch
if not torch.cuda.is_available():
    print("true")
else:
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    raw_gpu = torch.cuda.get_device_name(0).lower()
    if "t4" in raw_gpu or vram_gb <= 16.0:
        print("true")
    else:
        print("false")
')"

if [[ "${IS_SLOW_GPU}" == "true" ]]; then
  WARMUP="${WARMUP:-2}"
  ITERS="${ITERS:-5}"
else
  WARMUP="${WARMUP:-5}"
  ITERS="${ITERS:-20}"
fi

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
TEMP_LOG="$(mktemp -t dwdp_bench.XXXXXX.log)"

fail() {
  local status="$1"
  echo "[ERROR] Benchmark execution failed with status ${status}." >&2
  echo "[ERROR] Detailed log saved to: ${TEMP_LOG}" >&2
  exit "${status}"
}

echo "================================================================="
echo "        DWDP Master Benchmark & Profiling Launcher"
echo "================================================================="
echo "Model:            ${MODEL}"
echo "Requested Quant:  ${QUANT}"
echo "Batch Size:       ${BATCH_SIZE}"
echo "Sequence Length:  ${SEQ_LEN}"
echo "Max New Tokens:   ${MAX_NEW_TOKENS}"
echo "Warmup / Iters:   ${WARMUP} / ${ITERS} (Fast defaults on T4/VRAM <= 16GB)"
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

# Install DWDP in editable mode
echo "[INFO] Registering DWDP package in python environment..."
"${PYTHON_BIN}" -m pip install -q -e "${REPO_ROOT}" || true

# Ensure zip tool is available
if ! command -v zip >/dev/null 2>&1; then
  echo "[INFO] Installing zip utility..."
  apt-get update -qq && apt-get install -y -qq zip || true
fi

# ------------------------------------------------------------------------------
# Step 2: Validate PyTorch, VRAM Capacity, & Determine Dynamic Precision Tag
# ------------------------------------------------------------------------------
echo "[INFO] Validating PyTorch, VRAM Capacity, & Hardware Capabilities..."

SYS_INFO="$("${PYTHON_BIN}" - <<PY
import re, torch

prec = "fp16"
gpu = "cpu"
if torch.cuda.is_available():
    count = torch.cuda.device_count()
    raw_gpu = torch.cuda.get_device_name(0).lower()
    clean_gpu = raw_gpu
    for prefix in ["nvidia", "geforce", "rtx", "tesla"]:
        clean_gpu = clean_gpu.replace(prefix, "")
    clean_gpu = re.sub(r'[^a-z0-9]+', '_', clean_gpu).strip('_')
    if not clean_gpu:
        clean_gpu = "gpu"
        
    if count > 1:
        gpu = f"cluster--{count}x{clean_gpu}"
    else:
        gpu = f"1x{clean_gpu}"
    
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    capability = torch.cuda.get_device_capability(0)
    has_e4m3 = hasattr(torch, "float8_e4m3fn")
    
    total_tokens = ${SEQ_LEN} + ${MAX_NEW_TOKENS}
    kv_cache_est_gb = (2 * 32 * 32 * 128 * total_tokens * ${BATCH_SIZE} * 2) / (1024 ** 3)
    total_req_gb = 14.3 + kv_cache_est_gb + 1.5
    
    req_quant = "${QUANT}"
    if req_quant == "fp8":
        if vram_gb >= total_req_gb and capability >= (8, 9) and has_e4m3:
            prec = "e4m3"
        else:
            prec = "4bit"
    elif req_quant == "4bit":
        prec = "4bit"
    elif req_quant == "8bit":
        prec = "8bit"
    else:
        prec = "fp16"

print(f"{prec}:{gpu}")
PY
)"

PRECISION_TAG="${SYS_INFO%%:*}"
GPU_TAG="${SYS_INFO##*:}"

RESULTS_DIR="${REPO_ROOT}/benchmark-results/${PRECISION_TAG}_${GPU_TAG}_run_${TIMESTAMP}"
ZIP_NAME="DWDP_${PRECISION_TAG}_${GPU_TAG}_Benchmark_Results_${TIMESTAMP}.zip"
ZIP_PATH="${REPO_ROOT}/${ZIP_NAME}"

echo "[SELECTION] Active Precision Tag: ${PRECISION_TAG}"
echo "[SELECTION] Active GPU Tag:       ${GPU_TAG}"
echo "[SELECTION] Output Directory:    ${RESULTS_DIR}"
echo "[SELECTION] ZIP Archive Target:  ${ZIP_NAME}"

# ------------------------------------------------------------------------------
# Step 3: Run Benchmark & Profiling Pass
# ------------------------------------------------------------------------------
mkdir -p "${RESULTS_DIR}"
cd -- "${REPO_ROOT}"

echo "[INFO] Executing benchmark (${PRECISION_TAG}) and collecting profile metrics..."
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
  zip -r -q "${ZIP_NAME}" "benchmark-results/$(basename "${RESULTS_DIR}")"
  echo "================================================================="
  echo " SUCCESS: ${PRECISION_TAG} Benchmark Completed & Packaged!"
  echo "================================================================="
  echo " Zip Archive Created: ${ZIP_PATH}"
  echo " Benchmark Folder:   ${RESULTS_DIR}"
  echo "================================================================="

  # Automatically trigger Google Colab browser download if running in Colab
  "${PYTHON_BIN}" - <<PY
try:
    from google.colab import files
    print("[COLAB AUTO-DOWNLOAD] Triggering browser download for ${ZIP_NAME}...")
    files.download("${ZIP_PATH}")
except Exception:
    pass
PY
else
  echo "[WARNING] zip utility not available. Results saved in ${RESULTS_DIR}"
fi
