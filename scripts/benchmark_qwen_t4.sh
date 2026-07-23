#!/usr/bin/env bash
# Reproducible single-GPU Colab T4 benchmark for Qwen1.5-MoE-A2.7B.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="Qwen/Qwen1.5-MoE-A2.7B"
RESULTS_ROOT="${REPO_ROOT}/results/qwen_t4"
LAUNCH_TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
TEMP_LOG="$(mktemp -t dwdp_qwen_t4.XXXXXX.log)"

fail() {
  local status="$1"
  local failed_dir="${RESULTS_ROOT}/${LAUNCH_TIMESTAMP}_failed"
  mkdir -p "${failed_dir}/logs"
  cp "${TEMP_LOG}" "${failed_dir}/logs/benchmark.log"
  echo "Benchmark failed (exit ${status}). Log saved to ${failed_dir}/logs/benchmark.log" >&2
  exit "${status}"
}

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN}. Set PYTHON_BIN to a Python 3 executable." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Select a GPU runtime in Colab before running this script." >&2
  exit 1
fi

echo "== NVIDIA GPU =="
nvidia-smi

ensure_package() {
  local module="$1"
  local requirement="$2"
  if ! "${PYTHON_BIN}" -c "import ${module}" >/dev/null 2>&1; then
    echo "Installing ${requirement}"
    "${PYTHON_BIN}" -m pip install -U "${requirement}"
  fi
}

echo "== Python dependencies =="
ensure_package torch torch
ensure_package transformers 'transformers>=4.40,<5'
ensure_package accelerate 'accelerate>=0.26'
ensure_package safetensors safetensors

if ! "${PYTHON_BIN}" -c "from pathlib import Path; import DWDP; raise SystemExit(0 if Path(DWDP.__file__).resolve().is_relative_to(Path('${REPO_ROOT}').resolve()) else 1)" >/dev/null 2>&1; then
  echo "Installing DWDP in editable mode"
  "${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
fi

echo "== CUDA validation =="
"${PYTHON_BIN}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable. Use a Colab GPU runtime with a CUDA-enabled PyTorch build.")

properties = torch.cuda.get_device_properties(0)
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"vram_gib={properties.total_memory / (1024 ** 3):.2f}")
capability = torch.cuda.get_device_capability(0)
print(f"compute_capability={capability[0]}.{capability[1]}")
if capability[0] != 7 or capability[1] != 5:
    print("warning: this launcher is configured for T4, but the active GPU is not compute capability 7.5")
PY

mkdir -p "${RESULTS_ROOT}"
cd -- "${REPO_ROOT}"

echo "== Qwen T4 experiment =="
echo "model=${MODEL} dtype=fp16 batch_size=1 sequence_length=128 max_new_tokens=128 warmup=5 iterations=20"
echo "Logs are being captured in ${TEMP_LOG}"

set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_colab.py" \
  --model "${MODEL}" \
  --quantization fp16 \
  --batch-size 1 \
  --sequence-length 128 \
  --max-new-tokens 128 \
  --warmup 5 \
  --iters 20 \
  --seed 0 \
  --results-root "${RESULTS_ROOT}" \
  --profile 2>&1 | tee "${TEMP_LOG}"
benchmark_status="${PIPESTATUS[0]}"
set -e

if [[ "${benchmark_status}" -ne 0 ]]; then
  fail "${benchmark_status}"
fi

results_dir="$(awk -F= '/^results_dir=/{value=$2} END{print value}' "${TEMP_LOG}")"
if [[ -z "${results_dir}" || ! -d "${results_dir}" ]]; then
  echo "Benchmark completed without a valid results_dir marker." >&2
  fail 1
fi

cp "${TEMP_LOG}" "${results_dir}/logs/benchmark.log"
if [[ -f "${results_dir}/benchmark_config.json" ]]; then
  cp "${results_dir}/benchmark_config.json" "${results_dir}/config.json"
fi

echo "== Completed =="
echo "results_dir=${results_dir}"
echo "report=${results_dir}/report.md"
