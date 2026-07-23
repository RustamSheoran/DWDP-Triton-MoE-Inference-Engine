#!/usr/bin/env bash
set -euo pipefail

# Run from any directory:
# bash /content/DWDP-Triton-MoE-Inference-Engine/scripts/benchmark_triton_kernel_colab.sh --experts 64 --tokens 4096 --top-k 8
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN}. Set PYTHON_BIN to a Python 3 executable." >&2
  exit 1
fi

if ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
  echo "Installing PyTorch. Set TORCH_INDEX_URL to select a CUDA wheel index if needed."
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    "${PYTHON_BIN}" -m pip install -U torch --index-url "${TORCH_INDEX_URL}"
  else
    "${PYTHON_BIN}" -m pip install -U torch
  fi
fi

if ! "${PYTHON_BIN}" -c 'import triton' >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install -U triton
fi

"${PYTHON_BIN}" -m pip install -e "${REPO_ROOT}"
cd -- "${REPO_ROOT}"
exec "${PYTHON_BIN}" benchmarks/benchmark_grouped_matmul.py "$@"
