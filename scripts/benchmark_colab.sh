#!/usr/bin/env bash
set -euo pipefail

# Run from any directory: bash /content/repo/scripts/benchmark_colab.sh ...
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN}. Set PYTHON_BIN to a Python 3 executable." >&2
  exit 1
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -m pip install -q -U \
    'transformers>=4.40,<5' \
    'accelerate>=0.26' \
    'bitsandbytes>=0.43' \
    sentencepiece
  "${PYTHON_BIN}" -m pip install -q -e "${REPO_ROOT}"
fi

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_colab.py" "$@"
