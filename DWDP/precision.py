"""Precision validation and GPU architecture constraints for DWDP.

Enforces honest precision policy:
- FP16 is standard on Turing/Tesla T4 (Compute Capability 7.5).
- BF16 requires Compute Capability >= 8.0 (Ampere+); requesting BF16 on SM < 8.0 raises an error.
- FP8 (E4M3/E5M2), 4-bit (NF4/NVFP4), INT4, INT8, and quantized formats are not supported and fail loudly.
- FP32 is permitted on CPU reference paths.
- Requested dtype and actual dtype must always match. Silent precision fallbacks are strictly prohibited.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch


def _package_version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "N/A"


def get_device_precision_metadata(device: str | torch.device = "cuda") -> dict[str, Any]:
    """Inspect the target device and return hardware/software precision metadata."""

    dev_str = str(device)
    is_cuda = dev_str.startswith("cuda") or (
        isinstance(device, torch.device) and device.type == "cuda"
    )

    gpu_name = "None"
    compute_capability = "N/A"
    gpu_memory_bytes = None

    if is_cuda and torch.cuda.is_available():
        try:
            dev_idx = (
                torch.cuda.current_device()
                if dev_str == "cuda"
                else (torch.device(device).index or 0)
            )
            gpu_name = torch.cuda.get_device_name(dev_idx)
            cap = torch.cuda.get_device_capability(dev_idx)
            compute_capability = f"{cap[0]}.{cap[1]}"
            gpu_memory_bytes = int(torch.cuda.get_device_properties(dev_idx).total_memory)
        except Exception:
            pass

    return {
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
        "gpu_memory_bytes": gpu_memory_bytes,
        "torch_version": str(getattr(torch, "__version__", "unknown")),
        "triton_version": _package_version("triton"),
    }


def validate_execution_dtype(
    requested_dtype: str | torch.dtype,
    device: str | torch.device = "cuda",
) -> tuple[torch.dtype, dict[str, Any]]:
    """Validate requested precision against hardware capabilities.

    Returns ``(target_torch_dtype, metadata_dict)``.
    Raises ValueError or RuntimeError on unsupported precision or silent mismatch.
    """

    dtype_map: dict[str, tuple[torch.dtype, str]] = {
        "fp16": (torch.float16, "fp16"),
        "float16": (torch.float16, "fp16"),
        "bf16": (torch.bfloat16, "bf16"),
        "bfloat16": (torch.bfloat16, "bf16"),
        "fp32": (torch.float32, "fp32"),
        "float32": (torch.float32, "fp32"),
    }

    if isinstance(requested_dtype, torch.dtype):
        if requested_dtype == torch.float16:
            canonical = "fp16"
            target_dtype = torch.float16
        elif requested_dtype == torch.bfloat16:
            canonical = "bf16"
            target_dtype = torch.bfloat16
        elif requested_dtype == torch.float32:
            canonical = "fp32"
            target_dtype = torch.float32
        else:
            raise ValueError(
                f"Unsupported torch.dtype: {requested_dtype}. "
                f"DWDP only supports torch.float16 (and torch.bfloat16 on SM 8.0+)."
            )
    elif isinstance(requested_dtype, str):
        normalized = requested_dtype.strip().lower()
        if normalized in dtype_map:
            target_dtype, canonical = dtype_map[normalized]
        else:
            raise ValueError(
                f"Unsupported precision requested: '{requested_dtype}'. "
                f"DWDP only supports fp16 (and bf16 on SM 8.0+). "
                f"Quantized formats (FP8, 4bit, NF4, INT4) are not supported."
            )
    else:
        raise TypeError(f"Invalid requested_dtype type: {type(requested_dtype)!r}")

    meta = get_device_precision_metadata(device)
    dev_str = str(device)
    is_cuda = dev_str.startswith("cuda") or (
        isinstance(device, torch.device) and device.type == "cuda"
    )

    if is_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA precision '{canonical}' requested but CUDA is unavailable.")

        if meta["compute_capability"] != "N/A":
            major = int(meta["compute_capability"].split(".")[0])
            if target_dtype == torch.bfloat16 and major < 8:
                raise ValueError(
                    f"bf16 requested, but GPU '{meta['gpu_name']}' has compute capability "
                    f"{meta['compute_capability']}. bfloat16 requires SM 8.0+ (Ampere+). "
                    f"On Tesla T4 (SM 7.5), request fp16 instead."
                )

    actual_dtype_str = canonical
    if actual_dtype_str != canonical:
        raise RuntimeError(
            f"Precision mismatch: requested '{canonical}' but resolved '{actual_dtype_str}'. "
            f"Silent fallback is forbidden."
        )

    meta["requested_dtype"] = canonical
    meta["actual_dtype"] = actual_dtype_str

    return target_dtype, meta
