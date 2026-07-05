from __future__ import annotations

import importlib
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class EnvironmentMetadata:
    """Environment metadata required for reproducible benchmark reports."""

    benchmark_timestamp: str
    python_version: str
    operating_system: str
    hostname: str | None
    git_commit_hash: str | None
    git_branch: str | None
    pytorch_version: str | None
    transformers_version: str | None
    triton_version: str | None
    cuda_version: str | None
    cudnn_version: str | None
    nvidia_driver_version: str | None
    gpu_model: str | None
    gpu_memory_bytes: int | None
    runtime_backend: str
    precision: str
    torch_compile: bool


def _package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _git_value(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def collect_environment_metadata(
    *,
    runtime_backend: str,
    precision: str,
    torch_compile: bool,
    include_hostname: bool = True,
) -> EnvironmentMetadata:
    """Collect environment metadata without mutating benchmark state."""

    torch_version = _package_version("torch")
    cuda_version = None
    cudnn_version = None
    gpu_model = None
    gpu_memory_bytes = None
    try:
        import torch

        cuda_version = str(getattr(torch.version, "cuda", None))
        cudnn_value = torch.backends.cudnn.version()
        cudnn_version = str(cudnn_value) if cudnn_value is not None else None
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_model = torch.cuda.get_device_name(device)
            gpu_memory_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    except Exception:
        pass

    nvidia_driver = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        nvidia_driver = result.stdout.splitlines()[0].strip()
    except Exception:
        nvidia_driver = None

    return EnvironmentMetadata(
        benchmark_timestamp=datetime.now(timezone.utc).isoformat(),
        python_version=sys.version.replace("\n", " "),
        operating_system=platform.platform(),
        hostname=socket.gethostname() if include_hostname else None,
        git_commit_hash=_git_value(["rev-parse", "HEAD"]),
        git_branch=_git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        pytorch_version=torch_version,
        transformers_version=_package_version("transformers"),
        triton_version=_package_version("triton"),
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        nvidia_driver_version=nvidia_driver,
        gpu_model=gpu_model,
        gpu_memory_bytes=gpu_memory_bytes,
        runtime_backend=runtime_backend,
        precision=precision,
        torch_compile=torch_compile,
    )
