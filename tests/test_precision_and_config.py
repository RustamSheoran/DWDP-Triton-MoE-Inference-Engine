from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from DWDP.adapters.extractor import QwenMoEConfig, parse_qwen_moe_config
from DWDP.benchmarking.environment import EnvironmentMetadata
from DWDP.precision import get_device_precision_metadata, validate_execution_dtype


def test_parse_qwen_moe_config_valid() -> None:
    config_dict = {
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "head_dim": 128,
        "num_experts": 64,
        "num_experts_per_tok": 4,
        "hidden_size": 2048,
        "moe_intermediate_size": 1408,
    }
    cfg = parse_qwen_moe_config(config_dict)
    assert isinstance(cfg, QwenMoEConfig)
    assert cfg.num_layers == 24
    assert cfg.num_kv_heads == 16
    assert cfg.head_dim == 128
    assert cfg.num_experts == 64
    assert cfg.top_k == 4
    assert cfg.hidden_size == 2048
    assert cfg.moe_intermediate_size == 1408


def test_parse_qwen_moe_config_missing_required_raises() -> None:
    config_dict = {
        "num_hidden_layers": 24,
        "num_experts": 64,
    }
    with pytest.raises(ValueError, match="hidden_size"):
        parse_qwen_moe_config(config_dict)


def test_validate_execution_dtype_fp16() -> None:
    dtype, meta = validate_execution_dtype("fp16", device="cpu")
    assert dtype == torch.float16
    assert meta["requested_dtype"] == "fp16"
    assert meta["actual_dtype"] == "fp16"
    assert "torch_version" in meta
    assert "triton_version" in meta


def test_validate_execution_dtype_unsupported_raises() -> None:
    for bad_dtype in ("fp8", "e4m3", "e5m2", "4bit", "nf4", "nvfp4", "int4", "int8", "invalid"):
        with pytest.raises(ValueError, match="Unsupported precision"):
            validate_execution_dtype(bad_dtype, device="cpu")


def test_validate_execution_dtype_bf16_on_t4_raises() -> None:
    # Simulate a Tesla T4 GPU with compute capability (7, 5)
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"), \
         patch("torch.cuda.get_device_capability", return_value=(7, 5)), \
         patch("torch.cuda.get_device_properties") as mock_props:
        mock_props.return_value.total_memory = 16 * 1024**3
        with pytest.raises(ValueError, match="bfloat16 requires SM 8.0+"):
            validate_execution_dtype("bf16", device="cuda:0")


def test_validate_execution_dtype_bf16_on_ampere_succeeds() -> None:
    # Simulate an Ampere A100 GPU with compute capability (8, 0)
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-40GB"), \
         patch("torch.cuda.get_device_capability", return_value=(8, 0)), \
         patch("torch.cuda.get_device_properties") as mock_props:
        mock_props.return_value.total_memory = 40 * 1024**3
        dtype, meta = validate_execution_dtype("bf16", device="cuda:0")
        assert dtype == torch.bfloat16
        assert meta["requested_dtype"] == "bf16"
        assert meta["actual_dtype"] == "bf16"
        assert meta["compute_capability"] == "8.0"


def test_environment_metadata_precision_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Precision mismatch"):
        EnvironmentMetadata(
            benchmark_timestamp="2026-11-17T00:00:00",
            python_version="3.15",
            operating_system="Linux",
            hostname="localhost",
            git_commit_hash="deadbeef",
            git_branch="cleanup",
            pytorch_version="2.15",
            transformers_version=None,
            triton_version="3.9",
            cuda_version=None,
            cudnn_version=None,
            nvidia_driver_version=None,
            gpu_model="Tesla T4",
            gpu_memory_bytes=16 * 1024**3,
            runtime_backend="triton",
            precision="fp16",
            torch_compile=False,
            requested_dtype="fp8",
            actual_dtype="4bit",
        )
