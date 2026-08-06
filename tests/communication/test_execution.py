from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from DWDP.communication import ExecutionCommunicationEngine  # noqa: E402
from DWDP.executor import ExpertRegistry  # noqa: E402


class _FakeCudaParameter:
    device = torch.device("cuda:0")

    @staticmethod
    def data_ptr() -> int:
        return 1234


class _FakeCudaExpert(nn.Module):
    def parameters(self, recurse: bool = True):
        del recurse
        return iter((_FakeCudaParameter(),))


def test_local_weight_lookup_does_not_initialize_native_staging(monkeypatch) -> None:
    engine = ExecutionCommunicationEngine(ExpertRegistry([_FakeCudaExpert()]))

    def fail_if_initialized(_device) -> bool:
        raise AssertionError("local lookup must not allocate native staging")

    monkeypatch.setattr(engine, "_ensure_native_engine", fail_if_initialized)

    pointer = engine.getWeight(0)

    assert pointer.device_pointers == (1234,)
    assert engine._native_engine is None
