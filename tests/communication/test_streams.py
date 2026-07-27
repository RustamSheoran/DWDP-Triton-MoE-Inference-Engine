from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.communication import CudaStreams, StreamRole  # noqa: E402


def test_cpu_execution_does_not_allocate_cuda_streams() -> None:
    streams = CudaStreams()

    assert not streams.ensure("cpu")
    assert streams.device is None
    assert not streams.initialized
    with streams.use(StreamRole.COPY, "cpu"):
        pass


def test_disabled_streams_are_a_noop_for_cuda_device_request() -> None:
    streams = CudaStreams(enabled=False)

    assert not streams.ensure("cuda:0")
    assert not streams.initialized


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_stream_pair_is_persistent_and_device_bound() -> None:
    streams = CudaStreams()

    assert streams.ensure("cuda:0")
    assert streams.copy is streams.copy
    assert streams.compute is streams.compute
    assert streams.copy is not streams.compute
    assert streams.device == torch.device("cuda", 0)
