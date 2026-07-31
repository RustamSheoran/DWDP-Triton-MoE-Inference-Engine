"""CUDA Graph Execution Engine (TensorRT-LLM / vLLM style).

Captures static execution graphs during decode warmup iterations and replays
them during token generation passes with zero CPU overhead (0 ms CPU call tax).
"""

from __future__ import annotations

from typing import Any

import torch


class CUDAGraphRunner:
    """CUDA Graph wrapper for zero-CPU-overhead decode iterations."""

    def __init__(self, model: Any, warmup_steps: int = 3):
        self.model = model
        self.warmup_steps = warmup_steps
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_inputs: dict[str, torch.Tensor] = {}
        self.static_outputs: Any = None
        self._is_captured = False

    def capture(self, sample_inputs: dict[str, torch.Tensor]) -> None:
        """Warmup and capture static CUDA graph."""
        if not torch.cuda.is_available():
            return

        # Allocate static input buffers
        self.static_inputs = {
            k: v.clone() for k, v in sample_inputs.items() if isinstance(v, torch.Tensor)
        }

        # Warmup executions
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(self.warmup_steps):
                self.static_outputs = self.model(**self.static_inputs)
        torch.cuda.current_stream().wait_stream(s)

        # Capture Graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_outputs = self.model(**self.static_inputs)

        self._is_captured = True

    def replay(self, inputs: dict[str, torch.Tensor]) -> Any:
        """Replay static CUDA graph with zero CPU call overhead."""
        if not self._is_captured or self.graph is None:
            return self.model(**inputs)

        # Copy inputs into static memory buffers
        for k, v in inputs.items():
            if k in self.static_inputs and isinstance(v, torch.Tensor):
                self.static_inputs[k].copy_(v)

        # Replay CUDA Graph on GPU
        self.graph.replay()
        return self.static_outputs
