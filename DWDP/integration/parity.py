from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from DWDP.runtime import CorrectnessReport, DWDPRuntime, compare_tensors


@dataclass(slots=True)
class GenerationParityReport:
    """Generated-token parity result for native and DWDP generation paths."""

    token_parity: bool
    native_output: Any
    dwdp_output: Any


class RuntimeParityHarness:
    """Correctness harness comparing native outputs against DWDP outputs."""

    def __init__(self, runtime: DWDPRuntime, native_model: Any | None = None) -> None:
        self.runtime = runtime
        self.native_model = native_model

    def compare_hidden_states(
        self,
        reference: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ) -> CorrectnessReport:
        """Compare a reference hidden-state tensor with DWDP runtime output."""

        actual = self.runtime(hidden_states).hidden_states
        return CorrectnessReport(tensor=compare_tensors(reference, actual, rtol=rtol, atol=atol))

    def compare_generation(self, *args, **kwargs) -> GenerationParityReport:
        """Compare native and DWDP generated tokens using identical inputs."""

        if self.native_model is None:
            raise RuntimeError("native_model is required for generation parity")
        native_output = self.native_model.generate(*args, **kwargs)
        dwdp_output = self.runtime.generate(*args, **kwargs)
        token_parity = bool(torch.equal(native_output, dwdp_output)) if isinstance(native_output, torch.Tensor) else native_output == dwdp_output
        return GenerationParityReport(
            token_parity=token_parity,
            native_output=native_output,
            dwdp_output=dwdp_output,
        )
