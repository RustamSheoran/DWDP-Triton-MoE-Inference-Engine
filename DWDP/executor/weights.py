"""Storage-preserving expert weight layouts for optimized MoE backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import torch
from torch import nn


class WeightFormat(str, Enum):
    """Logical encoding of expert weights for backend selection."""

    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"
    FP8 = "fp8"
    INT4 = "int4"
    UNKNOWN = "unknown"


def infer_weight_format(weight: torch.Tensor) -> WeightFormat:
    """Classify a tensor dtype without imposing a backend-specific encoding."""

    if weight.dtype == torch.float16:
        return WeightFormat.FP16
    if weight.dtype == torch.bfloat16:
        return WeightFormat.BF16
    if weight.dtype == torch.float32:
        return WeightFormat.FP32
    if "float8" in str(weight.dtype):
        return WeightFormat.FP8
    if weight.dtype == torch.uint8:
        return WeightFormat.INT4
    return WeightFormat.UNKNOWN


@dataclass(frozen=True, slots=True)
class ExpertMajorMatrixView:
    """Logical ``[E, rows, columns]`` matrix layout backed by original tensors.

    Hugging Face stores Qwen experts as independent ``nn.Linear`` modules.
    Stacking their weights would duplicate model storage, so this view retains
    references to the original per-expert tensors while exposing a canonical
    expert-major shape to optimized backends.
    """

    expert_ids: tuple[int, ...]
    expert_weights: tuple[torch.Tensor, ...]
    name: str

    def __post_init__(self) -> None:
        if not self.expert_weights:
            raise ValueError(f"{self.name} requires at least one expert weight")
        if len(self.expert_ids) != len(self.expert_weights):
            raise ValueError(f"{self.name} expert ids and weights must have matching lengths")
        reference = self.expert_weights[0]
        if reference.ndim != 2:
            raise ValueError(f"{self.name} weights must be rank-2")
        for weight in self.expert_weights:
            if weight.ndim != 2:
                raise ValueError(f"{self.name} weights must be rank-2")
            if weight.shape != reference.shape:
                raise ValueError(f"{self.name} requires uniform expert weight shapes")
            if weight.dtype != reference.dtype or weight.device != reference.device:
                raise ValueError(f"{self.name} requires one dtype and device")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Logical expert-major shape without allocating stacked storage."""

        rows, columns = self.expert_weights[0].shape
        return len(self.expert_weights), rows, columns

    @property
    def dtype(self) -> torch.dtype:
        """Common expert weight dtype."""

        return self.expert_weights[0].dtype

    @property
    def device(self) -> torch.device:
        """Common expert weight device."""

        return self.expert_weights[0].device

    @property
    def format(self) -> WeightFormat:
        """Storage encoding available to an execution backend."""

        return infer_weight_format(self.expert_weights[0])

    @property
    def storage_pointers(self) -> tuple[int, ...]:
        """Underlying data pointers, useful for no-duplication validation."""

        return tuple(weight.data_ptr() for weight in self.expert_weights)

    def for_expert(self, expert_id: int) -> torch.Tensor:
        """Return the original matrix for a global expert id."""

        try:
            return self.expert_weights[self.expert_ids.index(int(expert_id))]
        except ValueError as exc:
            raise KeyError(f"{self.name} has no expert {expert_id}") from exc

    def materialize(self) -> torch.Tensor:
        """Explicitly stack weights for a backend that accepts the memory cost.

        This method is never used by provider construction or the Triton
        skeleton. Calling it creates new storage by design.
        """

        return torch.stack(self.expert_weights, dim=0)


@dataclass(frozen=True, slots=True)
class FusedGateUpWeightView:
    """Logical Qwen SwiGLU ``[E, 2I, H]`` projection without concatenation."""

    gate_weights: ExpertMajorMatrixView
    up_weights: ExpertMajorMatrixView

    def __post_init__(self) -> None:
        if self.gate_weights.expert_ids != self.up_weights.expert_ids:
            raise ValueError("gate and up projections must use the same expert ids")
        if self.gate_weights.shape != self.up_weights.shape:
            raise ValueError("gate and up projections must have identical shapes")
        if self.gate_weights.dtype != self.up_weights.dtype or self.gate_weights.device != self.up_weights.device:
            raise ValueError("gate and up projections must use one dtype and device")

    @property
    def shape(self) -> tuple[int, int, int]:
        """Logical grouped-GEMM shape ``[E, 2 * I, H]``."""

        experts, intermediate_size, hidden_size = self.gate_weights.shape
        return experts, 2 * intermediate_size, hidden_size

    @property
    def dtype(self) -> torch.dtype:
        """Common projection dtype."""

        return self.gate_weights.dtype

    @property
    def device(self) -> torch.device:
        """Common projection device."""

        return self.gate_weights.device

    @property
    def format(self) -> WeightFormat:
        """Storage encoding available to an execution backend."""

        return self.gate_weights.format

    def for_expert(self, expert_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return original gate and up matrices for one expert."""

        return self.gate_weights.for_expert(expert_id), self.up_weights.for_expert(expert_id)

    def materialize(self) -> torch.Tensor:
        """Explicitly concatenate and stack ``[E, 2I, H]`` weights."""

        return torch.stack(
            tuple(
                torch.cat((gate, up), dim=0)
                for gate, up in zip(self.gate_weights.expert_weights, self.up_weights.expert_weights)
            ),
            dim=0,
        )


class ExpertWeightProvider(ABC):
    """Backend-independent interface for structured MoE expert weights."""

    @property
    @abstractmethod
    def expert_ids(self) -> tuple[int, ...]:
        """Global expert ids represented by this provider."""

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Input and output hidden size."""

    @property
    @abstractmethod
    def intermediate_size(self) -> int:
        """SwiGLU intermediate width per projection."""

    @property
    @abstractmethod
    def gate_up_weights(self) -> FusedGateUpWeightView:
        """Logical expert-major gate/up layout ``[E, 2I, H]``."""

    @property
    @abstractmethod
    def down_weights(self) -> ExpertMajorMatrixView:
        """Logical expert-major down projection layout ``[E, H, I]``."""

    @property
    @abstractmethod
    def has_bias(self) -> bool:
        """Whether any projection has a bias tensor."""


@dataclass(frozen=True, slots=True)
class QwenSwiGLUWeightProvider(ExpertWeightProvider):
    """Storage-preserving Qwen MoE SwiGLU weight provider.

    It holds tensor references only. No parameter is cloned, concatenated, or
    registered a second time, so the source Hugging Face module remains the
    sole owner of parameter storage.
    """

    _expert_ids: tuple[int, ...]
    _gate_up_weights: FusedGateUpWeightView
    _down_weights: ExpertMajorMatrixView
    gate_biases: tuple[torch.Tensor | None, ...]
    up_biases: tuple[torch.Tensor | None, ...]
    down_biases: tuple[torch.Tensor | None, ...]

    def __post_init__(self) -> None:
        if self._expert_ids != self._gate_up_weights.gate_weights.expert_ids:
            raise ValueError("gate/up expert ids must match provider expert ids")
        if self._expert_ids != self._down_weights.expert_ids:
            raise ValueError("down expert ids must match provider expert ids")
        if len(self.gate_biases) != len(self._expert_ids):
            raise ValueError("gate bias count must match expert count")
        if len(self.up_biases) != len(self._expert_ids):
            raise ValueError("up bias count must match expert count")
        if len(self.down_biases) != len(self._expert_ids):
            raise ValueError("down bias count must match expert count")
        experts, two_intermediate, hidden_size = self._gate_up_weights.shape
        down_experts, down_hidden_size, intermediate_size = self._down_weights.shape
        if experts != down_experts or hidden_size != down_hidden_size or two_intermediate != 2 * intermediate_size:
            raise ValueError("Qwen SwiGLU projection shapes are incompatible")

    @property
    def expert_ids(self) -> tuple[int, ...]:
        """Global expert ids in expert-major order."""

        return self._expert_ids

    @property
    def num_experts(self) -> int:
        """Number of represented experts."""

        return len(self._expert_ids)

    @property
    def hidden_size(self) -> int:
        """Input and output hidden size."""

        return self._gate_up_weights.shape[2]

    @property
    def intermediate_size(self) -> int:
        """Per-projection SwiGLU intermediate width."""

        return self._down_weights.shape[2]

    @property
    def gate_up_weights(self) -> FusedGateUpWeightView:
        """Logical ``[E, 2I, H]`` gate/up layout."""

        return self._gate_up_weights

    @property
    def down_weights(self) -> ExpertMajorMatrixView:
        """Logical ``[E, H, I]`` down projection layout."""

        return self._down_weights

    @property
    def has_bias(self) -> bool:
        """Whether any Qwen projection has a bias."""

        return any(bias is not None for bias in (*self.gate_biases, *self.up_biases, *self.down_biases))

    @property
    def weight_format(self) -> WeightFormat:
        """Common parameter encoding for backend dispatch."""

        return self._gate_up_weights.format

    @property
    def owns_weight_storage(self) -> bool:
        """Whether provider construction allocated parameter storage."""

        return False


def _linear_projection(expert: nn.Module, name: str) -> nn.Module:
    try:
        projection = getattr(expert, name)
    except AttributeError as exc:
        raise ValueError(f"Qwen SwiGLU expert is missing '{name}'") from exc
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise ValueError(f"Qwen SwiGLU projection '{name}' must expose a tensor weight")
    return projection


def build_qwen_swiglu_weight_provider(
    experts: tuple[tuple[int, nn.Module], ...],
) -> QwenSwiGLUWeightProvider:
    """Build a provider from Qwen-style experts without copying parameters."""

    if not experts:
        raise ValueError("Qwen SwiGLU provider requires at least one expert")
    expert_ids = tuple(expert_id for expert_id, _ in experts)
    gate_modules = tuple(_linear_projection(expert, "gate_proj") for _, expert in experts)
    up_modules = tuple(_linear_projection(expert, "up_proj") for _, expert in experts)
    down_modules = tuple(_linear_projection(expert, "down_proj") for _, expert in experts)
    gate_view = ExpertMajorMatrixView(expert_ids, tuple(module.weight for module in gate_modules), "gate_proj")
    up_view = ExpertMajorMatrixView(expert_ids, tuple(module.weight for module in up_modules), "up_proj")
    down_view = ExpertMajorMatrixView(expert_ids, tuple(module.weight for module in down_modules), "down_proj")
    return QwenSwiGLUWeightProvider(
        _expert_ids=expert_ids,
        _gate_up_weights=FusedGateUpWeightView(gate_view, up_view),
        _down_weights=down_view,
        gate_biases=tuple(getattr(module, "bias", None) for module in gate_modules),
        up_biases=tuple(getattr(module, "bias", None) for module in up_modules),
        down_biases=tuple(getattr(module, "bias", None) for module in down_modules),
    )
