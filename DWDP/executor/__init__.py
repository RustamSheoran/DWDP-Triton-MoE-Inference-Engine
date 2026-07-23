"""Expert execution primitives for the DWDP MoE runtime."""

from .config import ExecutorConfig
from .experts import ExpertBatch, ExpertExecutionContext, ExpertRegistry
from .outputs import ExecutionMetadata, ExecutionStatistics, ExecutorOutput, ExpertOutput, OutputMetadata
from .pytorch import PyTorchExecutor
from .registry import build_executor, get_executor_class, register_executor
from .triton import TritonExpertExecutor
from .weights import ExpertMajorMatrixView, ExpertWeightProvider, FusedGateUpWeightView, QwenSwiGLUWeightProvider, WeightFormat
from .workspace import ExecutorWorkspace

__all__ = [
    "ExecutionMetadata",
    "ExecutionStatistics",
    "ExecutorConfig",
    "ExecutorOutput",
    "ExecutorWorkspace",
    "ExpertBatch",
    "ExpertExecutionContext",
    "ExpertMajorMatrixView",
    "ExpertOutput",
    "ExpertRegistry",
    "ExpertWeightProvider",
    "FusedGateUpWeightView",
    "OutputMetadata",
    "PyTorchExecutor",
    "QwenSwiGLUWeightProvider",
    "TritonExpertExecutor",
    "WeightFormat",
    "build_executor",
    "get_executor_class",
    "register_executor",
]
