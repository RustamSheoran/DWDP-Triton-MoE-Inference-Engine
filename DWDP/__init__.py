"""DWDP package root."""

from . import (
    # Import runtime first: adapter modules depend on runtime.config and
    # importing adapters before runtime creates a package initialization cycle.
    runtime,
    adapters,
    benchmarking,
    dispatcher,
    executor,
    integration,
    merger,
    profiling,
    router,
    scheduler,
)
from .runtime import DWDPRuntime, RuntimeConfig
from .precision import get_device_precision_metadata, validate_execution_dtype

__all__ = [
    "DWDPRuntime",
    "RuntimeConfig",
    "adapters",
    "benchmarking",
    "dispatcher",
    "executor",
    "get_device_precision_metadata",
    "integration",
    "merger",
    "profiling",
    "router",
    "runtime",
    "scheduler",
    "validate_execution_dtype",
]
