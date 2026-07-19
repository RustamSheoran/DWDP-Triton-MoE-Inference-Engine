"""DWDP package root."""

from . import (
    # Import runtime first: adapter modules depend on runtime.config and
    # importing adapters before runtime creates a package initialization cycle.
    runtime,
    adapters,
    benchmarking,
    comms_planner,
    dispatcher,
    executor,
    integration,
    merger,
    profiling,
    router,
    scheduler,
)
from .runtime import DWDPRuntime, RuntimeConfig

__all__ = [
    "DWDPRuntime",
    "RuntimeConfig",
    "adapters",
    "benchmarking",
    "comms_planner",
    "dispatcher",
    "executor",
    "integration",
    "merger",
    "profiling",
    "router",
    "runtime",
    "scheduler",
]
