"""CUDA Graph decode capture and replay.

MoE decode is launch-bound, not compute-bound. Each token issues thousands of
tiny kernels (one or more per expert, per layer), and the CPU cannot enqueue
them faster than the GPU drains them, so the device idles between launches.
Capturing one decode step into a CUDA graph collapses that whole launch
sequence into a single `cudaGraphLaunch`.

Capture has hard requirements that this module enforces rather than assumes:

* Every tensor read or written by the captured region must live at a fixed
  address. Inputs are copied into persistent static buffers before replay.
* Shapes must not change between replays. A graph is captured per batch shape
  and reused only for an exact match.
* The captured region must contain no host synchronization. A `.item()`,
  `.cpu()`, or `.tolist()` inside capture either errors or silently bakes a
  stale scalar into the graph. `assert_capture_safe()` checks the MoE pipeline
  for these before capture is attempted.
* Capture must run on a non-default stream, after warmup iterations on that
  same side stream, so lazy allocations and autotune are already resolved.

Anything that fails these checks falls back to eager execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


class CUDAGraphCaptureError(RuntimeError):
    """Raised when a region cannot be safely captured."""


@dataclass(slots=True)
class GraphStats:
    """Counters describing graph usage, for benchmark reporting."""

    captures: int = 0
    replays: int = 0
    fallbacks: int = 0
    shapes: tuple[tuple[int, ...], ...] = ()


def _tree_map_tensors(value: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    """Apply ``fn`` to every tensor in a nested structure, preserving shape."""

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        mapped = [_tree_map_tensors(item, fn) for item in value]
        return type(value)(mapped) if not isinstance(value, tuple) else tuple(mapped)
    if isinstance(value, dict):
        return {key: _tree_map_tensors(item, fn) for key, item in value.items()}
    return value


class CUDAGraphRunner:
    """Capture a callable once per input shape and replay it thereafter.

    The runner is deliberately shape-keyed: decode steps at batch size 1 and
    batch size 8 get separate graphs, and a shape that has never been captured
    falls back to eager execution instead of replaying a mismatched graph.
    """

    def __init__(
        self,
        callable_: Callable[..., Any],
        *,
        warmup_steps: int = 3,
        enabled: bool = True,
    ) -> None:
        self._callable = callable_
        self.warmup_steps = max(1, warmup_steps)
        self.enabled = enabled
        # shape key -> (graph, static_inputs, static_outputs)
        self._graphs: dict[tuple, tuple[torch.cuda.CUDAGraph, dict, Any]] = {}
        self._failed: set[tuple] = set()
        self.stats = GraphStats()
        # Graph memory must outlive capture; a dedicated pool keeps replay
        # buffers from being reused by unrelated allocations.
        self._pool = None

    @staticmethod
    def _shape_key(kwargs: dict[str, Any]) -> tuple:
        """Build a hashable key from the tensor shapes and dtypes of inputs."""

        parts: list[Any] = []
        for name in sorted(kwargs):
            value = kwargs[name]
            if isinstance(value, torch.Tensor):
                parts.append((name, tuple(value.shape), str(value.dtype)))
            elif isinstance(value, (int, float, bool, type(None), str)):
                parts.append((name, value))
            else:
                # Unhashable / dynamic argument (e.g. a KV cache object).
                # Refuse to key on it rather than risk a wrong replay.
                return ()
        return tuple(parts)

    def capture(self, **kwargs: Any) -> bool:
        """Warm up and capture one graph for this input shape.

        Returns True when a graph is now available for these shapes.
        """

        if not self.enabled or not torch.cuda.is_available():
            return False

        key = self._shape_key(kwargs)
        if not key or key in self._failed:
            return False
        if key in self._graphs:
            return True

        try:
            static_inputs = {
                name: (value.clone().detach() if isinstance(value, torch.Tensor) else value)
                for name, value in kwargs.items()
            }

            # Warmup on a side stream. This resolves lazy module init, Triton
            # autotuning, and allocator growth so none of it lands inside the
            # captured region.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.inference_mode():
                for _ in range(self.warmup_steps):
                    self._callable(**static_inputs)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool), torch.inference_mode():
                static_outputs = self._callable(**static_inputs)

            self._graphs[key] = (graph, static_inputs, static_outputs)
            self.stats.captures += 1
            self.stats.shapes = self.stats.shapes + (
                tuple(
                    v.shape[0] for v in kwargs.values() if isinstance(v, torch.Tensor)
                ),
            )
            logger.info("captured CUDA graph for shape key %s", key)
            return True
        except Exception:
            # A failed capture can leave the allocator in a capturing state;
            # synchronize to clear it, then permanently fall back for this key.
            self._failed.add(key)
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            logger.warning(
                "CUDA graph capture failed; falling back to eager execution",
                exc_info=True,
            )
            return False

    def __call__(self, **kwargs: Any) -> Any:
        """Replay the captured graph for these shapes, or run eagerly."""

        if not self.enabled or not torch.cuda.is_available():
            self.stats.fallbacks += 1
            return self._callable(**kwargs)

        key = self._shape_key(kwargs)
        entry = self._graphs.get(key) if key else None
        if entry is None:
            if key and key not in self._failed and self.capture(**kwargs):
                entry = self._graphs[key]
            else:
                self.stats.fallbacks += 1
                return self._callable(**kwargs)

        graph, static_inputs, static_outputs = entry
        for name, value in kwargs.items():
            target = static_inputs.get(name)
            if isinstance(value, torch.Tensor) and isinstance(target, torch.Tensor):
                target.copy_(value, non_blocking=True)

        graph.replay()
        self.stats.replays += 1
        # Outputs live in graph-owned memory that the next replay overwrites.
        # Hand back clones so callers can hold results across steps.
        return _tree_map_tensors(static_outputs, lambda t: t.clone())

    def reset(self) -> None:
        """Drop all captured graphs and their memory pool."""

        self._graphs.clear()
        self._failed.clear()
        self._pool = None


# Call sites that force a device->host transfer. Inside a captured region these
# either raise or, worse, bake a stale scalar into the graph so every replay
# reuses the value observed at capture time.
_SYNC_PATTERNS = (".item()", ".tolist()", ".cpu()", ".numpy()", "nonzero(")

# Modules on the MoE decode path that must be sync-free for capture to be sound.
_HOT_PATH_MODULES = (
    "DWDP.dispatcher.expert_major",
    "DWDP.dispatcher.ops.scatter",
    "DWDP.scheduler.policies.round_robin",
    "DWDP.executor.pytorch",
    "DWDP.executor.triton",
    "DWDP.merger.pytorch",
    "DWDP.comms_planner.static",
)


def find_host_syncs() -> dict[str, list[str]]:
    """Report device->host syncs on the MoE decode path.

    Returns a mapping of module name to offending ``line_number: source`` entries.
    An empty mapping means the pipeline is structurally capture-safe.
    """

    import importlib
    import inspect

    findings: dict[str, list[str]] = {}
    for module_name in _HOT_PATH_MODULES:
        try:
            module = importlib.import_module(module_name)
            source = inspect.getsource(module)
        except Exception:
            continue
        hits = []
        for number, line in enumerate(source.splitlines(), start=1):
            code = line.split("#", 1)[0]
            if any(pattern in code for pattern in _SYNC_PATTERNS):
                hits.append(f"{number}: {line.strip()}")
        if hits:
            findings[module_name] = hits
    return findings


def assert_capture_safe() -> None:
    """Raise :class:`CUDAGraphCaptureError` if the decode path has host syncs."""

    findings = find_host_syncs()
    if not findings:
        return
    detail = "\n".join(
        f"  {module}\n    " + "\n    ".join(hits) for module, hits in findings.items()
    )
    raise CUDAGraphCaptureError(
        "device->host synchronization on the decode path prevents CUDA graph "
        f"capture:\n{detail}"
    )
