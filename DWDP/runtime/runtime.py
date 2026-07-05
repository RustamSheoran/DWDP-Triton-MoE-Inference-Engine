from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from DWDP.adapters import BaseModelAdapter, build_adapter, get_adapter_class
from DWDP.comms_planner import CommunicationPlannerConfig, build_communication_planner
from DWDP.dispatcher import DispatcherConfig, build_dispatcher
from DWDP.executor import ExecutorConfig, ExpertRegistry, build_executor
from DWDP.merger import MergerConfig, build_merger
from DWDP.router import RouterConfig, build_router
from DWDP.scheduler import SchedulerConfig, build_scheduler

from .config import RuntimeConfig
from .context import RuntimeContext
from .correctness import CorrectnessReport, compare_tensors
from .pipeline import RuntimePipelineOutput
from .profiler import RuntimeProfiler
from .registry import register_runtime


class DWDPRuntime(nn.Module):
    """Orchestrates the complete DWDP Router-to-Merger MoE pipeline."""

    def __init__(
        self,
        *,
        router: nn.Module,
        dispatcher: nn.Module,
        scheduler: nn.Module,
        comms_planner: nn.Module,
        executor: nn.Module,
        merger: nn.Module,
        config: RuntimeConfig | None = None,
        adapter: BaseModelAdapter | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RuntimeConfig()
        self.router = router
        self.dispatcher = dispatcher
        self.scheduler = scheduler
        self.comms_planner = comms_planner
        self.executor = executor
        self.merger = merger
        self.adapter = adapter
        self.context = RuntimeContext.from_config(self.config)

    @classmethod
    def build_reference(
        cls,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        experts: Mapping[int, nn.Module] | Sequence[nn.Module],
        config: RuntimeConfig | None = None,
        router: nn.Module | None = None,
    ) -> "DWDPRuntime":
        """Build a single-GPU reference runtime from explicit MoE components."""

        runtime_config = config or RuntimeConfig()
        expert_registry = experts if isinstance(experts, ExpertRegistry) else ExpertRegistry(experts)
        router_module = router or build_router(
            RouterConfig(
                hidden_size=hidden_size,
                num_experts=num_experts,
                top_k=top_k,
                router_type=runtime_config.router_type,
            )
        )
        return cls(
            router=router_module,
            dispatcher=build_dispatcher(
                DispatcherConfig(
                    num_experts=num_experts,
                    dispatcher_type=runtime_config.dispatcher_type,
                )
            ),
            scheduler=build_scheduler(
                SchedulerConfig(
                    scheduling_policy=runtime_config.scheduling_policy,
                    deterministic=runtime_config.deterministic,
                    enable_workspace=runtime_config.enable_workspace,
                )
            ),
            comms_planner=build_communication_planner(
                CommunicationPlannerConfig(
                    planner_policy=runtime_config.communication_policy,
                    deterministic=runtime_config.deterministic,
                    enable_workspace=runtime_config.enable_workspace,
                    world_size=runtime_config.world_size,
                    local_rank=runtime_config.local_rank,
                )
            ),
            executor=build_executor(
                ExecutorConfig(
                    backend=runtime_config.executor_backend,
                    dtype=runtime_config.dtype,
                    enable_workspace=runtime_config.enable_workspace,
                    enable_statistics=runtime_config.enable_statistics,
                    deterministic=runtime_config.deterministic,
                ),
                expert_registry,
            ),
            merger=build_merger(
                MergerConfig(
                    backend=runtime_config.merger_backend,
                    enable_workspace=runtime_config.enable_workspace,
                    enable_statistics=runtime_config.enable_statistics,
                    deterministic=runtime_config.deterministic,
                )
            ),
            config=runtime_config,
        )

    @classmethod
    def wrap(cls, model: Any, *, config: RuntimeConfig | None = None, tokenizer: Any | None = None) -> "DWDPRuntime":
        """Wrap an existing Hugging Face model with a DWDP adapter."""

        runtime_config = config or RuntimeConfig()
        adapter = build_adapter(runtime_config.adapter, model=model, tokenizer=tokenizer, config=runtime_config)
        return adapter.create_runtime()

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, config: RuntimeConfig | None = None, **kwargs) -> "DWDPRuntime":
        """Load a model through the configured adapter and return a DWDP runtime."""

        runtime_config = config or RuntimeConfig()
        adapter_cls = get_adapter_class(runtime_config.adapter)
        adapter = adapter_cls.from_pretrained(model_name_or_path, config=runtime_config, **kwargs)
        return adapter.create_runtime()

    def forward(self, hidden_states: torch.Tensor) -> RuntimePipelineOutput:
        """Execute one complete DWDP MoE layer pipeline."""

        profiler = RuntimeProfiler(enabled=self.config.enable_profiling)
        profiler.start()
        workspaces = self.context.workspaces

        with profiler.record("router"):
            router_output = self.router(hidden_states)
        with profiler.record("dispatcher"):
            dispatch_plan = self.dispatcher(
                router_output,
                workspace=workspaces.dispatch if workspaces is not None else None,
            )
        with profiler.record("scheduler"):
            execution_plan = self.scheduler(
                dispatch_plan,
                workspace=workspaces.scheduler if workspaces is not None else None,
            )
        with profiler.record("comms_planner"):
            communication_plan = self.comms_planner(
                execution_plan,
                workspace=workspaces.comms if workspaces is not None else None,
            )
        with profiler.record("executor"):
            executor_output = self.executor(
                hidden_states,
                dispatch_plan,
                execution_plan,
                communication_plan,
                workspace=workspaces.executor if workspaces is not None else None,
            )
        with profiler.record("merger"):
            merger_output = self.merger(
                executor_output,
                workspace=workspaces.merger if workspaces is not None else None,
            )

        workspace_bytes = workspaces.estimated_bytes() if workspaces is not None else 0
        return RuntimePipelineOutput(
            router_output=router_output,
            dispatch_plan=dispatch_plan,
            execution_plan=execution_plan,
            communication_plan=communication_plan,
            executor_output=executor_output,
            merger_output=merger_output,
            profile=profiler.finish(workspace_bytes=workspace_bytes),
        )

    def generate(self, *args, **kwargs):
        """Generate text/tokens through the adapter when a full model is wrapped."""

        if self.adapter is None:
            raise RuntimeError("generate() requires a model adapter; use DWDPRuntime.wrap or from_pretrained")
        return self.adapter.generate(*args, **kwargs)

    def compile(self) -> "DWDPRuntime":
        """Compile compile-capable stage modules with `torch.compile`."""

        if not self.config.torch_compile:
            return self
        self.router = torch.compile(self.router)
        self.dispatcher = torch.compile(self.dispatcher)
        self.scheduler = torch.compile(self.scheduler)
        self.comms_planner = torch.compile(self.comms_planner)
        self.executor = torch.compile(self.executor)
        self.merger = torch.compile(self.merger)
        return self

    def profile(self, hidden_states: torch.Tensor):
        """Run a profiled forward pass and return the profile object."""

        from dataclasses import asdict

        previous = self.config
        object.__setattr__(self, "config", RuntimeConfig(**{**asdict(previous), "enable_profiling": True}))
        try:
            return self.forward(hidden_states).profile
        finally:
            object.__setattr__(self, "config", previous)

    def benchmark(self, hidden_states: torch.Tensor, *, warmup: int = 5, iters: int = 20) -> dict[str, float]:
        """Benchmark repeated forward passes without executing generation."""

        import time

        with torch.no_grad():
            for _ in range(warmup):
                self.forward(hidden_states)
            if hidden_states.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iters):
                self.forward(hidden_states)
            if hidden_states.device.type == "cuda":
                torch.cuda.synchronize()
        latency = (time.perf_counter() - start) / iters
        tokens = hidden_states.numel() // hidden_states.shape[-1]
        return {
            "latency_us": latency * 1e6,
            "tokens_per_second": tokens / latency,
            "workspace_bytes": float(self.context.workspaces.estimated_bytes() if self.context.workspaces else 0),
        }

    def validate_against(self, reference: torch.Tensor, hidden_states: torch.Tensor, *, rtol: float = 1e-4, atol: float = 1e-4) -> CorrectnessReport:
        """Compare DWDP hidden-state output against a reference tensor."""

        actual = self.forward(hidden_states).hidden_states
        return CorrectnessReport(tensor=compare_tensors(reference, actual, rtol=rtol, atol=atol))


register_runtime("dwdp", DWDPRuntime)
