from __future__ import annotations

from typing import Any

import torch
from torch import nn

from DWDP.dispatcher import DispatcherConfig, build_dispatcher
from DWDP.executor import ExecutorConfig, build_executor
from DWDP.merger import MergerConfig, build_merger
from DWDP.comms_planner import CommunicationPlannerConfig, build_communication_planner
from DWDP.router import LinearTopKRouter, RouterConfig
from DWDP.scheduler import SchedulerConfig, build_scheduler
from DWDP.runtime.config import RuntimeConfig
from DWDP.runtime.context import RuntimeContext

from .extractor import MoELayerSpec, discover_qwen_moe_layers
from .huggingface import HuggingFaceAdapter, _DelegatingRuntime
from .patcher import ModulePatcher
from .registry import register_adapter, register_model_adapter


class _HFProjectionRouter(LinearTopKRouter):
    """Use the original HF projection when its weights are bitsandbytes-packed."""

    def __init__(self, config: RouterConfig, projection: nn.Module) -> None:
        super().__init__(config)
        self.projection = projection

    def compute_router_logits(self, flat_hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.projection(flat_hidden_states)
        if self.config.score_scale != 1.0:
            router_logits = router_logits * self.config.score_scale
        return router_logits


class DWDPMoEBlock(nn.Module):
    """Hugging Face MoE block replacement backed by the DWDP pipeline."""

    def __init__(self, spec: MoELayerSpec, config: RuntimeConfig) -> None:
        super().__init__()
        self.layer_name = spec.name
        self.hidden_size = spec.hidden_size
        self.num_experts = spec.num_experts
        self.top_k = spec.top_k
        self.config = config
        self.returns_router_logits = spec.returns_router_logits

        router_config = RouterConfig(
            hidden_size=spec.hidden_size,
            num_experts=spec.num_experts,
            top_k=spec.top_k,
            bias=spec.gate.bias is not None,
            topk_sorted=False,
            renormalize=bool(getattr(spec.module, "norm_topk_prob", True)),
        )
        if torch.is_floating_point(spec.gate.weight):
            self.router = LinearTopKRouter(router_config)
            self.router.weight = spec.gate.weight
            if spec.gate.bias is not None:
                self.router.bias = spec.gate.bias
        else:
            # A bitsandbytes Linear4bit/Linear8bit weight cannot be passed to
            # torch.nn.functional.linear directly. Keep the HF projection so
            # bitsandbytes performs the dequantization during the matmul.
            self.router = _HFProjectionRouter(router_config, spec.gate)

        self.shared_expert = getattr(spec.module, "shared_expert", None)
        self.shared_expert_gate = getattr(spec.module, "shared_expert_gate", None)
        self.context = RuntimeContext.from_config(config)

        self.dispatcher = build_dispatcher(
            DispatcherConfig(
                num_experts=spec.num_experts,
                dispatcher_type=config.dispatcher_type,
            )
        )
        self.scheduler = build_scheduler(
            SchedulerConfig(
                scheduling_policy=config.scheduling_policy,
                deterministic=config.deterministic,
                enable_workspace=config.enable_workspace,
            )
        )
        self.comms_planner = build_communication_planner(
            CommunicationPlannerConfig(
                planner_policy=config.communication_policy,
                deterministic=config.deterministic,
                enable_workspace=config.enable_workspace,
                world_size=config.world_size,
                local_rank=config.local_rank,
            )
        )
        self.executor = build_executor(
            ExecutorConfig(
                backend=config.executor_backend,
                dtype=config.dtype,
                enable_workspace=config.enable_workspace,
                enable_statistics=config.enable_statistics,
                deterministic=config.deterministic,
            ),
            spec.experts,
        )
        self.merger = build_merger(
            MergerConfig(
                backend=config.merger_backend,
                enable_workspace=config.enable_workspace,
                enable_statistics=config.enable_statistics,
                deterministic=config.deterministic,
                apply_routing_weights=False,
            )
        )

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Execute the MoE block through DWDP while preserving HF signature."""

        del args, kwargs
        workspaces = self.context.workspaces
        with torch.autograd.profiler.record_function("dwdp.router"):
            router_output = self.router(hidden_states)
        with torch.autograd.profiler.record_function("dwdp.dispatcher"):
            dispatch_plan = self.dispatcher(
                router_output,
                workspace=workspaces.dispatch if workspaces is not None else None,
            )
        with torch.autograd.profiler.record_function("dwdp.scheduler"):
            execution_plan = self.scheduler(
                dispatch_plan,
                workspace=workspaces.scheduler if workspaces is not None else None,
            )
        with torch.autograd.profiler.record_function("dwdp.comms_planner"):
            communication_plan = self.comms_planner(
                execution_plan,
                workspace=workspaces.comms if workspaces is not None else None,
            )
        with torch.autograd.profiler.record_function("dwdp.executor"):
            executor_output = self.executor(
                hidden_states,
                dispatch_plan,
                execution_plan,
                communication_plan,
                workspace=workspaces.executor if workspaces is not None else None,
            )
        with torch.autograd.profiler.record_function("dwdp.merger"):
            merger_output = self.merger(
                executor_output,
                workspace=workspaces.merger if workspaces is not None else None,
            )
        output = merger_output.hidden_states

        if self.shared_expert is not None:
            shared = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared
            output = output + shared

        if self.returns_router_logits:
            return output, router_output.router_logits
        return output


class Qwen15MoEAdapter(HuggingFaceAdapter):
    """Automatic DWDP adapter for Qwen1.5/Qwen2-style Hugging Face MoE models."""

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, *, config: RuntimeConfig | None = None, **kwargs) -> "Qwen15MoEAdapter":
        """Load a supported Hugging Face Qwen MoE model and patch its MoE blocks."""

        base = super().from_pretrained(model_name_or_path, config=config, **kwargs)
        adapter = cls(model=base.model, tokenizer=base.tokenizer, config=base.config)
        adapter.patch_model()
        return adapter

    @classmethod
    def supports(cls, model: Any) -> bool:
        """Return whether this adapter supports the provided HF model."""

        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        class_name = type(model).__name__.lower()
        architectures = " ".join(str(item).lower() for item in getattr(config, "architectures", ()) or ())
        if "qwen" not in model_type and "qwen" not in class_name and "qwen" not in architectures:
            return False
        return bool(discover_qwen_moe_layers(model))

    def patch_model(self) -> int:
        """Automatically discover and replace Qwen MoE blocks."""

        if self.model is None:
            raise RuntimeError("cannot patch without a Hugging Face model")
        if hasattr(self, "_patcher") and self._patcher.records:
            return len(self._patcher.records)
        specs = discover_qwen_moe_layers(self.model)
        if not specs:
            raise ValueError("no supported Qwen MoE layers were discovered")
        self._patcher = ModulePatcher()
        self.moe_layer_specs = specs
        for spec in specs:
            replacement = DWDPMoEBlock(spec, self.config)
            self._patcher.replace(
                name=spec.name,
                parent=spec.parent,
                child_name=spec.child_name,
                replacement=replacement,
            )
        return len(specs)

    def restore_model(self) -> int:
        """Restore native Hugging Face MoE blocks."""

        if not hasattr(self, "_patcher"):
            return 0
        return self._patcher.restore()

    def create_runtime(self):
        """Return a HF-compatible runtime wrapper around the patched model."""

        if self.model is None:
            raise RuntimeError("adapter has no wrapped model")
        if not hasattr(self, "_patcher") or not self._patcher.records:
            self.patch_model()
        return _DelegatingRuntime(adapter=self, config=self.config)


register_adapter("qwen15_moe", Qwen15MoEAdapter)
register_model_adapter(("qwen2_moe", "qwen1.5-moe", "qwenmoe", "qwen2moe"), Qwen15MoEAdapter)
