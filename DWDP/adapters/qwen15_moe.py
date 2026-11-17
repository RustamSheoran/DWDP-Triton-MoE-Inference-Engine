from __future__ import annotations

from typing import Any

import torch
from torch import nn

try:
    import torch.distributed as dist
except ImportError:  # pragma: no cover - torch always ships distributed on CUDA builds
    dist = None

from DWDP.dispatcher import DispatcherConfig, build_dispatcher
from DWDP.executor import ExecutorConfig, build_executor
from DWDP.merger import MergerConfig, build_merger
from DWDP.router import LinearTopKRouter, MetadataLevel, RouterConfig
from DWDP.scheduler import SchedulerConfig, SchedulerMetadataLevel, build_scheduler
from DWDP.runtime.config import RuntimeConfig
from DWDP.runtime.context import RuntimeContext

from .extractor import MoELayerSpec, discover_qwen_moe_layers
from .huggingface import HuggingFaceAdapter, _DelegatingRuntime
from .patcher import ModulePatcher
from .registry import register_adapter, register_model_adapter


class DWDPMoEBlock(nn.Module):
    """Hugging Face MoE block replacement backed by the DWDP pipeline."""

    def __init__(self, spec: MoELayerSpec, config: RuntimeConfig, *, context: RuntimeContext | None = None) -> None:
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
            metadata_level=MetadataLevel.COUNTS,
        )
        if not torch.is_floating_point(spec.gate.weight):
            raise ValueError(
                "DWDP does not support quantized/bitsandbytes weights. Only fp16 (or fp32 for CPU reference) is supported."
            )
        self.router = LinearTopKRouter(router_config)
        self.router.weight = spec.gate.weight
        if spec.gate.bias is not None:
            self.router.bias = spec.gate.bias

        self.shared_expert = getattr(spec.module, "shared_expert", None)
        self.shared_expert_gate = getattr(spec.module, "shared_expert_gate", None)
        self.context = context if context is not None else RuntimeContext.from_config(config)
        # Populated by attach_expert_table() after distributed handle exchange.
        self.expert_table = None
        # Set by enable_cuda_graphs(); None means eager execution.
        self._graph_runner = None

        # The reference "counting_scatter" dispatch computes destination slots
        # in a Python loop over every token-expert assignment on the host. That
        # is the largest single source of decode overhead and it blocks CUDA
        # graph capture. Prefer the device-resident Triton path when it is
        # actually available; the two are equivalence-tested.
        dispatch_algorithm = "counting_scatter"
        if torch.cuda.is_available():
            try:
                from DWDP.dispatcher.kernels.triton import TRITON_AVAILABLE

                if TRITON_AVAILABLE:
                    dispatch_algorithm = "triton_counting_scatter"
            except ImportError:
                pass

        self.dispatcher = build_dispatcher(
            DispatcherConfig(
                num_experts=spec.num_experts,
                dispatcher_type=config.dispatcher_type,
                validate_inputs=False,
                algorithm=dispatch_algorithm,
            )
        )
        self.scheduler = build_scheduler(
            SchedulerConfig(
                scheduling_policy=config.scheduling_policy,
                deterministic=config.deterministic,
                enable_workspace=config.enable_workspace,
                metadata_level=SchedulerMetadataLevel.MINIMAL,
                enable_dependency_metadata=False,
                enable_barrier_metadata=False,
            )
        )
        self.executor = build_executor(
            ExecutorConfig(
                backend=config.executor_backend,
                dtype=config.dtype,
                enable_workspace=config.enable_workspace,
                enable_statistics=config.enable_statistics,
                enable_profiling=config.enable_profiling,
                deterministic=config.deterministic,
                # The merger below runs with apply_routing_weights=False, so it
                # consumes weighted outputs and never reads the packed copy.
                # Skipping it saves a [num_tokens * top_k, hidden] buffer per
                # forward.
                materialize_packed_outputs=False,
                accumulate_token_outputs=config.executor_backend == "pytorch",
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

    def _dwdp_pipeline(self, hidden_states: torch.Tensor):
        """Run router -> dispatch -> schedule -> plan -> execute -> merge.

        Returns ``(hidden_states, router_logits)``. Only these two tensors
        cross the CUDA graph boundary; the rest of RouterOutput stays inside so
        replay does not clone metadata the caller never reads. Returning the
        logits rather than recomputing them outside avoids running the router
        twice per token, since this adapter reports router logits by default.

        Kept free of host syncs so the region is capturable. The remote-expert
        prefetch loop lives in ``forward`` because it calls ``.tolist()``.
        """

        workspaces = self.context.workspaces
        router_output = self.router(hidden_states)
        dispatch_plan = self.dispatcher(
            router_output,
            workspace=workspaces.dispatch if workspaces is not None else None,
        )
        execution_plan = self.scheduler(
            dispatch_plan,
            workspace=workspaces.scheduler if workspaces is not None else None,
        )
        executor_output = self.executor(
            hidden_states,
            dispatch_plan,
            execution_plan,
            workspace=workspaces.executor if workspaces is not None else None,
        )
        merger_output = self.merger(
            executor_output,
            workspace=workspaces.merger if workspaces is not None else None,
        )
        return merger_output.hidden_states, router_output.router_logits

    def enable_cuda_graphs(self, warmup_steps: int = 3) -> bool:
        """Route the DWDP pipeline through a shape-keyed CUDA graph runner.

        Returns True when graph execution is active for this block. Capture is
        lazy: the first forward at each new shape captures, later ones replay.
        """

        if not torch.cuda.is_available():
            return False
        from DWDP.runtime.cuda_graph import CUDAGraphRunner

        self._graph_runner = CUDAGraphRunner(
            lambda hidden_states: self._dwdp_pipeline(hidden_states),
            warmup_steps=warmup_steps,
        )
        return True

    def _launch_shared_expert(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """Start shared-expert work before routed execution when CUDA allows."""

        if self.shared_expert is None:
            return None
        streams = getattr(self.context, "streams", None)
        if streams is None or not getattr(streams, "initialized", False):
            return None
        shared_stream = streams.compute
        shared_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(shared_stream):
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
                if torch.is_grad_enabled():
                    shared_output = gate * shared_output
                else:
                    shared_output.mul_(gate)
        return shared_output

    def _add_shared_expert(
        self,
        routed_output: torch.Tensor,
        hidden_states: torch.Tensor,
        shared_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine shared-expert output after routed execution completes."""

        if self.shared_expert is None:
            return routed_output
        if shared_output is None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
                if torch.is_grad_enabled():
                    shared_output = gate * shared_output
                else:
                    shared_output.mul_(gate)
        else:
            streams = getattr(self.context, "streams", None)
            if streams is not None and getattr(streams, "initialized", False):
                torch.cuda.current_stream().wait_stream(streams.compute)
        if torch.is_grad_enabled():
            return routed_output + shared_output
        routed_output.add_(shared_output)
        return routed_output

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Execute the MoE block through DWDP while preserving HF signature."""

        del args, kwargs
        self.context.prepare_for(hidden_states.device)
        workspaces = self.context.workspaces
        shared_output = self._launch_shared_expert(hidden_states)

        # Hybrid executor: decode-EP below the threshold, prefill-DWDP above it.
        if not self.config.enable_profiling:
            if self._graph_runner is not None:
                output_hidden, router_logits = self._graph_runner(
                    hidden_states=hidden_states
                )
            else:
                router_output = self.router(hidden_states)
                router_logits = router_output.router_logits
                dispatch_plan = self.dispatcher(
                    router_output,
                    workspace=workspaces.dispatch if workspaces is not None else None,
                )
                execution_plan = self.scheduler(
                    dispatch_plan,
                    workspace=workspaces.scheduler if workspaces is not None else None,
                )
                executor_output = self.executor(
                    hidden_states,
                    dispatch_plan,
                    execution_plan,
                    workspace=workspaces.executor if workspaces is not None else None,
                )
                merger_output = self.merger(
                    executor_output,
                    workspace=workspaces.merger if workspaces is not None else None,
                )
                output_hidden = merger_output.hidden_states
        else:
            with torch.autograd.profiler.record_function("dwdp.router"):
                router_output = self.router(hidden_states)
                router_logits = router_output.router_logits
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
            with torch.autograd.profiler.record_function("dwdp.executor"):
                executor_output = self.executor(
                    hidden_states,
                    dispatch_plan,
                    execution_plan,
                    workspace=workspaces.executor if workspaces is not None else None,
                )
            with torch.autograd.profiler.record_function("dwdp.merger"):
                merger_output = self.merger(
                    executor_output,
                    workspace=workspaces.merger if workspaces is not None else None,
                )
            output_hidden = merger_output.hidden_states
        output = self._add_shared_expert(output_hidden, hidden_states, shared_output)

        if self.returns_router_logits:
            return output, router_logits
        return output


class Qwen15MoEAdapter(HuggingFaceAdapter):
    """Automatic DWDP adapter for Qwen1.5/Qwen2-style Hugging Face MoE models."""

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, *, config: RuntimeConfig | None = None, **kwargs
    ) -> "Qwen15MoEAdapter":
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
        architectures = " ".join(
            str(item).lower() for item in getattr(config, "architectures", ()) or ()
        )
        if (
            "qwen" not in model_type
            and "qwen" not in class_name
            and "qwen" not in architectures
        ):
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
        shared_context = RuntimeContext.from_config(self.config)

        blocks: list[DWDPMoEBlock] = []
        for spec in specs:
            replacement = DWDPMoEBlock(spec, self.config, context=shared_context)
            blocks.append(replacement)
            self._patcher.replace(
                name=spec.name,
                parent=spec.parent,
                child_name=spec.child_name,
                replacement=replacement,
            )

        return len(specs)

    def enable_cuda_graphs(self, warmup_steps: int = 3) -> int:
        """Enable CUDA graph capture on every patched MoE block.

        Returns the number of blocks now backed by a graph runner. Capture is
        lazy: the first forward at each input shape captures, later ones replay.
        """

        if not hasattr(self, "_patcher") or not self._patcher.records:
            raise RuntimeError("patch_model() must run before enabling CUDA graphs")

        enabled = 0
        for record in self._patcher.records:
            block = record.replacement
            if isinstance(block, DWDPMoEBlock) and block.enable_cuda_graphs(
                warmup_steps=warmup_steps
            ):
                enabled += 1
        return enabled

    def graph_statistics(self) -> dict[str, int]:
        """Aggregate capture/replay counters across patched blocks."""

        totals = {"captures": 0, "replays": 0, "fallbacks": 0}
        if not hasattr(self, "_patcher"):
            return totals
        for record in self._patcher.records:
            runner = getattr(record.replacement, "_graph_runner", None)
            if runner is None:
                continue
            totals["captures"] += runner.stats.captures
            totals["replays"] += runner.stats.replays
            totals["fallbacks"] += runner.stats.fallbacks
        return totals

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
register_model_adapter(
    ("qwen2_moe", "qwen1.5-moe", "qwenmoe", "qwen2moe"), Qwen15MoEAdapter
)
