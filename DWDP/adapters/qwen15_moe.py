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
from DWDP.comms_planner import CommunicationPlannerConfig, build_communication_planner
from DWDP.router import LinearTopKRouter, MetadataLevel, RouterConfig
from DWDP.scheduler import SchedulerConfig, SchedulerMetadataLevel, build_scheduler
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
        self.context = context if context is not None else RuntimeContext.from_config(config)
        # Populated by attach_expert_table() after distributed handle exchange.
        self.expert_table = None
        # Set by enable_cuda_graphs(); None means eager execution.
        self._graph_runner = None

        self._shared_stream: torch.cuda.Stream | None = None
        if self.shared_expert is not None and torch.cuda.is_available():
            self._shared_stream = torch.cuda.Stream()

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
        self.comms_planner = build_communication_planner(
            CommunicationPlannerConfig(
                planner_policy=config.communication_policy,
                deterministic=config.deterministic,
                enable_workspace=config.enable_workspace,
                world_size=config.world_size,
                local_rank=config.local_rank,
                enable_prefetch_metadata=False,
                enable_overlap_metadata=False,
                enable_topology_metadata=False,
                enable_cost_model=False,
                enable_statistics=False,
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

    def attach_expert_table(self, table) -> None:
        """Bind the post-handle-exchange GlobalExpertTable to this layer."""

        self.expert_table = table
        attach = getattr(self.comms_planner, "attach_expert_table", None)
        if attach is not None:
            attach(table)

    def _ep_decode_forward(self, hidden_states: torch.Tensor):
        """Decode-phase Expert-Parallel path (Project 1 hybrid executor).

        At small batch sizes the compute-to-weight ratio collapses, so
        streaming a whole expert's weights over NVLink can no longer be hidden
        behind the GEMM. EP is the cheaper side of the trade here: it moves the
        few token activations instead, via two all_to_all exchanges (dispatch
        then combine) around purely local expert compute.
        """

        router_output = self.router(hidden_states)
        topk_indices = router_output.topk_indices
        topk_weights = router_output.topk_weights

        world_size = self.config.world_size
        rank = self.config.local_rank
        num_tokens, hidden_size = hidden_states.shape
        experts_per_rank = max(1, (self.num_experts + world_size - 1) // world_size)

        # Flatten (token, slot) pairs and bucket them by owning rank.
        flat_experts = topk_indices.reshape(-1)
        flat_tokens = torch.arange(
            num_tokens, device=hidden_states.device
        ).repeat_interleave(topk_indices.shape[1])
        flat_weights = topk_weights.reshape(-1)
        target_ranks = torch.clamp(flat_experts // experts_per_rank, max=world_size - 1)

        order = torch.argsort(target_ranks, stable=True)
        sorted_ranks = target_ranks[order]
        send_tokens = hidden_states[flat_tokens[order]]
        send_experts = flat_experts[order]

        send_counts = torch.bincount(sorted_ranks, minlength=world_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)

        send_list = send_counts.tolist()
        recv_list = recv_counts.tolist()
        total_recv = int(sum(recv_list))

        # Dispatch: activations and their expert ids to the owning ranks.
        recv_tokens = torch.empty(
            (total_recv, hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        dist.all_to_all_single(
            recv_tokens, send_tokens, recv_list, send_list
        )
        recv_experts = torch.empty(
            total_recv, device=hidden_states.device, dtype=send_experts.dtype
        )
        dist.all_to_all_single(
            recv_experts, send_experts, recv_list, send_list
        )

        # Local compute: every received token is routed to a locally-owned expert.
        computed = torch.zeros_like(recv_tokens)
        if total_recv:
            local_lo = rank * experts_per_rank
            local_hi = min(self.num_experts, (rank + 1) * experts_per_rank)
            for expert_id in range(local_lo, local_hi):
                mask = recv_experts == expert_id
                if not bool(mask.any()):
                    continue
                module = self.executor.communication_engine.getWeight(expert_id).module
                computed[mask] = module(recv_tokens[mask]).to(computed.dtype)

        # Combine: results travel back along the reversed partition.
        returned = torch.empty_like(send_tokens)
        dist.all_to_all_single(
            returned, computed, send_list, recv_list
        )

        # Undo the sort, apply routing weights, and reduce the top-k slots.
        weighted = returned * flat_weights[order].unsqueeze(-1).to(returned.dtype)
        output = torch.zeros_like(hidden_states)
        output.index_add_(0, flat_tokens[order], weighted)

        return output, router_output

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
        communication_plan = self.comms_planner(
            execution_plan,
            workspace=workspaces.comms if workspaces is not None else None,
        )
        executor_output = self.executor(
            hidden_states,
            dispatch_plan,
            execution_plan,
            communication_plan,
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

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """Execute the MoE block through DWDP while preserving HF signature."""

        del args, kwargs
        workspaces = self.context.workspaces
        # Stream-Overlapped Shared Expert Execution
        shared_output = None
        if self.shared_expert is not None and self._shared_stream is not None:
            self._shared_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._shared_stream):
                shared_output = self.shared_expert(hidden_states)
                if self.shared_expert_gate is not None:
                    shared_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
        elif self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                shared_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output

        # Hybrid executor: decode-EP below the threshold, prefill-DWDP above it.
        if (
            hidden_states.shape[0] <= self.config.ep_batch_size_threshold
            and self.config.world_size > 1
            and dist is not None
            and dist.is_available()
            and dist.is_initialized()
        ):
            output, router_output = self._ep_decode_forward(hidden_states)
            if shared_output is not None:
                if self._shared_stream is not None:
                    torch.cuda.current_stream().wait_stream(self._shared_stream)
                output = output + shared_output
            if self.returns_router_logits:
                return output, router_output.router_logits
            return output

        if not self.config.enable_profiling:
            # CUDA graph path when enabled. On single-GPU the prefetch loop is
            # a no-op (empty remote list), so it stays outside to keep the
            # captured region free of the .tolist() device->host sync.
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
                communication_plan = self.comms_planner(
                    execution_plan,
                    workspace=workspaces.comms if workspaces is not None else None,
                )

                for expert_id in communication_plan.remote_expert_ids.tolist():
                    self.executor.communication_engine.prefetch(expert_id)
                for expert_id in communication_plan.remote_expert_ids.tolist():
                    self.executor.communication_engine.wait(expert_id)

                executor_output = self.executor(
                    hidden_states,
                    dispatch_plan,
                    execution_plan,
                    communication_plan,
                    workspace=workspaces.executor if workspaces is not None else None,
                )

                self.executor.communication_engine.swap_buffers()
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
            with torch.autograd.profiler.record_function("dwdp.comms_planner"):
                communication_plan = self.comms_planner(
                    execution_plan,
                    workspace=workspaces.comms if workspaces is not None else None,
                )
                
            with torch.autograd.profiler.record_function("dwdp.comms_prefetch"):
                for expert_id in communication_plan.remote_expert_ids.tolist():
                    self.executor.communication_engine.prefetch(expert_id)
                for expert_id in communication_plan.remote_expert_ids.tolist():
                    self.executor.communication_engine.wait(expert_id)

            with torch.autograd.profiler.record_function("dwdp.executor"):
                executor_output = self.executor(
                    hidden_states,
                    dispatch_plan,
                    execution_plan,
                    communication_plan,
                    workspace=workspaces.executor if workspaces is not None else None,
                )
                
            with torch.autograd.profiler.record_function("dwdp.comms_swap"):
                self.executor.communication_engine.swap_buffers()
            with torch.autograd.profiler.record_function("dwdp.merger"):
                merger_output = self.merger(
                    executor_output,
                    workspace=workspaces.merger if workspaces is not None else None,
                )
            output_hidden = merger_output.hidden_states
        output = output_hidden

        if shared_output is not None:
            if self._shared_stream is not None:
                torch.cuda.current_stream().wait_stream(self._shared_stream)
            output = output + shared_output

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

        self._bootstrap_distributed(specs, blocks)
        return len(specs)

    def _bootstrap_distributed(
        self, specs: tuple[MoELayerSpec, ...], blocks: list[DWDPMoEBlock]
    ) -> None:
        """Register CUDA IPC handles at load time, as the Project 1 spec requires.

        Each MoE layer owns its own expert modules and therefore its own
        communication engine, so the handle exchange runs per layer. The
        process group itself is initialized once by the first call.
        """

        if self.config.world_size <= 1:
            return

        from DWDP.communication.distributed import bootstrap_dwdp_distributed

        self.expert_tables = []
        for spec, block in zip(specs, blocks):
            engine = block.executor.communication_engine
            # The engine is normally constructed lazily on the first getWeight
            # call; IPC registration has to happen before the first forward.
            engine.ensure_native()
            table = bootstrap_dwdp_distributed(
                experts=block.executor.communication_engine._registry,
                communication_engine=engine,
                world_size=self.config.world_size,
                rank=self.config.local_rank,
            )
            if table is None:
                continue
            self.expert_tables.append(table)
            block.attach_expert_table(table)

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
