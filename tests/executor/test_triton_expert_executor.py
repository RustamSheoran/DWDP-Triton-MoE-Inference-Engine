from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from DWDP.comms_planner import (  # noqa: E402
    CommunicationCostModel,
    CommunicationGraph,
    CommunicationPlan,
    CommunicationStatistics,
    DependencyMetadata as CommunicationDependencyMetadata,
    OverlapPlan,
    PrefetchPlan,
    SynchronizationMetadata as CommunicationSynchronizationMetadata,
    TopologyMetadata,
)
from DWDP.dispatcher import DispatchMetadata, DispatchPlan, ExpertAssignments  # noqa: E402
from DWDP.executor import (  # noqa: E402
    ExecutorConfig,
    ExpertRegistry,
    PyTorchExecutor,
    TritonExpertExecutor,
    build_executor,
)
from DWDP.executor.extractors import extract_qwen_swiglu_weight_provider  # noqa: E402
from DWDP.scheduler import (  # noqa: E402
    DependencyMetadata as SchedulerDependencyMetadata,
    ExecutionPlan,
    SchedulerStatistics,
    SynchronizationMetadata as SchedulerSynchronizationMetadata,
)


class QwenSwiGLUExpert(torch.nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def make_registry() -> ExpertRegistry:
    torch.manual_seed(7)
    return ExpertRegistry([QwenSwiGLUExpert(4, 8), QwenSwiGLUExpert(4, 8)])


def make_dispatch_plan() -> DispatchPlan:
    assignments = ExpertAssignments(
        expert_ids=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        packed_token_indices=torch.tensor([0, 2, 1, 3], dtype=torch.int64),
        packed_routing_weights=torch.tensor([0.5, 1.0, 0.25, 0.75], dtype=torch.float32),
    )
    return DispatchPlan(
        assignments=assignments,
        metadata=DispatchMetadata(
            num_tokens=4,
            num_assignments=4,
            num_experts=2,
            top_k=1,
            token_shape=(4,),
            expert_counts=torch.tensor([2, 2], dtype=torch.int64),
            expert_offsets=torch.tensor([0, 2, 4], dtype=torch.int64),
            token_permutation=torch.arange(4, dtype=torch.int64),
            inverse_permutation=torch.arange(4, dtype=torch.int64),
            destination_positions=torch.arange(4, dtype=torch.int64),
            stable_order=True,
            algorithm="test",
        ),
    )


def make_execution_plan() -> ExecutionPlan:
    order = torch.tensor([0, 1], dtype=torch.int64)
    return ExecutionPlan(
        execution_order=order,
        expert_queue=order,
        expert_starts=torch.tensor([0, 2], dtype=torch.int64),
        expert_ends=torch.tensor([2, 4], dtype=torch.int64),
        expert_counts=torch.tensor([2, 2], dtype=torch.int64),
        execution_priority=order,
        stream_assignments=torch.zeros(2, dtype=torch.int64),
        batches=(),
        synchronization=SchedulerSynchronizationMetadata(torch.zeros(2, dtype=torch.bool)),
        dependencies=SchedulerDependencyMetadata(torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)),
        statistics=SchedulerStatistics(2, 2, 0, 2, 4, 2, 2, "round_robin"),
        scheduling_policy="round_robin",
        deterministic=True,
    )


def make_communication_plan() -> CommunicationPlan:
    empty_i64 = torch.empty(0, dtype=torch.int64)
    empty_f32 = torch.empty(0, dtype=torch.float32)
    return CommunicationPlan(
        local_expert_ids=torch.tensor([0, 1], dtype=torch.int64),
        remote_expert_ids=empty_i64,
        graph=CommunicationGraph((), (), empty_i64, empty_i64, empty_i64),
        communication_descriptors=(),
        transfer_descriptors=(),
        communication_groups=(),
        topology=TopologyMetadata(0, 1, 0, torch.tensor([0], dtype=torch.int64), torch.tensor([0], dtype=torch.int64), None, None, None, (), ((0,),), "single_gpu", 0.0, 0.0),
        synchronization=CommunicationSynchronizationMetadata(empty_i64, empty_i64, empty_i64, empty_i64),
        dependencies=CommunicationDependencyMetadata(empty_i64, empty_i64, empty_i64),
        prefetch=PrefetchPlan(empty_i64, empty_i64, empty_f32),
        overlap=OverlapPlan(empty_i64, empty_i64, empty_f32),
        cost_model=CommunicationCostModel((), 0, 0.0, 0.0, 0.0),
        statistics=CommunicationStatistics(2, 0, 0, 0, 0, 0, 0, 0.0, "static"),
        planner_policy="static",
        deterministic=True,
    )


def test_qwen_provider_preserves_original_weight_storage() -> None:
    registry = make_registry()
    provider = extract_qwen_swiglu_weight_provider(registry)

    assert provider.gate_up_weights.shape == (2, 16, 4)
    assert provider.down_weights.shape == (2, 4, 8)
    assert not provider.owns_weight_storage
    for expert_id in registry.expert_ids:
        expert = registry.get(expert_id)
        gate, up = provider.gate_up_weights.for_expert(expert_id)
        assert gate.data_ptr() == expert.gate_proj.weight.data_ptr()
        assert up.data_ptr() == expert.up_proj.weight.data_ptr()
        assert provider.down_weights.for_expert(expert_id).data_ptr() == expert.down_proj.weight.data_ptr()


def test_qwen_provider_materialization_is_explicit() -> None:
    provider = extract_qwen_swiglu_weight_provider(make_registry())
    fused = provider.gate_up_weights.materialize()

    assert fused.shape == (2, 16, 4)
    assert fused.data_ptr() not in provider.gate_up_weights.gate_weights.storage_pointers


def test_triton_skeleton_matches_pytorch_executor() -> None:
    registry = make_registry()
    reference = PyTorchExecutor(ExecutorConfig(), registry)
    triton = TritonExpertExecutor(ExecutorConfig(backend="triton"), registry)
    hidden_states = torch.randn(4, 4)
    dispatch_plan = make_dispatch_plan()
    execution_plan = make_execution_plan()
    communication_plan = make_communication_plan()

    reference_output = reference(hidden_states, dispatch_plan, execution_plan, communication_plan)
    triton_output = triton(hidden_states, dispatch_plan, execution_plan, communication_plan)

    assert torch.allclose(triton_output.packed_expert_outputs, reference_output.packed_expert_outputs)
    assert torch.allclose(triton_output.weighted_expert_outputs, reference_output.weighted_expert_outputs)
    assert triton_output.backend == "triton_reference_fallback"


def test_registry_builds_triton_skeleton_for_qwen_experts() -> None:
    executor = build_executor(ExecutorConfig(backend="triton"), make_registry())

    assert isinstance(executor, TritonExpertExecutor)


def test_triton_skeleton_rejects_non_qwen_experts() -> None:
    with pytest.raises(ValueError, match="Qwen-style"):
        build_executor(ExecutorConfig(backend="triton"), [torch.nn.Identity()])
