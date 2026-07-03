from __future__ import annotations

import torch

from DWDP.scheduler.execution import ExecutionPlan

from .base import BaseCommunicationPlanner
from .config import CommunicationPlannerConfig
from .cost_model import CommunicationCostModel
from .graph import CommunicationGraph
from .kernels import reference_static_communication_plan
from .metadata import (
    CommunicationPlan,
    CommunicationStatistics,
    DependencyMetadata,
    OverlapPlan,
    PrefetchPlan,
    SynchronizationMetadata,
)
from .registry import register_communication_planner
from .topology import CommunicationDomain, TopologyMetadata
from .utils import empty_float32, empty_int64, validate_execution_plan
from .workspace import CommunicationPlannerWorkspace
from .ops import build_single_gpu_topology_tensors


class StaticCommunicationPlanner(BaseCommunicationPlanner):
    """Static single-GPU communication planner.

    This planner emits the full CommunicationPlan schema while describing zero
    remote communication for the current single-GPU runtime.
    """

    def __init__(self, config: CommunicationPlannerConfig) -> None:
        super().__init__(config)

    def forward(
        self,
        execution_plan: ExecutionPlan,
        workspace: CommunicationPlannerWorkspace | None = None,
    ) -> CommunicationPlan:
        """Build a communication blueprint from an execution plan."""

        validate_execution_plan(execution_plan)
        active_workspace = workspace if self.config.enable_workspace else None

        (
            local_expert_ids,
            remote_expert_ids,
            node_ids,
            edge_src,
            edge_dst,
        ) = reference_static_communication_plan(
            execution_plan.expert_queue,
            workspace=active_workspace,
        )

        topology = self._build_topology(
            execution_plan.expert_queue.device,
            active_workspace,
        )
        graph = CommunicationGraph(
            nodes=(),
            edges=(),
            node_ids=node_ids,
            edge_src=edge_src,
            edge_dst=edge_dst,
        )
        synchronization = self._build_synchronization_metadata(
            execution_plan.expert_queue.device,
            active_workspace,
        )
        dependencies = self._build_dependency_metadata(
            execution_plan.expert_queue.device,
            active_workspace,
        )
        prefetch = self._build_prefetch_plan(
            execution_plan.expert_queue.device,
            active_workspace,
        )
        overlap = self._build_overlap_plan(
            execution_plan.expert_queue.device,
            active_workspace,
        )
        cost_model = self._build_cost_model()
        statistics = self._build_statistics(
            execution_plan,
            local_expert_ids=local_expert_ids,
            remote_expert_ids=remote_expert_ids,
            graph=graph,
            cost_model=cost_model,
        )

        return CommunicationPlan(
            local_expert_ids=local_expert_ids,
            remote_expert_ids=remote_expert_ids,
            graph=graph,
            communication_descriptors=(),
            transfer_descriptors=(),
            communication_groups=(),
            topology=topology,
            synchronization=synchronization,
            dependencies=dependencies,
            prefetch=prefetch,
            overlap=overlap,
            cost_model=cost_model,
            statistics=statistics,
            planner_policy=self.config.planner_policy,
            deterministic=self.config.deterministic,
        )

    def _empty_int64(
        self,
        device: torch.device,
        workspace: CommunicationPlannerWorkspace | None,
    ) -> torch.Tensor:
        if workspace is None:
            return empty_int64(device)
        return workspace.get_empty_int64(device=device)

    def _empty_float32(
        self,
        device: torch.device,
        workspace: CommunicationPlannerWorkspace | None,
    ) -> torch.Tensor:
        if workspace is None:
            return empty_float32(device)
        return workspace.get_empty_float32(device=device)

    def _build_topology(
        self,
        device: torch.device,
        workspace: CommunicationPlannerWorkspace | None,
    ) -> TopologyMetadata:
        if self.config.enable_topology_metadata:
            gpu_ids, numa_domains = build_single_gpu_topology_tensors(
                world_size=self.config.world_size,
                device=device,
                workspace=workspace,
            )
            communication_domains = (
                CommunicationDomain(
                    domain_id=0,
                    domain_type="local",
                    gpu_ids=(self.config.local_gpu_id,),
                    bandwidth_gbps=self.config.default_link_bandwidth_gbps,
                    latency_us=self.config.default_link_latency_us,
                ),
            )
            locality_groups = ((self.config.local_gpu_id,),)
        else:
            gpu_ids = self._empty_int64(device, workspace)
            numa_domains = self._empty_int64(device, workspace)
            communication_domains = ()
            locality_groups = ()

        return TopologyMetadata(
            local_gpu_id=self.config.local_gpu_id,
            world_size=self.config.world_size,
            local_rank=self.config.local_rank,
            gpu_ids=gpu_ids,
            numa_domains=numa_domains,
            nvlink_connectivity=None,
            nvswitch_domains=None,
            pcie_hierarchy=None,
            communication_domains=communication_domains,
            locality_groups=locality_groups,
            fabric="single_gpu" if self.config.world_size == 1 else "unspecified",
            default_link_bandwidth_gbps=self.config.default_link_bandwidth_gbps,
            default_link_latency_us=self.config.default_link_latency_us,
        )

    def _build_synchronization_metadata(
        self,
        device: torch.device,
        workspace: CommunicationPlannerWorkspace | None,
    ) -> SynchronizationMetadata:
        empty = self._empty_int64(device, workspace)
        return SynchronizationMetadata(
            barrier_node_ids=empty,
            cuda_event_ids=empty,
            stream_wait_edges=empty,
            synchronization_points=empty,
        )

    def _build_dependency_metadata(
        self,
        device: torch.device,
        workspace: CommunicationPlannerWorkspace | None,
    ) -> DependencyMetadata:
        empty = self._empty_int64(device, workspace)
        return DependencyMetadata(
            dependency_src=empty,
            dependency_dst=empty,
            dependency_type=empty,
        )

    def _build_prefetch_plan(
        self,
        device: torch.device,
        workspace: CommunicationPlannerWorkspace | None,
    ) -> PrefetchPlan:
        empty_i64 = self._empty_int64(device, workspace)
        empty_f32 = self._empty_float32(device, workspace)
        return PrefetchPlan(
            prefetch_expert_ids=empty_i64,
            prefetch_priorities=empty_i64,
            prefetch_windows_us=empty_f32,
        )

    def _build_overlap_plan(
        self,
        device: torch.device,
        workspace: CommunicationPlannerWorkspace | None,
    ) -> OverlapPlan:
        empty_i64 = self._empty_int64(device, workspace)
        empty_f32 = self._empty_float32(device, workspace)
        return OverlapPlan(
            communication_node_ids=empty_i64,
            compute_batch_ids=empty_i64,
            overlap_windows_us=empty_f32,
        )

    def _build_cost_model(self) -> CommunicationCostModel:
        return CommunicationCostModel(
            estimates=(),
            total_estimated_bytes=0,
            total_estimated_latency_us=0.0,
            critical_path_us=0.0,
            estimated_bandwidth_gbps=0.0,
        )

    def _build_statistics(
        self,
        execution_plan: ExecutionPlan,
        *,
        local_expert_ids: torch.Tensor,
        remote_expert_ids: torch.Tensor,
        graph: CommunicationGraph,
        cost_model: CommunicationCostModel,
    ) -> CommunicationStatistics:
        if not self.config.enable_statistics:
            return CommunicationStatistics(
                num_local_experts=0,
                num_remote_experts=0,
                num_communication_nodes=0,
                num_communication_edges=0,
                num_transfer_descriptors=0,
                num_communication_groups=0,
                total_estimated_bytes=0,
                total_estimated_latency_us=0.0,
                planner_policy=self.config.planner_policy,
            )

        return CommunicationStatistics(
            num_local_experts=local_expert_ids.numel(),
            num_remote_experts=remote_expert_ids.numel(),
            num_communication_nodes=len(graph.nodes),
            num_communication_edges=len(graph.edges),
            num_transfer_descriptors=0,
            num_communication_groups=0,
            total_estimated_bytes=cost_model.total_estimated_bytes,
            total_estimated_latency_us=cost_model.total_estimated_latency_us,
            planner_policy=self.config.planner_policy,
        )


register_communication_planner("static", StaticCommunicationPlanner)
