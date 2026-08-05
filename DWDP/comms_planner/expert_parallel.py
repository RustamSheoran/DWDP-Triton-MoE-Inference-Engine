"""Expert-Parallel / DWDP communication planner for multi-GPU NVLink nodes.

Project 1 spec: experts are sharded evenly across ranks, attention stays
replicated, and remote expert *weights* are streamed over NVLink via
asynchronous peer-to-peer copies rather than dispatching token activations.

This planner materializes the plan the C++ engine consumes:

* Expert ownership comes from the :class:`GlobalExpertTable` when distributed
  startup has registered one, otherwise from an even static partition.
* Peer reachability is queried through the compiled ``PeerTopology`` so the
  plan distinguishes true NVLink/P2P paths from staged host copies.
* Remote weights are decomposed into 2 MB transfer slices assigned
  round-robin across destination ranks. This is the documented mitigation for
  NVLink port contention: it stops every rank from hammering slice 0 of the
  same source GPU copy engine simultaneously.
* Slice tasks are submitted to the compiled ``TransferScheduler`` so the
  priority/coalescing logic in C++ owns issue order.
"""

from __future__ import annotations

import logging

import torch

from DWDP.scheduler.execution import ExecutionPlan

from .base import BaseCommunicationPlanner
from .config import CommunicationPlannerConfig
from .cost_model import CommunicationCostModel, CommunicationCostEstimate
from .graph import CommunicationGraph
from .metadata import (
    CommunicationDescriptor,
    CommunicationGroup,
    CommunicationPlan,
    CommunicationStatistics,
    DependencyMetadata,
    OverlapPlan,
    PrefetchPlan,
    SynchronizationMetadata,
    TransferDescriptor,
)
from .registry import register_communication_planner
from .topology import CommunicationDomain, TopologyMetadata
from .workspace import CommunicationPlannerWorkspace

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on compiled extension
    import dwdp_communication_ext as _native  # type: ignore[import-not-found]

    _NATIVE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _native = None
    _NATIVE_AVAILABLE = False

# NVLink transfer slice granularity. Must match kSliceSize in buffers.cu so the
# Python-side plan and the C++ copy loop agree on slice boundaries.
SLICE_BYTES = 2 * 1024 * 1024


class ExpertParallelPlanner(BaseCommunicationPlanner):
    """DWDP weight-streaming planner for intra-node multi-GPU inference."""

    def __init__(self, config: CommunicationPlannerConfig) -> None:
        super().__init__(config)
        self.num_gpus = max(1, config.world_size)
        self.local_rank = config.local_rank
        self.local_gpu_id = config.local_gpu_id
        # GlobalExpertTable is attached by the adapter after handle exchange.
        self.expert_table = None
        self._peer_topology = None
        self._scheduler = None
        if _NATIVE_AVAILABLE:
            try:
                self._peer_topology = _native.PeerTopology()
                self._scheduler = _native.TransferScheduler()
            except Exception:  # pragma: no cover - defensive
                logger.warning("failed to construct native topology/scheduler", exc_info=True)

    def attach_expert_table(self, table) -> None:
        """Bind the post-handle-exchange GlobalExpertTable to this planner."""

        self.expert_table = table

    def close(self) -> None:
        """Release the native scheduler queue."""

        if self._scheduler is not None:
            try:
                self._scheduler.close()
            except Exception:  # pragma: no cover
                pass
            self._scheduler = None

    def _can_access(self, source_gpu: int, destination_gpu: int) -> bool:
        """Return whether source_gpu is P2P-reachable from destination_gpu."""

        if source_gpu == destination_gpu:
            return True
        if self._peer_topology is None:
            return False
        try:
            return bool(self._peer_topology.canAccess(source_gpu, destination_gpu))
        except Exception:  # pragma: no cover - defensive
            return False

    def _ownership(self, num_experts: int) -> tuple[list[int], list[int], dict[int, int]]:
        """Resolve (local, remote, expert_id -> owner_rank) expert ownership."""

        owners: dict[int, int] = {}
        if self.expert_table is not None:
            for expert_id in range(num_experts):
                try:
                    owners[expert_id] = self.expert_table.get_mapping(expert_id).owner_rank
                except KeyError:
                    owners[expert_id] = expert_id % self.num_gpus
        else:
            experts_per_rank = max(1, (num_experts + self.num_gpus - 1) // self.num_gpus)
            for expert_id in range(num_experts):
                owners[expert_id] = min(expert_id // experts_per_rank, self.num_gpus - 1)

        local = [e for e in range(num_experts) if owners[e] == self.local_rank]
        remote = [e for e in range(num_experts) if owners[e] != self.local_rank]
        return local, remote, owners

    def _expert_bytes(self, expert_id: int) -> int:
        """Return the byte size of one expert's weights when known."""

        if self.expert_table is None:
            return 0
        try:
            return int(self.expert_table.get_mapping(expert_id).size_bytes)
        except KeyError:
            return 0
    def _build_topology(self, device: torch.device) -> TopologyMetadata:
        """Describe the intra-node NVLink fabric using real peer reachability."""

        gpu_ids = torch.arange(self.num_gpus, device=device, dtype=torch.int64)
        numa_domains = torch.zeros(self.num_gpus, device=device, dtype=torch.int64)

        connectivity = torch.zeros(
            (self.num_gpus, self.num_gpus), device=device, dtype=torch.int64
        )
        for source in range(self.num_gpus):
            for destination in range(self.num_gpus):
                if self._can_access(source, destination):
                    connectivity[destination, source] = 1

        peers = tuple(
            gpu
            for gpu in range(self.num_gpus)
            if self._can_access(gpu, self.local_gpu_id)
        )
        domains = (
            CommunicationDomain(
                domain_id=0,
                domain_type="nvlink" if len(peers) > 1 else "local",
                gpu_ids=peers or (self.local_gpu_id,),
                bandwidth_gbps=self.config.default_link_bandwidth_gbps,
                latency_us=self.config.default_link_latency_us,
            ),
        )

        return TopologyMetadata(
            local_gpu_id=self.local_gpu_id,
            world_size=self.num_gpus,
            local_rank=self.local_rank,
            gpu_ids=gpu_ids,
            numa_domains=numa_domains,
            nvlink_connectivity=connectivity,
            nvswitch_domains=None,
            pcie_hierarchy=None,
            communication_domains=domains,
            locality_groups=(peers or (self.local_gpu_id,),),
            fabric="nvlink" if self.num_gpus > 1 else "single_gpu",
            default_link_bandwidth_gbps=self.config.default_link_bandwidth_gbps,
            default_link_latency_us=self.config.default_link_latency_us,
        )

    def forward(
        self,
        execution_plan: ExecutionPlan,
        workspace: CommunicationPlannerWorkspace | None = None,
    ) -> CommunicationPlan:
        """Build a DWDP weight-streaming plan with round-robin NVLink slices."""

        del workspace
        device = execution_plan.expert_queue.device
        num_experts = int(execution_plan.expert_queue.numel())

        local_ids, remote_ids, owners = self._ownership(num_experts)

        descriptors: list[CommunicationDescriptor] = []
        transfers: list[TransferDescriptor] = []
        estimates: list[CommunicationCostEstimate] = []
        prefetch_ids: list[int] = []
        prefetch_priorities: list[int] = []
        total_bytes = 0

        # Round-robin the *starting* slice per destination rank so concurrent
        # ranks reading the same source GPU enter its copy engine at different
        # offsets instead of all contending for slice 0.
        for position, expert_id in enumerate(remote_ids):
            owner_rank = owners[expert_id]
            size_bytes = self._expert_bytes(expert_id)
            num_slices = max(1, (size_bytes + SLICE_BYTES - 1) // SLICE_BYTES)
            p2p = self._can_access(owner_rank, self.local_gpu_id)
            op_type = "p2p_weight_copy" if p2p else "staged_weight_copy"
            # Higher priority for experts that appear earlier in the queue.
            priority = num_experts - position

            for index in range(num_slices):
                slice_index = (self.local_rank + index) % num_slices
                offset = slice_index * SLICE_BYTES
                slice_bytes = (
                    min(SLICE_BYTES, size_bytes - offset) if size_bytes else 0
                )
                descriptor_id = len(descriptors)
                descriptors.append(
                    CommunicationDescriptor(
                        descriptor_id=descriptor_id,
                        op_type=op_type,
                        source_gpu=owner_rank,
                        destination_gpu=self.local_gpu_id,
                        source_expert_id=expert_id,
                        destination_expert_id=expert_id,
                        start=offset,
                        end=offset + slice_bytes,
                        count=slice_bytes,
                        priority=priority,
                        stream_id=1,  # dedicated high-priority copy stream
                    )
                )
                transfers.append(
                    TransferDescriptor(
                        transfer_id=descriptor_id,
                        descriptor_id=descriptor_id,
                        source_gpu=owner_rank,
                        destination_gpu=self.local_gpu_id,
                        source_expert_id=expert_id,
                        destination_expert_id=expert_id,
                        estimated_bytes=slice_bytes,
                        priority=priority,
                    )
                )
                estimates.append(
                    CommunicationCostEstimate(
                        descriptor_id=descriptor_id,
                        estimated_bytes=slice_bytes,
                        estimated_latency_us=self.config.default_link_latency_us,
                        estimated_bandwidth_gbps=self.config.default_link_bandwidth_gbps,
                        communication_priority=priority,
                        critical_path_us=0.0,
                        transfer_duration_us=0.0,
                        prefetch_window_us=0.0,
                        overlap_estimate_us=0.0,
                    )
                )
                total_bytes += slice_bytes

            prefetch_ids.append(expert_id)
            prefetch_priorities.append(priority)

            # Hand issue-ordering to the C++ scheduler, which coalesces
            # duplicate expert submissions and orders by (priority, sequence).
            if self._scheduler is not None:
                try:
                    self._scheduler.submit(expert_id, priority)
                except Exception:  # pragma: no cover - defensive
                    logger.debug("scheduler submit failed for expert %d", expert_id, exc_info=True)

        groups = tuple(
            CommunicationGroup(
                group_id=index,
                descriptor_ids=torch.tensor(
                    [d.descriptor_id for d in descriptors if d.source_gpu == source_gpu],
                    device=device,
                    dtype=torch.int64,
                ),
                source_gpu=source_gpu,
                destination_gpu=self.local_gpu_id,
                domain_id=0,
                priority=0,
            )
            for index, source_gpu in enumerate(
                sorted({d.source_gpu for d in descriptors})
            )
        )

        empty_i64 = torch.tensor([], device=device, dtype=torch.int64)
        empty_f32 = torch.tensor([], device=device, dtype=torch.float32)
        node_ids = torch.tensor(
            [d.descriptor_id for d in descriptors], device=device, dtype=torch.int64
        )

        cost_model = CommunicationCostModel(
            estimates=tuple(estimates),
            total_estimated_bytes=total_bytes,
            total_estimated_latency_us=0.0,
            critical_path_us=0.0,
            estimated_bandwidth_gbps=self.config.default_link_bandwidth_gbps,
        )

        return CommunicationPlan(
            local_expert_ids=torch.tensor(local_ids, device=device, dtype=torch.int64),
            remote_expert_ids=torch.tensor(remote_ids, device=device, dtype=torch.int64),
            graph=CommunicationGraph(
                nodes=(),
                edges=(),
                node_ids=node_ids,
                edge_src=empty_i64,
                edge_dst=empty_i64,
            ),
            communication_descriptors=tuple(descriptors),
            transfer_descriptors=tuple(transfers),
            communication_groups=groups,
            topology=self._build_topology(device),
            synchronization=SynchronizationMetadata(
                barrier_node_ids=node_ids,
                cuda_event_ids=empty_i64,
                stream_wait_edges=empty_i64,
                synchronization_points=empty_i64,
            ),
            dependencies=DependencyMetadata(
                dependency_src=empty_i64,
                dependency_dst=empty_i64,
                dependency_type=empty_i64,
            ),
            prefetch=PrefetchPlan(
                prefetch_expert_ids=torch.tensor(
                    prefetch_ids, device=device, dtype=torch.int64
                ),
                prefetch_priorities=torch.tensor(
                    prefetch_priorities, device=device, dtype=torch.int64
                ),
                prefetch_windows_us=empty_f32,
            ),
            overlap=OverlapPlan(
                communication_node_ids=node_ids,
                compute_batch_ids=empty_i64,
                overlap_windows_us=empty_f32,
            ),
            cost_model=cost_model,
            statistics=CommunicationStatistics(
                num_local_experts=len(local_ids),
                num_remote_experts=len(remote_ids),
                num_communication_nodes=len(descriptors),
                num_communication_edges=0,
                num_transfer_descriptors=len(transfers),
                num_communication_groups=len(groups),
                total_estimated_bytes=total_bytes,
                total_estimated_latency_us=0.0,
                planner_policy=self.config.planner_policy,
            ),
            planner_policy=self.config.planner_policy,
            deterministic=self.config.deterministic,
        )
