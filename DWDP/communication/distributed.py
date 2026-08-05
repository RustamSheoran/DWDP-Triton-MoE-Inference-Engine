import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

try:
    import dwdp_communication_ext as _native  # type: ignore[import-not-found]
    _NATIVE_AVAILABLE = True
except ImportError:
    _native = None
    _NATIVE_AVAILABLE = False

if TYPE_CHECKING:
    from DWDP.executor.experts import ExpertRegistry

logger = logging.getLogger(__name__)


@dataclass
class ExpertMapping:
    expert_id: int
    owner_rank: int
    device_id: int
    ipc_handle: bytes | None  # None for local experts
    size_bytes: int


class GlobalExpertTable:
    """Master mapping table built globally on each device (Project 1 spec)."""

    def __init__(self, local_rank: int, world_size: int) -> None:
        self.local_rank = local_rank
        self.world_size = world_size
        self._table: dict[int, ExpertMapping] = {}

    def register_local_experts(self, experts: "ExpertRegistry") -> None:
        """Register local experts and generate IPC handles if possible."""
        for expert_id in experts.expert_ids:
            module = experts.get(expert_id)
            ptr = None
            size_bytes = 0
            device_id = 0

            # Find first CUDA parameter like in ExecutionCommunicationEngine
            for parameter in module.parameters(recurse=True):
                if parameter.device.type == "cuda":
                    ptr = parameter.data_ptr()
                    size_bytes = parameter.nelement() * parameter.element_size()
                    device_id = parameter.device.index if parameter.device.index is not None else 0
                    break

            ipc_handle = None
            if ptr is not None and _NATIVE_AVAILABLE:
                try:
                    ipc_handle = _native.export_ipc_handle(ptr)
                except Exception as e:
                    logger.warning("Failed to export IPC handle for expert %d: %s", expert_id, e)

            mapping = ExpertMapping(
                expert_id=expert_id,
                owner_rank=self.local_rank,
                device_id=device_id,
                ipc_handle=ipc_handle,
                size_bytes=size_bytes,
            )
            self._table[expert_id] = mapping

    def exchange_handles(self) -> None:
        """Exchange local expert mappings with all other ranks via all_gather_object."""
        if self.world_size <= 1 or not _DIST_AVAILABLE or not dist.is_initialized():
            return

        local_mappings = [
            mapping for mapping in self._table.values() 
            if mapping.owner_rank == self.local_rank
        ]

        gathered_mappings: list[list[ExpertMapping]] = [[] for _ in range(self.world_size)]
        
        try:
            dist.all_gather_object(gathered_mappings, local_mappings)
            
            for rank_mappings in gathered_mappings:
                for mapping in rank_mappings:
                    if mapping.expert_id not in self._table:
                        self._table[mapping.expert_id] = mapping
        except Exception as e:
            logger.error("Failed to exchange IPC handles: %s", e)

    def get_mapping(self, expert_id: int) -> ExpertMapping:
        """Get the global mapping for a specific expert."""
        if expert_id not in self._table:
            raise KeyError(f"Expert {expert_id} not found in global table")
        return self._table[expert_id]

    def is_local(self, expert_id: int) -> bool:
        """Check if an expert is local to this rank."""
        return self.get_mapping(expert_id).owner_rank == self.local_rank

    def remote_expert_ids(self) -> list[int]:
        """Return a list of all remote expert IDs."""
        return [
            expert_id for expert_id, mapping in self._table.items() 
            if mapping.owner_rank != self.local_rank
        ]

    def local_expert_ids(self) -> list[int]:
        """Return a list of all local expert IDs."""
        return [
            expert_id for expert_id, mapping in self._table.items() 
            if mapping.owner_rank == self.local_rank
        ]


def initialize_distributed(world_size: int, rank: int, backend: str = 'nccl') -> None:
    """Initialize torch.distributed process group."""
    if not _DIST_AVAILABLE:
        logger.warning("torch.distributed is not available, skipping initialization")
        return
    
    if world_size <= 1:
        return

    if dist.is_initialized():
        logger.info("torch.distributed process group is already initialized")
        return

    try:
        dist.init_process_group(backend=backend, world_size=world_size, rank=rank)
        logger.info("Initialized torch.distributed process group (rank %d/%d, backend %s)", rank, world_size, backend)
    except Exception as e:
        logger.error("Failed to initialize process group: %s", e)
        raise


def cleanup_distributed() -> None:
    """Destroy process group."""
    if _DIST_AVAILABLE and dist.is_initialized():
        try:
            dist.destroy_process_group()
            logger.info("Destroyed torch.distributed process group")
        except Exception as e:
            logger.error("Failed to destroy process group: %s", e)


def bootstrap_dwdp_distributed(
    experts: "ExpertRegistry",
    communication_engine: Any,
    world_size: int,
    rank: int,
) -> GlobalExpertTable | None:
    """Bootstrap DWDP distributed: init process group, build expert table, exchange IPC handles.

    This is the entry point called by the adapter after model load but before
    the first forward pass. It:

    1. Initializes torch.distributed if world_size > 1
    2. Builds the GlobalExpertTable and registers local experts
    3. Exchanges IPC handles via all_gather_object
    4. Registers remote IPC experts with the C++ CommunicationEngine

    Returns the table so the adapter can attach it to the comms_planner.
    """
    if world_size <= 1:
        return None

    initialize_distributed(world_size, rank)

    table = GlobalExpertTable(rank, world_size)
    table.register_local_experts(experts)
    table.exchange_handles()

    registered = 0
    for expert_id in table.remote_expert_ids():
        mapping = table.get_mapping(expert_id)
        if mapping.ipc_handle is None:
            continue
        if communication_engine.register_remote_expert(
            expert_id, mapping.ipc_handle, mapping.size_bytes
        ):
            registered += 1

    logger.info(
        "DWDP distributed bootstrap complete on rank %d/%d: "
        "%d local experts, %d remote experts, %d IPC handles registered",
        rank,
        world_size,
        len(table.local_expert_ids()),
        len(table.remote_expert_ids()),
        registered,
    )
    return table
