from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ExecutorWorkspace:
    """Reusable reference and persistent-execution buffers.

    Tensor storage is owned by this workspace only for temporary outputs,
    intermediates, and metadata queues. Model weights and caller activations
    remain externally owned and are referenced through TensorList pointers.
    """

    packed_expert_outputs: torch.Tensor | None = None
    weighted_expert_outputs: torch.Tensor | None = None
    gathered_activations: torch.Tensor | None = None
    temporary_outputs: torch.Tensor | None = None
    # Grouped Triton execution keeps its transient SwiGLU values here.  This
    # workspace owns the allocation; TensorList only stores its address.
    intermediate_activations: torch.Tensor | None = None
    fp8_input_activations: torch.Tensor | None = None
    # TensorList metadata has host staging and device-resident SoA buffers.
    # They are grown together and reused across decode iterations.
    tensorlist_device_fields: dict[str, torch.Tensor] | None = None
    tensorlist_host_fields: dict[str, torch.Tensor] | None = None
    _tensorlist_device_buffer: torch.Tensor | None = None
    _tensorlist_host_buffer: torch.Tensor | None = None
    _tensorlist_capacity: int = 0
    _tensorlist_launch_key: tuple[int, int, int] | None = None
    _tensorlist_launch_dimensions: tuple[int, int, int] | None = None
    _tensorlist_schedule_host: torch.Tensor | None = None
    _tensorlist_provider_ids: tuple[int, ...] | None = None
    _tensorlist_provider_positions: dict[int, int] | None = None
    # Persistent tile queue storage.  These are metadata only: all addresses
    # consumed by a work item remain in TensorList and ultimately refer to
    # caller/model-owned tensors.
    persistent_gather_descriptor_indices: torch.Tensor | None = None
    persistent_gather_tile_m: torch.Tensor | None = None
    persistent_gather_tile_n: torch.Tensor | None = None
    persistent_down_descriptor_indices: torch.Tensor | None = None
    persistent_down_tile_m: torch.Tensor | None = None
    persistent_down_tile_n: torch.Tensor | None = None
    persistent_gather_counter: torch.Tensor | None = None
    persistent_down_counter: torch.Tensor | None = None
    _persistent_gather_host: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
        None
    )
    _persistent_down_host: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    _persistent_gather_device_buffer: torch.Tensor | None = None
    _persistent_gather_host_buffer: torch.Tensor | None = None
    _persistent_down_device_buffer: torch.Tensor | None = None
    _persistent_down_host_buffer: torch.Tensor | None = None
    _persistent_queue_capacity: int = 0
    # The scheduler stores its compact execution description on the device.
    # Keep the last host copy here so repeated decode shapes do not force a
    # device-to-host read before every expert launch.
    _schedule_tensors: tuple[torch.Tensor, ...] | None = None
    _schedule_rows: tuple[tuple[int, int, int, int, int, int], ...] | None = None
    _host_copy_event: torch.cuda.Event | None = None

    def _ensure_2d(
        self,
        name: str,
        rows: int,
        cols: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        tensor = getattr(self, name)
        if (
            tensor is None
            or tensor.shape[0] < rows
            or tensor.shape[1] != cols
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            pin = device.type == "cpu" and torch.cuda.is_available()
            tensor = torch.empty(rows, cols, dtype=dtype, device=device, pin_memory=pin)
            setattr(self, name, tensor)
        return tensor[:rows, :cols]

    def get_output_buffers(
        self,
        num_assignments: int,
        output_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        materialize_packed: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return the packed and weighted output buffers.

        Both are ``[num_assignments, output_size]`` where ``num_assignments``
        is ``num_tokens * top_k``, so each is top_k times larger than a
        token-major buffer. The Merger reads only one of them. When
        ``materialize_packed`` is False the packed buffer is not allocated and
        ``None`` is returned in its place.
        """

        weighted = self._ensure_2d(
            "weighted_expert_outputs",
            num_assignments,
            output_size,
            dtype=dtype,
            device=device,
        )
        if not materialize_packed:
            return None, weighted

        packed = self._ensure_2d(
            "packed_expert_outputs",
            num_assignments,
            output_size,
            dtype=dtype,
            device=device,
        )
        return packed, weighted

    def get_gather_buffer(
        self,
        rows: int,
        hidden_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a reusable gathered activation buffer."""

        return self._ensure_2d(
            "gathered_activations",
            rows,
            hidden_size,
            dtype=dtype,
            device=device,
        )

    def get_intermediate_buffer(
        self,
        rows: int,
        cols: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return reusable Qwen SwiGLU intermediate storage."""

        return self._ensure_2d(
            "intermediate_activations", rows, cols, dtype=dtype, device=device
        )

    def get_fp8_input_buffer(
        self,
        rows: int,
        cols: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Return FP8 activation storage for one input quantization per forward."""

        return self._ensure_2d(
            "fp8_input_activations", rows, cols, dtype=dtype, device=device
        )

    def ensure_tensorlist_capacity(self, required: int, *, device: torch.device) -> int:
        """Reserve contiguous TensorList SoA buffers without owning tensor data.

        Metadata capacity grows geometrically.  Logical TensorList size resets
        on every forward, so steady-state decode construction performs no
        allocation.  Host staging avoids one CUDA write per descriptor field.
        """

        if required < 0:
            raise ValueError("TensorList capacity must be non-negative")
        from .tensor_list import tensorlist_field_names

        needs_new_buffers = (
            self.tensorlist_device_fields is None
            or self.tensorlist_host_fields is None
            or any(
                field.device != device
                for field in self.tensorlist_device_fields.values()
            )
            or required > self._tensorlist_capacity
        )
        if needs_new_buffers:
            capacity = max(required, max(1, self._tensorlist_capacity * 2))
            names = tensorlist_field_names()
            num_fields = len(names)
            
            self._tensorlist_device_buffer = torch.empty(
                (num_fields, capacity), dtype=torch.int64, device=device
            )
            self._tensorlist_host_buffer = torch.empty(
                (num_fields, capacity),
                dtype=torch.int64,
                device="cpu",
                pin_memory=device.type == "cuda",
            )
            
            self.tensorlist_device_fields = {
                name: self._tensorlist_device_buffer[i]
                for i, name in enumerate(names)
            }
            self.tensorlist_host_fields = {
                name: self._tensorlist_host_buffer[i]
                for i, name in enumerate(names)
            }
            self._tensorlist_capacity = capacity
        return self._tensorlist_capacity

    def get_tensorlist_launch_dimensions(
        self, max_tokens: int, output_size: int, intermediate_size: int
    ) -> tuple[int, int, int]:
        """Cache the host launch metadata for repeated grouped decode shapes."""

        key = (max_tokens, output_size, intermediate_size)
        if key != self._tensorlist_launch_key:
            self._tensorlist_launch_key = key
            self._tensorlist_launch_dimensions = key
        assert self._tensorlist_launch_dimensions is not None
        return self._tensorlist_launch_dimensions

    def get_tensorlist_schedule(
        self,
        expert_queue: torch.Tensor,
        expert_starts: torch.Tensor,
        expert_counts: torch.Tensor,
        execution_priority: torch.Tensor,
        stream_assignments: torch.Tensor,
    ) -> torch.Tensor:
        """Copy the compact scheduler rows into reusable host staging storage."""

        rows = (
            expert_queue,
            expert_starts,
            expert_counts,
            execution_priority,
            stream_assignments,
        )
        count = expert_queue.numel()
        if any(row.numel() != count for row in rows):
            raise ValueError("TensorList scheduler fields must have equal lengths")
        if (
            self._tensorlist_schedule_host is None
            or self._tensorlist_schedule_host.shape[1] < count
        ):
            capacity = max(
                count,
                1
                if self._tensorlist_schedule_host is None
                else self._tensorlist_schedule_host.shape[1] * 2,
            )
            self._tensorlist_schedule_host = torch.empty(
                5,
                capacity,
                dtype=torch.int64,
                device="cpu",
                pin_memory=expert_queue.is_cuda,
            )
        schedule = self._tensorlist_schedule_host[:, :count]
        for index, row in enumerate(rows):
            schedule[index].copy_(row, non_blocking=True)
        # CPU reads below must observe asynchronous CUDA copies.
        if expert_queue.is_cuda:
            torch.cuda.current_stream(expert_queue.device).synchronize()
        return schedule

    def get_tensorlist_provider_positions(
        self, expert_ids: tuple[int, ...]
    ) -> dict[int, int]:
        """Cache the provider's global-id to storage-position lookup."""

        if expert_ids != self._tensorlist_provider_ids:
            self._tensorlist_provider_ids = expert_ids
            self._tensorlist_provider_positions = {
                expert_id: position for position, expert_id in enumerate(expert_ids)
            }
        assert self._tensorlist_provider_positions is not None
        return self._tensorlist_provider_positions

    def get_tensorlist_provider_pointers_cpu(self, provider) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if provider.expert_ids != self._tensorlist_provider_ids:
            self.get_tensorlist_provider_positions(provider.expert_ids)
            
            max_id = max(provider.expert_ids) if provider.expert_ids else 0
            gate_ptrs = torch.zeros(max_id + 1, dtype=torch.int64)
            up_ptrs = torch.zeros(max_id + 1, dtype=torch.int64)
            down_ptrs = torch.zeros(max_id + 1, dtype=torch.int64)
            gate_lds = torch.zeros(max_id + 1, dtype=torch.int64)
            up_lds = torch.zeros(max_id + 1, dtype=torch.int64)
            down_lds = torch.zeros(max_id + 1, dtype=torch.int64)
            
            for expert_id, pos in self._tensorlist_provider_positions.items():
                gate = provider.gate_up_weights.gate_weights.expert_weights[pos]
                up = provider.gate_up_weights.up_weights.expert_weights[pos]
                down = provider.down_weights.expert_weights[pos]
                gate_ptrs[expert_id] = gate.data_ptr()
                up_ptrs[expert_id] = up.data_ptr()
                down_ptrs[expert_id] = down.data_ptr()
                gate_lds[expert_id] = gate.stride(0)
                up_lds[expert_id] = up.stride(0)
                down_lds[expert_id] = down.stride(0)
                
            self._tensorlist_provider_ptrs_cpu = (gate_ptrs, up_ptrs, down_ptrs, gate_lds, up_lds, down_lds)
            
        return self._tensorlist_provider_ptrs_cpu

    def ensure_persistent_queue_capacity(
        self, required: int, *, device: torch.device
    ) -> int:
        """Reserve reusable device and host SoA storage for persistent tiles.

        Queue entries contain only ``(descriptor_index, tile_m, tile_n)``.
        Pointer and shape metadata is intentionally not duplicated: the kernel
        resolves it from TensorList after atomically claiming an entry.
        """

        needs_allocation = (
            self.persistent_gather_descriptor_indices is None
            or self.persistent_gather_descriptor_indices.device != device
            or required > self._persistent_queue_capacity
        )
        if needs_allocation:
            capacity = max(required, max(1, self._persistent_queue_capacity * 2))

            self._persistent_gather_device_buffer = torch.empty(
                (3, capacity), dtype=torch.int64, device=device
            )
            self._persistent_gather_host_buffer = torch.empty(
                (3, capacity), dtype=torch.int64, device="cpu", pin_memory=device.type == "cuda"
            )
            self._persistent_down_device_buffer = torch.empty(
                (3, capacity), dtype=torch.int64, device=device
            )
            self._persistent_down_host_buffer = torch.empty(
                (3, capacity), dtype=torch.int64, device="cpu", pin_memory=device.type == "cuda"
            )

            gather = tuple(self._persistent_gather_device_buffer[i] for i in range(3))
            down = tuple(self._persistent_down_device_buffer[i] for i in range(3))
            (
                self.persistent_gather_descriptor_indices,
                self.persistent_gather_tile_m,
                self.persistent_gather_tile_n,
            ) = gather
            (
                self.persistent_down_descriptor_indices,
                self.persistent_down_tile_m,
                self.persistent_down_tile_n,
            ) = down
            
            self._persistent_gather_host = tuple(
                self._persistent_gather_host_buffer[i] for i in range(3)
            )
            self._persistent_down_host = tuple(
                self._persistent_down_host_buffer[i] for i in range(3)
            )

            self.persistent_gather_counter = torch.zeros(
                1, dtype=torch.int32, device=device
            )
            self.persistent_down_counter = torch.zeros(
                1, dtype=torch.int32, device=device
            )
            self._persistent_queue_capacity = capacity
        return self._persistent_queue_capacity

    def get_schedule_rows(
        self,
        expert_queue: torch.Tensor,
        expert_starts: torch.Tensor,
        expert_ends: torch.Tensor,
        expert_counts: torch.Tensor,
        execution_priority: torch.Tensor,
        stream_assignments: torch.Tensor,
    ) -> tuple[tuple[int, int, int, int, int, int], ...]:
        """Return host execution rows with one device synchronization at most.

        Calling ``Tensor.item()`` for every scheduler field forces CUDA to
        synchronize per scalar.  Materializing the small schedule once makes
        the host loop independent of device scalar reads.  Holding tensor
        references, rather than a pointer-only key, keeps the cache safe when
        the allocator recycles storage for a later plan.
        """

        tensors = (
            expert_queue,
            expert_starts,
            expert_ends,
            expert_counts,
            execution_priority,
            stream_assignments,
        )
        if (
            self._schedule_tensors is not None
            and all(
                cached is current
                for cached, current in zip(self._schedule_tensors, tensors)
            )
            and self._schedule_rows is not None
        ):
            return self._schedule_rows

        # stack performs one compact device copy before tolist transfers the
        # complete schedule, replacing six scalar D2H synchronizations per
        # active expert.
        values = torch.stack(tensors).cpu().tolist()
        rows = tuple(zip(*values))
        self._schedule_tensors = tensors
        self._schedule_rows = rows
        return rows

    def estimated_bytes(self) -> int:
        """Estimate allocated workspace bytes."""

        total = 0
        for tensor in (
            self.packed_expert_outputs,
            self.weighted_expert_outputs,
            self.gathered_activations,
            self.temporary_outputs,
            self.intermediate_activations,
            self.fp8_input_activations,
            self.persistent_gather_descriptor_indices,
            self.persistent_gather_tile_m,
            self.persistent_gather_tile_n,
            self.persistent_down_descriptor_indices,
            self.persistent_down_tile_m,
            self.persistent_down_tile_n,
            self.persistent_gather_counter,
            self.persistent_down_counter,
        ):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        for fields in (self.tensorlist_device_fields, self.tensorlist_host_fields):
            if fields is not None:
                total += sum(
                    tensor.numel() * tensor.element_size() for tensor in fields.values()
                )
        if self._tensorlist_schedule_host is not None:
            total += (
                self._tensorlist_schedule_host.numel()
                * self._tensorlist_schedule_host.element_size()
            )
        for fields in (self._persistent_gather_host, self._persistent_down_host):
            if fields is not None:
                total += sum(
                    tensor.numel() * tensor.element_size() for tensor in fields
                )
        return total
