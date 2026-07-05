from __future__ import annotations

import torch

from DWDP.executor.outputs import ExecutorOutput

from .base import BaseMerger
from .config import MergerConfig
from .kernels import reference_merge
from .metadata import MergeMetadata, MergeStatistics, TimingMetadata, WorkspaceMetadata
from .outputs import MergerOutput
from .registry import register_merger
from .utils import num_tokens_from_shape, validate_executor_output
from .workspace import MergerWorkspace


class PyTorchMerger(BaseMerger):
    """Reference PyTorch backend for output reconstruction."""

    def __init__(self, config: MergerConfig) -> None:
        super().__init__(config)

    def forward(
        self,
        executor_output: ExecutorOutput,
        workspace: MergerWorkspace | None = None,
    ) -> MergerOutput:
        """Reconstruct final hidden states from packed expert outputs."""

        if self.config.validate_shapes:
            validate_executor_output(executor_output)

        metadata = executor_output.output_metadata
        source = executor_output.packed_expert_outputs
        used_weighted = False
        if self.config.apply_routing_weights:
            source = source * metadata.packed_routing_weights.unsqueeze(-1)
        else:
            source = executor_output.weighted_expert_outputs
            used_weighted = True

        num_tokens = num_tokens_from_shape(metadata.token_shape)
        output_size = source.shape[-1]
        active_workspace = workspace if self.config.enable_workspace else None
        assignment_out = None
        merged_out = None
        if active_workspace is not None:
            assignment_out = active_workspace.get_assignment_buffer(
                source.shape[0],
                output_size,
                dtype=source.dtype,
                device=source.device,
            )
            merged_out = active_workspace.get_merged_buffer(
                num_tokens,
                output_size,
                dtype=source.dtype,
                device=source.device,
            )

        _, merged_flat = reference_merge(
            source,
            metadata.inverse_permutation,
            num_tokens=num_tokens,
            top_k=metadata.top_k,
            assignment_out=assignment_out,
            merged_out=merged_out,
        )
        hidden_states = merged_flat.reshape(*metadata.token_shape, output_size).contiguous()

        merge_metadata = MergeMetadata(
            token_shape=metadata.token_shape,
            assignment_shape=(num_tokens, metadata.top_k),
            inverse_permutation=metadata.inverse_permutation,
            top_k=metadata.top_k,
            deterministic=self.config.deterministic,
        )
        statistics = MergeStatistics(
            num_tokens=num_tokens,
            num_assignments=source.shape[0],
            top_k=metadata.top_k,
            output_size=output_size,
            used_weighted_executor_outputs=used_weighted,
            backend=self.config.backend,
        )
        workspace_metadata = WorkspaceMetadata(
            used_workspace=active_workspace is not None,
            workspace_bytes=active_workspace.estimated_bytes() if active_workspace is not None else 0,
        )
        return MergerOutput(
            hidden_states=hidden_states,
            metadata=merge_metadata,
            statistics=statistics,
            timing=TimingMetadata(),
            workspace=workspace_metadata,
            backend=self.config.backend,
            deterministic=self.config.deterministic,
        )


register_merger("pytorch", PyTorchMerger)
