from __future__ import annotations


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
        used_weighted = False
        premerged = executor_output.merged_expert_outputs
        if premerged is not None:
            if self.config.apply_routing_weights:
                raise ValueError(
                    "apply_routing_weights=True is incompatible with executor-side accumulation"
                )
            source = None
            used_weighted = True
        elif self.config.apply_routing_weights:
            source = executor_output.packed_expert_outputs
            if source is None:
                raise ValueError(
                    "apply_routing_weights=True requires the executor to "
                    "materialize packed outputs; set "
                    "ExecutorConfig(materialize_packed_outputs=True)"
                )
            source = source * metadata.packed_routing_weights.unsqueeze(-1)
        else:
            source = executor_output.weighted_expert_outputs
            if source is None:
                raise ValueError("executor did not provide weighted expert outputs")
            used_weighted = True

        num_tokens = num_tokens_from_shape(metadata.token_shape)
        output_size = (
            premerged.shape[-1] if premerged is not None else source.shape[-1]
        )

        if premerged is not None:
            merged_flat = premerged
        else:
            merged_flat = reference_merge(
                source,
                metadata.packed_token_indices,
                num_tokens=num_tokens,
            )
        hidden_states = merged_flat.reshape(
            *metadata.token_shape, output_size
        ).contiguous()

        merge_metadata = MergeMetadata(
            token_shape=metadata.token_shape,
            assignment_shape=(num_tokens, metadata.top_k),
            inverse_permutation=metadata.inverse_permutation,
            top_k=metadata.top_k,
            deterministic=self.config.deterministic,
        )
        statistics = MergeStatistics(
            num_tokens=num_tokens,
            num_assignments=metadata.packed_token_indices.numel(),
            top_k=metadata.top_k,
            output_size=output_size,
            used_weighted_executor_outputs=used_weighted,
            backend=self.config.backend,
        )
        workspace_metadata = WorkspaceMetadata(
            used_workspace=False,
            workspace_bytes=0,
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
