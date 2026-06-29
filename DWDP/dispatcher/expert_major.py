from __future__ import annotations

from DWDP.router.types import RouterOutput

from .assignments import ExpertAssignments
from .base import BaseDispatcher
from .config import DispatcherConfig
from .kernels import reference_expert_major_dispatch
from .metadata import DispatchMetadata
from .plan import DispatchPlan
from .registry import register_dispatcher
from .utils import flatten_router_output, maybe_reuse_router_metadata, validate_router_output
from .workspace import DispatchWorkspace


class ExpertMajorDispatcher(BaseDispatcher):
    """Convert token-major routing results into expert-major layout."""

    def __init__(self, config: DispatcherConfig) -> None:
        super().__init__(config)

    def forward(
        self,
        router_output: RouterOutput,
        workspace: DispatchWorkspace | None = None,
    ) -> DispatchPlan:
        """Build an expert-major dispatch plan from router output."""

        if self.config.validate_inputs:
            validate_router_output(router_output, self.config.num_experts)

        flat_expert_indices, flat_routing_weights, num_tokens, top_k, token_shape = (
            flatten_router_output(router_output)
        )

        router_counts = None
        router_offsets = None
        if self.config.reuse_router_metadata:
            router_counts, router_offsets = maybe_reuse_router_metadata(
                router_output,
                num_experts=self.config.num_experts,
            )

        (
            expert_counts,
            expert_offsets,
            token_permutation,
            inverse_permutation,
            packed_expert_ids,
            packed_token_indices,
            packed_routing_weights,
        ) = reference_expert_major_dispatch(
            flat_expert_indices,
            flat_routing_weights,
            num_experts=self.config.num_experts,
            top_k=top_k,
            algorithm=self.config.algorithm,
            stable_order=self.config.stable_order,
            workspace=workspace,
            router_counts=router_counts,
            router_offsets=router_offsets,
        )

        assignments = ExpertAssignments(
            expert_ids=packed_expert_ids,
            packed_token_indices=packed_token_indices,
            packed_routing_weights=packed_routing_weights,
        )
        metadata = DispatchMetadata(
            num_tokens=num_tokens,
            num_assignments=flat_expert_indices.numel(),
            num_experts=self.config.num_experts,
            top_k=top_k,
            token_shape=token_shape,
            expert_counts=expert_counts,
            expert_offsets=expert_offsets,
            token_permutation=token_permutation,
            inverse_permutation=inverse_permutation,
            destination_positions=inverse_permutation,
            stable_order=self.config.stable_order,
            algorithm=self.config.algorithm,
        )
        return DispatchPlan(assignments=assignments, metadata=metadata)


register_dispatcher("expert_major", ExpertMajorDispatcher)
