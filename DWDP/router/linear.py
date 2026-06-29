from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .base import BaseRouter
from .config import MetadataLevel, RouterConfig
from .kernels import reference_topk_routing
from .metadata import build_routing_metadata
from .registry import register_router
from .types import RouterOutput
from .utils import flatten_token_dims, restore_token_dims, validate_hidden_states


class LinearTopKRouter(BaseRouter):
    """Standard linear projection + softmax + top-k MoE router."""

    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(self, config: RouterConfig) -> None:
        super().__init__(config)
        self.weight = nn.Parameter(torch.empty(config.num_experts, config.hidden_size))
        if config.bias:
            self.bias = nn.Parameter(torch.empty(config.num_experts))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize router parameters."""

        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def compute_router_logits(self, flat_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project flattened token states into expert logits."""

        router_logits = F.linear(flat_hidden_states, self.weight, self.bias)
        if self.config.score_scale != 1.0:
            router_logits = router_logits * self.config.score_scale
        return router_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        metadata_level: MetadataLevel | None = None,
    ) -> RouterOutput:
        """Route hidden states to their top-k experts.

        Args:
            hidden_states: Tensor with shape [batch, seq, hidden] or any
                token-major prefix ending in hidden size.
            metadata_level: Optional override for metadata materialization.
        """

        validate_hidden_states(hidden_states, self.config.hidden_size)
        flat_hidden_states, token_shape = flatten_token_dims(hidden_states)
        router_logits_2d = self.compute_router_logits(flat_hidden_states)

        routing_probabilities_2d, topk_indices_2d, topk_weights_2d = (
            reference_topk_routing(
                router_logits_2d,
                top_k=self.config.top_k,
                softmax_dtype=self.config.softmax_dtype,
                probability_dtype=self.config.probability_dtype,
                topk_sorted=self.config.topk_sorted,
                renormalize=self.config.renormalize,
                eps=self.config.eps,
            )
        )

        active_metadata_level = metadata_level or self.config.metadata_level
        metadata = build_routing_metadata(
            topk_indices=topk_indices_2d,
            topk_weights=topk_weights_2d,
            num_experts=self.config.num_experts,
            level=active_metadata_level,
        )

        return RouterOutput(
            router_logits=restore_token_dims(router_logits_2d, token_shape),
            routing_probabilities=restore_token_dims(
                routing_probabilities_2d,
                token_shape,
            ),
            topk_indices=restore_token_dims(topk_indices_2d, token_shape),
            topk_weights=restore_token_dims(topk_weights_2d, token_shape),
            metadata=metadata,
        )


register_router("linear_topk", LinearTopKRouter)
