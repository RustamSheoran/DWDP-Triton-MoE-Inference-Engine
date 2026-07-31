"""Multi-Head Latent Attention (MLA) Matrix Absorption Module (DeepSeek-V2/V3 style).

Fuses low-rank query projection (W_UQ) and key projection (W_UK) matrices at load time:
    W_absorbed = W_UQ @ W_UK.T
Eliminates 50% of attention projection matrix multiplications during decode passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLAAbsorbedAttention(nn.Module):
    """Absorbs MLA key and query projections into a single fused projection matrix."""

    def __init__(self, w_uq: torch.Tensor, w_uk: torch.Tensor):
        super().__init__()
        # Pre-multiply and absorb W_UQ and W_UK at load time
        # Shape: [num_heads, q_head_dim, kv_lora_rank]
        with torch.no_grad():
            self.register_buffer("w_absorbed", torch.matmul(w_uq, w_uk.transpose(-1, -2)))

    def forward(self, compressed_kv: torch.Tensor) -> torch.Tensor:
        """Compute absorbed query-key projection in a single matmul pass."""
        return torch.matmul(compressed_kv, self.w_absorbed.transpose(-1, -2))


def absorb_mla_weights(w_uq: torch.Tensor, w_uk: torch.Tensor) -> torch.Tensor:
    """Pre-multiply and return absorbed MLA query-key weight tensor."""
    return torch.matmul(w_uq, w_uk.transpose(-1, -2))
