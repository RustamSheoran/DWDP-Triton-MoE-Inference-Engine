"""DWDP Modernized Triton MoE Engine Package.

Provides clean, verifiable, and high-performance FP16 MoE kernels:
- reference: Pure PyTorch reference implementation for correctness and testing.
- align: Token sorting and block alignment.
- kernels: Fused SwiGLU GEMM and Down-Projection Triton kernels.
- layer: Complete end-to-end MoE layer orchestration.
"""

from .align import AlignedMoEData, align_tokens_by_expert
from .reference import moe_reference

__all__ = ["AlignedMoEData", "align_tokens_by_expert", "moe_reference"]
