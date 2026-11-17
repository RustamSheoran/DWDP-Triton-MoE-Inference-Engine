"""DWDP Modernized Triton MoE Engine Package.

Provides clean, verifiable, and high-performance FP16 MoE kernels:
- reference: Pure PyTorch reference implementation for correctness and testing.
- align: Token sorting and block alignment.
- kernels: Fused SwiGLU GEMM and Down-Projection Triton kernels.
- layer: Complete end-to-end MoE layer orchestration.
"""

from .align import AlignedMoEData, align_tokens_by_expert
from .kernels import launch_down_proj, launch_fused_swiglu
from .layer import TritonMoELayer, triton_moe_layer
from .reference import moe_reference

__all__ = [
    "AlignedMoEData",
    "TritonMoELayer",
    "align_tokens_by_expert",
    "launch_down_proj",
    "launch_fused_swiglu",
    "moe_reference",
    "triton_moe_layer",
]
