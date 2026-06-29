"""Kernel entry points for future fused or Triton implementations."""

from .fused import reference_topk_routing

__all__ = ["reference_topk_routing"]
