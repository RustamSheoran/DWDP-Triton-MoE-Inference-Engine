"""PagedAttention KV Cache Manager (vLLM style).

Allocates virtual fixed-size memory blocks for Key-Value Cache tensors,
eliminating VRAM memory fragmentation and enabling high-concurrency decoding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVCacheBlock:
    block_id: int
    ref_count: int = 0


class PagedKVCacheManager:
    """Virtual Memory Block Manager for KV Cache Allocation."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        self.free_blocks: list[int] = list(range(num_blocks))
        self.used_blocks: dict[int, KVCacheBlock] = {}

        # Physical KV cache memory allocation [num_blocks, num_layers, 2, block_size, num_heads, head_dim]
        if torch.cuda.is_available() and device.startswith("cuda"):
            self.kv_cache = torch.empty(
                (num_blocks, num_layers, 2, block_size, num_heads, head_dim),
                dtype=dtype,
                device=device,
            )
        else:
            self.kv_cache = None

    def allocate(self, num_needed_blocks: int) -> list[int]:
        """Allocate physical memory blocks for a sequence."""
        if len(self.free_blocks) < num_needed_blocks:
            raise RuntimeError(
                f"Out of KV Cache memory: requested {num_needed_blocks} blocks, but only {len(self.free_blocks)} available."
            )

        allocated = []
        for _ in range(num_needed_blocks):
            block_id = self.free_blocks.pop(0)
            self.used_blocks[block_id] = KVCacheBlock(block_id=block_id, ref_count=1)
            allocated.append(block_id)
        return allocated

    def free(self, block_ids: list[int]) -> None:
        """Free physical memory blocks back to the pool."""
        for block_id in block_ids:
            if block_id in self.used_blocks:
                block = self.used_blocks[block_id]
                block.ref_count -= 1
                if block.ref_count <= 0:
                    del self.used_blocks[block_id]
                    self.free_blocks.append(block_id)
