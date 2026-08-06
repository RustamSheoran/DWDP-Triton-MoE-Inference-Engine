#!/usr/bin/env python3
"""Example: Enable CUDA graphs on Qwen1.5-MoE for faster decode.

CUDA graphs collapse thousands of kernel launches per token into a single
`cudaGraphLaunch`, eliminating CPU orchestration overhead. The first forward
at each input shape triggers capture (slow), all subsequent forwards at that
shape replay instantly.

Run this on a CUDA-capable machine to see the speedup.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

from DWDP.adapters import Qwen15MoEAdapter

# Load model and enable graphs
print("Loading Qwen1.5-MoE-A2.7B with 4-bit quantization...")
adapter = Qwen15MoEAdapter.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B-4bit",
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")

print("Enabling CUDA graphs on all MoE blocks...")
num_enabled = adapter.enable_cuda_graphs(warmup_steps=3)
print(f"  → {num_enabled} blocks now use graphs")

# Generate text
prompt = "Explain how CUDA graphs work in one sentence:"
inputs = tokenizer(prompt, return_tensors="pt").to(adapter.model.device)

print("\nGenerating (first few tokens trigger capture, rest replay)...")
with torch.inference_mode():
    output_ids = adapter.model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
    )

generated = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(f"\n{prompt}{generated}")

# Show graph statistics
stats = adapter.graph_statistics()
print(f"\nGraph statistics: {stats}")
print(f"  → {stats['captures']} shapes captured")
print(f"  → {stats['replays']} replays (these were ~zero-overhead)")
print(f"  → {stats['fallbacks']} eager fallbacks")
