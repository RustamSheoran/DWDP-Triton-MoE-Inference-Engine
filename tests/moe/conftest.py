"""Pytest configuration for tests/moe.

Ensures TRITON_INTERPRET is set only when CUDA is unavailable (CPU test environment).
On GPU environments, Triton compiles and executes native GPU kernels.
"""

from __future__ import annotations

import os
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"
