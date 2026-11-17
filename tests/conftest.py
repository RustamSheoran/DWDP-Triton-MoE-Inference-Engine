"""Global pytest configuration for DWDP test suite."""

import os
import torch

# If running on a CPU-only environment, enable Triton interpreter mode so that
# Triton kernels can be executed and tested on CPU without requiring a GPU driver.
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"
