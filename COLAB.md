# Running DWDP on Google Colab (Tesla T4)

Follow these steps to run the DWDP GPU parity tests and MoE layer benchmark on a Google Colab Tesla T4 instance.

### 1. Open Google Colab & Select T4 GPU
- Open [Google Colab](https://colab.research.google.com).
- Go to **Runtime → Change runtime type**.
- Under **Hardware accelerator**, select **T4 GPU**, then click **Save**.

### 2. Copy and Run This Single Cell
Paste the following block into a code cell and run it:

```python
%cd /content
!rm -rf /content/repo
!git clone -b cleanup https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine.git repo
%cd /content/repo
!bash scripts/run_on_gpu.sh --diagnose
import os
from google.colab import files
z = "moe_t4_results.zip"
assert os.path.exists(z), "No results zip was created; read the output above for the failure."
files.download(z)
```

### What this cell does:
1. Clones the `cleanup` branch of the repository.
2. Runs `scripts/run_on_gpu.sh`, which:
   - Verifies an NVIDIA GPU with CUDA capability is active.
   - Installs DWDP in editable mode (`--no-deps` to preserve Colab's matched torch/triton setup).
   - Runs the GPU MoE test suite (`tests/moe/test_fused_moe_kernel.py`).
   - Runs `scripts/bench_moe_layer.py` measuring latency across token counts $T \in [1, 8, 64, 512, 4096]$ across 4 baselines.
   - Packages the results into `moe_t4_results.zip`.
3. Checks that `moe_t4_results.zip` exists and automatically initiates browser download.
