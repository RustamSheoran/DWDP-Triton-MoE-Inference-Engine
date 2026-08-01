"""DWDP build script.

When PyTorch and CUDA are available, compiles the native C++/CUDA
communication engine into `dwdp_communication_ext`. Falls back to a
pure-Python package otherwise (CPU-only, no native engine).
"""

import os
import glob
from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    if torch.cuda.is_available() or os.environ.get("FORCE_CUDA", "0") == "1":
        comm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DWDP", "communication")
        sources = sorted(
            glob.glob(os.path.join(comm_dir, "*.cpp"))
            + glob.glob(os.path.join(comm_dir, "*.cu"))
        )
        if sources:
            ext_modules.append(
                CUDAExtension(
                    name="dwdp_communication_ext",
                    sources=sources,
                    include_dirs=[comm_dir],
                    extra_compile_args={
                        "cxx": ["-O3", "-std=c++20"],
                        "nvcc": ["-O3", "-std=c++20", "--extended-lambda"],
                    },
                )
            )
            cmdclass["build_ext"] = BuildExtension

except ImportError:
    pass  # PyTorch not installed; pure-Python package

setup(
    name="dwdp",
    version="0.1.0",
    description="Distributed Weight Data Parallelism reference MoE inference runtime",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
