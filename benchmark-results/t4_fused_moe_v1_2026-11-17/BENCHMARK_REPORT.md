# DWDP Fused MoE Layer Benchmark Results (Tesla T4)

- **GPU**: Tesla T4 (SM 7.5)
- **PyTorch**: 2.14.1+cu130
- **Shapes**: Hidden=2048, Intermediate=1408, Experts=64, Top-k=4, Dtype=float16
- **Single GEMM Ratio (Triton/cuBLAS)**: 17.40x

| Tokens (T) | Naive Loop (a) | Sorted Matmul (b) | Triton Path (c) | Roofline cuBLAS (d) | Launches (c) | Speedup vs (b) |
|---|---|---|---|---|---|---|
| 1 | 4.701 ms | 3.391 ms | **5.535 ms** | 0.135 ms | 181 | **0.61x** |
| 8 | 13.852 ms | 7.753 ms | **24.579 ms** | 0.104 ms | 574 | **0.32x** |
| 64 | 31.851 ms | 16.871 ms | **70.417 ms** | 0.145 ms | 1540 | **0.24x** |
| 512 | 44.126 ms | 18.172 ms | **92.638 ms** | 0.964 ms | 1563 | **0.20x** |
| 4096 | 35.970 ms | 26.265 ms | **402.318 ms** | 16.031 ms | 1755 | **0.07x** |
