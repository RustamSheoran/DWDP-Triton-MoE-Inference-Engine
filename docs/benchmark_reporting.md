# Benchmark Reporting

## Overview

DWDP benchmark reporting treats every benchmark run as a reproducible experiment. A report writer creates a fresh timestamped directory and writes human-readable and machine-readable artifacts without overwriting previous results.

```text
results/
    2026-07-05_14-32-18_qwen15moe_a2.7b_hf_vs_dwdp_h100/
        report.md
        report.json
        benchmark_config.json
        environment.json
        profiler.json
        correctness.json
        runtime_statistics.json
        metadata.json
        logs/
        plots/
```

## Architecture

`DWDP/benchmarking/config.py` defines benchmark and generation configuration schemas.

`environment.py` records Python, OS, package, CUDA, GPU, git, precision, backend, and compile metadata when invoked by a benchmark runner.

`metrics.py` defines performance, memory, correctness, runtime breakdown, and runtime statistics payloads.

`experiment.py` creates fresh experiment directories and never overwrites existing results.

`report.py` renders `BenchmarkReport` into Markdown.

`writer.py` writes all required JSON sidecars and the Markdown report.

`runtime_stats.py` extracts research-oriented runtime statistics from a `RuntimePipelineOutput`.

## Methodology

Benchmark runners should construct one `BenchmarkConfig` shared by Hugging Face and DWDP. Both backends must use identical:

- model
- checkpoint
- tokenizer
- prompt
- random seed
- precision
- batch size
- sequence length
- generation length
- sampling configuration

The reporting layer does not execute benchmarks. It records the configuration and measured results supplied by benchmark harnesses.

## Report Files

`report.md` is the human-readable summary with:

- Benchmark Summary
- Environment
- Configuration
- Performance Results
- Runtime Breakdown
- Correctness Validation
- Memory Usage
- Profiling Summary
- Notes

`report.json` contains the full structured report. Sidecar JSON files isolate configuration, environment, profiler payload, correctness metrics, runtime statistics, and metadata for downstream analysis.

## Future Extensions

The schema reserves extension points for:

- multi-GPU experiments
- NVLink and PCIe statistics
- communication overlap
- weight prefetch statistics
- CUDA stream timelines
- Nsight exports
- power and energy metrics
- scaling efficiency

The reporting format is intended to remain stable as DWDP evolves from reference PyTorch to `torch.compile`, Triton, CUDA, grouped GEMM, persistent kernels, distributed execution, and research optimizations.
