There is no GPU available offline in this environment, so avoid CUDA-dependent runs here. Python is available for CPU-only checks and documentation work.

## How To Use

Install the package in editable mode:

```bash
pip install -e .
```

Run generation through the installed command:

```bash
dwdp run --model /path/to/model --backend dwdp --prompt "Hello"
```

Equivalent module form:

```bash
python -m dwdp run --model /path/to/model --backend dwdp --prompt "Hello"
```

Benchmark Hugging Face against DWDP:

```bash
dwdp benchmark --model /path/to/model --backend hf --compare dwdp
```

### Google Colab T4 benchmark

For a real-model comparison on a Colab T4, select **Runtime > Change runtime type > T4 GPU**, clone this repository, and run:

```bash
git clone https://github.com/RustamSheoran/DWDP-Triton-MoE-Inference-Engine.git
cd DWDP-Triton-MoE-Inference-Engine
bash scripts/benchmark_colab.sh
```

The script installs the required packages, loads `Qwen/Qwen1.5-MoE-A2.7B` with 4-bit NF4 bitsandbytes quantization, and benchmarks both the native Transformers implementation and the DWDP-patched implementation. It uses the same prompt and generation settings for both runs, unloads the first model before loading the second, and prints latency, tokens/sec, and sample output.

Use a custom prompt or change the benchmark length:

```bash
bash scripts/benchmark_colab.sh \
  --prompt "Explain mixture-of-experts inference in one paragraph." \
  --max-new-tokens 64 --warmup 2 --iters 5
```

The default 4-bit mode is intended for a 16 GB T4. An 8-bit run is available when there is enough free VRAM:

```bash
bash scripts/benchmark_colab.sh --quantization 8bit
```

Save the machine-readable result as well as the console output:

```bash
bash scripts/benchmark_colab.sh --output-json results/colab_t4.json
```

To rerun without reinstalling packages, use `SKIP_INSTALL=1`. The script requires CUDA and is expected to be run on a Colab GPU, not in a CPU-only checkout.

Profile one generation pass:

```bash
dwdp profile --model /path/to/model --prompt "Hello"
```

Use the runtime from Python:

```python
from dwdp import DWDPRuntime

runtime = DWDPRuntime.from_pretrained("/path/to/model")
output = runtime.generate("Hello")
```

Wrap an already loaded Hugging Face model:

```python
from transformers import AutoModelForCausalLM
from dwdp import DWDPRuntime

model = AutoModelForCausalLM.from_pretrained("/path/to/model")
runtime = DWDPRuntime.wrap(model)
```

Bind one explicit MoE layer when you want to build the reference runtime directly:

```python
runtime = DWDPRuntime.build_reference(
    hidden_size=4096,
    num_experts=64,
    top_k=2,
    experts=experts,
)
```

The command names map directly to the CLI modules:

- `dwdp run` loads a model and calls generation through the runtime wrapper.
- `dwdp benchmark` runs repeated generation and prints latency and throughput.
- `dwdp profile` enables the runtime profiler for one pass.
- `python -m dwdp ...` and `dwdp ...` are equivalent entrypoints; the first runs the package directly and the second uses the installed console script.
- `DWDPRuntime.wrap(...)` keeps an existing HF model and routes supported MoE work through DWDP.
- `DWDPRuntime.build_reference(...)` is the lower-level path for explicit expert modules and direct pipeline testing.

## Benchmarking

There are two benchmark paths in this repo.

The CLI path is lightweight:

```bash
dwdp benchmark --model /path/to/model --backend hf --compare dwdp
```

It runs repeated generation, prints average latency and tokens/sec, and is meant for quick comparisons.

The reporting package is the reproducible experiment path. It creates a timestamped results directory with Markdown and JSON artifacts:

- `report.md`
- `report.json`
- `benchmark_config.json`
- `environment.json`
- `profiler.json`
- `correctness.json`
- `runtime_statistics.json`
- `metadata.json`

Typical Python flow:

```python
from dwdp.benchmarking import (
    BackendPerformance,
    BenchmarkConfig,
    BenchmarkReport,
    BenchmarkReportWriter,
    CorrectnessMetrics,
    EnvironmentMetadata,
    GenerationConfig,
    MemoryMetrics,
    PerformanceComparison,
    ReportMetadata,
    RuntimeBreakdown,
    RuntimeStatistics,
)

report = BenchmarkReport(
    metadata=ReportMetadata(experiment_name="qwen15moe_hf_vs_dwdp"),
    config=BenchmarkConfig(
        model_name="Qwen/Qwen1.5-MoE-A2.7B",
        checkpoint="/path/to/model",
        prompt="Hello",
        batch_size=1,
        sequence_length=16,
        generation=GenerationConfig(max_new_tokens=32),
        dtype="float16",
        device="cuda",
        random_seed=0,
    ),
    environment=EnvironmentMetadata(...),
    performance=PerformanceComparison(
        huggingface=BackendPerformance(
            backend="hf",
            tokens_per_second=12.3,
            memory=MemoryMetrics(),
        ),
        dwdp=BackendPerformance(
            backend="dwdp",
            tokens_per_second=18.7,
            memory=MemoryMetrics(),
        ),
        runtime_breakdown=RuntimeBreakdown(),
    ),
    correctness=CorrectnessMetrics(torch_allclose=True),
    runtime_statistics=RuntimeStatistics(),
)

BenchmarkReportWriter(results_root="results").write(report)
```

Why the two paths exist:

- `dwdp benchmark` is for quick timing while you are iterating.
- `DWDP.benchmarking` is for full reports, saved artifacts, and later analysis.
- The reporting layer does not execute the benchmark itself; it writes the data you collected from your harness.

## Router Module

The repository now includes a production-oriented MoE router package under `DWDP/router`.

The router is responsible only for expert selection:

- router logits
- routing probabilities
- top-k expert indices
- normalized routing weights
- routing metadata

It does not perform dispatch, expert execution, scheduling, communication, or output merging.

Detailed engineering documentation is available in [docs/router.md](docs/router.md). A package-local overview is available in [DWDP/router/README.md](DWDP/router/README.md).

## Dispatcher Module

The repository also includes a production-oriented dispatcher package under `DWDP/dispatcher`.

The dispatcher consumes completed router output and converts token-major routing assignments into an expert-major physical layout. It is responsible for:

- expert-major grouping
- per-expert counts
- expert offsets
- token permutation
- inverse permutation
- packed token indices
- packed routing weights
- reusable dispatch metadata

It does not perform routing, expert execution, communication, scheduling, or output merging.

Detailed engineering documentation is available in [docs/dispatcher.md](docs/dispatcher.md). A package-local overview is available in [DWDP/dispatcher/README.md](DWDP/dispatcher/README.md).

## Scheduler Module

The scheduler package under `DWDP/scheduler` consumes `DispatchPlan` and produces `ExecutionPlan`.

The scheduler is responsible only for execution planning:

- expert execution order
- expert work queues
- expert-major execution ranges
- execution priorities
- stream assignment placeholders
- dependency metadata placeholders
- synchronization metadata placeholders
- scheduler statistics

It does not execute experts, move tensors, launch communication, inspect router output, inspect model weights, or merge outputs.

Detailed engineering documentation is available in [docs/scheduler.md](docs/scheduler.md). A package-local overview is available in [DWDP/scheduler/README.md](DWDP/scheduler/README.md).

## Comms Planner Module

The communication planner package under `DWDP/comms_planner` consumes `ExecutionPlan` and produces `CommunicationPlan`.

The Comms Planner is responsible only for communication planning metadata:

- local and remote expert classification
- communication graph metadata
- transfer descriptors
- communication groups
- topology metadata
- dependency metadata
- synchronization placeholders
- prefetch placeholders
- overlap placeholders
- communication cost estimates
- communication statistics

It does not execute communication, move tensors, allocate communication buffers, prefetch weights, execute experts, launch CUDA kernels, launch collectives, or mutate Scheduler output.

Detailed engineering documentation is available in [docs/comms_planner.md](docs/comms_planner.md). A package-local overview is available in [DWDP/comms_planner/README.md](DWDP/comms_planner/README.md).

## Executor Module

The executor package under `DWDP/executor` consumes hidden states, `DispatchPlan`, `ExecutionPlan`, and `CommunicationPlan`, then produces `ExecutorOutput`.

The Executor is responsible only for expert computation:

- gather token activations for scheduled expert ranges
- execute expert modules
- apply routing weights
- write packed expert outputs
- emit metadata required by the future Merger

It does not route, dispatch, schedule, plan communication, execute communication, or merge outputs.

Detailed engineering documentation is available in [docs/executor.md](docs/executor.md). A package-local overview is available in [DWDP/executor/README.md](DWDP/executor/README.md).

## Merger Module

The merger package under `DWDP/merger` consumes `ExecutorOutput` and reconstructs the final hidden states for the next Transformer layer.

The Merger is responsible only for output reconstruction:

- restore token-major assignment order
- accumulate Top-K expert outputs
- optionally apply routing weights
- reshape back to the original token layout
- emit merge statistics and metadata

It does not route, dispatch, schedule, plan communication, execute communication, execute experts, or inspect upstream runtime plans.

Detailed engineering documentation is available in [docs/merger.md](docs/merger.md). A package-local overview is available in [DWDP/merger/README.md](DWDP/merger/README.md).

## Runtime Integration Layer

The runtime integration package under `DWDP/runtime` orchestrates the complete MoE pipeline:

Router -> Dispatcher -> Scheduler -> Comms Planner -> Executor -> Merger

The runtime owns stage modules and reusable workspaces, exposes a HF-style `DWDPRuntime` API, and provides adapter, profiling, correctness, CLI, and benchmark scaffolding.

Detailed engineering documentation is available in [docs/runtime.md](docs/runtime.md). A package-local overview is available in [DWDP/runtime/README.md](DWDP/runtime/README.md).

## Hugging Face Adapter Layer

The adapter package under `DWDP/adapters` automatically detects supported Hugging Face MoE models and replaces only their MoE blocks with DWDP-backed execution.

The current automatic target is Qwen1.5/Qwen2-style MoE blocks. The adapter preserves native Hugging Face tokenization, attention, KV cache, generation, sampling, checkpoint loading, and non-MoE layers.

Detailed engineering documentation is available in [docs/adapters.md](docs/adapters.md). A package-local overview is available in [DWDP/adapters/README.md](DWDP/adapters/README.md).

## Benchmark Reporting

The benchmark reporting package under `DWDP/benchmarking` provides reproducible experiment directories, structured JSON artifacts, Markdown reports, environment capture, correctness metrics, runtime statistics, and future extension points for long-term performance tracking.

Detailed documentation is available in [docs/benchmark_reporting.md](docs/benchmark_reporting.md).
