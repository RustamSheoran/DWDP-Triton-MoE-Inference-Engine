# Executor

## Overview

The `DWDP.executor` package is the first DWDP runtime layer that performs model computation.

Input:

- hidden states
- `DispatchPlan`
- `ExecutionPlan`
- `CommunicationPlan`

Output:

- `ExecutorOutput`

The Executor does not perform routing, dispatch planning, token reordering policy, scheduling, communication planning, communication execution, or output merging. Those responsibilities belong to earlier or later runtime stages.

## Runtime Position

```mermaid
flowchart TD
    A[RouterOutput] --> B[Dispatcher]
    B --> C[DispatchPlan]
    C --> D[Scheduler]
    D --> E[ExecutionPlan]
    E --> F[Comms Planner]
    F --> G[CommunicationPlan]
    G --> H[Executor]
    H --> I[ExecutorOutput]
    I --> J[Merger]
```

## Architecture

```text
DWDP/executor/
  __init__.py
  base.py
  config.py
  experts.py
  metadata.py
  outputs.py
  pytorch.py
  triton.py
  weights.py
  extractors/
    qwen.py
  registry.py
  utils.py
  workspace.py
  ops/
    __init__.py
    reference.py
  kernels/
    __init__.py
    reference.py
  backends/
    __init__.py
```

## PyTorch Backend

`PyTorchExecutor` is the reference backend. It:

1. flattens hidden states into token-major 2D form
2. validates finalized planning artifacts
3. iterates experts exactly in `ExecutionPlan.expert_queue` order
4. gathers hidden states using `DispatchPlan.assignments.packed_token_indices`
5. executes the registered expert module
6. applies routing weights
7. writes packed outputs and weighted outputs
8. returns `ExecutorOutput`

The backend supports arbitrary expert modules that follow the standard interface:

```text
expert(hidden_states: Tensor) -> Tensor
```

## Public API

### `ExecutorConfig`

Immutable execution configuration.

Fields include:

- `backend`
- `dtype`
- `enable_workspace`
- `enable_statistics`
- `enable_profiling`
- `deterministic`
- `max_tokens_per_expert`
- future distributed and async placeholders

### `ExpertRegistry`

Container mapping global expert ids to `torch.nn.Module` expert implementations.

### `PyTorchExecutor`

Reference local expert execution backend.

### `ExecutorOutput`

Packed executor result consumed by the future Merger.

Contains:

- `packed_expert_outputs`
- `weighted_expert_outputs`
- per-expert output descriptors
- output metadata
- execution metadata
- statistics
- timing placeholder
- workspace metadata

## Workspace

`ExecutorWorkspace` reuses buffers for:

- packed expert outputs
- weighted expert outputs
- gathered activations
- temporary outputs

This avoids repeated allocation during inference iterations and keeps the API compatible with future CUDA Graph constraints.

## Kernel Boundaries

Current replacement boundary:

```text
kernels/reference.py::reference_execute_expert
```

Future backends can replace PyTorch internals with:

- Triton kernels
- CUDA kernels
- grouped GEMM
- persistent kernels
- FP8 execution
- TensorRT execution
- multi-stream execution
- distributed expert execution

without changing Executor inputs or `ExecutorOutput`.

## Optimized Weight Representation

`ExpertWeightProvider` is the internal optimized-execution ABI. The initial
`QwenSwiGLUWeightProvider` discovers Qwen-style `gate_proj`, `up_proj`, and
`down_proj` modules and exposes logical expert-major layouts:

```text
gate_up_weights: [E, 2I, H]
down_weights:    [E, H, I]
```

Hugging Face experts normally own separate parameter tensors. The provider
therefore exposes storage-preserving matrix views rather than eagerly calling
`torch.stack` or `torch.cat`; each logical expert-major entry references the
original parameter storage. Explicit `materialize()` methods exist only for a
future backend that intentionally chooses a packed-weight memory tradeoff.

`TritonExpertExecutor` is registered as `backend="triton"`. It validates and
owns this provider, but currently delegates execution to `PyTorchExecutor` and
reports `triton_reference_fallback`. No Triton kernel is launched yet.

## Grouped GEMM Prototype

`kernels/grouped_matmul.py` contains the first real Triton kernel. It computes
an expert-major projection from activations `[N, K]`, physical packed weights
`[E, O, K]`, and dispatcher `expert_offsets [E + 1]`, producing `[N, O]` in
the same deterministic expert-major order. It has a PyTorch reference and is
tested independently of full expert MLP execution.

The storage-preserving provider remains the default representation. The
prototype uses explicit `materialize_expert_major_weights()` when it needs a
physical `[E, O, K]` tensor, and `TritonExpertExecutor.forward()` does not yet
call this path. This keeps the first kernel measurable without introducing
silent model-weight duplication into the runtime.

## Tests and Benchmark

Tests live in `tests/executor/test_pytorch_executor.py`.

Benchmark scaffold:

```text
benchmarks/benchmark_executor.py
```

The benchmark measures executor latency, tokens/sec, workspace reuse, expert execution, routing weight application, and output collection for the reference PyTorch backend.
