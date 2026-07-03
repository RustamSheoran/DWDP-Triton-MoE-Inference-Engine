There is no GPU available offline in this environment, so avoid CUDA-dependent runs here. Python is available for CPU-only checks and documentation work.

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
