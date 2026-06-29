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
