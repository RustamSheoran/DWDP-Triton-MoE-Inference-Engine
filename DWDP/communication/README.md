# DWDP Communication Runtime

`CommunicationEngine` coordinates expert residency. `WeightManager` owns only expert metadata and validated resident-state transitions. `CacheManager` owns resident capacity, LRU ordering, pin counts, and eviction. `IPCManager` owns imported CUDA IPC mappings. `PeerTopology` caches peer-access capability and enables supported peer paths.

`TransferScheduler` coalesces requests by expert id and owns transfer lifecycle:

```text
CREATED -> QUEUED -> RUNNING -> COMPLETED
                    |           
                    +-> FAILED
                    +-> QUEUED (bounded retry)
QUEUED -> CANCELLED
```

The worker-facing scheduler API returns one shared task for duplicate requests. Completion callbacks observe the terminal state. A failed task does not close the scheduler.

Communication paths are selected by `CommunicationPolicy`:

```text
resident -> LOCAL
IPC mapping -> IPC
peer-accessible remote source -> P2P
otherwise -> COPY
```

All transfer completion is represented by CUDA events. Compute enqueues a wait on the selected expert event and never performs a device-wide or stream-wide synchronization.

Public runtime operations:

- `registerExpert`
- `registerIPCExpert`
- `prefetch`
- `wait`
- `isResident`
- `getResidentPointer`
- `release`
- `shutdown`
