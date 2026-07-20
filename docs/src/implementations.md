<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# Distributor Implementations

DynaMPI provides three full-featured distributors plus a minimal lock-free
parallel-for helper.

---

## NaiveMPIWorkDistributor

**Best for:** small-to-medium process counts, simplicity, ordered results.

Two-sided `MPI_Send` / `MPI_Recv` between the manager and each worker.
Workers send an initial REQUEST, then the manager assigns one task at a time.
Each incoming RESULT signals that the worker is ready for the next task.

### Protocol

```
Worker:                              Manager:
  send REQUEST ───────────────────→   (worker queued as free)
  probe() ←───────────────────────   send TASK
  recv TASK
  execute task
  send RESULT ───────────────────→   recv RESULT
                                     store result
                                     (worker queued as free again)
  probe() ←───────────────────────   send TASK
  ...repeat...
  probe() ←───────────────────────   send DONE (no more tasks)
```

- **Communication:** Two-sided `MPI_Send` / `MPI_Recv`
- **Ordering:** Strictly ordered by task ID (`ordered = true`)
- **Prioritization:** Supported with `enable_prioritization`

---

## MPIDynamicWorkDistributor

**Best for:** large process counts (100+ ranks), multi-node clusters.  This is
the default distributor (`mpi_manager_worker_distribution` and the type to
construct for dynamic workloads).

Implemented by `HierarchicalMPIWorkDistributor` (same type).

Organises ranks into a tree.  Leaf workers communicate only with their local
*node coordinator*; coordinators batch requests and results to/from the
manager.  Two topology modes:

### Topology

```
  Manager ──┬── Coordinator 0 ──┬── Worker₀
            │                   ├── Worker₁
            │                   └── Worker₂
            ├── Coordinator 1 ──┬── Worker₃
            │                   └── Worker₄
            └── Coordinator 2 ──┬── Worker₅
                                └── Worker₆
```

#### `coordinator_per_node = true` (default)

Uses `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` to discover physical nodes.
One *node coordinator* per node (local rank 0).  Manager + coordinators form
a *leader group*.  Workers talk to their coordinator via shared memory; the
manager is excluded from its own node's local group.

#### `coordinator_per_node = false`

Virtual tree built from rank ordering.  Fan-out defaults to `max(2, √N)`.

### Protocol (node coordinator)

```
while not done:
    send REQUEST_BATCH(n) to parent   // n = children × multiplier
    while task_queue not empty:
        if free children available:
            dequeue child, send TASK_BATCH
        else:
            recv from anyone:
                REQUEST  → push child to free stack
                RESULT   → return batch to parent, request more
                TASK     → execute locally, send RESULT up
                DONE     → propagate to children, exit
```

- **Communication:** Two-sided `MPI_Send` / `MPI_Recv` with batching
- **Ordering:** Not guaranteed (`ordered = false`)
- **Prioritization:** Not yet implemented
- **Batching:** Coordinators batch requests and results to amortise overhead

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message_batch_size` | `optional<size_t>` | auto | Tasks per batch |
| `max_workers_per_coordinator` | `optional<int>` | `max(2, √N)` | Children per node |
| `batch_size_multiplier` | `int` | `2` | Prefetch = children × multiplier |
| `coordinator_per_node` | `bool` | `true` | Physical-node topology |

---

## LockFreeMPIWorkDistributor

**Best for:** fine-grained tasks where passive-target RMA progress is strong,
and you want to avoid a two-sided request/response handshake per task.

Workers claim work by atomically advancing a shared counter on the manager's
MPI window (`MPI_Fetch_and_op`) and deposit results with `MPI_Put` under
`MPI_Win_lock_all`.  Supports arbitrary `TaskT` / `ResultT` (with fixed
capacity limits), incremental `insert_task(s)` / `run_tasks`, and ordered
results.

There is also **`MinimalLockFreeMPIWorkDistributor<ResultT>`**: a smaller
API for embarrassingly parallel index loops (`size_t → ResultT`) that claims
indices with one atomic counter and gathers results once at the end.

### Protocol (sketch)

```
Workers (lock_all once):
  while true:
    idx ← Fetch_and_op(+1, head)
    if idx >= total_tasks: exit (or wait for more / shutdown)
    result = worker_function(task[idx])
    Put(result) → manager; signal completion

Manager:
  insert_task(s) bumps total_tasks / publishes payloads in the window
  poll / gather completed results into run_tasks() return value
  finalize() sets finished flag and drains remaining work
```

- **Communication:** Passive-target RMA (`MPI_Win_lock_all`, `Fetch_and_op`, `Put`)
- **Ordering:** Ordered by task ID (`ordered = true`)
- **Prioritization:** Not supported (priority argument is ignored if enabled)

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_tasks` | `int` | `8192` | Lifetime capacity of task/result tables |
| `max_task_count` | `int` | `256` | Max elements per resizable `TaskT` |
| `max_result_count` | `int` | `256` | Max elements per resizable `ResultT` |

Design background (historical fence-based prototype and MPICH notes): see
[Lock-Free Design](mpi_compare_and_swap_design.md).

---

## Choosing a Distributor

| Scenario | Use |
|----------|-----|
| < ~64 ranks, need ordered results / prioritization | `NaiveMPIWorkDistributor` |
| 100+ ranks, multi-node (default) | `MPIDynamicWorkDistributor` |
| Fine-grained tasks, good RMA progress | `LockFreeMPIWorkDistributor` |
| Static index parallel-for only | `MinimalLockFreeMPIWorkDistributor` |

## Comparison

| Feature | Naive | MPIDynamic (hierarchical) | LockFree |
|---------|-------|---------------------------|----------|
| Communication | Two-sided | Two-sided + batching | Passive RMA |
| Ordered results | Yes | No | Yes |
| Task prioritisation | Yes | No | No |
| Statistics | Yes | Yes | Yes |
| Node-aware topology | No | Yes | No |
| Manager bottleneck | O(W) messages | O(coordinators) | Atomic claim + Put |
| Max practical ranks | ~64 | ~1000+ | MPI-RMA dependent |
