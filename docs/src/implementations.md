<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# Distributor Implementations

DynaMPI provides three distributor implementations.

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
- **Ordering:** Strictly ordered by task ID
- **Worker count:** One `int64_t` per worker, plus a stack of free ranks

---

## OneSidedMPIWorkDistributor

**Best for:** large task payloads where one-sided RMA avoids the two-sided
message path.

Uses MPI one-sided `MPI_Put` with active-target (`MPI_Win_fence`)
synchronisation.  The manager exposes a result slot per worker; each worker
exposes a single task slot.  Three fences per round.

### Protocol

```
// One round — all ranks participate in each fence:

MPI_Win_fence()          // start epoch A: workers publish
  worker: MPI_Put(result into manager's result_slot)
MPI_Win_fence()          // start epoch B: manager processes
  manager: read result_slots (local load), collect completed results
  manager: for each idle worker, write task, MPI_Put(TaskPending, worker_slot)
MPI_Win_fence()          // start epoch C: workers fetch
  worker: read worker_slot (local load)
          if TaskPending: copy task payload, execute between epochs
          if Shutdown:   exit
```

**Window layout:**

| Process | Window |
|---------|--------|
| Manager | `[result_slot₀] [result_slot₁] ... [result_slot_{W-1}]` — per-worker headers + result payloads |
| Worker | `[task_slot]` — header + task payload |

- **Communication:** One-sided `MPI_Put` (no `MPI_Send`/`MPI_Recv`)
- **Ordering:** Strictly ordered by task ID
- **Latency:** 3 fences (barriers) per round; worker executes between epochs C and A

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_task_count` | `int` | `1024` | Max element count for resizable task types |
| `max_result_count` | `int` | `1024` | Max element count for resizable result types |

---

## HierarchicalMPIWorkDistributor

**Best for:** large process counts (100+ ranks), multi-node clusters.  This is
the default distributor.

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

#### coordinator_per_node = true (default)

Uses `MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` to discover physical nodes.
One *node coordinator* per node (local rank 0).  Manager + coordinators form
a *leader group*.  Workers talk to their coordinator via shared memory; the
manager is excluded from its own node's local group.

#### coordinator_per_node = false

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
- **Batching:** Coordinators batch requests and results to amortise overhead

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `message_batch_size` | `optional<size_t>` | auto | Tasks per batch |
| `max_workers_per_coordinator` | `optional<int>` | `max(2, √N)` | Children per node |
| `batch_size_multiplier` | `int` | `2` | Prefetch = children × multiplier |
| `coordinator_per_node` | `bool` | `true` | Physical-node topology |

---

## Choosing a Distributor

| Scenario | Use |
|----------|-----|
| < 64 ranks, simple workloads | `NaiveMPIWorkDistributor` |
| Large task payloads, shared memory | `OneSidedMPIWorkDistributor` |
| 100+ ranks, multi-node | `HierarchicalMPIWorkDistributor` (default) |
| Heterogeneous task durations | `HierarchicalMPIWorkDistributor` with `coordinator_per_node=true` |

## Comparison

| Feature | Naive | OneSided | Hierarchical |
|---------|-------|----------|--------------|
| Communication | Two-sided | One-sided RMA | Two-sided + batching |
| Ordered results | Yes | Yes | No |
| Task prioritisation | Yes | Yes | No |
| Statistics | Yes | Yes | Yes |
| Node-aware topology | No | No | Yes |
| Manager bottleneck | O(W) | O(W) fence latency | O(log W) |
| Max practical ranks | ~64 | ~32 | ~1000+ |
