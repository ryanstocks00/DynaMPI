<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# Lock-Based Work Stealing — Design

!!! warning "MPICH compatibility"
    `MPI_Win_sync` under `MPI_Win_lock_all` does not make remote
    `MPI_Fetch_and_op` results visible to local loads on MPICH 4.0
    (ch4:ofi) without async progress threads.  Workers atomically
    increment the counter but the manager's sync+load always sees the
    initial value.  This design requires either async progress or the
    two-window architecture described in the Implementation Note below.
    The fence-based `OneSidedMPIWorkDistributor` is the recommended
    approach for now.

## Rationale

The fence-based `OneSidedMPIWorkDistributor` uses three `MPI_Win_fence`
calls per round — collective barriers.  This replaces them with
`MPI_Win_lock_all(SHARED)` and passive-target RMA.

For the first implementation, tasks are indices (`size_t`).  The atomic
counter doubles as the task distributor — no ring buffer needed.

## Data Layout

Single MPI window, manager only.  Workers use `MPI_Win_create(nullptr, 0)`.

```
  Offset 0:   head_idx     (int64)   ← MPI_Fetch_and_op(SUM) by workers
  Offset 8:   total_tasks  (int64)   ← written by manager
  Offset 16:  done_count   (int64)   ← MPI_Fetch_and_op(SUM) by workers
  Offset 24:  result slots  (one per rank)
```

Result slot: `[task_id: int64_t | result: ResultT]`

No per-slot state needed — the manager only reads results after all tasks
are done, so there's no need to distinguish in-flight from completed slots.

## Synchronisation

`MPI_Win_lock_all(MPI_LOCK_SHARED, win)` once at startup.  Never unlocked
until shutdown.  All locks SHARED — no actor ever blocks another.

| Call | Purpose |
|------|---------|
| `MPI_Win_flush(manager)` | Complete my own RMA |
| `MPI_Win_sync(win)` | Manager: make remote writes locally visible, and vice versa |

## Protocol

### Worker

```
lock_all(SHARED, win)    // once

while true:
    // --- Claim ---
    Fetch_and_op(+1, &my_idx, SUM, head_off)
    Get(&total, total_off)
    flush(manager)

    if my_idx >= total:
        break                   // no more tasks (or shutdown)

    // --- Execute (no MPI) ---
    result = worker_function(my_idx)

    // --- Publish (one Put, one flush) ---
    buf = [my_idx, bytes(result)]
    Put(buf, sizeof(buf), MPI_BYTE, manager, my_result_slot_off)
    flush(manager)

    // --- Signal completion ---
    Fetch_and_op(+1, &ignored, SUM, done_count_off)
    flush(manager)
```

Per task: 3 RMA calls (2 × Fetch_and_op, 1 × Get, 1 × Put), 3 flushes.

### Manager

```
lock_all(SHARED, win)    // once

head_idx = 0;  total_tasks = n_tasks;  done_count = 0
sync(win)

// Wait until all tasks are done
while done_count < n_tasks:
    Get(&done_count, done_count_off)
    flush(manager)
    if nothing_changed:
        mpi_progress_spin(50us)

// All done — collect results
sync(win)
for each rank r:
    task_id = result_slot[r].task_id
    result  = result_slot[r].result
    store(task_id, result)

// Shutdown: prevent new claims, then workers will see my_idx >= total and exit
total_tasks = head_idx
sync(win)
```

### Incremental task insertion

```
insert_tasks(n_more):
    total_tasks += n_more
    sync(win)
```

Bumps `total_tasks` — workers polling the counter claim the new indices.
The `done_count` increases alongside, so the manager stays in its wait
loop until the new tasks are also complete.

## Correctness

### Result visibility

The worker publishes with a single `Put(buf, result_slot)` then signals
via `Fetch_and_op(+1, done_count)`.  The manager polls `done_count` with
`Get` + `flush`.  When `done_count == total_tasks`, all results have been
delivered.  The manager calls `Win_sync` — a memory barrier — then reads
every result slot from its own buffer.  Each slot is guaranteed to contain
the published data because `flush` completed the Put and `Win_sync` made
it visible.

### Claim atomicity

`MPI_Fetch_and_op(SUM)` on `head_idx` is atomic.  Each worker gets a
unique, monotonically increasing index.  The manager sets `total_tasks` to
the number of available tasks.  When `my_idx >= total_tasks`, the worker
exits — no more work.

### Shutdown

The manager sets `total_tasks = head_idx` (prevents new claims), calls
`Win_sync`.  Workers see the updated limit and exit.  All ranks call
`MPI_Win_unlock_all` + `MPI_Win_free`.

---

### Manager as Worker

Between sync cycles the manager atomically increments `head_idx` and calls
`worker_function(my_idx)` locally — no MPI needed.

## Comparison

| | OneSided (fence) | This (lock_all) |
|---|---|---|
| Barriers per round | 3 (collective) | 0 |
| Lock/unlock per task | 0 | 0 (lock_all once) |
| RMA per task | 2 Put (implicit) | 1 Fetch_and_op + 1 Get + 1 Put |
| Flushes per task | 0 | 2 |
| Memory overhead | O(W) slots + per-worker windows | Two int64 + O(W) result slots |
| Arbitrary task types | Yes | No (indices only; ring buffer planned) |

## Correctness

### Result publication handoff

The worker publishes the entire result slot in a **single contiguous
`Put`**: `[DONE | padding | task_id | result_bytes]`.

`MPI_Put` is non-blocking — data may be delivered to the target window
in any order and at any time.  The handoff uses two steps:

| Step | Call | Effect |
|------|------|--------|
| 1 | `MPI_Win_flush` (worker) | Put is fully delivered to the target window. The bytes are in memory on the manager. |
| 2 | `MPI_Win_sync` (manager) | Memory barrier — makes delivered bytes visible to the manager's local loads. |

After step 1, the data is in the window but the manager's CPU may still
see stale cache lines.  After step 2, local loads see the current
window contents.  The state field acts as a flag: if the manager reads
`DONE`, the accompanying data bytes were delivered by the same `Put`
and are already present.

```
Worker:   Put(buf, N) → flush     // deliver
Manager:  Win_sync → read state   // make visible, then check
```

No partial visibility: `Win_sync` operates on the whole window.  Either
the Put was delivered before the sync (all bytes visible) or it wasn't
(the old state is read).  No ordering between separate Puts is relied
upon.

### Claim ordering

`MPI_Fetch_and_op(SUM)` on `head_idx` is atomic.  Each worker receives a
unique, monotonically increasing index.  The manager sets `total_tasks` to
the number of available tasks.  When `head_idx` reaches `total_tasks`, all
tasks have been claimed.  The manager sets `total_tasks = head_idx` to
signal shutdown — workers polling the counter will see no more work.

---

## Implementation Plan

1. `LockFreeMPIWorkDistributor<TaskT, ResultT, Options...>` class.
   Restrict `TaskT = size_t` for v1.
2. `MPI_Win_create` on manager, `nullptr` on workers.
3. `MPI_Win_lock_all(SHARED)` in constructor.
4. Window: `[head_idx: int64 | total_tasks: int64 | done_count: int64 | result_slots...]`
5. Result slot: `[task_id: int64 | result: ResultT]` — no state flag needed.
6. `insert_task(size_t)` / `insert_tasks(vector<size_t>)` bump `total_tasks`.
7. Manager polls `done_count`, collects when it reaches `total_tasks`.
8. Shutdown: `total_tasks = head_idx`, `Win_sync`, `Win_unlock_all`, `Win_free`.
