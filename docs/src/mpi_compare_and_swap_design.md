<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# Lock-Free RMA Distributor — Design Notes

These notes document the design that became `MinimalLockFreeMPIWorkDistributor`.
For the user-facing summary, see [Implementations](implementations.md).

!!! note "LockFreeMPIWorkDistributor removed"
    The arbitrary-payload, CAS-based `LockFreeMPIWorkDistributor` this
    document originally also covered has been removed: its result path
    (`MPI_Barrier` + `MPI_Gather`/`MPI_Gatherv` per round) scales badly
    under heavy oversubscription, measured taking 50+ seconds (sometimes
    timing out entirely) for a trivial test at 150-204 ranks on 2 nodes.
    Use `AsyncPutLockFreeMPIWorkDistributor` or one of the `Hierarchical*`
    distributors instead — no collectives on the hot path, and batching
    happens at coordinator levels rather than per flat claim.

!!! note "Historical context"
    An earlier fence-based one-sided prototype (`OneSidedMPIWorkDistributor`)
    used three `MPI_Win_fence` barriers per round.  That class is **not** in
    the current tree; passive-target `MPI_Win_lock_all` is the shipped RMA
    path.

!!! warning "MPICH compatibility"
    Early `MPI_Win_sync` + `MPI_Fetch_and_op` experiments under
    `MPI_Win_lock_all` did not make remote atomic updates visible to local
    loads on some MPICH 4.0 (ch4:ofi) configurations without async progress.
    The current lock-free distributors use flush/gather patterns that have
    been exercised in CI; still validate RMA progress on your target MPI
    before relying on lock-free at scale.

## Rationale

Passive-target RMA avoids a two-sided request/response handshake per task.
Workers claim indices with `MPI_Fetch_and_op` and results are gathered once
at the end.

`MinimalLockFreeMPIWorkDistributor` restricts tasks to loop indices
(`size_t`) -- see `AsyncPutLockFreeMPIWorkDistributor` (and the
`Hierarchical*` variants) for arbitrary task/result payloads in
fixed-capacity window slots.

## Data Layout (conceptual)

Manager window (simplified):

```
  Offset 0:   head_idx / claim counter   (int64)  ← Fetch_and_op by workers
```

Exact offsets live in `include/dynampi/impl/lockfree_distributor.hpp`.

## Synchronisation

`MPI_Win_lock_all(MPI_MODE_NOCHECK)` (or equivalent) once at startup.
Workers use `MPI_Win_flush` to complete their own RMA.  The manager gathers
completed results and eventually signals shutdown via a finished flag.

## Protocol (minimal index parallel-for)

```
lock_all once

while true:
    idx ← Fetch_and_op(+1, head)
    if idx >= n_tasks: break
    local_results.append(idx, worker_function(idx))

gather_sorted(local_results) on manager
```

## Comparison (fence prototype vs lock-free)

| | Fence prototype (removed) | Lock-free (current) |
|---|---|---|
| Barriers per round | 3 collective fences | 0 |
| Lock/unlock per task | 0 | 0 (`lock_all` once) |
| RMA per task | Puts in fence epochs | Fetch_and_op (+ one gather at the end) |
| Index-only fast path | — | `MinimalLockFreeMPIWorkDistributor` |

## Implementation checklist (shipped)

1. `MinimalLockFreeMPIWorkDistributor<ResultT>` for index parallel-for.
2. Manager window created on the root; workers attach with a placeholder base.
3. `Config::comm` / `manager_rank` only -- no per-task capacity limits needed
   since the task payload is just the loop index.
