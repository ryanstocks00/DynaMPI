<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# Lock-Free RMA Design Notes

Design background for the three RMA-based distributors:
`LockFreeRMAWorkDistributor` and
`HierarchicalLockFreeRMAWorkDistributor`.  For the user-facing summary
and configuration, see [Implementations](implementations.md).

## Why one-sided at all

A two-sided distributor spends one request/response round trip per task, and
every one of those messages lands on the manager.  Passive-target RMA replaces
the handshake with an atomic on a shared counter: a worker claims work without
the manager doing anything, so the manager's involvement drops from "two messages
per task" to "occasionally read the window".

The cost is that everything the protocol needs — task payloads, results,
completion status — has to live in a preallocated window with a fixed layout, and
memory visibility has to be established explicitly rather than falling out of
message matching.

All three distributors open their windows with
`MPI_Win_lock_all(MPI_MODE_NOCHECK)` once at construction and hold that epoch for
the distributor's lifetime.  There are no fences, no per-task lock/unlock, and no
collectives on the hot path.

## Minimal index parallel-for

```text
Manager window:
  offset 0:  head / claim counter (int64)   ← MPI_Fetch_and_op(MPI_SUM) by every rank
```

```text
lock_all once
broadcast n

while true:
    idx ← fetch_and_op(+1, head); flush
    if idx >= n: break
    local.append(idx, worker_function(idx))

gather_sorted(local) on the manager        # one MPI_Gatherv, at the end
```

Results are packed as `[int64 index][int64 count][data]` records, gathered with a
single `MPI_Gatherv`, then sorted by index.  That terminal gather is the design's
one collective and its scaling limit.

## Lock-free RMA window protocol

`LockFreeRMAWorkDistributor` generalises this to arbitrary payloads and
removes the gather entirely.

### Layout

```text
[ head ][ total ][ finished ]   control block, 24 bytes, int64 each
[ task slot 0 ][ task slot 1 ] ... × max_tasks
[ result slot 0 ][ result slot 1 ] ... × max_tasks
[ log 0 ][ log 1 ] ...             int64 each, × max_tasks
```

Task and result slots are `[int64 count][data]`, padded to an 8-byte stride so
every slot is independently addressable.  Exact offsets are in
`include/dynampi/impl/lockfree_rma_distributor.hpp`; the hierarchical
variant reuses the identical layout per level via `detail::LockFreeRMALevel`.

| Field | Written by | Read by | Purpose |
|-------|-----------|---------|---------|
| `head` | claimants, `fetch_and_op(SUM)` | manager | How far claiming has progressed |
| `total` | manager, `fetch_and_op(REPLACE)` | claimants | How many tasks are published |
| `finished` | manager, `fetch_and_op(REPLACE)` | claimants | No further tasks will be published |
| task table | manager, bulk `Put` | claimants, `Get` | Task payloads |
| result table | claimants, bulk `Put` | manager, `Get` | Result payloads |
| completion log | claimants, plain `Put` | manager, `Get` | Which claimed ranges are fully written |

### Ordering rules

Passive-target RMA is non-blocking, so every step names the completion it needs:

- `MPI_Win_flush_local` completes an operation **at the origin** — enough before
  reading the value returned by a `fetch_and_op` or a `Get`.
- `MPI_Win_flush` completes it **at the target** — required before another rank
  may observe a `Put` or an atomic.

Two orderings are load-bearing:

1. **Publish:** `Put(task slots)` → `atomic REPLACE(total)` → `flush`.  `total` is
   the only gate on claiming, so bumping it last means a claimant can never see
   a task index whose payload is still in flight.
2. **Complete:** `Put(result data)` → `flush` → `Put(log[start] = count)` →
   `flush`.  The intervening flush is what portably orders data before flag.

### Why a completion log

The log turns "is this result ready?" into a single contiguous-prefix scan
instead of a per-slot poll, and it needs no atomics: `fetch_and_op` on `head`
already gives a claimant exclusive ownership of its start index, so the log entry
at that index has exactly one writer.  A zero entry is the untouched sentinel —
a real entry is always `count >= 1`.

The manager harvests in three round trips regardless of how many batches turn out
to be ready:

```text
atomic read(head)                       # how far claiming got
Get(log[frontier .. head))              # one bulk read
scan forward while log[i] != 0
Get(result[frontier .. confirmed_end))  # one bulk read
```

A gap — a batch still in flight — simply stops the scan for that call.
Already-complete entries beyond the gap are picked up on a later pass, at the
cost of re-scanning that stretch of log, which is cheap because the scan is one
bulk `Get` either way.

### Claiming under a moving `total`

Claims are unconditional `fetch_and_add`s, so aggregate claim rate can outrun
publication and a claimed index can land beyond `total`.  Claimants therefore
carry a `pending_start`/`pending_end` remainder, resolved on later iterations as
`total` catches up, and clamp the ready boundary into `[start, end]` — clamping to
`min(end, total)` alone can produce a boundary *below* the claim's own start,
which would make a rank write results into a range another claimant owns.

Symmetrically, observing `finished` means "`total` will not grow", not "stop
claiming": a rank that has seen `finished` must still be able to claim a genuine
unclaimed gap between its cached head and the final `total`.

### Self-targeted RMA

The manager reads its own window through the RMA API rather than plain loads.
Under `MPI_WIN_SEPARATE` — the model MS-MPI always uses — a window owner is not
guaranteed to observe another rank's completed RMA writes through local loads at
all.  Going through `Get`/`Fetch_and_op` is correct under both memory models.

The same reasoning lets a rank be both owner *and* claimant of a level in the
hierarchical variant: a self-targeted RMA operation behaves exactly like a remote
one, just over a loopback.

### Progress

Where a rank must spin on remote state, it calls `MPI_Iprobe` between polls
rather than re-flushing the window.  Every RMA primitive here already flushes its
target before returning, so an extra `MPI_Win_flush_all` would only add
contention; `MPI_Iprobe` drives the two-sided progress engine, which some stacks
(MS-MPI in particular) require before self-targeted `Get`s ever observe a change.
Idle ranks additionally back off for tens of microseconds — staggered by rank on
Windows — so oversubscribed ranks do not flood the window as a thundering herd.

!!! warning "Validate RMA progress on your MPI"
    Early `MPI_Win_sync` + `MPI_Fetch_and_op` experiments under
    `MPI_Win_lock_all` did not make remote atomic updates visible to local loads
    on some MPICH 4.0 (ch4:ofi) configurations without asynchronous progress.
    The shipped distributors avoid that pattern and are exercised in CI, but
    behaviour at scale is stack-dependent — measure before relying on it.

## Composing levels

`HierarchicalLockFreeRMAWorkDistributor` instantiates the protocol above
once per tree level (`detail::LockFreeRMALevel`), which is possible precisely
because nothing in it is collective: publishing, claiming, writing and harvesting
are independent one-sided operations with no requirement that all claimants
participate together.

The claim and write steps are exposed as separate non-blocking calls
(`try_claim()` / `write_result_range()`) rather than a single claim-compute-write
loop, because a coordinator does not compute what it claims — it republishes into
the level below and writes results back only once they return.  A leaf worker
simply calls both back to back, reassembling the flat class's loop from the same
two primitives.

Size-1 levels skip `MPI_Win_create` entirely (some MPI implementations reject
window creation on singleton communicators) and fall back to plain loads and
stores on the owner's buffer, which is correct because owner and claimant are
then the same process.
