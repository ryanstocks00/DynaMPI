<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# Distributor Implementations

DynaMPI ships six distributors.  Five share the full manager–worker interface
(`insert_task*` / `run_tasks` / `run_worker` / `finalize`); the sixth is a
deliberately minimal parallel-for helper with its own small API.

| Class | Header | Communication | Topology | Ordered |
|-------|--------|---------------|----------|---------|
| [`NaiveWorkDistributor`](#naiveworkdistributor) | `impl/naive_distributor.hpp` | Two-sided `Send`/`Recv` | Flat | **Yes** |
| [`HierarchicalWorkDistributor`](#hierarchicalworkdistributor-dynamicworkdistributor) | `impl/hierarchical_distributor.hpp` | Two-sided, batched | Tree | No |
| [`LockFreeRMAWorkDistributor`](#lockfreermaworkdistributor) | `impl/lockfree_rma_distributor.hpp` | Passive-target RMA | Flat | **Yes** |
| [`HierarchicalLockFreeRMAWorkDistributor`](#hierarchicallockfreermaworkdistributor) | `impl/hierarchical_lockfree_rma_distributor.hpp` | Passive-target RMA | Tree | **Yes** |

Only `NaiveWorkDistributor`, `HierarchicalWorkDistributor` are pulled in by `<dynampi/dynampi.hpp>`.
The other three need their own `#include` — see [Headers](api.md#headers).

---

## Shared behaviour

### Result ordering

Ordering is exposed as a compile-time constant, `Distributor::ordered`:

- **`NaiveWorkDistributor`, `LockFreeRMAWorkDistributor`,
  `HierarchicalLockFreeRMAWorkDistributor` (`ordered == true`)** — the manager
  buffers results by task ID (a result table slot for the RMA distributors,
  keyed the same way as the task it answers) and only releases a contiguous
  prefix, so returned results are always in insertion order.  A slow task
  therefore holds back every result behind it, all the way up the tree for the
  hierarchical variant.
- **`HierarchicalWorkDistributor` (`ordered == false`)** — results are
  returned as they are confirmed complete, roughly submission order but not
  guaranteed.

If you need to know which task a result came from and the distributor is
unordered, carry the identity in the result type (e.g. `std::pair<size_t, T>`).

### Lifecycle

1. **Construction is collective** over `Config::comm`.  The communicator is
   duplicated, so DynaMPI traffic never collides with the caller's.
2. With `auto_run_workers = true` (the default), every non-manager rank enters
   `run_worker()` inside the constructor and does not return until shutdown is
   signalled.  Set it to `false` if you want to call `run_worker()` yourself.
3. The manager drives distribution with `run_tasks()` /
   `finish_remaining_tasks()`.
4. `finalize()` signals shutdown to the workers.  The destructor calls it if you
   have not.
5. **Destruction is collective too** — the RMA distributors free windows and
   barrier inside their destructors.

### Failing tasks

A task that throws does not abort the job.  Every distributor catches whatever
escapes the worker function, substitutes a default-constructed `ResultT` so the
task still occupies its slot in the result set, and carries the message back to
the manager out of band — a `TASK_ERROR` message alongside the result for the
two-sided distributors, a small fixed-size table in the window for the RMA ones.

What the manager then does is up to `Config::rethrow_task_errors`:

- **`true` (default) — propagate.**  `run_tasks()` / `finish_remaining_tasks()`
  throw `dynampi::TaskFailure`, carrying the failing rank and message.  It is
  thrown *before* results are drained, so whoever catches it can call again and
  still collect everything gathered so far.  Each failure is thrown once;
  `finalize()` and the destructors never throw.
- **`false` — recover.**  Distribution runs to completion and the failures come
  back from `take_task_errors()` as a `std::vector<TaskError>`, oldest first.

```cpp
dynampi::DynamicWorkDistributor<int, double>::Config config;
config.rethrow_task_errors = false;
dynampi::DynamicWorkDistributor<int, double> distributor(work, config);

if (distributor.is_root_manager()) {
  distributor.insert_tasks({1, 2, 3});
  auto results = distributor.finish_remaining_tasks();   // one entry per task
  for (const auto& failure : distributor.take_task_errors()) {
    std::cerr << "rank " << failure.worker_rank << ": " << failure.message << "\n";
  }
}
```

Messages are truncated to `dynampi::kMaxTaskErrorMessage` (240) characters, and
the RMA distributors keep at most 16 records per window — beyond that the
failures still happen and the run still completes, but the extra messages are
dropped.  Anything not thrown or taken by the time the distributor is destroyed
is reported to `stderr` rather than vanishing.

### Single-rank runs

Every distributor handles a communicator of size 1 by executing tasks inline on
the manager, with no MPI window or message traffic.  Tests and serial debugging
therefore work unchanged.

---

## NaiveWorkDistributor

**Best for:** small communicators, ordered results, task priorities, and as a
reference implementation when debugging a workload.

The manager talks to every worker directly.  Workers announce themselves with an
empty `REQUEST`, then receive one task at a time; each `RESULT` doubles as the
signal that the worker is free again.

```text
Worker                                Manager
  send REQUEST ───────────────────►   push worker onto free stack
  probe / recv TASK  ◄────────────    send TASK (pop free worker)
  execute
  send RESULT ────────────────────►   recv RESULT, store by task ID
                                      push worker onto free stack
  probe / recv TASK  ◄────────────    send TASK
  ...
  probe / recv DONE  ◄────────────    finalize(): send DONE to every worker
```

- **Ordering:** strict — results are buffered by task ID and released as a
  contiguous prefix.
- **Prioritization:** supported, and this is the *only* distributor where it
  works.  With `dynampi::enable_prioritization`, `insert_task(task, priority)`
  feeds a `std::priority_queue` and higher priorities are dispatched first.
  Note that `insert_tasks(vector)` is disabled in this mode.
- **Variable-length payloads:** supported for both `TaskT` and `ResultT`
  (`std::vector<T>`, `std::string`, …).
- **Statistics:** `Aggregated` and `Detailed` both supported (see
  [Statistics](api.md#statistics)).  `worker_task_counts` is indexed by worker
  index — the rank with the manager removed.

### Cost model

The manager handles two messages per task and is the only rank doing so, which
makes it the bottleneck at a few hundred ranks.  There is no batching and no
prefetching: a worker is idle for one manager round trip between finishing a
task and receiving the next.

### Configuration

Beyond the [common fields](api.md#common-config-fields) there are none that take
effect.

---

## HierarchicalWorkDistributor (`DynamicWorkDistributor`)

**Best for:** the general multi-node case.  This is the default used by
`mpi_manager_worker_distribution`, and `dynampi::DynamicWorkDistributor` is an
alias for it.

Ranks are arranged in a tree.  Leaf workers only ever talk to their node
manager; node managers exchange *batches* with their own parent, so the root
manager's message count scales with the number of node managers rather than the
number of ranks.

```text
  Root manager ──┬── Node manager 0 ──┬── Worker 0
                 │       (node 0)     ├── Worker 1
                 │                    └── Worker 2
                 ├── Node manager 1 ──┬── Worker 3
                 │       (node 1)     └── Worker 4
                 └── Node manager 2 ──┬── Worker 5
                         (node 2)     └── Worker 6
```

### Topology

**`manager_per_node = true` (default).**
`MPI_Comm_split_type(MPI_COMM_TYPE_SHARED)` discovers physical nodes.  Local rank
0 of each node becomes that node's manager; the root manager is excluded from its
own node's local group so it is never also a node manager.  The root manager plus
the node managers form the *leader layer*.

Two knobs reshape this:

- `max_local_group_size > 0` splits large nodes into several contiguous local
  groups, producing more (smaller) node managers.  This reduces per-manager
  contention, and lets a single-node job exercise the multi-manager paths.
- `max_upper_fanout` caps how many direct children any leader-layer rank may
  have.  If there are more managers than that, they are recursively grouped
  into a k-ary tree of intermediate leaders, so not even the manager exceeds the
  cap.  `-1` (default) is auto: stay flat at ≤ 32 managers, otherwise use the
  smallest power of two ≥ `sqrt(manager_count)`.  `0` disables grouping
  entirely (one flat leader layer).

**`manager_per_node = false`.**  A virtual tree derived purely from rank
order, with fan-out `max_workers_per_manager` (default `max(2, sqrt(N))`).
Useful when shared-memory splitting is unavailable or misleading (e.g. under
simulators).

### Protocol

Leaf workers use the unbatched `REQUEST` / `TASK` / `RESULT` exchange.
Managers use the batched one:

```text
manager:
  top_up_pipeline(1)                       # one REQUEST_BATCH(n) to parent
  loop until DONE:
    release any prefetched batch into the work queue
    wait for TASK_BATCH if the queue is empty (flushing results while waiting)
    mark round active                      # later replies are quarantined
    top_up_pipeline(pipeline_depth - 1)    # ask for the next batches now
    while queue non-empty:
      pop a free child, send it TASK_BATCH (or TASK for a leaf)
      otherwise block until a child reports in
    end round; send whatever results have accumulated up as RESULT_BATCH
  propagate DONE to children, drain in-flight results, flush
```

Two details matter for throughput:

- **Pipelining.** A manager keeps `pipeline_depth` batch requests in flight
  (default 2 = double buffering) so children never wait a full parent round trip
  between batches.  Replies that arrive mid-round are quarantined until the round
  boundary, so a round can never overshoot into the next round's tasks.  Deeper
  pipelines hide more latency but commit tasks further ahead, which coarsens load
  balancing — tasks already handed to one manager cannot be reassigned to an
  idle sibling.
- **Partial result flushing.** Results are forwarded upward at every round
  boundary, whether or not that round is complete.  Waiting for a round's slowest
  straggler would otherwise stall results from children that finished long ago.

Batch size is `subtree_leaf_count × batch_size_multiplier` — the number of leaf
workers a manager is responsible for feeding, including its own node's, not
the number of children it sends to.  The two differ at every level above the
node managers, since a leader's leader-layer children are themselves
managers fronting whole nodes: at 2048 nodes with 7 ranks per node a leader
has 69 direct children but 384 leaf workers beneath it, and the gap grows by a
factor of the fanout per level.  `HierarchicalLockFreeRMAWorkDistributor` sizes
its claims the same way (`setup_upper_chain()`'s `feed_width`).

### Constraints

!!! note "Variable-length payloads are supported"
    `TaskT` and `ResultT` may be `std::vector<T>`, `std::string`, or any other
    type with `MPI_Type<T>::resize_required == true`.  Single tasks and results
    are sent with a probed count; batches are packed into a length-prefixed flat
    buffer (`detail::pack_variable_batch`), so no separate code path is needed
    and there is no per-element cap of the kind the RMA distributors impose.

!!! warning "Prioritization is not supported here"
    The `enable_prioritization` option compiles and `insert_task(task, priority)`
    exists, but any manager that receives a `TASK_BATCH` hits
    `DYNAMPI_UNIMPLEMENTED` (undefined behaviour under `NDEBUG`).  It happens to
    work only in degenerate topologies where no manager has children.

### Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_workers_per_manager` | `optional<int>` | `max(2, sqrt(N))` | Fan-out of the virtual tree. Only used when `manager_per_node == false`. |
| `batch_size_multiplier` | `int` | `1` | Batch request size = subtree leaf workers × this. |
| `pipeline_depth` | `int` | `2` | Batch requests kept outstanding, including the active round. `1` disables prefetching. |
| `manager_per_node` | `bool` | `true` | Use shared-memory node discovery instead of a rank-order virtual tree. |
| `max_local_group_size` | `int` | `0` | `> 0` splits nodes into local groups of at most this size. `manager_per_node` only. |
| `max_upper_fanout` | `int` | `-1` | Max direct leader-layer children. `-1` auto, `0` disabled, `> 0` explicit. `manager_per_node` only. |

**Statistics:** `track_statistics<Aggregated>` and `<Detailed>` are both
supported.  `Statistics::worker_task_counts` is populated by an `MPI_Gather`
inside `finalize()`, so it is `std::nullopt` until then.

---

## LockFreeRMAWorkDistributor

**Best for:** fine-grained tasks where two-sided hand-off latency dominates.  No
collective calls anywhere in the hot path.

The manager exposes one RMA window containing a control block, a task table, a
result table and a completion log.  Everything is passive-target
(`MPI_Win_lock_all(MPI_MODE_NOCHECK)` once at construction) — workers never wait
on the manager to make progress.

```text
manager publish (insert_tasks):
  Put[task slots]  →  atomic REPLACE(total)  →  flush          # 2 round trips per batch

worker (per claim, claim width 1):
  fetch_and_add(head, 1) + flush_local        # claim an index
  Get(task slot)         + flush_local        # read the payload
  compute
  Put(result slot)       + flush              # data durable at the manager
  Put(log[start] = count)+ flush              # publishes the result, after data

manager harvest (run_tasks):
  atomic read(head)                           # how far claiming got
  Get(log[frontier .. head))                  # one bulk read
  scan the contiguous prefix; Get(result[frontier .. confirmed))
```

The completion log is what makes the result path correct without atomics on the
data: `fetch_and_add` already gives a worker exclusive ownership of its start
index, so the log entry is a plain `Put`, and the intervening flush is what
portably orders "data written" before "flag published".  The manager only
advances past a contiguous prefix, so a straggling batch stops that harvest and
is picked up on a later pass.

Harvesting reads the manager's *own* window through the RMA API rather than
plain loads: under `MPI_WIN_SEPARATE` (always the case on MS-MPI) local loads are
not guaranteed to observe remote writes at all.

- **Ordering:** guaranteed, via the contiguous-prefix harvest above -- same
  cost as `NaiveWorkDistributor`'s ordering, a straggler holds back every
  result behind it.
- **Prioritization:** not supported.  `insert_task(task, priority)` exists for
  interface compatibility and ignores the priority.
- **Variable-length payloads:** supported, within the per-slot capacity below.
- **Statistics:** `track_statistics<Detailed>` supported; `Statistics` carries
  `comm_statistics` only (no per-worker task counts).
- **Extra API:** `gather_once()` performs exactly one harvest pass and returns
  whatever is ready, instead of looping until everything is collected.

### Capacity

The window is preallocated, so capacities are hard limits, not hints.

| Field | Default | Meaning |
|-------|---------|---------|
| `max_tasks` | `8192` | **Lifetime** total of published tasks, not a concurrent depth. Exceeding it throws `std::length_error`. |
| `max_task_count` | `256` | Max element count of a single task, for variable-length `TaskT`. Ignored for fixed-size types. |
| `max_result_count` | `256` | Same, for `ResultT`. |

Exceeding `max_task_count` / `max_result_count` trips an assertion in debug
builds and corrupts the window in release builds — size them for your worst case.

Manager window bytes are

```text
24 + max_tasks × (task_stride + result_stride + 8)

task_stride   = round_up_8(8 + max_task_count   × sizeof(task element))
result_stride = round_up_8(8 + max_result_count × sizeof(result element))
```

(For fixed-size types the count factor is 1.)  With `int` tasks and `double`
results at the defaults that is ~320 KiB; with `std::vector<int>` payloads at
`max_task_count = max_result_count = 256` it is ~16 MiB.

### Batching guidance

`insert_task()` publishes immediately — two RMA round trips for one task.
`insert_tasks(vector)` publishes the whole span in two round trips regardless of
size.  Always prefer the batch form; per-task publishing is the measured
bottleneck otherwise.

---

## HierarchicalLockFreeRMAWorkDistributor

**Best for:** large node counts where a single manager-owned window becomes the
ceiling.

Takes the lock-free RMA protocol above and instantiates it once *per level* of the
node-aware tree: one window at the leader level (root manager ↔ node managers), plus an
independent window per node (manager ↔ its local workers).  Each level is an
independent `detail::LockFreeRMALevel`, so claiming, publishing and harvesting stay
purely one-sided — composing hierarchically reintroduces no collectives.

A node manager plays two roles simultaneously: **owner** of its local level
and **claimant** of the level above.  It does not compute tasks itself.  Instead
each `BridgeHop` runs, non-blocking, per iteration:

1. Claim a range from the parent level (subject to backpressure) and republish it
   into the child level, queueing a `{parent_start, child_len}` relay entry.
2. Harvest whatever the child level has confirmed and append it to a relay buffer.
3. Write the available prefix of the front relay entry back to the parent, in as
   many partial `write_result_range` calls as it takes.

Two properties are load-bearing:

- **Backpressure.** Outstanding relayed tasks are capped at
  `8 × parent claim width`.  Because the parent's harvest is a contiguous-prefix
  scan, relays flush strictly FIFO — claiming far ahead adds *ordering* latency,
  not just buffering, since a caught-up entry near the back of the queue cannot
  flush until everything ahead of it does.
- **Partial flushing.** A relay entry is written back incrementally rather than
  all-or-nothing.  Since claim width scales with the local worker count, an
  all-or-nothing flush would relay nothing until every task of a wide claim
  finished — an artificial barrier whose cost grows with task duration.

Ranks alone on a node skip the local level and claim/compute directly against the
level above (`run_leaf_leader_worker`).  Under `max_upper_fanout` grouping, a
promoted group leader both owns its group's window and claims from it via
self-targeted RMA.

The constructor ends with an `MPI_Barrier`: level setup runs a chain of
`MPI_Comm_split` calls whose cost grows with rank count, and without the barrier
a caller that starts timing when the manager's constructor returns would measure
other ranks still finishing setup.

- **Ordering:** guaranteed, recursively at every level of the relay chain --
  see [Result ordering](#result-ordering).
- **Prioritization:** not supported (no priority overload at all).
- **Statistics:** **not supported.**  The `Options...` pack is accepted and
  ignored, and there is no `get_statistics()`.
- **Extra API:** `gather_once()`, as above.

### Configuration

| Field | Default | Description |
|-------|---------|-------------|
| `max_tasks` | `8192` | Lifetime task capacity of each **upper-level** window. |
| `max_local_tasks` | `8192` | Lifetime task capacity of each **node-local** window. |
| `max_task_count` / `max_result_count` | `256` | Per-slot element caps, as for the flat variant. |
| `max_local_group_size` | `0` | `> 0` splits nodes into smaller local groups. |
| `max_upper_fanout` | `-1` | `-1` auto (flat at ≤ 32 managers, else smallest power of two ≥ `sqrt(manager_count)`), `0` disabled, `> 0` explicit. |

Memory is the flat formula applied per window, so a manager pays for its
local window *and* its share of the upper ones.

---

## Feature matrix

| | Naive | Hierarchical | LockFreeRMA | Hier. LockFreeRMA |
|---|---|---|---|---|
| Communication | Two-sided | Two-sided, batched | Passive RMA | Passive RMA, per level |
| Collectives on the hot path | No | No | No | No |
| Ordered results | **Yes** | No | **Yes** | **Yes** |
| Arbitrary `TaskT` / `ResultT` | Yes | Yes | Yes | Yes |
| Variable-length payloads | Yes | Yes | Yes (capped) | Yes (capped) |
| [Custom structs](api.md#custom-types) | Yes | Yes | Yes | Yes |
| Prioritization | **Yes** | No | No (ignored) | No |
| Statistics | `Aggregated`, `Detailed` | `Aggregated`, `Detailed` | `Aggregated`, `Detailed` | **None** |
| Node-aware topology | No | Yes | No | Yes |
| Incremental insertion | Yes | Yes | Yes | Yes |
| Preallocated capacity limit | No | No | `max_tasks` | `max_tasks`, `max_local_tasks` |
| Manager load per task | 2 messages | O(1/batch), per manager | O(1/batch), per manager | ~3 RMA ops | ~3 RMA ops, per level | 1 atomic |

---

## Benchmarking

Measured throughput depends heavily on the machine, the MPI stack and the task
duration, so pick a distributor by measuring rather than from a table.  The
repository ships drivers for exactly that (build with
`-DDYNAMPI_BUILD_BENCHMARKS=ON`):

| Benchmark | Measures |
|-----------|----------|
| `weak_scaling_distribution_rate` | Task hand-off rate at fixed rank count, across all four full distributors |
| `asymptotic_distribution_throughput` | Throughput vs. task count |
| `shutdown_time` | Cost of finalisation at scale |
| `rma_atomic_microbench` | Raw `MPI_Fetch_and_op` ceiling for the RMA path |
| `pingpong`, `timer_resolution` | Baseline latency and clock granularity |

```bash
mpirun -n 64 ./build/benchmark/weak_scaling_distribution_rate -D lockfree_rma -t 100 -d 10
```

`-D` selects `naive`, `hierarchical`, `lockfree_rma` or
`hierarchical_lockfree_rma`; `-t` is the expected task duration in
microseconds and `-d` the measurement window in seconds.  The two RMA
distributors preallocate their windows for `--max_tasks` tasks (default 500M,
roughly 19 GiB on the owning rank), so lower it to run them anywhere with less
memory than a compute node — the run warns and stops early if it exhausts the
table before `-d` elapses.  Launch scripts for Frontier and Aurora (which take
the same value as `MAX_TASKS`), plus plotting helpers, live under
`benchmark/scripts/`.

For orientation, the design work behind these classes found the flat lock-free RMA
distributor plateauing once a single manager window saturated, while the
hierarchical topologies kept climbing with node count — which is the entire
motivation for `HierarchicalLockFreeRMAWorkDistributor`.  Reproduce on
your own target before committing to a choice.
