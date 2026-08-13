<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# API Reference

Everything lives in namespace `dynampi`.

## Headers

| Header | Provides |
|--------|----------|
| `<dynampi/task_error.hpp>` | `TaskError`, `TaskFailure`, `kMaxTaskErrorMessage` (pulled in by every distributor) |
| `<dynampi/impl/naive_distributor.hpp>` | `NaiveWorkDistributor` |
| `<dynampi/impl/hierarchical_distributor.hpp>` | `HierarchicalWorkDistributor` |
| `<dynampi/impl/lockfree_rma_distributor.hpp>` | `LockFreeRMAWorkDistributor` |
| `<dynampi/impl/hierarchical_lockfree_rma_distributor.hpp>` | `HierarchicalLockFreeRMAWorkDistributor` |
| `<dynampi/version.hpp>` | `dynampi::version` |
| `<dynampi/mpi/mpi_types.hpp>` | `MPI_Type<T>` — the trait you specialise for custom payloads |
| `<dynampi/mpi/mpi_communicator.hpp>` | `MPICommunicator`, `CommStatistics`, `StatisticsMode`, `track_statistics` |
| `<dynampi/mpi/mpi_error.hpp>` | `DYNAMPI_MPI_CHECK` |
| `<dynampi/utilities/timer.hpp>` | `Timer` |
| `<dynampi/utilities/assert.hpp>` | `DYNAMPI_ASSERT*` |
| `<dynampi/utilities/printing.hpp>` | `operator<<` for STL containers |

Every header is usable with a plain `-I include`; nothing requires CMake or
predefined macros.

## `mpi_manager_worker_distribution`

```cpp
template <typename ResultT,
          template <typename, typename, typename...> typename Distributor =
              HierarchicalWorkDistributor>
std::optional<std::vector<ResultT>> mpi_manager_worker_distribution(
    size_t n_tasks,
    std::function<ResultT(size_t)> worker_function,
    MPI_Comm comm = MPI_COMM_WORLD,
    int manager_rank = 0);
```

The one-call entry point.  Distributes tasks `0 .. n_tasks-1` across `comm`.
Collective: **every** rank must call it.  Workers run the worker loop internally
and return `std::nullopt`; the manager returns the results.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_tasks` | — | Number of tasks; the task value is the index itself |
| `worker_function` | — | `size_t → ResultT`, called on the worker rank |
| `comm` | `MPI_COMM_WORLD` | Communicator (duplicated internally) |
| `manager_rank` | `0` | Rank that distributes and collects |

`Distributor` may be any of the five full distributors.  The default is
`HierarchicalWorkDistributor`, so **results are unordered** — pass
`NaiveWorkDistributor` if you need `(*result)[i]` to be task `i`.

```cpp
auto ordered = dynampi::mpi_manager_worker_distribution<
    double, dynampi::NaiveWorkDistributor>(100, work);
```


## `DynamicWorkDistributor`

```cpp
template <typename TaskT, typename ResultT, typename... Options>
using DynamicWorkDistributor =
    HierarchicalWorkDistributor<TaskT, ResultT, Options...>;
```

An alias, not a distinct type.  Use it when you want "the recommended default"
to track future changes.

## The distributor interface

The five full distributors share this shape:

```cpp
template <typename TaskT, typename ResultT, typename... Options>
class Distributor {
 public:
  struct Config;
  struct RunConfig;
  static const bool ordered;

  explicit Distributor(std::function<ResultT(TaskT)> worker_function,
                       Config config = {});
  ~Distributor();   // calls finalize() if you have not
};
```


### Construction and lifecycle

Construction and destruction are both **collective** over `Config::comm`.  The
communicator is duplicated internally, so DynaMPI messages never interfere with
the caller's own traffic.

With `auto_run_workers = true` (default) non-manager ranks enter `run_worker()`
inside the constructor and only return once shutdown is signalled — typically
when the manager's distributor is destroyed.  Set it to `false` to control that
yourself:

```cpp
Distributor::Config cfg;
cfg.auto_run_workers = false;
Distributor dist(work, cfg);

if (dist.is_root_manager()) {
  dist.insert_tasks(tasks);
  auto results = dist.finish_remaining_tasks();
} else {
  dist.run_worker();      // returns when the manager finalizes
}
```

### Common `Config` fields

Present on all five full distributors:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `comm` | `MPI_Comm` | `MPI_COMM_WORLD` | Communicator to distribute over |
| `manager_rank` | `int` | `0` | Rank that owns the queue and collects results |
| `auto_run_workers` | `bool` | `true` | Non-manager ranks run the worker loop from the constructor |

Distributor-specific fields — tree shape, pipelining, RMA window capacities —
are documented per class in [Implementations](implementations.md).

### `RunConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_num_tasks` | `size_t` | `SIZE_MAX` | Return once this many results are ready |
| `allow_more_than_target_tasks` | `bool` | `true` | If `false`, clip the returned vector to `target_num_tasks`; the excess stays buffered for the next call |
| `max_seconds` | `optional<double>` | `nullopt` | Soft deadline for this call |

Two semantics worth knowing:

- On the ordered distributors (`NaiveWorkDistributor` and the two RMA
  distributors), `target_num_tasks` counts *contiguous* results from the front
  of the ordered stream.  On `HierarchicalWorkDistributor` (unordered), it
  counts however many have arrived, in any order.
- `max_seconds` is checked between messages, not asynchronously.  A call already
  blocked in a probe or an RMA wait will overshoot the deadline until the next
  event arrives.

### Methods

| Method | Returns | Notes |
|--------|---------|-------|
| `is_root_manager()` | `bool` | Safe on every rank |
| `insert_task(TaskT)` | `void` | Manager only. Disabled when `enable_prioritization` is set |
| `insert_task(const TaskT&, double priority)` | `void` | Manager only. Requires `enable_prioritization` (see the availability table) |
| `insert_tasks(const std::vector<TaskT>&)` | `void` | Manager only. Preferred over repeated `insert_task` on the RMA distributors |
| `insert_tasks(Range&&)` | `void` | Hierarchical variants only; any `std::ranges::input_range` |
| `run_tasks(RunConfig)` | `std::vector<ResultT>` | Manager only. Drives distribution and returns completed results |
| `finish_remaining_tasks()` | `std::vector<ResultT>` | Manager only. `run_tasks({})` — runs until everything outstanding is collected |
| `gather_once()` | `std::vector<ResultT>` | Manager only, lock-free RMA distributors. One harvest pass, no retry loop |
| `run_worker()` | `void` | Non-manager ranks. Returns when shutdown is signalled |
| `remaining_tasks_count()` | `size_t` | Manager only. See the note below |
| `finalize()` | `void` | Signals shutdown. Called by the destructor if you do not |
| `get_statistics()` | `const Statistics&` | Manager only. Requires a `track_statistics` option |

!!! note "`remaining_tasks_count()` is not uniform"
    On the two-sided distributors it is the number of *unallocated* tasks still
    in the manager's queue, excluding tasks already dispatched.  On
    `LockFreeRMAWorkDistributor` it is published-minus-returned, and on
    `HierarchicalLockFreeRMAWorkDistributor` it is the number still
    outstanding in the tree.  Use it as a progress indicator, not as an exact
    cross-distributor quantity.

### Availability

| | Naive | Hierarchical | LockFreeRMA | Hier. LockFreeRMA |
|---|---|---|---|---|
| `insert_task(TaskT)` | ✓ | ✓ | ✓ | ✓ |
| `insert_task(task, priority)` | ✓ | ✓ ¹ | ✓ ² | — |
| `insert_tasks(vector)` | ✓ | ✓ | ✓ | ✓ |
| `insert_tasks(Range)` | — | ✓ | — | — |
| `run_tasks` / `finish_remaining_tasks` | ✓ | ✓ | ✓ | ✓ |
| `gather_once` | — | — | ✓ | ✓ |
| `run_worker` / `finalize` | ✓ | ✓ | ✓ | ✓ |
| `get_statistics` | ✓ | ✓ | ✓ | — |
| `num_workers()` | — | — | ✓ | — |
| `static ordered` | `true` | `false` | `true` | `true` |
| `static prioritize_tasks` | — | ✓ | — | — |

¹ Compiles, but the hierarchical batch path raises `DYNAMPI_UNIMPLEMENTED` as
soon as a manager has children — treat prioritization as naive-only.
² Accepted for interface compatibility; the priority is ignored.

## Options

Options are empty tag types passed through the `Options...` pack.  Unrecognised
options are ignored, so a generic wrapper can pass the same pack to every
distributor.

```cpp
using Prioritised = dynampi::NaiveWorkDistributor<
    int, double, dynampi::enable_prioritization>;

using Instrumented = dynampi::DynamicWorkDistributor<
    int, double, dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>;
```

| Option | Effect |
|--------|--------|
| `enable_prioritization` | Swaps the FIFO queue for a `std::priority_queue` and enables `insert_task(task, priority)` (descending priority). Functional on `NaiveWorkDistributor` only. |
| `track_statistics<Mode>` | `Mode` is `StatisticsMode::None` (default), `Aggregated`, or `Detailed`. Defaults to `Detailed` if written as `track_statistics<>`. |

### Choosing a statistics mode

| Mode | Message and byte counters | Per-rank task counts | `send_time` / `recv_time` |
|------|---------------------------|----------------------|---------------------------|
| `None` | — | — | — |
| `Aggregated` | ✓ | ✓ | — |
| `Detailed` | ✓ | ✓ | ✓ |

`Aggregated` only increments counters, with no clock reads in the hot path —
that is the mode to use while benchmarking.  `Detailed` additionally accumulates
the wall time spent inside the very calls that bump `send_count` / `recv_count`,
which costs two `MPI_Wtime()` calls per message.

## Statistics

`get_statistics()` is only declared when a `track_statistics` option requests a
mode other than `None`, and may only be called on the manager.  The returned
reference stays valid for the distributor's lifetime.

```cpp
struct CommStatistics {
  int    send_count, recv_count, collective_count, atomic_count;
  size_t bytes_sent, bytes_received, atomic_bytes;
  double send_time, recv_time;   // Detailed only; zero under Aggregated

  double average_send_size() const;
  double average_receive_size() const;
  void   reset();
};
```

`CommStatistics` counts only the **calling rank's own** traffic; it is not
reduced across the communicator.  For the RMA distributors, `put_bytes` /
`get_bytes` count one logical transfer each (even when chunked internally), and
`MPI_Fetch_and_op` is counted separately under `atomic_count` / `atomic_bytes`.

`send_time` / `recv_time` cover exactly the calls that increment
`send_count` / `recv_count` — the blocking sends and receives, the `MPI_Isend`
post, and `MPI_Put` / `MPI_Get`.  Time spent blocked in `MPI_Probe` waiting for
work to arrive, and time inside `MPI_Fetch_and_op`, are not included.

Each distributor defines its own nested `Statistics`:

```cpp
// NaiveWorkDistributor
struct Statistics {
  const CommStatistics& comm_statistics;
  std::vector<size_t>   worker_task_counts;  // indexed by worker index
};

// HierarchicalWorkDistributor
struct Statistics {
  const CommStatistics&                     comm_statistics;
  std::optional<std::vector<size_t>>        worker_task_counts;  // by world rank
};

// LockFreeRMAWorkDistributor
struct Statistics {
  const CommStatistics& comm_statistics;
};
```

Read the fields you need rather than assuming a shared layout:

```cpp
const auto& stats = dist.get_statistics();
std::cout << stats.comm_statistics.bytes_sent << "\n";
```

Support and caveats:

| Distributor | `Aggregated` | `Detailed` | Notes |
|-------------|--------------|------------|-------|
| `NaiveWorkDistributor` | ✓ | ✓ | `worker_task_counts` is indexed by worker index (this rank's communicator rank with the manager removed) |
| `HierarchicalWorkDistributor` | ✓ | ✓ | `worker_task_counts` is filled by an `MPI_Gather` inside `finalize()`, so it is `nullopt` before then, and holds locally-executed task counts by world rank |
| `LockFreeRMAWorkDistributor` | ✓ | ✓ | Communication counters only |
| `HierarchicalLockFreeRMAWorkDistributor` | — | — | Options accepted and ignored; no `get_statistics()` |

!!! warning "Statistics enable a collective in `finalize()`"
    On the hierarchical two-sided distributors, `finalize()` performs an
    `MPI_Gather` when statistics are enabled.  Every rank must therefore reach
    `finalize()` (directly or via its destructor).

## Task and result types

`TaskT` and `ResultT` must have a `dynampi::MPI_Type<T>` specialization.  Shipped
out of the box:

- all MPI-mappable scalars (`char`, `short`, `int`, `long`, `long long` and
  unsigned variants, `float`, `double`, `long double`, `bool`, `std::byte`);
- `std::vector<T>` for any supported scalar `T` (**not** `std::vector<bool>`,
  which is bit-packed — a `static_assert` catches it);
- `std::string`.

`MPI_Type<T>::resize_required` is `false` for fixed-size types and `true` for the
variable-length ones.  That flag decides where a type may be used:

| Payload | Naive | Hierarchical | LockFreeRMA / Hier. LockFreeRMA |
|---------|-------|---------------|----------------------------------|
| Fixed-size scalar | ✓ | ✓ | ✓ |
| Fixed-size struct | ✓ | ✓ | ✓ |
| Variable-length | ✓ | ✓ | ✓, up to `max_task_count` / `max_result_count` elements |

### Custom types

Specialise `MPI_Type` for your own type.  DynaMPI moves a value as `count()`
elements of `value`, read and written through `ptr()`, so the storage must be
contiguous.

```cpp
struct Point { double x, y, z; };
static_assert(sizeof(Point) == 3 * sizeof(double));

template <>
struct dynampi::MPI_Type<Point> {
  inline static const MPI_Datatype value = MPI_DOUBLE;
  inline static const bool resize_required = false;  // fixed size

  static int   count(const Point&) noexcept { return 3; }
  static void  resize(Point&, int) noexcept {}       // fixed size: nothing to do
  static void* ptr(Point& p) noexcept { return &p; }
  static const void* ptr(const Point& p) noexcept { return &p; }
};
```

That works with **every** distributor.  A value may span several datatype
elements — three `MPI_DOUBLE`s here — and everything that sizes a buffer per
value accounts for it: the fixed-width RMA window slots, and the element count
of a batched `std::vector<T>` message.

Genuinely variable-length types work the same way with `resize_required = true`,
`count()` returning the current element count, and `resize(value, n)` growing the
storage to `n` elements — exactly how the shipped `std::vector<T>` and
`std::string` specializations are written.

Two requirements to be aware of:

- **The whole object must be sent.** `count() * MPI_Type_size(value)` has to
  equal `sizeof(T)` for a fixed-size type, and the storage at `ptr()` has to be
  contiguous.
- **The object must be a whole number of elements.** A 12-byte struct described
  by an 8-byte datatype cannot be expressed as *N* elements; distributors detect
  this in their constructor and throw `std::invalid_argument` naming the type,
  in release builds too.  Describe such a type with a datatype that fits — three
  `MPI_FLOAT`s rather than one and a half `MPI_DOUBLE`s — or with `MPI_BYTE` and
  `count() == sizeof(T)`.

[examples/04_custom_task_type.cpp](https://github.com/ryanstocks00/DynaMPI/blob/main/examples/04_custom_task_type.cpp)
is a complete working version of this.

## Version

```cpp
#include <dynampi/dynampi.hpp>

namespace dv = dynampi::version;

dv::string;                  // "v0.0.1"  (std::string_view)
dv::major; dv::minor; dv::patch;   // 0, 0, 1
dv::is_at_least(0, 0, 1);    // constexpr bool
dv::compile_date();          // __DATE__ " " __TIME__ of the translation unit
dv::commit_hash();           // "abc1234" or "abc1234-dirty"
```

`major`/`minor`/`patch` and `commit_hash()` come from the macros the CMake target
defines.

## Utilities

### `Timer`

```cpp
#include <dynampi/utilities/timer.hpp>

dynampi::Timer timer;                              // starts immediately
dynampi::Timer paused{dynampi::Timer::AutoStart::No};

timer.elapsed();   // std::chrono::duration<double>, valid while running
timer.stop();      // pause, returns total elapsed
timer.start();     // resume (accumulates)
timer.reset();     // back to zero, restarts unless AutoStart::No

std::cout << timer;  // "12.345 seconds"
```

Backed by `std::chrono::high_resolution_clock`.  `start()` on an already-running
timer and `stop()` on a stopped one trip `assert`s.

### Assertions

```cpp
#include <dynampi/utilities/assert.hpp>

DYNAMPI_ASSERT(cond);                    // optional trailing message args
DYNAMPI_ASSERT(cond, "context: ", value);
DYNAMPI_ASSERT_EQ(a, b);                 // also NE, LT, LE, GT, GE
DYNAMPI_ASSERT_EQ(a, b, "while merging");
DYNAMPI_FAIL("unreachable state");
DYNAMPI_UNIMPLEMENTED();
```

On failure these print `DynaMPI assertion failed on rank N: …` with the failing
expression, both operand values for the binary forms, and (where
`std::source_location` is available) function, file and line — then **throw**
`std::runtime_error`.  They are compiled out entirely under `NDEBUG`, at which
point `DYNAMPI_FAIL` / `DYNAMPI_UNIMPLEMENTED` degrade to
`__builtin_unreachable()`, so do not rely on them for release-build control flow.

Message arguments are streamed with `operator<<`, so anything printable works —
including the container printers below.

### MPI error checking

```cpp
#include <dynampi/mpi/mpi_error.hpp>

DYNAMPI_MPI_CHECK(MPI_Send, (buf, count, type, dest, tag, comm));
```

Calls the function and, on any result other than `MPI_SUCCESS`, throws
`std::runtime_error` containing the MPI error string, the call site expression,
and file:line.  Note the argument list is a single parenthesised group.

### Container printing

```cpp
#include <dynampi/utilities/printing.hpp>

using namespace dynampi;  // required: the operators live in namespace dynampi

std::cout << std::vector<int>{1, 2, 3};        // [1, 2, 3]
std::cout << std::optional<int>{42};           // Some(42)
std::cout << std::optional<int>{};             // None
std::cout << std::pair{1, 'a'};                // (1, a)
```

Overloads exist for `std::vector`, `std::array`, `std::span`, `std::set`,
`std::pair`, `std::tuple`, `std::optional` and `std::byte`.

Because these operate on standard-library types, argument-dependent lookup will
not find them — bring them in with `using namespace dynampi;` (or a targeted
`using dynampi::operator<<;`) in the scope where you stream.  Assertion message
arguments are an exception: `DYNAMPI_ASSERT(cond, some_vector)` streams from
inside `namespace dynampi`, so container printing works there with no
using-declaration.

### MPI wrappers

`dynampi::MPICommunicator<Options...>` and `dynampi::MPIGroup` are the RAII
wrappers the distributors are built on.  `MPICommunicator` owns (or references) a
communicator, offers typed `send` / `isend` / `recv` / `probe` / `broadcast` /
`gather` and RMA helpers (`put_bytes`, `get_bytes`, `fetch_and_op`), records
`CommStatistics` when a `track_statistics` option is supplied, and provides
`split()` / `split_by_node()`.  `MPIGroup` wraps `MPI_Group` for rank
translation.  They are public and usable directly, but the distributors are the
supported surface.
