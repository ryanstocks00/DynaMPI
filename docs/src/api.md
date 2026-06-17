<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# API Reference

## `dynampi::mpi_manager_worker_distribution`

```cpp
template <typename ResultT,
          template <typename, typename, typename...> typename Distributor =
              HierarchicalMPIWorkDistributor>
std::optional<std::vector<ResultT>> mpi_manager_worker_distribution(
    size_t n_tasks,
    std::function<ResultT(size_t)> worker_function,
    MPI_Comm comm = MPI_COMM_WORLD,
    int manager_rank = 0);
```

The primary entry point.  Distributes `n_tasks` tasks (indices `0..n_tasks-1`)
across the MPI communicator.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_tasks` | — | Number of tasks (0, 1, 2, ...) |
| `worker_function` | — | `size_t → ResultT` |
| `comm` | `MPI_COMM_WORLD` | MPI communicator |
| `manager_rank` | `0` | Manager rank |
| **Returns** | | `vector<ResultT>` on manager, `nullopt` on workers |

## `dynampi::MPIDynamicWorkDistributor`

Type alias for the default distributor:

```cpp
template <typename TaskT, typename ResultT, typename... Options>
using MPIDynamicWorkDistributor =
    HierarchicalMPIWorkDistributor<TaskT, ResultT, Options...>;
```

## Distributor Classes

All distributors share a common interface.  The template signature is:

```cpp
template <typename TaskT, typename ResultT, typename... Options>
class Distributor;
```

### Common `Config` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `comm` | `MPI_Comm` | `MPI_COMM_WORLD` | MPI communicator |
| `manager_rank` | `int` | `0` | Rank that manages distribution |
| `auto_run_workers` | `bool` | `true` | Start workers automatically in constructor |

### Common `RunConfig` Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_num_tasks` | `size_t` | `SIZE_MAX` | Stop once this many results are ready |
| `allow_more_than_target_tasks` | `bool` | `true` | If false, clip return to `target_num_tasks` |
| `max_seconds` | `optional<double>` | `nullopt` | Time limit for this call |

### Common Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_root_manager()` | `bool` | Whether this rank is the manager |
| `insert_task(TaskT)` | `void` | Add one task (manager only) |
| `insert_tasks(vector)` | `void` | Add multiple tasks (manager only) |
| `run_tasks(RunConfig)` | `vector<ResultT>` | Process tasks, collect results (manager) |
| `finish_remaining_tasks()` | `vector<ResultT>` | Process all remaining tasks (manager) |
| `run_worker()` | `void` | Enter worker loop (non-manager) |
| `remaining_tasks_count()` | `size_t` | Tasks still in the queue (manager) |
| `finalize()` | `void` | Signal shutdown to all workers |
| `get_statistics()` | `Statistics` | Communication statistics |

!!! note "Distributor-Specific Config"
    See [Implementations](implementations.md) for additional configuration
    fields on each distributor class.

## Options

Template options customise distributor behaviour via the `Options...`
parameter pack:

```cpp
// Task prioritisation
using Dist = NaiveMPIWorkDistributor<int, double,
    dynampi::enable_prioritization>;
// Tasks inserted with insert_task(task, priority) are processed in
// descending priority order.

// Statistics tracking
using Dist = NaiveMPIWorkDistributor<int, double,
    dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>;
// Tracks bytes sent/received, message counts, per-rank task counts.
```

| Option | Effect |
|--------|--------|
| `enable_prioritization` | Enables `insert_task(task, priority)` |
| `track_statistics<None>` | No statistics (default) |
| `track_statistics<Aggregated>` | Per-rank task counts |
| `track_statistics<Detailed>` | Full communication statistics |

## Version

```cpp
#include <dynampi/dynampi.hpp>

namespace dv = dynampi::version;

dv::string;                    // "v0.0.1"
dv::major;  dv::minor;  dv::patch;  // 0, 0, 1
dv::is_at_least(0, 0, 1);    // true
dv::compile_date();           // "Jun  6 2026 12:34:56"
dv::commit_hash();            // "abc1234" or "abc1234-dirty"
```

## Statistics

```cpp
struct CommStatistics {
    int send_count, recv_count, collective_count;
    size_t bytes_sent, bytes_received;
    double send_time, recv_time;
    double average_send_size() const;
    double average_receive_size() const;
};

struct RMAStatistics {          // OneSided distributor only
    int put_count, get_count;
    size_t bytes_put, bytes_get;
};

struct Statistics {
    CommStatistics comm_statistics;
    RMAStatistics rma_statistics;         // zero for non-RMA distributors
    std::vector<size_t> worker_task_counts;  // per-rank task counts
};
```

## Utilities

### Timer

```cpp
#include <dynampi/utilities/timer.hpp>

dynampi::Timer timer;                       // starts automatically
timer.elapsed();                             // std::chrono::duration<double>
timer.stop();                                // pause
timer.reset();                               // restart from zero
std::cout << timer;                          // "12.345 seconds"
```

### Assertions

```cpp
#include <dynampi/utilities/assert.hpp>

// These are no-ops in release builds (-DNDEBUG):
DYNAMPI_ASSERT(condition);                   // abort with file:line on failure
DYNAMPI_ASSERT_EQ(a, b);                    // assert a == b, print both values
DYNAMPI_ASSERT_NE(a, b);                    // assert a != b
DYNAMPI_ASSERT_GT(a, b);                    // assert a > b
DYNAMPI_ASSERT_GE(a, b);                    // assert a >= b
DYNAMPI_ASSERT_LT(a, b);                    // assert a < b
DYNAMPI_ASSERT_LE(a, b);                    // assert a <= b
DYNAMPI_UNIMPLEMENTED();                    // always fails — marks unimplemented paths
```

### MPI Error Checking

```cpp
// Wraps MPI calls and throws std::runtime_error with file:line on failure:
DYNAMPI_MPI_CHECK(MPI_Send, (buf, count, type, dest, tag, comm));
```

### Printing

```cpp
#include <dynampi/utilities/printing.hpp>

// std::ostream operators for STL containers:
std::vector<int> v{1, 2, 3};
std::cout << v;  // "[1, 2, 3]"

std::optional<int> o = 42;
std::cout << o;  // "Some(42)"
```
