<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# DynaMPI

**Header-only C++20 library for dynamic MPI task distribution.**

DynaMPI distributes tasks across MPI ranks at runtime.  A single *manager*
rank holds a queue of tasks; *worker* ranks pull work as they become
available.  The manager collects results in task-submission order.

## Quick Start

```cpp
#include <dynampi/dynampi.hpp>

// Worker function: TaskT → ResultT
auto work = [](int task) -> double { return std::sqrt(static_cast<double>(task)); };

// Single function call — manager gets results, workers return nullopt
auto results = dynampi::mpi_manager_worker_distribution<double>(
    100,    // number of tasks
    work    // worker function
);

if (results.has_value()) {
    for (double r : *results) {
        std::cout << r << "\n";
    }
}
```

## API Reference

### `dynampi::mpi_manager_worker_distribution`

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

Distributes `n_tasks` tasks (indices `0..n_tasks-1`) across the MPI
communicator.  Returns the results on the manager rank, `std::nullopt` on
workers.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_tasks` | — | Number of tasks to distribute |
| `worker_function` | — | `size_t → ResultT` — called with the task index |
| `comm` | `MPI_COMM_WORLD` | MPI communicator |
| `manager_rank` | `0` | Rank that manages distribution and collects results |

**Template parameters:**
- `ResultT` — the type each task produces
- `Distributor` — which distributor implementation to use (default: `HierarchicalMPIWorkDistributor`)

### `dynampi::MPIDynamicWorkDistributor`

Type alias for the default distributor:

```cpp
template <typename TaskT, typename ResultT, typename... Options>
using MPIDynamicWorkDistributor = HierarchicalMPIWorkDistributor<TaskT, ResultT, Options...>;
```

### Direct Distributor Usage

For fine-grained control (incremental task insertion, batch result
collection, statistics):

```cpp
#include <dynampi/impl/naive_distributor.hpp>

using Distributor = dynampi::NaiveMPIWorkDistributor<int, double>;

auto work = [](int task) -> double { return std::sqrt(task); };
Distributor::Config cfg;
cfg.comm = MPI_COMM_WORLD;
cfg.auto_run_workers = false;

Distributor dist(work, cfg);

if (dist.is_root_manager()) {
    dist.insert_tasks({1, 2, 3, 4, 5});

    // Collect first 3 results, leave remaining tasks in-flight
    auto batch = dist.run_tasks({
        .target_num_tasks = 3,
        .allow_more_than_target_tasks = false
    });

    // Collect remaining results
    auto rest = dist.finish_remaining_tasks();
} else {
    dist.run_worker();
}
```

#### Common Config Fields

All distributors share these configuration fields:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `comm` | `MPI_Comm` | `MPI_COMM_WORLD` | MPI communicator |
| `manager_rank` | `int` | `0` | Manager rank |
| `auto_run_workers` | `bool` | `true` | If true, workers start automatically in the constructor |

#### Common RunConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `target_num_tasks` | `size_t` | `max` | Stop once this many results are ready |
| `allow_more_than_target_tasks` | `bool` | `true` | If false, clip return to `target_num_tasks` |
| `max_seconds` | `optional<double>` | `nullopt` | Stop if this many seconds have elapsed |

#### Common Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `is_root_manager()` | `bool` | Whether this rank is the manager |
| `insert_task(TaskT)` | `void` | Add one task to the queue (manager only) |
| `insert_tasks(vector)` | `void` | Add multiple tasks (manager only) |
| `run_tasks(RunConfig)` | `vector<ResultT>` | Process tasks, return completed results (manager) |
| `finish_remaining_tasks()` | `vector<ResultT>` | Process all remaining tasks (manager) |
| `run_worker()` | `void` | Enter worker loop (non-manager ranks) |
| `remaining_tasks_count()` | `size_t` | Tasks still in the queue (manager) |
| `finalize()` | `void` | Signal shutdown to workers |
| `get_statistics()` | `Statistics` | Communication statistics (required `track_statistics` option) |

### Options

Template options customise distributor behaviour:

```cpp
// Enable task prioritisation
using Dist = NaiveMPIWorkDistributor<int, double, dynampi::enable_prioritization>;

// Enable statistics tracking
using Dist = NaiveMPIWorkDistributor<int, double,
    dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>;

// Combine options
using Dist = HierarchicalMPIWorkDistributor<int, double,
    dynampi::enable_prioritization,
    dynampi::track_statistics<dynampi::StatisticsMode::Aggregated>>;
```

| Option | Values | Description |
|--------|--------|-------------|
| `prioritize_tasks_t` | `enable_prioritization` | Tasks submitted with `insert_task(task, priority)` are processed in priority order |
| `track_statistics_t` | `track_statistics<None>`, `<Aggregated>`, `<Detailed>` | Track communication volume and per-rank task counts |

### Version Info

```cpp
#include <dynampi/dynampi.hpp>

dynampi::version::string;           // "v0.0.1"
dynampi::version::major;            // 0
dynampi::version::is_at_least(0,0,1); // true
dynampi::version::compile_date();   // "Jun  6 2026 12:34:56"
dynampi::version::commit_hash();    // "abc1234" or "abc1234-dirty"
```
