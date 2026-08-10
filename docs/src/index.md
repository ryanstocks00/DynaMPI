<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# DynaMPI

**Header-only C++20 library for manager–worker dynamic load distribution over MPI.**

DynaMPI hands out a stream of tasks to MPI ranks at runtime instead of
partitioning them up front.  One rank acts as the *manager* and owns the task
queue; the remaining ranks act as *workers* and pull work as they finish what
they already have.  This keeps every rank busy when task costs are irregular or
unknown in advance.

Six distributors implement that contract with different communication
strategies — from a 400-line two-sided loop to a tree of one-sided RMA windows
that sustains millions of task hand-offs per second.  They share one interface,
so switching between them is a single type change.

## Quick start

### Fixed set of index tasks

```cpp
#include <cmath>
#include <iostream>

#include <dynampi/dynampi.hpp>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    auto work = [](size_t task) -> double {
      return std::sqrt(static_cast<double>(task));
    };

    // Collective: every rank calls it. Workers loop internally and return
    // std::nullopt; the manager returns the results.
    auto results = dynampi::mpi_manager_worker_distribution<double>(100, work);

    if (results.has_value()) {
      for (double r : *results) std::cout << r << "\n";
    }
  }
  MPI_Finalize();
}
```

!!! warning "Results are unordered by default"
    The default distributor is hierarchical, and hierarchical results come back
    in completion order — `(*results)[i]` is **not** the result of task `i`.
    Either carry the index inside your result type, or use
    `NaiveWorkDistributor` (see
    [Result ordering](implementations.md#result-ordering)).

### Tasks discovered as you go

When the task set is not known up front, construct a distributor directly and
alternate insertion with collection:

```cpp
#include <dynampi/dynampi.hpp>

using Task = int;
using Result = double;

dynampi::DynamicWorkDistributor<Task, Result> dist(
    [](Task t) -> Result { return std::sqrt(static_cast<double>(t)); });

if (dist.is_root_manager()) {
  dist.insert_tasks({1, 2, 3, 4, 5});
  auto first = dist.finish_remaining_tasks();   // 5 results

  dist.insert_tasks({6, 7, 8});
  auto second = dist.finish_remaining_tasks();  // 3 results
}
// Non-manager ranks entered the worker loop inside the constructor
// (auto_run_workers defaults to true) and return here once the
// manager's distributor is destroyed.
```

## Mental model

| Concept | Meaning |
|---------|---------|
| **Manager** | Rank `Config::manager_rank` (default `0`). Owns the queue, collects results, and is the only rank allowed to call `insert_task*` / `run_tasks` / `finish_remaining_tasks`. |
| **Worker** | Every other rank. Runs `run_worker()` — automatically from the constructor unless you set `auto_run_workers = false`. |
| **Coordinator** | In the hierarchical distributors, an intermediate rank (one per node by default) that relays batches of tasks down and results up, so the manager never talks to every rank directly. |
| **Task** | A `TaskT` value. Must be sendable — see [task and result types](api.md#task-and-result-types). |
| **Result** | Whatever the worker function returns. Collected on the manager. |

Construction is collective over `Config::comm`, and so is destruction: every
rank must construct and destroy the distributor.

## Choosing a distributor

| Situation | Use |
|-----------|-----|
| Small communicator, need results in task order or task priorities | [`NaiveWorkDistributor`](implementations.md#naiveworkdistributor) |
| General purpose, multi-node, fixed-size task/result types | [`DynamicWorkDistributor`](implementations.md#hierarchicalworkdistributor-dynamicworkdistributor) (default) |
| Fine-grained tasks, want the highest hand-off rate at moderate scale | [`LockFreeRMAWorkDistributor`](implementations.md#lockfreermaworkdistributor) |
| Same, but at large node counts where one manager window saturates | [`HierarchicalLockFreeRMAWorkDistributor`](implementations.md#hierarchicallockfreermaworkdistributor) |
| A plain parallel-for over `0 .. n-1` | [`MinimalLockFreeWorkDistributor`](implementations.md#minimallockfreeworkdistributor) |

The full comparison, including protocol descriptions and the constraints each
distributor imposes, is in [Implementations](implementations.md).

## Where to go next

- **[Examples](https://github.com/ryanstocks00/DynaMPI/tree/main/examples)** —
  seven short runnable programs, one per concept.
- **[Installation](install.md)** — requirements, CMake, and the header-only path.
- **[API Reference](api.md)** — every public type, method, config field and option.
- **[Implementations](implementations.md)** — how each distributor works and when to pick it.
- **[Lock-free RMA design](lockfree_rma_design.md)** — window layouts and
  synchronisation rules behind the RMA distributors.

Licensed under the Apache License 2.0.
