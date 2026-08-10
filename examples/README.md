<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# DynaMPI examples

Standalone MPI programs, each focused on one thing. They depend on nothing
beyond MPI and the DynaMPI headers.

| Example | Shows |
|---------|-------|
| [01_index_tasks.cpp](01_index_tasks.cpp) | `mpi_manager_worker_distribution` — the one-call entry point |
| [02_incremental_tasks.cpp](02_incremental_tasks.cpp) | Inserting tasks as you discover them, chunked collection with `RunConfig`, driving `run_worker()` yourself |
| [03_ordered_and_prioritized.cpp](03_ordered_and_prioritized.cpp) | Task-ordered results and task priorities, both `NaiveWorkDistributor` |
| [04_custom_task_type.cpp](04_custom_task_type.cpp) | Specialising `dynampi::MPI_Type` for your own struct, and tagging results with their task index |
| [05_lockfree_rma.cpp](05_lockfree_rma.cpp) | The one-sided RMA distributors, window capacity, and `gather_once()` |
| [06_statistics.cpp](06_statistics.cpp) | `track_statistics`, and what `Aggregated` gives you that `Detailed` does not |
| [07_parallel_for.cpp](07_parallel_for.cpp) | `MinimalLockFreeWorkDistributor` for an index-only parallel-for |

## Building

With CMake, as part of the repository:

```bash
cmake -B build -DDYNAMPI_BUILD_EXAMPLES=ON
cmake --build build -j
mpirun -n 4 ./build/examples/01_index_tasks
```

Or directly, since DynaMPI is header-only and needs no CMake:

```bash
mpic++ -std=c++20 -I../include 01_index_tasks.cpp -o 01_index_tasks
mpirun -n 4 ./01_index_tasks
```

Every example runs correctly at any rank count, including `-n 1` (each
distributor falls back to executing tasks inline on the manager).

Concepts are explained in full in the
[documentation](https://ryanstocks00.github.io/DynaMPI/).
