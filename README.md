<!--
  SPDX-FileCopyrightText: 2025 Ryan Stocks and QDX Technologies
  SPDX-License-Identifier: Apache-2.0
 -->

# DynaMPI

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![codecov](https://codecov.io/github/ryanstocks00/dynampi/graph/badge.svg?token=S65RFJ3FCX)](https://codecov.io/github/ryanstocks00/dynampi)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/7bb14fa81aeb4bd5b59ca62cc3a80975)](https://app.codacy.com/gh/ryanstocks00/DynaMPI/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.14%2B-green.svg)](https://cmake.org/)
![Repo Size](https://img.shields.io/github/repo-size/ryanstocks00/DynaMPI)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Stability - Alpha](https://img.shields.io/badge/stability-alpha-f4d03f.svg)](https://github.com/mkenney/software-guides/blob/master/STABILITY-BADGES.md#alpha)
[![REUSE status](https://api.reuse.software/badge/github.com/ryanstocks00/dynampi)](https://api.reuse.software/info/github.com/ryanstocks00/dynampi)

## CI Status

[![Linux GCC](https://github.com/ryanstocks00/DynaMPI/workflows/Linux%20GCC/badge.svg)](https://github.com/ryanstocks00/DynaMPI/actions/workflows/linux-gcc.yml)
[![Linux Clang](https://github.com/ryanstocks00/DynaMPI/workflows/Linux%20Clang/badge.svg)](https://github.com/ryanstocks00/DynaMPI/actions/workflows/linux-clang.yml)
[![Linux Intel](https://github.com/ryanstocks00/DynaMPI/workflows/Linux%20Intel/badge.svg)](https://github.com/ryanstocks00/DynaMPI/actions/workflows/linux-intel.yml)
[![Windows](https://github.com/ryanstocks00/DynaMPI/workflows/Windows/badge.svg)](https://github.com/ryanstocks00/DynaMPI/actions/workflows/windows.yml)
[![macOS](https://github.com/ryanstocks00/DynaMPI/workflows/macOS/badge.svg)](https://github.com/ryanstocks00/DynaMPI/actions/workflows/macos.yml)
[![SMPI](https://github.com/ryanstocks00/DynaMPI/workflows/SMPI/badge.svg)](https://github.com/ryanstocks00/DynaMPI/actions/workflows/smpi.yml)
[![Sanitizers](https://github.com/ryanstocks00/DynaMPI/workflows/Sanitizers/badge.svg)](https://github.com/ryanstocks00/DynaMPI/actions/workflows/sanitizers.yml)
[![Pre-commit](https://github.com/ryanstocks00/DynaMPI/workflows/Pre-commit/badge.svg)](https://github.com/ryanstocks00/DynaMPI/actions/workflows/pre-commit.yml)

---

Header-only C++20 library for efficient manager–worker dynamic load distribution over MPI.

One rank manages a task queue; the rest pull work as they finish, keeping every
rank busy when task costs are irregular or unknown up front. Six distributors
implement that contract behind one interface, from a plain two-sided loop to a
tree of one-sided RMA windows.

Licensed under the Apache License 2.0.

## Usage

### Static number of tasks

```cpp
#include <cassert>
#include <dynampi/dynampi.hpp>

auto worker_task = [](size_t task) -> size_t { return task * task; };
auto result = dynampi::mpi_manager_worker_distribution<size_t>(4, worker_task);
if (result.has_value()) {
  // Manager: one result per task. The default distributor is hierarchical and
  // unordered — result[i] is *not* task i. Pass NaiveWorkDistributor as the
  // second template argument if you need task-index order.
  assert(result->size() == 4);
}
```

### Dynamic / incremental tasks

When the task set is not known up front, construct a distributor directly and
alternate `insert_tasks` with `run_tasks` / `finish_remaining_tasks`:

```cpp
#include <cmath>
#include <dynampi/dynampi.hpp>

using Task = int;
using Result = double;

auto worker_task = [](Task task) -> Result {
  return std::sqrt(static_cast<double>(task));
};

dynampi::DynamicWorkDistributor<Task, Result> distributor(worker_task);
if (distributor.is_root_manager()) {
  distributor.insert_tasks({1, 2, 3, 4, 5});
  auto results = distributor.finish_remaining_tasks();
  // results.size() == 5

  distributor.insert_tasks({6, 7, 8});
  results = distributor.finish_remaining_tasks();
  // results.size() == 3
}
```

With `auto_run_workers = true` (the default), non-manager ranks enter the worker
loop in the constructor. Construction and destruction are collective.

### Distributors

| Class | Communication | Ordered | Notes |
|-------|---------------|---------|-------|
| `NaiveWorkDistributor` | Two-sided | Yes | Task priorities, variable-length payloads; manager-bound at scale |
| `DynamicWorkDistributor` (hierarchical) | Two-sided, batched | No | Default. Node-aware tree; fixed-size payloads only |
| `HierarchicalNonBlockingWorkDistributor` | Two-sided, batched, `Isend` | No | A/B variant of the above |
| `LockFreeRMAWorkDistributor` | Passive-target RMA | No | No collectives on the hot path; preallocated window |
| `HierarchicalLockFreeRMAWorkDistributor` | Passive-target RMA, per level | No | RMA protocol applied per tree level, for large node counts |
| `MinimalLockFreeWorkDistributor` | RMA counter + `Gatherv` | Yes | Parallel-for over `0 .. n-1` |

A task that throws is reported to the manager rather than aborting the job:
`run_tasks` rethrows it as `dynampi::TaskFailure`, or set
`Config::rethrow_task_errors = false` and collect failures from
`take_task_errors()`.

Optional compile-time features: task prioritization
(`dynampi::enable_prioritization`), statistics tracking
(`dynampi::track_statistics<Mode>`), and custom payloads via
`dynampi::MPI_Type`.

See the [documentation](https://ryanstocks00.github.io/DynaMPI/) for how each
distributor works, its constraints, and its configuration, and
[`examples/`](examples/) for short runnable programs covering each concept.

## Installation

DynaMPI depends only on MPI. Copy `include/` into your project — no macros to
define, nothing to link — or consume it with CMake:

```cmake
include(FetchContent)
FetchContent_Declare(
    dynampi
    GIT_REPOSITORY https://github.com/ryanstocks00/DynaMPI.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(dynampi)
target_link_libraries(my_target PRIVATE dynampi)
```
