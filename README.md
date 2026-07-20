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

Licensed under the Apache License 2.0.

## Usage

### Static number of tasks

```cpp
#include <dynampi/dynampi.hpp>

auto worker_task = [](size_t task) -> size_t { return task * task; };
auto result = dynampi::mpi_manager_worker_distribution<size_t>(4, worker_task);
if (result.has_value()) {
  // Manager: results for tasks 0..3. Default distributor is hierarchical
  // (completion order is not guaranteed — sort if you need task-index order).
  assert(result->size() == 4);
}
```

The default distributor is `MPIDynamicWorkDistributor` (hierarchical). Pass another
distributor as a template argument if needed (e.g. `NaiveMPIWorkDistributor` for
strictly ordered results on small communicator sizes).

### Dynamic / incremental tasks

When the task set is not known up front, use `MPIDynamicWorkDistributor`
and alternate `insert_tasks` with `run_tasks` / `finish_remaining_tasks`:

```cpp
#include <dynampi/dynampi.hpp>

using Task = int;
using Result = std::vector<int>;

auto worker_task = [](Task task) -> Result {
  return Result{task, task * task, task * task * task};
};

dynampi::MPIDynamicWorkDistributor<Task, Result> distributor(worker_task);
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
loop in the constructor. Optional compile-time features include task
prioritization (naive distributor), statistics tracking, and custom MPI
datatypes via `dynampi::MPI_Type`.

See the [documentation](https://trailblaze-software.github.io/DynaMPI/) for
distributor choice (naive, hierarchical, lock-free RMA) and configuration.

## Installation

DynaMPI depends only on MPI. Copy `include/` into your project, or consume it
with CMake:

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
