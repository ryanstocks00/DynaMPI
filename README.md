<!--
  SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
  SPDX-License-Identifier: MIT
 -->

# DynaMPI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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

Library for efficient manager-worker dynamic load distribution using MPI

This project is licensed under the terms of the MIT license.

## Usage

For dynamic load distribution of a static number of tasks, we provide the simple API

```
template <typename ResultT>
std::optional<std::vector<ResultT>> mpi_manager_worker_distribution(
    size_t n_tasks, std::function<ResultT(size_t)> worker_function, MPI_Comm comm = MPI_COMM_WORLD,
    int manager_rank = 0);
```

Allowing 
```cpp
#include <dynampi/dynampi.hpp>

auto worker_task = [](size_t task) -> size_t { return task * task; };
auto result = dynampi::mpi_manager_worker_distribution<size_t>(4, worker_task);
if (result.has_value())
    assert(result == std::vector<size_t>({0, 1, 4, 9}));
```

The order of the result is guaranteed to be in order of the task indexes.

It is common for the number of tasks to not be static. It can also be inefficient to form all tasks prior to 

```
template <typename TaskT, typename ResultT>
class MPIDynampicWorkDistributor {

```

Allowing
```
typedef int Task;
typedef std::vector<double> Result;
auto worker_task = [](Task task) -> Result { return task * task; };
{
    dynampi::MPIDynamicWorkDistributor work_distributer(worker_task);
    if (work_distributer.is_manager()) {
        work_distributer.insert_tasks({1, 2, 3, 4, 5});
        work_distributer.insert_tasks({6, 7, 8});
    }
}
```

This allows the manager to begin distributing tasks before all of the tasks have been formed. The manager can also alternate between inserting tasks and receiving results for when task formation is dependent on the results of previous tasks. There are many configuration additional options including work prioritization, custom datatypes, and error handling.

## Installation

DynaMPI is a header-only library with dependence only on MPI, so you can simply copy the include folder into your project. Alternatively, DynaMPI can be installed using CMake:

```cmake
include(FetchContent)
FetchContent_Declare(
    dynampi
    GIT_REPOSITORY https://github.com/ryanstocks00/DynaMPI.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(dynampi)
add_subdirectory(${dynampi_SOURCE_DIR} ${dynampi_BINARY_DIR})
```
