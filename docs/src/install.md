<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# Installation

DynaMPI is **header-only**.  Its only dependency is MPI — there is nothing to
build or link beyond your MPI implementation.

## Requirements

| Dependency | Minimum |
|------------|---------|
| C++ compiler | C++20: GCC 11+, Clang 14+, MSVC 2022+, Intel oneAPI (IntelLLVM) |
| MPI | MPICH 3.4+, Open MPI 4.0+, Intel MPI 2021+, MS-MPI 10+ |
| CMake | 3.14+ (only to consume the package or build tests/benchmarks) |

CI covers Linux (GCC, Clang, Intel), macOS, Windows (MS-MPI), SimGrid/SMPI, and
sanitizer builds.

!!! note "RMA distributors and MPI progress"
    The lock-free distributors use passive-target RMA
    (`MPI_Win_lock_all` + `MPI_Fetch_and_op` / `MPI_Put` / `MPI_Get`).  Some MPI
    builds make remote RMA progress only when the two-sided engine is driven;
    DynaMPI handles this internally, but validate your target stack at scale
    before relying on them for production runs.

## CMake

```cmake
# As a subdirectory
add_subdirectory(path/to/DynaMPI)
target_link_libraries(my_target PRIVATE dynampi)
```

```cmake
# Or via FetchContent
include(FetchContent)
FetchContent_Declare(
    DynaMPI
    GIT_REPOSITORY https://github.com/ryanstocks00/DynaMPI.git
    GIT_TAG main            # pin a tag or commit for reproducible builds
)
FetchContent_MakeAvailable(DynaMPI)
target_link_libraries(my_target PRIVATE dynampi)
```

The `dynampi` INTERFACE target adds the include directory, links `MPI::MPI_CXX`,
requires C++20 (`cxx_std_20`), and defines `DYNAMPI_COMMIT_HASH` plus the
`DYNAMPI_VERSION_*` macros.

When DynaMPI is consumed as a subproject, `DYNAMPI_BUILD_TESTS`,
`DYNAMPI_BUILD_BENCHMARKS` and `DYNAMPI_BUILD_EXAMPLES` all default to `OFF`, so
it costs you no extra build time.

## Without CMake

Copy `include/` into your project and add it to the include path.  No macros to
define, nothing to link beyond MPI:

```bash
mpic++ -std=c++20 -I/path/to/DynaMPI/include my_program.cpp
```

The only thing you lose is `dynampi::version::commit_hash()`, which reports
`"unknown"` outside a CMake build since nothing else can determine it.

## Building the examples

```bash
cmake -B build -DDYNAMPI_BUILD_EXAMPLES=ON
cmake --build build -j
mpirun -n 4 ./build/examples/01_index_tasks
```

Seven short standalone programs covering the entry point, incremental task
insertion, ordering and priorities, custom payload types, the RMA distributors,
statistics, and the index parallel-for.  See
[examples/README.md](https://github.com/ryanstocks00/DynaMPI/tree/main/examples)
for what each one demonstrates.  They all run correctly at any rank count,
including `-n 1`.

## Building the tests

```bash
git clone https://github.com/ryanstocks00/DynaMPI.git
cd DynaMPI
cmake -B build -DDYNAMPI_BUILD_TESTS=ON
cmake --build build -j
```

```bash
./build/test/unit_test              # serial unit tests
mpirun -n 4 ./build/test/mpi_test   # MPI distributor tests
ctest --test-dir build              # everything, as registered
```

`ctest` registers the MPI suite at 1, 2, 3, 4, 8, 16 and 64 ranks, since several
topology paths only appear at particular rank counts.  Cap that on a small
machine with `-DDYNAMPI_MAX_MPI_RANK=8` (or the `DYNAMPI_MAX_MPI_RANK`
environment variable).

## Building the benchmarks

```bash
cmake -B build -DDYNAMPI_BUILD_BENCHMARKS=ON
cmake --build build -j

mpirun -n 64 ./build/benchmark/weak_scaling_distribution_rate \
  -D hierarchical -t 100 -d 10
```

Benchmarks fetch `cxxopts` at configure time for argument parsing.  See
[Benchmarking](implementations.md#benchmarking) for what each driver measures.
Site launch and submit scripts live under `benchmark/frontier/` and
`benchmark/aurora/`; machine-independent helpers, including the plotting
scripts, stay in `benchmark/scripts/`
(`pip install -r benchmark/scripts/requirements.txt`).
