<!--
  SPDX-FileCopyrightText: 2026 Ryan Stocks
  SPDX-License-Identifier: Apache-2.0
-->

# Installation

DynaMPI is a **header-only** C++20 library.  It requires an MPI
implementation (MPICH, Open MPI, Intel MPI, or MS-MPI) and a C++20 compiler.

## Requirements

| Dependency | Minimum Version |
|------------|----------------|
| C++ compiler | GCC 11+, Clang 14+, MSVC 2022+ |
| MPI | MPICH 3.4+, Open MPI 4.0+, Intel MPI 2021+, MS-MPI 10+ |
| CMake | 3.14+ (for building tests) |

## CMake

```cmake
# From a subdirectory
add_subdirectory(path/to/DynaMPI)
target_link_libraries(my_target PRIVATE dynampi)

# Or via FetchContent
include(FetchContent)
FetchContent_Declare(
    DynaMPI
    GIT_REPOSITORY https://github.com/Trailblaze-Software/DynaMPI.git
    GIT_TAG main
)
FetchContent_MakeAvailable(DynaMPI)
target_link_libraries(my_target PRIVATE dynampi)
```

## Manual (header-only)

DynaMPI can be used without CMake — add the `include/` directory to your
compiler's include path and link against MPI:

```bash
g++ -std=c++20 -I/path/to/DynaMPI/include my_program.cpp -lmpi
```

## Building Tests

```bash
git clone https://github.com/Trailblaze-Software/DynaMPI.git
cd DynaMPI
cmake -B build -DDYNAMPI_BUILD_TESTS=ON
cmake --build build

# Run unit tests
./build/test/unit_test

# Run MPI tests with 4 ranks
mpirun -n 4 ./build/test/mpi_test
```
