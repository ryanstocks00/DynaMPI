# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)
project(MyTests LANGUAGES CXX)

# Enable testing
include(CTest)
enable_testing()
set(CTEST_OUTPUT_ON_FAILURE TRUE)

# --------------------------------------------------------------------
# Check if using Clang and add appropriate flags for GoogleTest
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # Clang has stricter inline optimization that can cause issues with GoogleTest
  add_compile_options(-Wno-inline)
  message(
    STATUS
      "Clang detected, added -Wno-error=inline to suppress GoogleTest warnings")
endif()

# --------------------------------------------------------------------
# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip)
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt # cmake-lint: disable=C0103
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# --------------------------------------------------------------------
# Create test executables from directories using glob
function(create_test target src_dir link_lib)
  file(GLOB test_src CONFIGURE_DEPENDS "${src_dir}/*.cpp")
  add_executable(${target} ${test_src})
  target_link_libraries(${target} ${link_lib})
endfunction()

if(NOT CMAKE_CXX_COMPILER MATCHES "smpicxx")
  create_test(unit_test unit "GTest::gtest_main;dynampi")
  add_test(NAME unit_test COMMAND unit_test)
endif()

create_test(mpi_test mpi "GTest::gtest;dynampi")

# --------------------------------------------------------------------
# Detect MPI implementation and configure oversubscribe flags
set(MPI_OVERSUBSCRIBE_FLAG "")
if(MPIEXEC_EXECUTABLE)
  execute_process(
    COMMAND ${MPIEXEC_EXECUTABLE} --version
    OUTPUT_VARIABLE MPI_VERSION_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "Open MPI|OpenRTE" IS_OPENMPI "${MPI_VERSION_OUTPUT}")
  string(REGEX MATCH "HYDRA|MPICH" IS_MPICH "${MPI_VERSION_OUTPUT}")
  string(REGEX MATCH "Intel.*MPI" IS_INTEL "${MPI_VERSION_OUTPUT}")
  string(REGEX MATCH "Microsoft MPI" IS_MICROSOFT "${MPI_VERSION_OUTPUT}")

  if(IS_OPENMPI)
    set(MPI_OVERSUBSCRIBE_FLAG "--oversubscribe")
    message(STATUS "Detected MPI implementation with oversubscribe support")
  else()
    message(
      STATUS
        "Detected MPI implementation without explicit oversubscribe requirement"
    )
  endif()

  message(STATUS "MPI version output: ${MPI_VERSION_OUTPUT}")
  message(STATUS "MPI oversubscribe flag: '${MPI_OVERSUBSCRIBE_FLAG}'")
endif()

# --------------------------------------------------------------------
# Add MPI tests with varying ranks
function(add_mpi_test test_name num_procs)
  set(command
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG}
      ${num_procs}
      ${MPI_OVERSUBSCRIBE_FLAG}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:mpi_test>
      ${MPIEXEC_POSTFLAGS})
  add_test(NAME ${test_name} COMMAND ${command})
endfunction()

foreach(rank 1 2 3 4 8)
  add_mpi_test(mpi_test_${rank}_rank ${rank})
endforeach()
