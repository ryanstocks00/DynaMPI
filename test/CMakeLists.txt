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
# Fetch GoogleTest (pin past v1.17.0: Clang 21 -Wcharacter-conversion in
# gtest-printers.h — google/googletest@fa8438a)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG fa8438ae6b70c57010177de47a9f13d7041a6328
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt # cmake-lint: disable=C0103
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Directory-level -Werror from the parent must not fail third-party builds.
foreach(_gtest_tgt gtest gtest_main gmock gmock_main)
  if(TARGET ${_gtest_tgt})
    target_compile_options(
      ${_gtest_tgt}
      PRIVATE
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-error>
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-Wno-character-conversion>)
  endif()
endforeach()

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
    string(REGEX MATCH "(Open MPI|OpenRTE)[^0-9]*([0-9]+)" OPENMPI_VERSION_MATCH
                 "${MPI_VERSION_OUTPUT}")
    if(CMAKE_MATCH_2 AND CMAKE_MATCH_2 LESS 5)
      set(OPENMPI_NEEDS_SPLIT_WIN_WORKAROUND TRUE)
      message(STATUS "Open MPI ${CMAKE_MATCH_2}: using osc/pt2pt for split-window tests")
    endif()
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
set(DYNAMPI_MAX_MPI_RANK 64 CACHE STRING "Max MPI ranks for ctest MPI runs")
if(DEFINED ENV{DYNAMPI_MAX_MPI_RANK})
  set(DYNAMPI_MAX_MPI_RANK "$ENV{DYNAMPI_MAX_MPI_RANK}")
endif()

function(add_mpi_test test_name num_procs)
  set(gtest_args "")
  if(OPENMPI_NEEDS_SPLIT_WIN_WORKAROUND)
    list(
      APPEND
      gtest_args
      "--gtest_filter=-HierarchicalAsyncPutLockFree.GroupedUpperHierarchy:HierarchicalAsyncPutLockFree.AutoFanoutUpperHierarchy"
    )
  endif()
  set(command
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG}
      ${num_procs}
      ${MPI_OVERSUBSCRIBE_FLAG}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:mpi_test>
      ${gtest_args}
      ${MPIEXEC_POSTFLAGS})
  add_test(NAME ${test_name} COMMAND ${command})
endfunction()

foreach(rank 1 2 3 4 8 16 64)
  if(rank GREATER DYNAMPI_MAX_MPI_RANK)
    message(STATUS "Skipping mpi_test_${rank}_rank (rank > ${DYNAMPI_MAX_MPI_RANK})")
  else()
  add_mpi_test(mpi_test_${rank}_rank ${rank})
  endif()
endforeach()

# Open MPI 4's osc/rdma component can race while creating windows on disjoint
# communicators from one shared-memory node. Run only the synthetic grouped
# topology tests with osc/pt2pt; forcing that legacy backend on the full
# lock-free suite can stall at high rank counts. Open MPI 5 fixed the rdma path
# and removed the pt2pt fallback.
if(OPENMPI_NEEDS_SPLIT_WIN_WORKAROUND AND DYNAMPI_MAX_MPI_RANK GREATER_EQUAL 8)
  add_test(
    NAME mpi_test_8_rank_grouped
    COMMAND
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG}
      8
      ${MPI_OVERSUBSCRIBE_FLAG}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:mpi_test>
      "--gtest_filter=HierarchicalAsyncPutLockFree.GroupedUpperHierarchy"
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(mpi_test_8_rank_grouped
                       PROPERTIES ENVIRONMENT "OMPI_MCA_osc=pt2pt")
endif()
if(OPENMPI_NEEDS_SPLIT_WIN_WORKAROUND AND DYNAMPI_MAX_MPI_RANK GREATER_EQUAL 64)
  add_test(
    NAME mpi_test_64_rank_auto_fanout
    COMMAND
      ${MPIEXEC_EXECUTABLE}
      ${MPIEXEC_NUMPROC_FLAG}
      64
      ${MPI_OVERSUBSCRIBE_FLAG}
      ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:mpi_test>
      "--gtest_filter=HierarchicalAsyncPutLockFree.AutoFanoutUpperHierarchy"
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(mpi_test_64_rank_auto_fanout
                       PROPERTIES ENVIRONMENT "OMPI_MCA_osc=pt2pt")
endif()
