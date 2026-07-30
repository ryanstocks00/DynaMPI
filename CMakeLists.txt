# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.14)

# Single source of truth for the version is include/dynampi/version.hpp, which
# defines it as #ifndef fallbacks so the headers stay usable without CMake.
# Parse it here rather than duplicating the numbers.
file(READ "${CMAKE_CURRENT_LIST_DIR}/include/dynampi/version.hpp"
     DYNAMPI_VERSION_HEADER)
foreach(part MAJOR MINOR PATCH)
  string(REGEX MATCH "#define DYNAMPI_VERSION_${part} ([0-9]+)" _matched
                     "${DYNAMPI_VERSION_HEADER}")
  if(NOT _matched)
    message(
      FATAL_ERROR
        "Could not parse DYNAMPI_VERSION_${part} from include/dynampi/version.hpp"
    )
  endif()
  set(DYNAMPI_VERSION_${part} "${CMAKE_MATCH_1}")
endforeach()

project(
  DynaMPI
  VERSION
    ${DYNAMPI_VERSION_MAJOR}.${DYNAMPI_VERSION_MINOR}.${DYNAMPI_VERSION_PATCH}
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(IS_BASE_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(IS_BASE_PROJECT ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(DYNAMPI_BUILD_TESTS "Build tests for DynaMPI" ${IS_BASE_PROJECT})
option(DYNAMPI_BUILD_BENCHMARKS "Build benchmarks for DynaMPI"
       ${IS_BASE_PROJECT})
option(DYNAMPI_BUILD_EXAMPLES "Build examples for DynaMPI" ${IS_BASE_PROJECT})

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET)

# Check for uncommitted changes
execute_process(
  COMMAND git diff-index --quiet HEAD --
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_DIRTY
  ERROR_QUIET)

# Append '-dirty' if there are changes
if(GIT_COMMIT_HASH AND NOT GIT_DIRTY EQUAL 0)
  set(GIT_COMMIT_HASH "${GIT_COMMIT_HASH}-dirty")
endif()

find_package(MPI REQUIRED)

add_library(dynampi INTERFACE)
target_include_directories(
  dynampi INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:include>)
# CMAKE_CXX_STANDARD above only applies to targets declared in this directory.
# Consumers that add_subdirectory() us and link dynampi need the requirement to
# travel with the target.
target_compile_features(dynampi INTERFACE cxx_std_20)
target_compile_definitions(
  dynampi
  INTERFACE DYNAMPI_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            DYNAMPI_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            DYNAMPI_VERSION_PATCH=${PROJECT_VERSION_PATCH})
# On the target rather than add_definitions() so it reaches consumers outside
# this directory too. Left undefined outside a git checkout, so version.hpp's
# "unknown" fallback applies rather than an empty string.
if(GIT_COMMIT_HASH)
  target_compile_definitions(dynampi
                             INTERFACE DYNAMPI_COMMIT_HASH="${GIT_COMMIT_HASH}")
endif()
if(NOT CMAKE_CXX_COMPILER MATCHES "smpicxx")
  target_link_libraries(dynampi INTERFACE MPI::MPI_CXX)
endif()

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror -fno-ms-extensions)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  add_compile_options(-diag-disable=10430 -Wno-unknown-warning-option)
endif()

set(CMAKE_CXX_FLAGS_COVERAGE "-O0 -g -coverage -fprofile-arcs -ftest-coverage \
    -fno-elide-constructors ${CMAKE_CXX_FLAGS_COVERAGE}")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS_COVERAGE
      "${CMAKE_CXX_FLAGS_COVERAGE} -fno-default-inline")
endif()

if(DYNAMPI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

if(DYNAMPI_BUILD_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

if(DYNAMPI_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
