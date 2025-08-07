# SPDX-FileCopyrightText: 2025 Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.14)

project(
  DynaMPI
  VERSION 0.0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(IS_BASE_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(IS_BASE_PROJECT ON)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

option(DYNAMPI_BUILD_TESTS "Build tests for DynaMPI" ${IS_BASE_PROJECT})
option(DYNAMPI_BUILD_BENCHMARKS "Build benchmarks for DynaMPI"
       ${IS_BASE_PROJECT})

execute_process(
  COMMAND git rev-parse HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Check for uncommitted changes
execute_process(
  COMMAND git diff-index --quiet HEAD --
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  RESULT_VARIABLE GIT_DIRTY)

# Append '-dirty' if there are changes
if(NOT GIT_DIRTY EQUAL 0)
  set(GIT_COMMIT_HASH "${GIT_COMMIT_HASH}-dirty")
endif()

add_definitions(-DDYNAMPI_COMMIT_HASH="${GIT_COMMIT_HASH}")

find_package(MPI REQUIRED)

add_library(dynampi INTERFACE)
target_include_directories(
  dynampi INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:include>)
target_compile_definitions(
  dynampi
  INTERFACE DYNAMPI_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            DYNAMPI_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            DYNAMPI_VERSION_PATCH=${PROJECT_VERSION_PATCH})
target_link_libraries(dynampi INTERFACE MPI::MPI_CXX)

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
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
