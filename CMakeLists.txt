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

add_library(dynampi INTERFACE)
target_include_directories(
  dynampi INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                    $<INSTALL_INTERFACE:include>)
target_compile_definitions(
  dynampi
  INTERFACE DYNAMPI_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
            DYNAMPI_VERSION_MINOR=${PROJECT_VERSION_MINOR}
            DYNAMPI_VERSION_PATCH=${PROJECT_VERSION_PATCH})

if(DYNAMPI_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
