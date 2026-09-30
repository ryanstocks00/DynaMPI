# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)
FetchContent_Declare(
    cxxopts
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.3.1.zip
)
FetchContent_MakeAvailable(cxxopts)

set(benchmarks
    asymptotic_distribution_throughput
    weak_scaling_distribution_rate
    load_balancing
    pingpong
    timer_resolution
    shutdown_time
    rma_atomic_microbench
    twosided_msgrate_microbench
)

foreach(benchmark IN LISTS benchmarks)
    add_executable(${benchmark}
        ${benchmark}.cpp
    )
    target_link_libraries(${benchmark}
        PRIVATE
        dynampi
        cxxopts::cxxopts
    )
endforeach()

# Optional ADLB arm of the load-balancing sweep. ADLB/X (the `lb` component of
# Swift/T) is an external reference, not a dependency, so this target only
# appears when -DADLB_ROOT=<prefix> -DCUTILS_ROOT=<prefix> (or the matching
# environment variables) point at a build of it.
set(ADLB_ROOT "$ENV{ADLB_ROOT}" CACHE PATH "Install prefix of ADLB/X (swift-t lb)")
set(CUTILS_ROOT "$ENV{CUTILS_ROOT}" CACHE PATH "Install prefix of ExM c-utils, which ADLB needs")

find_path(ADLB_INCLUDE_DIR adlb.h HINTS "${ADLB_ROOT}/include")
find_library(ADLB_LIBRARY NAMES adlb HINTS "${ADLB_ROOT}/lib")
find_path(CUTILS_INCLUDE_DIR c-utils.h HINTS "${CUTILS_ROOT}/include" "${ADLB_ROOT}/include")
find_library(CUTILS_LIBRARY NAMES exmcutils HINTS "${CUTILS_ROOT}/lib" "${ADLB_ROOT}/lib")

if(ADLB_INCLUDE_DIR AND ADLB_LIBRARY AND CUTILS_INCLUDE_DIR AND CUTILS_LIBRARY)
    message(STATUS "ADLB found (${ADLB_LIBRARY}); building load_balancing_adlb")
    add_executable(load_balancing_adlb load_balancing_adlb.cpp)
    target_include_directories(load_balancing_adlb PRIVATE
        ${ADLB_INCLUDE_DIR}
        ${CUTILS_INCLUDE_DIR}
    )
    target_link_libraries(load_balancing_adlb
        PRIVATE
        dynampi
        cxxopts::cxxopts
        ${ADLB_LIBRARY}
        ${CUTILS_LIBRARY}
        m
    )
else()
    message(STATUS "ADLB not found; skipping load_balancing_adlb "
                   "(set -DADLB_ROOT=<prefix> -DCUTILS_ROOT=<prefix> to build it)")
endif()
