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
    test_load_balancing
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
