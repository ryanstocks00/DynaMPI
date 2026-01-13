# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: MIT

include(FetchContent)
FetchContent_Declare(
    cxxopts
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.3.1.zip
)
FetchContent_MakeAvailable(cxxopts)

set(benchmarks
    asymptotic_distribution_throughput
    pingpong
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
