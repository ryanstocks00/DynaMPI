# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)
FetchContent_Declare(
    cxxopts
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    URL https://github.com/jarro2783/cxxopts/archive/refs/tags/v3.3.1.zip
)
FetchContent_MakeAvailable(cxxopts)

add_executable(asymptotic_distribution_throughput
    asymptotic_distribution_throughput.cpp
)
target_link_libraries(asymptotic_distribution_throughput
    PRIVATE
    dynampi
    cxxopts::cxxopts
)
