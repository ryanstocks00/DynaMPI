# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: MIT

add_executable(asymptotic_distribution_throughput
    asymptotic_distribution_throughput.cpp
)
target_link_libraries(asymptotic_distribution_throughput
    PRIVATE
    dynampi
)
