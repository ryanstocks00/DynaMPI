/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include <cassert>
#include <functional>
#include <optional>
#include <string_view>
#include <tuple>
#include <vector>

#include "dynampi/impl/hierarchical_distributor.hpp"
#include "dynampi/impl/naive_distributor.hpp"

namespace dynampi {

namespace version {

inline constexpr int major = DYNAMPI_VERSION_MAJOR;
inline constexpr int minor = DYNAMPI_VERSION_MINOR;
inline constexpr int patch = DYNAMPI_VERSION_PATCH;

// Macros for compile-time version string
#define DYNAMPI_STR_HELPER(x) #x
#define DYNAMPI_STR(x) DYNAMPI_STR_HELPER(x)
#define DYNAMPI_VERSION_STRING                                                                   \
  "v" DYNAMPI_STR(DYNAMPI_VERSION_MAJOR) "." DYNAMPI_STR(DYNAMPI_VERSION_MINOR) "." DYNAMPI_STR( \
      DYNAMPI_VERSION_PATCH)

inline constexpr std::string_view string = DYNAMPI_VERSION_STRING;

[[nodiscard]] constexpr bool is_at_least(int v_major, int v_minor, int v_patch) {
  return std::tie(major, minor, patch) >= std::tie(v_major, v_minor, v_patch);
}

[[nodiscard]] inline constexpr std::string_view compile_date() { return __DATE__ " " __TIME__; }

[[nodiscard]] inline constexpr std::string_view commit_hash() { return DYNAMPI_COMMIT_HASH; }

}  // namespace version

template <typename ResultT,
          template <typename, typename, typename...> typename T = HierarchicalMPIWorkDistributor>
std::optional<std::vector<ResultT>> mpi_manager_worker_distribution(
    size_t n_tasks, std::function<ResultT(size_t)> worker_function, MPI_Comm comm = MPI_COMM_WORLD,
    int manager_rank = 0) {
  T<size_t, ResultT> distributor(worker_function, {.comm = comm, .manager_rank = manager_rank});
  if (distributor.is_root_manager()) {
    for (size_t i = 0; i < n_tasks; ++i) {
      distributor.insert_task(i);
    }
    return distributor.finish_remaining_tasks();
  }
  return {};
}

template <typename TaskT, typename ResultT, typename... Options>
using MPIDynamicWorkDistributor = HierarchicalMPIWorkDistributor<TaskT, ResultT, Options...>;

}  // namespace dynampi
