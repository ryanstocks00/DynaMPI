/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include <cassert>
#include <functional>
#include <optional>
#include <vector>

#include "dynampi/impl/hierarchical_distributor.hpp"
#include "dynampi/impl/naive_distributor.hpp"
#include "dynampi/version.hpp"

namespace dynampi {

template <typename ResultT,
          template <typename, typename, typename...> typename T = HierarchicalWorkDistributor>
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
using DynamicWorkDistributor = HierarchicalWorkDistributor<TaskT, ResultT, Options...>;

}  // namespace dynampi
