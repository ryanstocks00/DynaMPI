/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#include <deque>
#include <queue>
#include <type_traits>
#include <utility>

namespace dynampi {

struct prioritize_tasks_t {
  static constexpr bool value = false;
};

struct enable_prioritization : public prioritize_tasks_t {
  static constexpr bool value = true;
};

template <typename TaskT, typename ResultT, typename... Options>
class BaseMPIWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
  };

 protected:
  static constexpr bool prioritize_tasks = get_option_value<prioritize_tasks_t, Options...>();
  using QueueT = std::conditional_t<prioritize_tasks, std::priority_queue<std::pair<double, TaskT>>,
                                    std::deque<TaskT>>;
};

}  // namespace dynampi
