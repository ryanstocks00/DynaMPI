

# File base\_distributor.hpp

[**File List**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**impl**](dir_87365955bfd9c1869b83a1fbd3fdb221.md) **>** [**base\_distributor.hpp**](base__distributor_8hpp.md)

[Go to the documentation of this file](base__distributor_8hpp.md)


```C++
/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
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
```


