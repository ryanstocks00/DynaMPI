

# File hierarchical\_topology\_detail.hpp

[**File List**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**impl**](dir_87365955bfd9c1869b83a1fbd3fdb221.md) **>** [**hierarchical\_topology\_detail.hpp**](hierarchical__topology__detail_8hpp.md)

[Go to the documentation of this file](hierarchical__topology__detail_8hpp.md)


```C++
/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Topology calculations shared by the two-sided and lock-free RMA
// hierarchical distributors. Transport-specific level ownership and task
// movement remain in their respective implementations.

#include <cmath>
#include <limits>
#include <optional>
#include <utility>

#include "../mpi/mpi_communicator.hpp"
#include "dynampi/mpi/mpi_error.hpp"

namespace dynampi::detail {

inline int resolve_upper_fanout(int manager_count, int configured_fanout) {
  if (configured_fanout < 0) {
    if (manager_count <= 32) return std::numeric_limits<int>::max();
    int fanout = 1;
    const double target = std::sqrt(static_cast<double>(manager_count));
    while (fanout < target) fanout *= 2;
    return fanout;
  }
  return configured_fanout > 0 ? configured_fanout : std::numeric_limits<int>::max();
}

template <typename Communicator>
std::optional<Communicator> split_local_worker_communicator(const Communicator& world_comm,
                                                            int manager_rank,
                                                            int max_local_group_size) {
  Communicator node_comm = world_comm.split_by_node();
  std::optional<Communicator> local_domain;
  if (max_local_group_size > 0 && node_comm.size() > max_local_group_size) {
    const int color = node_comm.rank() / max_local_group_size;
    auto partition = node_comm.split(color, node_comm.rank());
    local_domain.emplace(std::move(*partition));
  } else {
    local_domain.emplace(std::move(node_comm));
  }

  const int local_color = world_comm.rank() == manager_rank ? MPI_UNDEFINED : 0;
  return local_domain->split(local_color, world_comm.rank());
}

// Valid only on group rank 0. Every group member must call this collectively.
// Summing the actual widths matters because the root manager is excluded from
// its local worker communicator, so its node manager fronts one fewer leaf.
template <typename Communicator>
int sum_subtree_widths_to_group_leader(int local_width, const Communicator& group_comm) {
  int group_width = 0;
  DYNAMPI_MPI_CHECK(MPI_Reduce, (&local_width, &group_width, 1, MPI_INT, MPI_SUM, 0,
                                 static_cast<MPI_Comm>(group_comm)));
  return group_width;
}

}  // namespace dynampi::detail
```


