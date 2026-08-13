/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 *
 * NaiveWorkDistributor is the one distributor that returns results in task
 * order, and the one where task priorities work. Both are shown here.
 *
 *   mpirun -n 4 ./03_ordered_and_prioritized
 */

#include <dynampi/impl/naive_distributor.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    // --- Ordered results -------------------------------------------------
    //
    // `ordered == true`: the manager buffers results by task ID and only
    // releases a contiguous prefix, so results[i] is always task i. Variable
    // length payloads (std::vector, std::string) work here too, as they do on
    // every other distributor -- what the others lack is the ordering.
    {
      using Distributor = dynampi::NaiveWorkDistributor<int, std::string>;
      static_assert(Distributor::ordered);

      Distributor dist([](int n) { return std::string(static_cast<size_t>(n), '*'); });
      if (dist.is_root_manager()) {
        dist.insert_tasks({1, 2, 3, 4, 5});
        for (const auto& s : dist.finish_remaining_tasks()) std::cout << "ordered: " << s << "\n";
      }
    }

    // --- Prioritized tasks -----------------------------------------------
    //
    // enable_prioritization swaps the FIFO queue for a priority queue and
    // enables the two-argument insert_task(). Highest priority is dispatched
    // first, so results arrive in descending priority order. Note that
    // insert_tasks() is disabled in this mode.
    {
      using Distributor = dynampi::NaiveWorkDistributor<int, int, dynampi::enable_prioritization>;

      Distributor dist([](int n) { return n; });
      if (dist.is_root_manager()) {
        dist.insert_task(10, /*priority=*/1.0);
        dist.insert_task(30, /*priority=*/3.0);
        dist.insert_task(20, /*priority=*/2.0);

        std::cout << "prioritized:";
        for (int n : dist.finish_remaining_tasks()) std::cout << " " << n;
        std::cout << "  (expect 30 20 10)\n";
      }
    }
  }
  MPI_Finalize();
  return 0;
}
