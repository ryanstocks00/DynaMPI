/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 *
 * The one-sided RMA distributors: no collective calls on the hot path, at the
 * price of a preallocated window whose capacity you must size up front.
 *
 *   mpirun -n 4 ./05_lockfree_rma
 */

#include <cstddef>
#include <dynampi/impl/hierarchical_lockfree_rma_distributor.hpp>
#include <dynampi/impl/lockfree_rma_distributor.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    using Task = int;
    using Result = long;
    auto work = [](Task n) -> Result { return static_cast<Result>(n) * n; };

    // --- Flat: one manager-owned window ----------------------------------
    {
      using Distributor = dynampi::LockFreeRMAWorkDistributor<Task, Result>;

      Distributor::Config config;
      // max_tasks is a LIFETIME total, not a concurrent depth: the task table,
      // result table and completion log are all allocated for this many
      // entries at construction. Publishing more throws std::length_error.
      config.max_tasks = 128;
      // Only consulted for variable-length payloads (std::vector, std::string).
      config.max_task_count = 1;
      config.max_result_count = 1;

      Distributor dist(work, config);
      if (dist.is_root_manager()) {
        std::vector<Task> tasks;
        for (Task n = 1; n <= 100; ++n) tasks.push_back(n);

        // Publish in one call: insert_tasks() puts the whole span in two RMA
        // round trips, while insert_task() costs two round trips *per task*.
        dist.insert_tasks(tasks);

        auto results = dist.finish_remaining_tasks();
        std::cout << "flat lock-free RMA: " << results.size() << " results\n";

        try {
          std::vector<Task> overflow(64, 1);  // 100 + 64 > max_tasks
          dist.insert_tasks(overflow);
          std::cout << "  (expected a capacity error)\n";
        } catch (const std::length_error& e) {
          std::cout << "  capacity guard: " << e.what() << "\n";
        }
      }
    }

    // --- Hierarchical: one window per tree level -------------------------
    //
    // Same protocol, instantiated per level of the node-aware tree, so no
    // single window has to service every rank. Worth it once a single
    // manager-owned window becomes the ceiling; measure before switching.
    {
      using Distributor = dynampi::HierarchicalLockFreeRMAWorkDistributor<Task, Result>;

      Distributor::Config config;
      config.max_tasks = 256;        // per upper-level window
      config.max_local_tasks = 256;  // per node-local window
      // max_upper_fanout: -1 auto, 0 flat, >0 explicit cap on direct children.
      config.max_upper_fanout = -1;

      Distributor dist(work, config);
      if (dist.is_root_manager()) {
        std::vector<Task> tasks;
        for (Task n = 1; n <= 100; ++n) tasks.push_back(n);
        dist.insert_tasks(tasks);

        // gather_once() takes a single harvest snapshot and returns whatever
        // is ready, instead of looping until everything is collected.
        size_t snapshot = dist.gather_once().size();
        size_t rest = dist.finish_remaining_tasks().size();
        std::cout << "hierarchical lock-free RMA: " << snapshot << " ready immediately, " << rest
                  << " collected after\n";
      }
    }
  }
  MPI_Finalize();
  return 0;
}
