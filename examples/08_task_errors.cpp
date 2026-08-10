/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 *
 * What happens when a task throws: the job carries on either way, and the
 * caller chooses between propagating the failure and recovering from it.
 *
 *   mpirun -n 4 ./08_task_errors
 */

#include <dynampi/dynampi.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    using Task = int;
    using Result = int;

    // Task 3 fails. Nothing about the surrounding run changes.
    auto work = [](Task n) -> Result {
      if (n == 3) throw std::runtime_error("no square root of a negative mood");
      return n * n;
    };

    // --- Recover: run to completion, inspect the failures afterwards. ---
    {
      dynampi::DynamicWorkDistributor<Task, Result>::Config config;
      config.rethrow_task_errors = false;

      dynampi::DynamicWorkDistributor<Task, Result> dist(work, config);
      if (dist.is_root_manager()) {
        dist.insert_tasks({1, 2, 3, 4, 5});
        auto results = dist.finish_remaining_tasks();

        // One entry per task, so the failed one holds a default-constructed
        // Result rather than shifting everything after it.
        std::cout << "recovered " << results.size() << " results\n";
        for (const auto& failure : dist.take_task_errors()) {
          std::cout << "  task failed on rank " << failure.worker_rank << ": " << failure.message
                    << "\n";
        }
      }
    }

    // --- Propagate: the default. The manager sees a TaskFailure. ---
    {
      dynampi::DynamicWorkDistributor<Task, Result> dist(work);
      if (dist.is_root_manager()) {
        dist.insert_tasks({1, 2, 3, 4, 5});
        try {
          auto results = dist.finish_remaining_tasks();
          std::cout << "no failures: " << results.size() << " results\n";
        } catch (const dynampi::TaskFailure& e) {
          std::cout << "caught: " << e.what() << "\n";
          // Thrown before the results were handed over, so they are still
          // there for a caller that wants to carry on.
          std::cout << "  " << dist.finish_remaining_tasks().size()
                    << " results were still waiting\n";
        }
      }
    }
  }
  MPI_Finalize();
  return 0;
}
