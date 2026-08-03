/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tasks discovered while the run is already in progress, plus explicit control
 * over the worker loop and over how much each collection call returns.
 *
 *   mpirun -n 4 ./02_incremental_tasks
 */

#include <cstddef>
#include <dynampi/dynampi.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    using Task = int;
    using Result = int;

    // Expanding a task into more tasks: each task n emits its two children,
    // which the manager feeds back into the queue.
    auto work = [](Task n) -> Result { return n * n; };

    dynampi::MPIDynamicWorkDistributor<Task, Result>::Config config;
    config.auto_run_workers = false;  // we call run_worker() ourselves below

    dynampi::MPIDynamicWorkDistributor<Task, Result> dist(work, config);

    if (dist.is_root_manager()) {
      // --- Round 1: a seed batch, collected in full. ---
      dist.insert_tasks({1, 2, 3, 4, 5});
      std::cout << "queued " << dist.remaining_tasks_count() << " tasks\n";
      auto round1 = dist.finish_remaining_tasks();
      std::cout << "round 1: " << round1.size() << " results\n";

      // --- Round 2: collect in chunks rather than waiting for everything. ---
      std::vector<Task> more;
      for (Task n = 6; n <= 25; ++n) more.push_back(n);
      dist.insert_tasks(more);

      size_t collected = 0;
      while (collected < more.size()) {
        // Return as soon as 5 results are ready, and clip to exactly 5 -- the
        // surplus stays buffered for the next call. max_seconds bounds the
        // wait (checked between messages, so it is a soft deadline).
        auto chunk = dist.run_tasks(
            {.target_num_tasks = 5, .allow_more_than_target_tasks = false, .max_seconds = 5.0});
        if (chunk.empty()) break;  // nothing ready within the deadline
        collected += chunk.size();
        std::cout << "  chunk of " << chunk.size() << " (" << collected << "/" << more.size()
                  << " done)\n";
      }

      // Releases the workers from run_worker(). The destructor would do it for
      // us; calling it explicitly makes the shutdown point obvious.
      dist.finalize();
    } else {
      // Workers block here until the manager finalizes. With the default
      // auto_run_workers = true this happens inside the constructor instead.
      dist.run_worker();
    }
  }
  // The enclosing scope above matters: every rank's distributor must be
  // destroyed before MPI_Finalize, since destruction is collective.
  MPI_Finalize();
  return 0;
}
