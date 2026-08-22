/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 *
 * Instrumenting a run with track_statistics, and the difference between the
 * Aggregated and Detailed modes.
 *
 *   mpirun -n 4 ./06_statistics
 */

#include <dynampi/dynampi.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    using Task = int;
    using Result = int;
    auto work = [](Task n) { return n * n; };

    std::vector<Task> tasks;
    for (Task n = 1; n <= 200; ++n) tasks.push_back(n);

    // --- Aggregated: counters only ---------------------------------------
    //
    // Message and byte counts plus per-rank task counts, with no clock reads
    // in the hot path. This is the mode to use while measuring throughput.
    {
      using Distributor = dynampi::DynamicWorkDistributor<
          Task, Result, dynampi::track_statistics<dynampi::StatisticsMode::Aggregated>>;

      Distributor dist(work, {});
      if (dist.is_root_manager()) {
        dist.insert_tasks(tasks);
        auto results = dist.finish_remaining_tasks();
        std::cout << "aggregated: " << results.size() << " results\n";
      }

      // worker_task_counts is filled by an MPI_Gather inside finalize(), so it
      // stays empty until then -- and every rank must reach finalize() (the
      // destructor calls it) for that gather to complete.
      dist.finalize();

      if (dist.is_root_manager()) {
        const auto& stats = dist.get_statistics();
        std::cout << "  messages sent: " << stats.comm_statistics.send_count
                  << ", bytes: " << stats.comm_statistics.bytes_sent << "\n";
        if (stats.worker_task_counts.has_value()) {
          std::cout << "  tasks executed per rank:";
          for (size_t rank = 0; rank < stats.worker_task_counts->size(); ++rank) {
            std::cout << " [" << rank << "]=" << stats.worker_task_counts->at(rank);
          }
          std::cout << "\n";
        }
      }
    }

    // --- Detailed: counters plus timing ----------------------------------
    //
    // Everything Aggregated tracks, and additionally the wall time spent
    // inside the calls that increment those counters. The extra clock reads
    // sit in the hot path, so prefer Aggregated when you are benchmarking.
    {
      using Distributor = dynampi::NaiveWorkDistributor<
          Task, Result, dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>;

      Distributor dist(work, {});
      if (dist.is_root_manager()) {
        dist.insert_tasks(tasks);
        auto results = dist.finish_remaining_tasks();

        const auto& comm = dist.get_statistics().comm_statistics;
        std::cout << "detailed: " << results.size() << " results\n";
        std::cout << "  sent " << comm.send_count << " messages (" << comm.average_send_size()
                  << " bytes avg) in " << comm.send_time << " s\n";
        std::cout << "  received " << comm.recv_count << " messages in " << comm.recv_time
                  << " s\n";
      }
    }
  }
  MPI_Finalize();
  return 0;
}
