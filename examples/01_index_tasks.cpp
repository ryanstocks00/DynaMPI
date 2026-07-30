/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 *
 * The shortest possible DynaMPI program: distribute a fixed number of
 * index-valued tasks with a single collective call.
 *
 *   mpirun -n 4 ./01_index_tasks
 */

#include <cmath>
#include <cstddef>
#include <dynampi/dynampi.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    // Uneven cost per task: exactly the case dynamic distribution exists for.
    auto work = [](size_t task) -> double {
      double sum = 0.0;
      for (size_t i = 0; i < task * 1000; ++i) sum += std::sqrt(static_cast<double>(i));
      return sum;
    };

    // Collective -- every rank calls it. Non-manager ranks run the worker loop
    // internally and get back std::nullopt; the manager gets the results.
    auto results = dynampi::mpi_manager_worker_distribution<double>(64, work);

    if (results.has_value()) {
      std::cout << "collected " << results->size() << " results\n";

      // NOTE: the default distributor is hierarchical, so results come back in
      // completion order -- (*results)[i] is NOT the result of task i. See
      // 02_ordered_results.cpp if you need index alignment.
    }
  }
  MPI_Finalize();
  return 0;
}
