/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 *
 * MinimalLockFreeMPIWorkDistributor: a dynamically balanced parallel-for over
 * 0 .. n-1, where the task payload is just the loop index.
 *
 *   mpirun -n 4 ./07_parallel_for
 */

#include <cmath>
#include <cstddef>
#include <dynampi/impl/lockfree_distributor.hpp>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    // Every rank claims indices by atomically bumping one shared counter, so
    // there is no manager bottleneck -- but results are collected with a
    // single MPI_Gatherv at the end, which bounds this to roughly the low
    // hundreds of ranks. Config carries comm and manager_rank only.
    dynampi::MinimalLockFreeMPIWorkDistributor<double> dist([](size_t i) {
      double sum = 0.0;
      for (size_t k = 0; k < i; ++k) sum += std::sqrt(static_cast<double>(k));
      return sum;
    });

    // run() is collective: every rank must call it with the same task count.
    // Results come back sorted by index on the manager, empty on workers.
    std::vector<double> results = dist.run(1000);

    if (dist.is_root_manager()) {
      std::cout << "parallel-for over 1000 indices -> " << results.size() << " results\n";
      std::cout << "  results[0] = " << results[0] << " (expect 0)\n";
      std::cout << "  results are index-ordered, so results[i] is task i\n";
    }
  }
  MPI_Finalize();
  return 0;
}
