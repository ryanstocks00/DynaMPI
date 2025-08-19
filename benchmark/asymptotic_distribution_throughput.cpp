/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <dynampi/dynampi.hpp>
#include <dynampi/utilities/timer.hpp>
#include <iostream>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "Beginning testing dynamic MPI task distribution throughput" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  dynampi::Timer timer;

  auto worker_task = [](size_t task) -> size_t { return task * task; };
  auto result = dynampi::mpi_manager_worker_distribution<size_t>(1000ul, worker_task);

  if (rank == 0) {
    std::cout
        << "Dynamic MPI task distribution throughput test completed successfully. Total run time: "
        << timer << std::endl;
  }

  MPI_Finalize();
  return 0;
}
