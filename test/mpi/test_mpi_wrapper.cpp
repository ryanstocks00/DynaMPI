/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <dynampi/dynampi.hpp>

#include "mpi_test_environment.hpp"

TEST(MPI, PingPong) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int send_data = rank;
  int recv_data = -1;
  MPI_Sendrecv(&send_data, 1, MPI_INT, (rank + 1) % size, 0, &recv_data, 1, MPI_INT,
               (rank - 1 + size) % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  EXPECT_EQ(recv_data, (rank == 0) ? size - 1 : rank - 1);
}

TEST(DynamicDistribution, Naive) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes.";
  }
  typedef uint32_t TaskT;
  auto worker_task = [](TaskT task) -> double {
    // Simulate work
    return sqrt(static_cast<double>(task));
  };
  std::vector<TaskT> tasks(10);
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i] = static_cast<TaskT>(i);
  }
  dynampi::NaiveMPIWorkDistributor<TaskT, double> distributor(MPI_COMM_WORLD, worker_task);
  if (distributor.is_manager()) {
    for (int i = 0; i < 10; ++i) {
      distributor.insert_task(i);
    }
  }
  if (distributor.is_manager()) {
    auto results = distributor.distribute_tasks();
    EXPECT_EQ(results.size(), 10);
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_DOUBLE_EQ(results[i] * results[i], static_cast<double>(i));
    }
  } else {
    distributor.run_worker();
  }
}

TEST(DynamicDistribution, Naive2) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes.";
  }

  std::vector<size_t> tasks;
  if (MPIEnvironment::world_comm_rank() == 0) {
    tasks = {0, 1};
  }
  auto worker_task = [](size_t task) -> char { return "Hi"[task]; };
  std::vector<char> result =
      dynampi::mpi_manager_worker_distribution<size_t, char>(std::span<size_t>(tasks), worker_task);
  if (MPIEnvironment::world_comm_rank() == 0) {
    EXPECT_EQ(result.size(), 2);
  }
}
