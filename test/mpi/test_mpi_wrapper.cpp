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
  using TaskT = uint32_t;
  auto worker_task = [](TaskT task) -> double {
    // Simulate work
    return sqrt(static_cast<double>(task));
  };
  std::vector<TaskT> tasks(10);
  for (size_t i = 0; i < tasks.size(); ++i) {
    tasks[i] = static_cast<TaskT>(i);
  }
  dynampi::NaiveMPIWorkDistributor<TaskT, double,
                                   dynampi::DynamicDistributionConfig{.auto_run_workers = false}>
      distributor(worker_task, MPI_COMM_WORLD);

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
  auto worker_task = [](size_t task) -> char { return "Hi"[task]; };
  auto result = dynampi::mpi_manager_worker_distribution<char>(2, worker_task);
  if (MPIEnvironment::world_comm_rank() == 0) {
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result->size(), 2);
  } else {
    EXPECT_FALSE(result.has_value());
  }
}

TEST(DynamicDistribution, Example1) {
  auto worker_task = [](size_t task) -> size_t { return task * task; };
  auto result = dynampi::mpi_manager_worker_distribution<size_t>(4, worker_task);
  if (result.has_value()) {
    assert(result == std::vector<size_t>({0, 1, 4, 9}));
    EXPECT_EQ(result, std::vector<size_t>({0, 1, 4, 9}));
  }
}

TEST(DynamicDistribution, Example1_ManagerLast) {
  auto worker_task = [](size_t task) -> size_t { return task * task; };
  auto result = dynampi::mpi_manager_worker_distribution<size_t>(
      4, worker_task, MPI_COMM_WORLD, MPIEnvironment::world_comm_size() - 1);
  if (result.has_value()) {
    assert(result == std::vector<size_t>({0, 1, 4, 9}));
    EXPECT_EQ(result, std::vector<size_t>({0, 1, 4, 9}));
  }
}

TEST(DynamicDistribution, Example2) {
  using Task = int;
  using Result = std::vector<int>;
  auto worker_task = [](Task task) -> Result {
    return Result{task, task * task, task * task * task};
  };
  {
    dynampi::MPIDynamicWorkDistributor<Task, Result> work_distributer(worker_task);
    if (work_distributer.is_manager()) {
      work_distributer.insert_tasks({1, 2, 3, 4, 5});
      auto results = work_distributer.distribute_tasks();
      EXPECT_EQ(results.size(), 5);
      work_distributer.insert_tasks({6, 7, 8});
      results = work_distributer.distribute_tasks();
      EXPECT_EQ(results.size(), 8);
    }
  }
}
