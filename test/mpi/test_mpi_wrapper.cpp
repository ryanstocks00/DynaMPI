/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <dynampi/dynampi.hpp>
#include <vector>

#include "dynampi/mpi/mpi_communicator.hpp"
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

TEST(MPI, ErrorCheck) {
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  EXPECT_THROW(DYNAMPI_MPI_CHECK(MPI_Comm_rank, (MPI_COMM_NULL, nullptr)), std::runtime_error);
  try {
    DYNAMPI_MPI_CHECK(MPI_Comm_rank, (MPI_COMM_NULL, nullptr));
  } catch (const std::runtime_error &e) {
    EXPECT_TRUE(std::string(e.what()).find("MPI error in MPI_Comm_rank") != std::string::npos);
  }
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
  dynampi::NaiveMPIWorkDistributor<TaskT, double> distributor(
      worker_task, {.comm = MPI_COMM_WORLD, .auto_run_workers = false});
  if (distributor.is_manager()) {
    for (int i = 0; i < 10; ++i) {
      distributor.insert_task(i);
    }
  }
  if (distributor.is_manager()) {
    auto results = distributor.finish_remaining_tasks();
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
  for (int manager_rank : {0, MPIEnvironment::world_comm_size() - 1}) {
    auto worker_task = [](size_t task) -> size_t { return task * task; };
    auto result = dynampi::mpi_manager_worker_distribution<size_t>(4, worker_task, MPI_COMM_WORLD,
                                                                   manager_rank);
    if (result.has_value()) {
      assert(result == std::vector<size_t>({0, 1, 4, 9}));
      EXPECT_EQ(result, std::vector<size_t>({0, 1, 4, 9}));
    }
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
      auto results = work_distributer.finish_remaining_tasks();
      EXPECT_EQ(results, (std::vector<std::vector<int>>{
                             {1, 1, 1}, {2, 4, 8}, {3, 9, 27}, {4, 16, 64}, {5, 25, 125}}));
      work_distributer.insert_tasks({6, 7, 8});
      results = work_distributer.finish_remaining_tasks();
      EXPECT_EQ(results, (std::vector<std::vector<int>>{{1, 1, 1},
                                                        {2, 4, 8},
                                                        {3, 9, 27},
                                                        {4, 16, 64},
                                                        {5, 25, 125},
                                                        {6, 36, 216},
                                                        {7, 49, 343},
                                                        {8, 64, 512}}));
    }
  }
}

TEST(DynamicDistribution, PriorityQueue) {
  using Task = int;
  using Result = int;
  auto worker_task = [](Task task) -> Result { return task * task; };
  {
    dynampi::MPIDynamicWorkDistributor<Task, Result, dynampi::enable_prioritization>
        work_distributer(worker_task);
    if (work_distributer.is_manager()) {
      work_distributer.insert_task(1, 1.0);
      work_distributer.insert_task(7, 7.0);
      work_distributer.insert_task(3, 3.0);
      work_distributer.insert_task(6, 6.0);
      work_distributer.insert_task(2, 2.0);
      work_distributer.insert_task(4, 5.0);
      work_distributer.insert_task(5, 4.0);
      auto result = work_distributer.finish_remaining_tasks();
      EXPECT_EQ(result, (std::vector<int>{49, 36, 16, 25, 9, 4, 1}));
    }
  }
}

TEST(DynamicDistribution, Statistics) {
  using Task = int;
  using Result = int;
  auto worker_task = [](Task task) -> Result { return task * task; };
  {
    dynampi::MPIDynamicWorkDistributor<Task, Result,
                                       dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>
        work_distributer(worker_task);
    if (work_distributer.is_manager()) {
      work_distributer.insert_tasks({1, 2, 3, 4, 5});
      auto results = work_distributer.finish_remaining_tasks();
      size_t expected_size = 5;
      if (MPIEnvironment::world_comm_size() == 1) {
        expected_size = 0;
      }
      EXPECT_EQ(results, (std::vector<int>{1, 4, 9, 16, 25}));
      EXPECT_EQ(work_distributer.get_statistics().comm_statistics.send_count, expected_size);
      EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_sent,
                expected_size * sizeof(int));
      EXPECT_EQ(work_distributer.get_statistics().comm_statistics.recv_count,
                expected_size + MPIEnvironment::world_comm_size() - 1);
      EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_received,
                expected_size * sizeof(int));
      work_distributer.finalize();
      EXPECT_EQ(work_distributer.get_statistics().comm_statistics.send_count,
                expected_size + MPIEnvironment::world_comm_size() - 1);
      EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_sent,
                expected_size * sizeof(int));
      double expected_num_bytes = 0;
      if (MPIEnvironment::world_comm_size() > 1) {
        expected_num_bytes = static_cast<double>(expected_size * sizeof(int)) /
                             (expected_size + MPIEnvironment::world_comm_size() - 1);
      }
      EXPECT_DOUBLE_EQ(work_distributer.get_statistics().comm_statistics.average_receive_size(),
                       expected_num_bytes);
      EXPECT_DOUBLE_EQ(work_distributer.get_statistics().comm_statistics.average_send_size(),
                       expected_num_bytes);
    }
  }
}

TEST(MPICommunicatorWrapper, RankAndSizeCastGet) {
  dynampi::MPICommunicator<> comm(MPI_COMM_WORLD);
  int r1 = comm.rank();
  int s1 = comm.size();
  int r2, s2;
  MPI_Comm_rank(MPI_COMM_WORLD, &r2);
  MPI_Comm_size(MPI_COMM_WORLD, &s2);
  EXPECT_EQ(r1, r2);
  EXPECT_EQ(s1, s2);

  MPI_Comm raw = comm;  // test operator MPI_Comm
  EXPECT_NE(raw, MPI_COMM_NULL);
  EXPECT_EQ(comm.get(), raw);

  MPI_Barrier(comm);  // ensure implicit cast works in MPI call
}

TEST(MPICommunicatorWrapper, SendRecvAndStatistics) {
  using TrackedComm =
      dynampi::MPICommunicator<dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>;
  TrackedComm comm(MPI_COMM_WORLD);
  int rank = comm.rank();
  int size = comm.size();
  constexpr int tag = 7;
  const int value = 1234;

  if (size >= 2) {
    if (rank == 0) {
      comm.send(value, 1, tag);
    } else if (rank == 1) {
      int received = 0;
      comm.recv(received, 0, tag);
      EXPECT_EQ(received, value);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto &stats = comm.get_statistics();
    if (rank == 0) {
      EXPECT_EQ(stats.send_count, 1);
      EXPECT_EQ(stats.bytes_sent, sizeof(int));
      EXPECT_EQ(stats.recv_count, 0);
    } else if (rank == 1) {
      EXPECT_EQ(stats.recv_count, 1);
      EXPECT_EQ(stats.bytes_received, sizeof(int));
      EXPECT_EQ(stats.send_count, 0);
    } else {
      EXPECT_EQ(stats.send_count, 0);
      EXPECT_EQ(stats.recv_count, 0);
    }
  } else {
    const auto &stats = comm.get_statistics();
    EXPECT_EQ(stats.send_count, 0);
    EXPECT_EQ(stats.recv_count, 0);
  }
}

TEST(MPICommunicatorWrapper, BroadcastVector) {
  dynampi::MPICommunicator<> comm(MPI_COMM_WORLD);
  int rank = comm.rank();

  int n = 6;
  if (rank != 0) n = 0;
  comm.broadcast(n, 0);

  std::vector<int> vec;
  if (rank == 0) {
    vec.resize(n);
    for (int i = 0; i < n; ++i) vec[i] = i * i;
  } else {
    vec.resize(n);
  }
  comm.broadcast(vec, 0);

  EXPECT_EQ(vec.size(), n);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(vec[i], i * i);
  }
}

TEST(MPICommunicatorWrapper, SplitByColor) {
  dynampi::MPICommunicator<> comm(MPI_COMM_WORLD);
  int world_rank = comm.rank();
  int world_size = comm.size();
  int color = world_rank % 2;

  auto sub = comm.split(color);
  ASSERT_TRUE(sub.has_value());
  int sub_size = sub->size();
  int expected = (color == 0) ? (world_size + 1) / 2 : world_size / 2;
  EXPECT_EQ(sub_size, expected);

  int sub_rank = sub->rank();
  EXPECT_EQ(sub_rank, world_rank / 2);
}

TEST(MPICommunicatorWrapper, SplitUndefinedColor) {
  dynampi::MPICommunicator<> comm(MPI_COMM_WORLD);
  int world_rank = comm.rank();
  int color = (world_rank == 0) ? 0 : MPI_UNDEFINED;

  auto sub = comm.split(color);
  if (world_rank == 0) {
    ASSERT_TRUE(sub.has_value());
    EXPECT_EQ(sub->size(), 1);
    EXPECT_EQ(sub->rank(), 0);
  } else {
    EXPECT_FALSE(sub.has_value());
  }
}

TEST(MPICommunicatorWrapper, SplitByNode) {
  dynampi::MPICommunicator<> comm(MPI_COMM_WORLD);
  auto node = comm.split_by_node();
  int node_size = node.size();
  EXPECT_GE(node_size, 1);

  std::string node_name;
  node_name.resize(MPI_MAX_PROCESSOR_NAME);
  int size;
  MPI_Get_processor_name(node_name.data(), &size);
  node_name.resize(size);

  std::string root_node_name;
  if (node.rank() == 0) {
    root_node_name = node_name;
  }
  node.broadcast(root_node_name, 0);
  EXPECT_EQ(root_node_name, node_name);
}

TEST(MPICommunicatorWrapper, RecvEmptyMessage) {
  using TrackedComm =
      dynampi::MPICommunicator<dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>;
  TrackedComm comm(MPI_COMM_WORLD, TrackedComm::Reference);
  int rank = comm.rank();
  int size = comm.size();
  constexpr int tag = 77;

  if (size >= 2) {
    if (rank == 0) {
      MPI_Send(nullptr, 0, MPI_PACKED, 1, tag, MPI_COMM_WORLD);
    } else if (rank == 1) {
      comm.recv_empty_message(0, tag);
      const auto &stats = comm.get_statistics();
      EXPECT_EQ(stats.recv_count, 1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
