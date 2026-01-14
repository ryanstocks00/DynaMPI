/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <dynampi/dynampi.hpp>
#include <vector>

#include "dynampi/mpi/mpi_communicator.hpp"

TEST(MPI, ErrorCheck) {
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  EXPECT_THROW(DYNAMPI_MPI_CHECK(MPI_Comm_rank, (MPI_COMM_NULL, nullptr)), std::runtime_error);
  try {
    DYNAMPI_MPI_CHECK(MPI_Comm_rank, (MPI_COMM_NULL, nullptr));
  } catch (const std::runtime_error &e) {
    EXPECT_TRUE(std::string(e.what()).find("MPI error in MPI_Comm_rank") != std::string::npos);
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
