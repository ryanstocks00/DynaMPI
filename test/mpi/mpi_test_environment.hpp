/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

// Global test environment for MPI tests
class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { MPI_Init(nullptr, nullptr); }

  void TearDown() override { MPI_Finalize(); }

  static int world_comm_size() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  static int world_comm_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
};
