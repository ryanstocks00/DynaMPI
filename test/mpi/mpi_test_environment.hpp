/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>
#include <mpi.h>

#include <dynampi/utilities/debug_log.hpp>

// Global test environment for MPI tests
class MPIEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    MPI_Init(nullptr, nullptr);
    // Set error handler to return errors instead of aborting
    // This allows us to safely check if MPI operations succeed
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank = 0;
    int result = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (result == MPI_SUCCESS) {
      dynampi::get_debug_log() << "[TEST ENV] SetUp: MPI initialized on rank " << rank << std::endl;
    }
  }

  void TearDown() override {
    int rank = 0;
    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
      int result = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      if (result == MPI_SUCCESS) {
        dynampi::get_debug_log() << "[TEST ENV] TearDown: About to call MPI_Finalize on rank "
                                 << rank << std::endl;
        dynampi::get_debug_log().flush();
      }
      MPI_Finalize();
      // Note: Can't log after MPI_Finalize as MPI is no longer available
    }
  }

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
