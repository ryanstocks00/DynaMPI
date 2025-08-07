/*
 * SPDX-FileCopyrightText: 2025 Ryan Stocks <ryan.stocks00@gmail.com>
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
};
