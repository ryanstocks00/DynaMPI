/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include "mpi_test_environment.hpp"

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Register the MPI environment
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

  return RUN_ALL_TESTS();
}
