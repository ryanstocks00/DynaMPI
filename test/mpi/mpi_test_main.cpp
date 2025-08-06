#include "mpi_test_environment.hpp"
#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Register the MPI environment
  ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

  return RUN_ALL_TESTS();
}
