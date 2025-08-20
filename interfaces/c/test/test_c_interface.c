/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <inttypes.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dynampi.h"

// Bytes worker: doubles an int64 payload if present
static void test_worker_function(const unsigned char* in_data, size_t in_size,
                                 unsigned char** out_data, size_t* out_size) {
  int64_t x = 0;
  if (in_size == sizeof(int64_t)) memcpy(&x, in_data, sizeof(int64_t));
  int64_t y = x * 2;
  *out_size = sizeof(int64_t);
  *out_data = (unsigned char*)malloc(sizeof(int64_t));
  if (*out_data) *(int64_t*)(*out_data) = y;
}

int main(int argc, char* argv[]) {
  int rank, size;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("Process %d/%d: Starting DynaMPI C interface test\n", rank, size);

  // Test version functions
  printf("Process %d: DynaMPI version: %s\n", rank, dynampi_version_string());
  printf("Process %d: Major: %d, Minor: %d, Patch: %d\n", rank, dynampi_version_major(),
         dynampi_version_minor(), dynampi_version_patch());
  printf("Process %d: Commit hash: %s\n", rank, dynampi_commit_hash());
  printf("Process %d: Compile date: %s\n", rank, dynampi_compile_date());

  // Test configuration
  dynampi_config_t config = dynampi_default_config();
  printf("Process %d: Default config - comm: %p, manager_rank: %d, auto_run_workers: %d\n", rank,
         (void*)(uintptr_t)config.comm, config.manager_rank, config.auto_run_workers);

  // Test manager-worker distribution - all ranks participate; manager collects results
  // ALL processes must call this function for proper coordination
  printf("Process %d: Running manager-worker distribution test\n", rank);

  dynampi_buffer_t* results;
  size_t result_count;

  int ret = dynampi_manager_worker_distribution(10,  // 10 tasks
                                                test_worker_function, &results, &result_count,
                                                MPI_COMM_WORLD, 0);

  if (ret == 0) {
    if (results) {
      // Manager rank gets results
      printf("Process %d: Successfully processed %zu tasks\n", rank, result_count);

      // Print results
      for (size_t i = 0; i < result_count; i++) {
        int64_t val = 0;
        if (results[i].size == sizeof(int64_t)) memcpy(&val, results[i].data, sizeof(int64_t));
        printf("Process %d: Result[%zu] = %" PRId64 "\n", rank, i, val);
      }

      // Clean up results
      for (size_t i = 0; i < result_count; i++) {
        free(results[i].data);
      }
      free(results);
    } else {
      // Worker ranks don't get results
      printf("Process %d: Completed as worker\n", rank);
    }
  } else {
    printf("Process %d: Manager-worker distribution failed\n", rank);
  }

  // Synchronize all processes before proceeding
  MPI_Barrier(MPI_COMM_WORLD);

  // Test work distributor creation (ALL processes must create distributors for coordination)
  printf("Process %d: Testing work distributor creation\n", rank);

  dynampi_work_distributor_t* distributor =
      dynampi_create_work_distributor(test_worker_function, &config);

  printf("Process %d: Successfully created work distributor\n", rank);

  // Manager inserts a few tasks; workers just run
  if (dynampi_is_manager(distributor)) {
    int64_t vals[] = {100, 200, 300};
    for (size_t i = 0; i < 3; ++i) {
      dynampi_insert_task(distributor, (unsigned char*)&vals[i], sizeof(int64_t));
    }
  }

  // Manager collects results; workers run the worker loop
  if (dynampi_is_manager(distributor)) {
    size_t rc = 0;
    dynampi_buffer_t* bufs = dynampi_finish_remaining_tasks(distributor, &rc);
    if (bufs) {
      for (size_t i = 0; i < rc; ++i) free(bufs[i].data);
      free(bufs);
    }
  } else if (!config.auto_run_workers) {
    dynampi_run_worker(distributor);
  }

  // Clean up
  dynampi_destroy_work_distributor(distributor);
  printf("Process %d: Work distributor destroyed\n", rank);

  // Final synchronization
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}
