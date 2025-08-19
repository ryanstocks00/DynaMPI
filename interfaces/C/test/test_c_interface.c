/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dynampi.h"

// Simple worker function for testing
void* test_worker_function(void* task) {
  int64_t task_value = *(int64_t*)task;
  printf("Processing task: %ld\n", task_value);

  // Allocate result (just return the task value as a pointer)
  int64_t* result = malloc(sizeof(int64_t));
  *result = task_value * 2;  // Simple transformation
  return result;
}

// Cleanup function for results
void result_cleanup(void* result) {
  if (result) {
    free(result);
  }
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

  // Test manager-worker distribution - this is the main test that should work with multiple ranks
  // ALL processes must call this function for proper coordination
  printf("Process %d: Running manager-worker distribution test\n", rank);

  void** results;
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
        int64_t* result = (int64_t*)results[i];
        printf("Process %d: Result[%zu] = %ld\n", rank, i, *result);
      }

      // Clean up results
      for (size_t i = 0; i < result_count; i++) {
        result_cleanup(results[i]);
      }
      free(results);
    } else {
      // Worker ranks don't get results
      printf("Process %d: Completed as worker\n", rank);
    }
  } else {
    printf("Process %d: Manager-worker distribution failed\n", rank);
    const char* error = dynampi_get_last_error();
    if (error && *error) {
      printf("Process %d: Error: %s\n", rank, error);
    }
  }

  // Synchronize all processes before proceeding
  MPI_Barrier(MPI_COMM_WORLD);

  // Test work distributor creation (ALL processes must create distributors for coordination)
  printf("Process %d: Testing work distributor creation\n", rank);

  dynampi_work_distributor_t* distributor =
      dynampi_create_work_distributor(test_worker_function, &config);

  if (distributor) {
    printf("Process %d: Successfully created work distributor\n", rank);

    // Test basic functions
    if (dynampi_is_manager(distributor)) {
      printf("Process %d: Correctly identified as manager\n", rank);

      // Insert some tasks (only manager can do this)
      dynampi_insert_task(distributor, 100);
      dynampi_insert_task(distributor, 200);
      dynampi_insert_task_with_priority(distributor, 300, 1.5);

      printf("Process %d: Inserted tasks, remaining: %zu\n", rank,
             dynampi_remaining_tasks_count(distributor));
    } else {
      printf("Process %d: Correctly identified as worker\n", rank);
    }

    // ALL processes must call finish_remaining_tasks for proper coordination
    printf("Process %d: Coordinating work distribution\n", rank);
    size_t result_count;
    void** results = dynampi_finish_remaining_tasks(distributor, &result_count);

    if (results) {
      // Manager gets results
      printf("Process %d: Received %zu results from work distributor\n", rank, result_count);
      // Clean up results
      for (size_t i = 0; i < result_count; i++) {
        result_cleanup(results[i]);
      }
      free(results);
    } else {
      // Workers don't get results
      printf("Process %d: Completed work distribution coordination\n", rank);
    }

    // Clean up
    dynampi_destroy_work_distributor(distributor);
    printf("Process %d: Work distributor destroyed\n", rank);
  } else {
    printf("Process %d: Failed to create work distributor\n", rank);
    const char* error = dynampi_get_last_error();
    if (error && *error) {
      printf("Process %d: Error: %s\n", rank, error);
    }
  }

  // Final synchronization
  MPI_Barrier(MPI_COMM_WORLD);

  printf("Process %d: C interface test completed\n", rank);

  MPI_Finalize();
  return 0;
}
