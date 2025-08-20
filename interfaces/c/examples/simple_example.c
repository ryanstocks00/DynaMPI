/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Simple example demonstrating the DynaMPI C interface
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "dynampi.h"

// Worker function that processes a task
void* process_task(void* task) {
  int64_t task_value = *(int64_t*)task;

  // Simulate some work
  int64_t result = task_value * task_value;  // Square the task value

  // Allocate result
  int64_t* result_ptr = malloc(sizeof(int64_t));
  *result_ptr = result;

  return result_ptr;
}

int main(int argc, char* argv[]) {
  int rank, size;

  // Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  printf("Process %d/%d: Starting DynaMPI C interface example\n", rank, size);

  // Create configuration
  dynampi_config_t config = dynampi_default_config();
  config.manager_rank = 0;
  config.auto_run_workers = 1;

  if (rank == 0) {
    // Manager process
    printf("Process %d: Running as manager\n", rank);

    // Create work distributor
    dynampi_work_distributor_t* distributor =
        dynampi_create_work_distributor(process_task, &config);

    if (!distributor) {
      printf("Process %d: Failed to create work distributor: %s\n", rank, dynampi_get_last_error());
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Insert tasks
    for (int64_t i = 1; i <= 20; i++) {
      dynampi_insert_task(distributor, i);
    }

    printf("Process %d: Inserted 20 tasks\n", rank);

    // Wait for all tasks to complete
    size_t result_count;
    void** results = dynampi_finish_remaining_tasks(distributor, &result_count);

    if (results) {
      printf("Process %d: Completed %zu tasks\n", rank, result_count);

      // Print some results
      for (size_t i = 0; i < result_count && i < 5; i++) {
        int64_t* result = (int64_t*)results[i];
        printf("Process %d: Task %zu result = %ld\n", rank, i, *result);
      }

      if (result_count > 5) {
        printf("Process %d: ... and %zu more results\n", rank, result_count - 5);
      }

      // Clean up results
      for (size_t i = 0; i < result_count; i++) {
        free(results[i]);
      }
      free(results);
    } else {
      printf("Process %d: No results received\n", rank);
    }

    // Clean up
    dynampi_destroy_work_distributor(distributor);

  } else {
    // Worker processes will be handled automatically
    printf("Process %d: Running as worker\n", rank);
  }

  printf("Process %d: Example completed\n", rank);

  MPI_Finalize();
  return 0;
}
