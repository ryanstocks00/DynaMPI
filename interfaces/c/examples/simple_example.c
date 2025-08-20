/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 *
 * Simple example demonstrating the DynaMPI C interface
 */

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dynampi.h"

// Worker function that processes a task (bytes in -> bytes out)
static void example_worker(const unsigned char* in_data, size_t in_size, unsigned char** out_data,
                           size_t* out_size) {
  assert(in_size == sizeof(int64_t));
  (void)in_size;
  int64_t x = *(int64_t*)(in_data);
  int64_t y = x * x;  // square
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

  printf("Process %d/%d: Starting DynaMPI C interface example\n", rank, size);

  // Create configuration
  dynampi_config_t config = dynampi_default_config();

  // All ranks must create the distributor for coordination
  dynampi_work_distributor_t* distributor =
      dynampi_create_work_distributor(example_worker, &config);
  if (!distributor) {
    printf("Process %d: Failed to create work distributor\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (dynampi_is_manager(distributor)) {
    printf("Process %d: Running as manager\n", rank);

    // Insert tasks (payload: int64 values 1..20)
    for (int64_t i = 1; i <= 20; i++) {
      dynampi_insert_task(distributor, (unsigned char*)&i, sizeof(int64_t));
    }
    printf("Process %d: Inserted 20 tasks\n", rank);

    // Collect results
    size_t result_count = 0;
    dynampi_buffer_t* results = dynampi_finish_remaining_tasks(distributor, &result_count);
    if (results) {
      printf("Process %d: Completed %zu tasks\n", rank, result_count);
      for (size_t i = 0; i < result_count && i < 5; i++) {
        int64_t val = 0;
        if (results[i].size == sizeof(val)) memcpy(&val, results[i].data, sizeof(val));
        printf("Process %d: Task %zu result = %ld\n", rank, i, (long)val);
      }
      if (result_count > 5) {
        printf("Process %d: ... and %zu more results\n", rank, result_count - 5);
      }
      for (size_t i = 0; i < result_count; i++) free(results[i].data);
      free(results);
    } else {
      printf("Process %d: No results received (workers only)\n", rank);
    }
  } else {
    printf("Process %d: Running as worker\n", rank);
    // dynampi_run_worker(distributor);
  }

  // Clean up
  dynampi_destroy_work_distributor(distributor);

  printf("Process %d: Example completed\n", rank);

  MPI_Finalize();
  return 0;
}
