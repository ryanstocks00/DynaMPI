/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#ifndef DYNAMPI_C_H
#define DYNAMPI_C_H

#include <mpi.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version information
extern const char* dynampi_version_string(void);
extern int dynampi_version_major(void);
extern int dynampi_version_minor(void);
extern int dynampi_version_patch(void);
extern const char* dynampi_commit_hash(void);
extern const char* dynampi_compile_date(void);

// Work distributor handle
typedef struct dynampi_work_distributor dynampi_work_distributor_t;

// Configuration structure
typedef struct {
  MPI_Comm comm;
  int manager_rank;
  int auto_run_workers;
} dynampi_config_t;

// Worker: arbitrary bytes in -> bytes out. The implementation must allocate out_data (malloc) and
// set size
typedef void (*dynampi_worker_t)(const unsigned char* in_data, size_t in_size,
                                 unsigned char** out_data, size_t* out_size);

// Create and destroy work distributor
dynampi_work_distributor_t* dynampi_create_work_distributor(dynampi_worker_t worker_function,
                                                            const dynampi_config_t* config);

void dynampi_destroy_work_distributor(dynampi_work_distributor_t* distributor);

// Configuration helpers
dynampi_config_t dynampi_default_config(void);

// Work distribution functions
int dynampi_is_manager(const dynampi_work_distributor_t* distributor);
void dynampi_run_worker(dynampi_work_distributor_t* distributor);

// Submit a task (arbitrary bytes)
void dynampi_insert_task(dynampi_work_distributor_t* distributor, const unsigned char* data,
                         size_t size);

// Task execution and result collection (manager only)
typedef struct {
  unsigned char* data;
  size_t size;
} dynampi_buffer_t;

dynampi_buffer_t* dynampi_finish_remaining_tasks(dynampi_work_distributor_t* distributor,
                                                 size_t* result_count);

// Utility function: run a simple manager-worker pipeline for n_tasks where the input is the task
// index
int dynampi_manager_worker_distribution(size_t n_tasks, dynampi_worker_t worker_function,
                                        dynampi_buffer_t** results, size_t* result_count,
                                        MPI_Comm comm, int manager_rank);

#ifdef __cplusplus
}
#endif

#endif  // DYNAMPI_C_H
