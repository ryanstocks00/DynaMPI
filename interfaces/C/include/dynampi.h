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

// Function pointer types for C compatibility
typedef void* (*dynampi_worker_function_t)(void* task);

// Create and destroy work distributor
dynampi_work_distributor_t* dynampi_create_work_distributor(
    dynampi_worker_function_t worker_function, const dynampi_config_t* config);

void dynampi_destroy_work_distributor(dynampi_work_distributor_t* distributor);

// Configuration helpers
dynampi_config_t dynampi_default_config(void);
void dynampi_set_comm(dynampi_config_t* config, MPI_Comm comm);
void dynampi_set_manager_rank(dynampi_config_t* config, int rank);
void dynampi_set_auto_run_workers(dynampi_config_t* config, int auto_run);

// Work distribution functions
int dynampi_is_manager(const dynampi_work_distributor_t* distributor);
size_t dynampi_remaining_tasks_count(const dynampi_work_distributor_t* distributor);
void dynampi_insert_task(dynampi_work_distributor_t* distributor, int64_t task);
void dynampi_insert_task_with_priority(dynampi_work_distributor_t* distributor, int64_t task,
                                       double priority);
void dynampi_insert_tasks_array(dynampi_work_distributor_t* distributor, const int64_t* tasks,
                                size_t count);
void dynampi_run_worker(dynampi_work_distributor_t* distributor);

// Task execution and result collection
void** dynampi_finish_remaining_tasks(dynampi_work_distributor_t* distributor,
                                      size_t* result_count);

// Utility functions
int dynampi_manager_worker_distribution(size_t n_tasks, dynampi_worker_function_t worker_function,
                                        void*** results, size_t* result_count, MPI_Comm comm,
                                        int manager_rank);

// Error handling
const char* dynampi_get_last_error(void);
void dynampi_clear_last_error(void);

#ifdef __cplusplus
}
#endif

#endif  // DYNAMPI_C_H
