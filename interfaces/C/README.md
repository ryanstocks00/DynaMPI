<!--
  SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
  SPDX-License-Identifier: MIT
 -->

# DynaMPI C Interface

This directory contains the C interface for the DynaMPI library, providing C-compatible bindings for the C++ MPI work distribution functionality.

## Overview

The C interface allows C programs to use DynaMPI's dynamic work distribution capabilities without requiring C++ compilation. It provides:

- Work distributor creation and management
- Task insertion and execution
- Manager-worker pattern support
- Error handling and version information

## Building

The C interface is automatically built when building the main DynaMPI project:

```bash
mkdir build && cd build
cmake ..
make
```

This will create:
- `libdynampi_c.so` - The C interface shared library
- `c_interface_test` - A test executable

## Usage

### Basic Setup

```c
#include <mpi.h>
#include "dynampi.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    // Your code here

    MPI_Finalize();
    return 0;
}
```

### Creating a Work Distributor

```c
// Define a worker function
void* my_worker_function(void* task) {
    int64_t task_value = *(int64_t*)task;
    // Process the task...

    // Return result (allocate memory for it)
    int64_t* result = malloc(sizeof(int64_t));
    *result = task_value * 2;
    return result;
}

// Create configuration
dynampi_config_t config = dynampi_default_config();
config.manager_rank = 0;
config.auto_run_workers = 1;

// Create the work distributor
dynampi_work_distributor_t* distributor = dynampi_create_work_distributor(
    my_worker_function, &config);

if (!distributor) {
    const char* error = dynampi_get_last_error();
    printf("Failed to create distributor: %s\n", error);
    return 1;
}
```

### Adding Tasks

```c
// Add individual tasks
dynampi_insert_task(distributor, 100);
dynampi_insert_task(distributor, 200);

// Add task with priority
dynampi_insert_task_with_priority(distributor, 300, 1.5);

// Add multiple tasks
int64_t tasks[] = {400, 500, 600};
dynampi_insert_tasks_array(distributor, tasks, 3);
```

### Executing Tasks

```c
// Manager process
if (dynampi_is_manager(distributor)) {
    // Insert all tasks first
    // ... insert tasks ...

    // Execute all tasks and get results
    size_t result_count;
    void** results = dynampi_finish_remaining_tasks(distributor, &result_count);

    if (results) {
        // Process results
        for (size_t i = 0; i < result_count; i++) {
            int64_t* result = (int64_t*)results[i];
            printf("Result %zu: %ld\n", i, *result);
            free(result); // Clean up individual result
        }
        free(results); // Clean up results array
    }
}
```

### Using the Manager-Worker Pattern

```c
// Simple manager-worker distribution
void** results;
size_t result_count;

int ret = dynampi_manager_worker_distribution(
    100,                    // Number of tasks
    my_worker_function,     // Worker function
    &results,               // Output results array
    &result_count,          // Output result count
    MPI_COMM_WORLD,         // MPI communicator
    0                       // Manager rank
);

if (ret == 0 && results) {
    // Process results
    // ... handle results ...

    // Clean up
    for (size_t i = 0; i < result_count; i++) {
        free(results[i]);
    }
    free(results);
}
```

### Error Handling

```c
// Check for errors after any function call
if (some_dynampi_function() != expected_result) {
    const char* error = dynampi_get_last_error();
    if (error && *error) {
        printf("Error: %s\n", error);
    }

    // Clear the error for next operation
    dynampi_clear_last_error();
}
```

### Cleanup

```c
// Always clean up the work distributor
dynampi_destroy_work_distributor(distributor);
```

## API Reference

### Types

- `dynampi_work_distributor_t` - Opaque handle for work distributor
- `dynampi_config_t` - Configuration structure
- `dynampi_worker_function_t` - Function pointer type for worker functions

### Functions

#### Version Information
- `dynampi_version_string()` - Get version string
- `dynampi_version_major/minor/patch()` - Get version components
- `dynampi_commit_hash()` - Get git commit hash
- `dynampi_compile_date()` - Get compilation date

#### Configuration
- `dynampi_default_config()` - Get default configuration
- `dynampi_set_comm()` - Set MPI communicator
- `dynampi_set_manager_rank()` - Set manager rank
- `dynampi_set_auto_run_workers()` - Set auto-run workers flag

#### Work Distribution
- `dynampi_create_work_distributor()` - Create work distributor
- `dynampi_destroy_work_distributor()` - Destroy work distributor
- `dynampi_is_manager()` - Check if current process is manager
- `dynampi_remaining_tasks_count()` - Get count of remaining tasks

#### Task Management
- `dynampi_insert_task()` - Insert single task
- `dynampi_insert_task_with_priority()` - Insert task with priority
- `dynampi_insert_tasks_array()` - Insert multiple tasks
- `dynampi_run_worker()` - Run worker loop
- `dynampi_finish_remaining_tasks()` - Execute all remaining tasks

#### Utility
- `dynampi_manager_worker_distribution()` - Simple manager-worker pattern
- `dynampi_get_last_error()` - Get last error message
- `dynampi_clear_last_error()` - Clear last error

## Compilation

To compile a C program using the DynaMPI C interface:

```bash
mpicc -o my_program my_program.c -ldynampi_c -lmpi
```

Or with CMake:

```cmake
find_package(DynaMPIC REQUIRED)
target_link_libraries(my_program PRIVATE DynaMPI::dynampi_c)
```

## Notes

- The C interface uses `int64_t` for tasks and results internally
- Worker functions should return pointers to allocated memory
- Always check return values and handle errors appropriately
- Remember to free allocated memory for results
- The interface automatically handles MPI communication details

## Examples

See the `examples/` directory for complete working examples of the C interface usage.
