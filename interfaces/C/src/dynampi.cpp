/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include "dynampi.h"

#include <dynampi/dynampi.hpp>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Global error handling
static thread_local std::string last_error;

void set_error(const std::string& error) { last_error = error; }

// C++ wrapper class to manage the work distributor
class CWorkDistributor {
 public:
  using WorkerFunction = std::function<int64_t(int64_t)>;

  CWorkDistributor(WorkerFunction worker_func, const dynampi_config_t& config)
      : worker_function_(worker_func), config_(config) {
    // Create the C++ work distributor
    dynampi::NaiveMPIWorkDistributor<int64_t, int64_t>::Config cpp_config;
    cpp_config.comm = config.comm;
    cpp_config.manager_rank = config.manager_rank;
    cpp_config.auto_run_workers = false;  // Disable auto-run to avoid deadlock

    // Create a wrapper function that converts between types
    auto wrapper_func = [this](int64_t task) -> int64_t { return this->worker_function_(task); };

    distributor_ = std::make_unique<dynampi::NaiveMPIWorkDistributor<int64_t, int64_t>>(
        wrapper_func, cpp_config);
  }

  // Method to run worker if this is a worker rank
  void run_worker_if_needed() {
    if (!is_manager()) {
      run_worker();
    }
  }

  bool is_manager() const { return distributor_->is_manager(); }

  size_t remaining_tasks_count() const { return distributor_->remaining_tasks_count(); }

  void insert_task(int64_t task) { distributor_->insert_task(task); }

  void insert_task_with_priority(int64_t task, double priority) {
    distributor_->insert_task(task, priority);
  }

  void insert_tasks(const std::vector<int64_t>& tasks) { distributor_->insert_tasks(tasks); }

  void run_worker() { distributor_->run_worker(); }

  std::vector<int64_t> finish_remaining_tasks() {
    // For non-manager ranks, run the worker loop
    if (!is_manager()) {
      printf("Worker %d: Running worker loop for work distributor\n",
             config_.comm == MPI_COMM_WORLD ? 0 : 1);
      run_worker();
      printf("Worker %d: Completed worker loop for work distributor\n",
             config_.comm == MPI_COMM_WORLD ? 0 : 1);
      return {};  // Workers don't return results
    }
    return distributor_->finish_remaining_tasks();
  }

  ~CWorkDistributor() = default;

 private:
  WorkerFunction worker_function_;
  dynampi_config_t config_;
  std::unique_ptr<dynampi::NaiveMPIWorkDistributor<int64_t, int64_t>> distributor_;
};

// C interface implementations

extern "C" {

const char* dynampi_version_string(void) {
  try {
    static std::string version = std::string(dynampi::version::string);
    return version.c_str();
  } catch (...) {
    set_error("Failed to get version string");
    return "unknown";
  }
}

int dynampi_version_major(void) {
  try {
    return dynampi::version::major;
  } catch (...) {
    set_error("Failed to get major version");
    return -1;
  }
}

int dynampi_version_minor(void) {
  try {
    return dynampi::version::minor;
  } catch (...) {
    set_error("Failed to get minor version");
    return -1;
  }
}

int dynampi_version_patch(void) {
  try {
    return dynampi::version::patch;
  } catch (...) {
    set_error("Failed to get patch version");
    return -1;
  }
}

const char* dynampi_commit_hash(void) {
  try {
    static std::string hash = std::string(dynampi::version::commit_hash());
    return hash.c_str();
  } catch (...) {
    set_error("Failed to get commit hash");
    return "unknown";
  }
}

const char* dynampi_compile_date(void) {
  try {
    static std::string date = std::string(dynampi::version::compile_date());
    return date.c_str();
  } catch (...) {
    set_error("Failed to get compile date");
    return "unknown";
  }
}

dynampi_config_t dynampi_default_config(void) {
  dynampi_config_t config;
  config.comm = MPI_COMM_WORLD;
  config.manager_rank = 0;
  config.auto_run_workers = 1;
  return config;
}

void dynampi_set_comm(dynampi_config_t* config, MPI_Comm comm) {
  if (config) {
    config->comm = comm;
  }
}

void dynampi_set_manager_rank(dynampi_config_t* config, int rank) {
  if (config) {
    config->manager_rank = rank;
  }
}

void dynampi_set_auto_run_workers(dynampi_config_t* config, int auto_run) {
  if (config) {
    config->auto_run_workers = auto_run;
  }
}

dynampi_work_distributor_t* dynampi_create_work_distributor(
    dynampi_worker_function_t worker_function, const dynampi_config_t* config) {
  try {
    if (!worker_function) {
      set_error("Worker function cannot be null");
      return nullptr;
    }

    dynampi_config_t default_config = dynampi_default_config();
    if (config) {
      default_config = *config;
    }

    // Convert C function pointer to C++ function object
    auto worker_func = [worker_function](int64_t task) -> int64_t {
      return *(int64_t*)worker_function(&task);
    };

    auto* c_distributor = new CWorkDistributor(worker_func, default_config);

    return reinterpret_cast<dynampi_work_distributor_t*>(c_distributor);

  } catch (const std::exception& e) {
    set_error(std::string("Failed to create work distributor: ") + e.what());
    return nullptr;
  } catch (...) {
    set_error("Unknown error creating work distributor");
    return nullptr;
  }
}

void dynampi_destroy_work_distributor(dynampi_work_distributor_t* distributor) {
  try {
    if (distributor) {
      auto* c_distributor = reinterpret_cast<CWorkDistributor*>(distributor);
      delete c_distributor;
    }
  } catch (...) {
    set_error("Error destroying work distributor");
  }
}

int dynampi_is_manager(const dynampi_work_distributor_t* distributor) {
  try {
    if (!distributor) {
      set_error("Distributor cannot be null");
      return 0;
    }

    const auto* c_distributor = reinterpret_cast<const CWorkDistributor*>(distributor);
    return c_distributor->is_manager() ? 1 : 0;

  } catch (const std::exception& e) {
    set_error(std::string("Failed to check manager status: ") + e.what());
    return 0;
  } catch (...) {
    set_error("Unknown error checking manager status");
    return 0;
  }
}

size_t dynampi_remaining_tasks_count(const dynampi_work_distributor_t* distributor) {
  try {
    if (!distributor) {
      set_error("Distributor cannot be null");
      return 0;
    }

    const auto* c_distributor = reinterpret_cast<const CWorkDistributor*>(distributor);
    return c_distributor->remaining_tasks_count();

  } catch (const std::exception& e) {
    set_error(std::string("Failed to get remaining tasks count: ") + e.what());
    return 0;
  } catch (...) {
    set_error("Unknown error getting remaining tasks count");
    return 0;
  }
}

void dynampi_insert_task(dynampi_work_distributor_t* distributor, int64_t task) {
  try {
    if (!distributor) {
      set_error("Distributor cannot be null");
      return;
    }

    auto* c_distributor = reinterpret_cast<CWorkDistributor*>(distributor);
    c_distributor->insert_task(task);

  } catch (const std::exception& e) {
    set_error(std::string("Failed to insert task: ") + e.what());
  } catch (...) {
    set_error("Unknown error inserting task");
  }
}

void dynampi_insert_task_with_priority(dynampi_work_distributor_t* distributor, int64_t task,
                                       double priority) {
  try {
    if (!distributor) {
      set_error("Distributor cannot be null");
      return;
    }

    auto* c_distributor = reinterpret_cast<CWorkDistributor*>(distributor);
    c_distributor->insert_task_with_priority(task, priority);

  } catch (const std::exception& e) {
    set_error(std::string("Failed to insert task with priority: ") + e.what());
  } catch (...) {
    set_error("Unknown error inserting task with priority");
  }
}

void dynampi_insert_tasks_array(dynampi_work_distributor_t* distributor, const int64_t* tasks,
                                size_t count) {
  try {
    if (!distributor) {
      set_error("Distributor cannot be null");
      return;
    }

    if (!tasks && count > 0) {
      set_error("Tasks array cannot be null when count > 0");
      return;
    }

    auto* c_distributor = reinterpret_cast<CWorkDistributor*>(distributor);
    std::vector<int64_t> task_vector(tasks, tasks + count);
    c_distributor->insert_tasks(task_vector);

  } catch (const std::exception& e) {
    set_error(std::string("Failed to insert tasks array: ") + e.what());
  } catch (...) {
    set_error("Unknown error inserting tasks array");
  }
}

void dynampi_run_worker(dynampi_work_distributor_t* distributor) {
  try {
    if (!distributor) {
      set_error("Distributor cannot be null");
      return;
    }

    auto* c_distributor = reinterpret_cast<CWorkDistributor*>(distributor);
    c_distributor->run_worker();

  } catch (const std::exception& e) {
    set_error(std::string("Failed to run worker: ") + e.what());
  } catch (...) {
    set_error("Unknown error running worker");
  }
}

void** dynampi_finish_remaining_tasks(dynampi_work_distributor_t* distributor,
                                      size_t* result_count) {
  try {
    if (!distributor) {
      set_error("Distributor cannot be null");
      if (result_count) *result_count = 0;
      return nullptr;
    }

    if (!result_count) {
      set_error("Result count pointer cannot be null");
      return nullptr;
    }

    auto* c_distributor = reinterpret_cast<CWorkDistributor*>(distributor);

    // For non-manager ranks, run the worker loop
    if (!c_distributor->is_manager()) {
      c_distributor->run_worker_if_needed();
      *result_count = 0;
      return nullptr;
    }

    std::vector<int64_t> results = c_distributor->finish_remaining_tasks();

    *result_count = results.size();

    if (results.empty()) {
      return nullptr;
    }

    // Allocate memory for the results array
    void** result_array = static_cast<void**>(malloc(results.size() * sizeof(void*)));
    if (!result_array) {
      set_error("Failed to allocate memory for results");
      *result_count = 0;
      return nullptr;
    }

    // Copy results to the allocated array as pointers
    for (size_t i = 0; i < results.size(); i++) {
      int64_t* result_ptr = static_cast<int64_t*>(malloc(sizeof(int64_t)));
      if (result_ptr) {
        *result_ptr = results[i];
        result_array[i] = result_ptr;
      } else {
        set_error("Failed to allocate memory for individual result");
        // Clean up already allocated results
        for (size_t j = 0; j < i; j++) {
          free(result_array[j]);
        }
        free(result_array);
        *result_count = 0;
        return nullptr;
      }
    }

    return result_array;

  } catch (const std::exception& e) {
    set_error(std::string("Failed to finish remaining tasks: ") + e.what());
    if (result_count) *result_count = 0;
    return nullptr;
  } catch (...) {
    set_error("Unknown error finishing remaining tasks");
    if (result_count) *result_count = 0;
    return nullptr;
  }
}

int dynampi_manager_worker_distribution(size_t n_tasks, dynampi_worker_function_t worker_function,
                                        void*** results, size_t* result_count, MPI_Comm comm,
                                        int manager_rank) {
  try {
    if (!worker_function) {
      set_error("Worker function cannot be null");
      return -1;
    }

    if (!results) {
      set_error("Results pointer cannot be null");
      return -1;
    }

    if (!result_count) {
      set_error("Result count pointer cannot be null");
      return -1;
    }

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == manager_rank) {
      // Manager process - distribute tasks and collect results
      printf("Manager: Starting work distribution with %zu tasks to %d workers\n", n_tasks,
             size - 1);

      std::vector<int64_t> all_results(n_tasks);

      if (size == 1) {
        // Single rank case - process all tasks locally
        printf("Manager: Single rank mode, processing tasks locally\n");
        for (size_t i = 0; i < n_tasks; i++) {
          int64_t result = *(int64_t*)worker_function(&i);
          all_results[i] = result;
          printf("Manager: Processed task %zu -> result %ld\n", i, result);
        }
        printf("Manager: All %zu tasks completed locally\n", n_tasks);
      } else {
        // Multiple ranks - distribute to workers
        size_t tasks_sent = 0;
        size_t results_received = 0;

        // Track which worker is processing which task
        std::vector<int64_t> worker_current_task(size, -1);

        // Send initial tasks to all workers
        for (int worker = 0; worker < size; worker++) {
          if (worker != manager_rank && tasks_sent < n_tasks) {
            MPI_Send(&tasks_sent, 1, MPI_INT64_T, worker, 100, comm);
            worker_current_task[worker] = tasks_sent;
            tasks_sent++;
          }
        }

        // Continue distributing tasks and collecting results
        while (results_received < n_tasks) {
          MPI_Status status;
          int64_t result;

          // Receive result from any worker
          MPI_Recv(&result, 1, MPI_INT64_T, MPI_ANY_SOURCE, 200, comm, &status);
          int worker_rank = status.MPI_SOURCE;
          results_received++;

          // Get the task index this worker was processing
          int64_t task_idx = worker_current_task[worker_rank];
          printf("Manager: Received result %ld from worker %d for task %ld (%zu/%zu)\n", result,
                 worker_rank, task_idx, results_received, n_tasks);

          // Store result at the correct task index
          all_results[task_idx] = result;

          // Send next task to this worker if available
          if (tasks_sent < n_tasks) {
            MPI_Send(&tasks_sent, 1, MPI_INT64_T, worker_rank, 100, comm);
            worker_current_task[worker_rank] = tasks_sent;
            tasks_sent++;
          } else {
            // Send termination signal
            MPI_Send(&tasks_sent, 1, MPI_INT64_T, worker_rank, 999, comm);
          }
        }

        printf("Manager: All %zu tasks completed via workers\n", n_tasks);
      }

      // Convert results to C format
      *result_count = all_results.size();

      if (all_results.empty()) {
        *results = nullptr;
        return 0;
      }

      // Allocate memory for the results array
      void** result_array = static_cast<void**>(malloc(all_results.size() * sizeof(void*)));
      if (!result_array) {
        set_error("Failed to allocate memory for results");
        return -1;
      }

      // Copy results to the allocated array as pointers
      for (size_t i = 0; i < all_results.size(); i++) {
        int64_t* result_ptr = static_cast<int64_t*>(malloc(sizeof(int64_t)));
        if (result_ptr) {
          *result_ptr = all_results[i];
          result_array[i] = result_ptr;
        } else {
          set_error("Failed to allocate memory for individual result");
          // Clean up already allocated results
          for (size_t j = 0; j < i; j++) {
            free(result_array[j]);
          }
          free(result_array);
          return -1;
        }
      }
      *results = result_array;

      return 0;

    } else {
      // Worker process - receive tasks and send results
      printf("Worker %d: Starting work processing\n", rank);

      while (true) {
        MPI_Status status;
        int64_t task_idx;

        // Receive task from manager
        MPI_Recv(&task_idx, 1, MPI_INT64_T, manager_rank, MPI_ANY_TAG, comm, &status);

        if (status.MPI_TAG == 999) {
          // Termination signal
          printf("Worker %d: Received termination signal\n", rank);
          break;
        }

        printf("Worker %d: Processing task %ld\n", rank, task_idx);

        // Process the task
        int64_t result = *(int64_t*)worker_function(&task_idx);

        // Send result back to manager
        MPI_Send(&result, 1, MPI_INT64_T, manager_rank, 200, comm);

        printf("Worker %d: Sent result %ld for task %ld\n", rank, result, task_idx);
      }

      printf("Worker %d: Completed work processing\n", rank);

      *results = nullptr;
      *result_count = 0;
      return 0;
    }

  } catch (const std::exception& e) {
    set_error(std::string("Failed to run manager-worker distribution: ") + e.what());
    return -1;
  } catch (...) {
    set_error("Unknown error running manager-worker distribution");
    return -1;
  }
}

const char* dynampi_get_last_error(void) { return last_error.c_str(); }

void dynampi_clear_last_error(void) { last_error.clear(); }

}  // extern "C"
