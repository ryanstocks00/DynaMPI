/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <ranges>
#include <span>
#include <stack>
#include <string_view>
#include <tuple>
#include <vector>

#include "mpi/mpi_communicator.hpp"
#include "mpi/mpi_types.hpp"

namespace dynampi {

namespace version {

inline constexpr int major = DYNAMPI_VERSION_MAJOR;
inline constexpr int minor = DYNAMPI_VERSION_MINOR;
inline constexpr int patch = DYNAMPI_VERSION_PATCH;

// Macros for compile-time version string
#define DYNAMPI_STR_HELPER(x) #x
#define DYNAMPI_STR(x) DYNAMPI_STR_HELPER(x)
#define DYNAMPI_VERSION_STRING                                                                   \
  "v" DYNAMPI_STR(DYNAMPI_VERSION_MAJOR) "." DYNAMPI_STR(DYNAMPI_VERSION_MINOR) "." DYNAMPI_STR( \
      DYNAMPI_VERSION_PATCH)

inline constexpr std::string_view string = DYNAMPI_VERSION_STRING;

[[nodiscard]] constexpr bool is_at_least(int v_major, int v_minor, int v_patch) {
  return std::tie(major, minor, patch) >= std::tie(v_major, v_minor, v_patch);
}

[[nodiscard]] inline constexpr std::string_view compile_date() { return __DATE__ " " __TIME__; }

[[nodiscard]] inline constexpr std::string_view commit_hash() { return DYNAMPI_COMMIT_HASH; }

}  // namespace version

template <typename TaskT, typename ResultT>
class NaiveMPIWorkDistributor {
  MPICommunicator _communicator;
  std::function<ResultT(TaskT)> _worker_function;
  int _manager_rank;
  std::queue<TaskT> _task_queue;
  std::vector<int64_t> _worker_task_indices;
  std::vector<ResultT> _results;
  std::stack<size_t, std::vector<size_t>> _free_workers;

  size_t _tasks_sent = 0;
  size_t _results_received = 0;

  static constexpr int TASK_TAG = 0;
  static constexpr int DONE_TAG = 1;
  static constexpr int RESULT_TAG = 2;
  static constexpr int REQUEST_TAG = 3;

  using _task_type = MPI_Type<TaskT>;
  using _result_type = MPI_Type<ResultT>;

 public:
  NaiveMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function, MPI_Comm comm = MPI_COMM_WORLD,
                          int manager_rank = 0)
      : _communicator(comm, MPICommunicator::Owned),
        _worker_function(worker_function),
        _manager_rank(manager_rank) {
    if (is_manager()) _worker_task_indices.resize(_communicator.size() - 1, -1);
  }

  void run_worker() {
    assert(_communicator.rank() != _manager_rank && "Worker cannot run on the manager rank");
    MPI_Send(nullptr, 0, _task_type::value, _manager_rank, REQUEST_TAG, _communicator.get());
    while (true) {
      MPI_Status status;
      DYNAMPI_MPI_CHECK(MPI_Probe, (MPI_ANY_SOURCE, MPI_ANY_TAG, _communicator.get(), &status));
      if (status.MPI_TAG == DONE_TAG) {
        MPI_Recv(nullptr, 0, _task_type::value, _manager_rank, MPI_ANY_TAG, _communicator.get(), &status);
        break;
      }
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, _task_type::value, &count));
      TaskT message;
      _task_type::resize(message, count);
      MPI_Recv(_task_type::ptr(message), 1, _task_type::value, _manager_rank, MPI_ANY_TAG, _communicator.get(), &status);
      _tasks_sent++;
      assert(status.MPI_TAG == TASK_TAG && "Unexpected tag received in worker");
      ResultT result = _worker_function(message);
      MPI_Send(_result_type::ptr(result), _result_type::count(result), _result_type::value, _manager_rank, RESULT_TAG, _communicator.get());
      _results_received++;
    }
  }

  bool is_manager() const { return _communicator.rank() == _manager_rank; }

  size_t remaining_tasks_count() const {
    assert(_communicator.rank() == _manager_rank && "Only the manager can check remaining tasks");
    return _task_queue.size();
  }

  void insert_task(TaskT task) {
    assert(_communicator.rank() == _manager_rank && "Only the manager can distribute tasks");
    _task_queue.push(task);
  }

  void insert_tasks(std::span<const TaskT> tasks) {
    assert(_communicator.rank() == _manager_rank && "Only the manager can distribute tasks");
    for (const auto& task : tasks) {
      _task_queue.push(task);
    }
  }
  void insert_tasks(const std::vector<TaskT>& tasks) {
    insert_tasks(std::span<const TaskT>(tasks));
  }

  template <std::ranges::input_range Range>
  void insert_tasks(const Range& tasks) {
    assert(_communicator.rank() == _manager_rank && "Only the manager can distribute tasks");
    for (const auto& task : tasks) {
      _task_queue.push(task);
    }
  }

  void receive_from_any_worker() {
    assert(_communicator.rank() == _manager_rank &&
           "Only the manager can receive results and send tasks");
    MPI_Status status;
    DYNAMPI_MPI_CHECK(MPI_Probe, (MPI_ANY_SOURCE, MPI_ANY_TAG, _communicator.get(), &status));
    if (status.MPI_TAG == RESULT_TAG) {
      int64_t task_idx =
          _worker_task_indices[status.MPI_SOURCE - (status.MPI_SOURCE > _manager_rank)];
      _worker_task_indices[status.MPI_SOURCE - (status.MPI_SOURCE > _manager_rank)] = -1;
      assert(task_idx >= 0 && "Task index should be valid");
      if (static_cast<uint64_t>(task_idx) >= _results.size()) {
        _results.resize(task_idx + 1);
      }
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, _result_type::value, &count));
      _result_type::resize(_results[task_idx], count);
      DYNAMPI_MPI_CHECK(MPI_Recv, (_result_type::ptr(_results[task_idx]), count, _result_type::value, status.MPI_SOURCE,
                                   RESULT_TAG, _communicator.get(), &status));
      _results_received++;
    } else {
      assert(status.MPI_TAG == REQUEST_TAG && "Unexpected tag received in worker");
      DYNAMPI_MPI_CHECK(MPI_Recv, (nullptr, 0, _result_type::value, status.MPI_SOURCE, REQUEST_TAG,
                                   _communicator.get(), &status));
    }
    _free_workers.push(status.MPI_SOURCE);
  }

  [[nodiscard]] std::vector<ResultT> distribute_tasks() {
    assert(_communicator.rank() == _manager_rank && "Only the manager can distribute tasks");
    while (!_task_queue.empty()) {
      const TaskT& task = _task_queue.front();
      if (_communicator.size() > 1) {
        if (_free_workers.empty()) {
          // If no free workers, wait for a result to be received
          receive_from_any_worker();
        }
        int worker = _free_workers.top();
        _free_workers.pop();
        _worker_task_indices[idx_for_worker(worker)] = _tasks_sent;
        DYNAMPI_MPI_CHECK(MPI_Send, (_task_type::ptr(task), _task_type::count(task), _task_type::value, worker, TASK_TAG, _communicator.get()));
      } else {
        // If there's only one process, we just run the worker function directly
        _results.emplace_back(_worker_function(task));
        _results_received++;
      }
      _task_queue.pop();
      _tasks_sent++;
    }
    while (_free_workers.size() + 1 < static_cast<size_t>(_communicator.size())) {
      receive_from_any_worker();
    }
    assert(_results_received == _tasks_sent && "Not all tasks were processed by workers");
    assert(_results.size() == _tasks_sent && "Results size should match tasks sent");
    return _results;
  }

  void finalize() {
    assert(_communicator.rank() == _manager_rank &&
           "Only the manager can finalize the work distribution");
    assert(_free_workers.size() + 1 == static_cast<size_t>(_communicator.size()) &&
           "All workers should be free before finalizing");
    for (int i = 0; i < _communicator.size() - 1; i++) {
      DYNAMPI_MPI_CHECK(MPI_Send,
                        (nullptr, 0, _task_type::value, worker_for_idx(i), DONE_TAG, _communicator.get()));
    }
  }

  ~NaiveMPIWorkDistributor() {
    if (is_manager()) finalize();
    assert(_tasks_sent == _results_received && "Not all tasks were processed by workers");
  }

 private:
  int idx_for_worker(int worker_rank) const {
    assert(worker_rank != _manager_rank && "Manager rank should not be used as a worker rank");
    if (worker_rank < _manager_rank) {
      return worker_rank;
    } else {
      return worker_rank - 1;
    }
  }

  int worker_for_idx(int idx) const { return (idx < _manager_rank) ? idx : (idx + 1); }
};

template <typename ResultT>
std::optional<std::vector<ResultT>> mpi_manager_worker_distribution(
    size_t n_tasks, std::function<ResultT(size_t)> worker_function, MPI_Comm comm = MPI_COMM_WORLD,
    int manager_rank = 0) {
  NaiveMPIWorkDistributor<size_t, ResultT> distributor(worker_function, comm, manager_rank);
  if (distributor.is_manager()) {
    for (size_t i = 0; i < n_tasks; ++i) {
      distributor.insert_task(i);
    }
    return distributor.distribute_tasks();
  } else {
    distributor.run_worker();
    return {};
  }
}

template <typename TaskT, typename ResultT>
using MPIDynamicWorkDistributor = NaiveMPIWorkDistributor<TaskT, ResultT>;

}  // namespace dynampi
