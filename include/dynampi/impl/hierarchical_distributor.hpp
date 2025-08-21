/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iterator>
#include <ranges>
#include <span>
#include <stack>
#include <type_traits>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/base_distributor.hpp"

namespace dynampi {

template <typename TaskT, typename ResultT, typename... Options>
class HierarchicalMPIWorkDistributor : public BaseMPIWorkDistributor<TaskT, ResultT, Options...> {
  using Base = BaseMPIWorkDistributor<TaskT, ResultT, Options...>;

  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    std::optional<size_t> message_batch_size = std::nullopt;
    size_t max_workers_per_coordinator = 2;
  };

  static constexpr bool prioritize_tasks = Base::prioritize_tasks;

 private:
  typename Base::QueueT _unallocated_task_queue;
  std::vector<int64_t> _worker_current_task_indices;
  std::vector<ResultT> _results;
  std::stack<int, std::vector<int>> _free_worker_indices;

  size_t _tasks_sent = 0;
  size_t _results_received = 0;
  bool _finalized = false;

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();

  using MPICommunicator = dynampi::MPICommunicator<track_statistics<statistics_mode>>;
  MPICommunicator _communicator;
  std::function<ResultT(TaskT)> _worker_function;
  Config _config;

  enum Tag : int { TASK = 0, DONE = 1, RESULT = 2, REQUEST = 3, ERROR = 4 };

  struct Statistics {
    const CommStatistics& comm_statistics;
    std::vector<size_t> worker_task_counts;
  };

  using StatisticsT =
      std::conditional_t<statistics_mode == StatisticsMode::Detailed, Statistics, std::monostate>;

  StatisticsT _statistics;

  static StatisticsT create_statistics(const MPICommunicator& comm) {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{comm.get_statistics(), {}};
    } else {
      return {};
    }
  }

 public:
  explicit HierarchicalMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                          Config runtime_config = Config{})
      : _communicator(runtime_config.comm, MPICommunicator::Duplicate),
        _worker_function(worker_function),
        _config(runtime_config),
        _statistics{create_statistics(_communicator)} {
    if (is_root_manager()) _worker_current_task_indices.resize(_communicator.size() - 1, -1);
    if (_config.auto_run_workers && _communicator.rank() != _config.manager_rank) {
      run_worker();
    }
    if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
      if (is_root_manager()) _statistics.worker_task_counts.resize(_communicator.size(), 0);
    }
  }

  const StatisticsT& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    assert(is_root_manager() && "Only the manager can access statistics");
    return _statistics;
  }

  void run_worker() {
    assert(_communicator.rank() != _config.manager_rank && "Worker cannot run on the manager rank");
    using task_type = MPI_Type<TaskT>;
    _communicator.send(nullptr, _config.manager_rank, Tag::REQUEST);
    while (true) {
      MPI_Status status;
      DYNAMPI_MPI_CHECK(MPI_Probe, (MPI_ANY_SOURCE, MPI_ANY_TAG, _communicator.get(), &status));
      if (status.MPI_TAG == Tag::DONE) {
        _communicator.recv_empty_message(_config.manager_rank, Tag::DONE);
        break;
      }
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_type::value, &count));
      TaskT message;
      task_type::resize(message, count);
      _communicator.recv(message, _config.manager_rank, Tag::TASK);
      _tasks_sent++;
      ResultT result = _worker_function(message);
      _communicator.send(result, _config.manager_rank, Tag::RESULT);
      _results_received++;
    }
  }

  bool is_root_manager() const { return _communicator.rank() == _config.manager_rank; }

  size_t remaining_tasks_count() const {
    assert(_communicator.rank() == _config.manager_rank &&
           "Only the manager can check remaining tasks");
    return _unallocated_task_queue.size();
  }

  void insert_task(TaskT task)
    requires(!prioritize_tasks)
  {
    assert(_communicator.rank() == _config.manager_rank && "Only the manager can distribute tasks");
    _unallocated_task_queue.push_back(task);
  }
  void insert_task(const TaskT& task, double priority)
    requires(prioritize_tasks)
  {
    assert(_communicator.rank() == _config.manager_rank && "Only the manager can distribute tasks");
    _unallocated_task_queue.emplace(priority, task);
  }

  template <typename Range>
    requires std::ranges::input_range<Range> && (!prioritize_tasks)
  void insert_tasks(const Range& tasks) {
    assert(_communicator.rank() == _config.manager_rank && "Only the manager can distribute tasks");
    std::copy(std::ranges::begin(tasks), std::ranges::end(tasks),
              std::back_inserter(_unallocated_task_queue));
  }
  void insert_tasks(const std::vector<TaskT>& tasks)
    requires(!prioritize_tasks)
  {
    insert_tasks(std::span<const TaskT>(tasks));
  }

  void get_task_and_allocate() {
    const TaskT task = get_next_task_to_send();
    if (_communicator.size() > 1) {
      if (_free_worker_indices.empty()) {
        // If no free workers, wait for a result to be received
        receive_from_any_worker();
      }
      int worker = _free_worker_indices.top();
      _free_worker_indices.pop();
      _worker_current_task_indices[idx_for_worker(worker)] = _tasks_sent;
      if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
        _statistics.worker_task_counts[worker]++;
      }
      _communicator.send(task, worker, Tag::TASK);
    } else {
      // If there's only one process, we just run the worker function directly
      _results.emplace_back(_worker_function(task));
      _results_received++;
    }
    _tasks_sent++;
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() {
    assert(_communicator.rank() == _config.manager_rank && "Only the manager can distribute tasks");
    while (!_unallocated_task_queue.empty()) {
      get_task_and_allocate();
    }
    while (_free_worker_indices.size() + 1 < static_cast<size_t>(_communicator.size())) {
      receive_from_any_worker();
    }
    assert(_results_received == _tasks_sent && "Not all tasks were processed by workers");
    assert(_results.size() == _tasks_sent && "Results size should match tasks sent");
    return _results;
  }

  void finalize() {
    assert(!_finalized && "Work distribution already finalized");
    if (is_root_manager()) {
      send_done_to_workers();
      _finalized = true;
    }
  }

  ~HierarchicalMPIWorkDistributor() {
    if (!_finalized) {
      finalize();
    }
    assert(_tasks_sent == _results_received && "Not all tasks were processed by workers");
  }

 private:
  TaskT get_next_task_to_send() {
    assert(_communicator.rank() == _config.manager_rank && "Only the manager can get next task");
    assert(!_unallocated_task_queue.empty() && "There should be tasks available to send");
    TaskT task;
    if constexpr (std::is_same_v<decltype(_unallocated_task_queue), std::deque<TaskT>>) {
      task = _unallocated_task_queue.front();
      _unallocated_task_queue.pop_front();
    } else {
      task = _unallocated_task_queue.top().second;
      _unallocated_task_queue.pop();
    }
    return task;
  }

  void send_done_to_workers() {
    assert(_communicator.rank() == _config.manager_rank &&
           "Only the manager can finalize the work distribution");
    assert(_free_worker_indices.size() + 1 == static_cast<size_t>(_communicator.size()) &&
           "All workers should be free before finalizing");
    for (int i = 0; i < _communicator.size() - 1; i++) {
      _communicator.send(nullptr, worker_for_idx(i), Tag::DONE);
    }
  }

  int idx_for_worker(int worker_rank) const {
    assert(worker_rank != _config.manager_rank &&
           "Manager rank should not be used as a worker rank");
    if (worker_rank < _config.manager_rank) {
      return worker_rank;
    } else {
      return worker_rank - 1;
    }
  }

  int worker_for_idx(int idx) const { return (idx < _config.manager_rank) ? idx : (idx + 1); }

  void receive_from_any_worker() {
    assert(_communicator.rank() == _config.manager_rank &&
           "Only the manager can receive results and send tasks");
    assert(_communicator.size() > 1 &&
           "There should be at least one worker to receive results from");
    using result_type = MPI_Type<ResultT>;
    MPI_Status status;
    DYNAMPI_MPI_CHECK(MPI_Probe, (MPI_ANY_SOURCE, MPI_ANY_TAG, _communicator.get(), &status));
    if (status.MPI_TAG == Tag::RESULT) {
      int64_t task_idx = _worker_current_task_indices[status.MPI_SOURCE -
                                                      (status.MPI_SOURCE > _config.manager_rank)];
      _worker_current_task_indices[status.MPI_SOURCE - (status.MPI_SOURCE > _config.manager_rank)] =
          -1;
      assert(task_idx >= 0 && "Task index should be valid");
      if (static_cast<uint64_t>(task_idx) >= _results.size()) {
        _results.resize(task_idx + 1);
      }
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, result_type::value, &count));
      result_type::resize(_results[task_idx], count);
      _communicator.recv(_results[task_idx], status.MPI_SOURCE, Tag::RESULT);
      _results_received++;
    } else {
      assert(status.MPI_TAG == Tag::REQUEST && "Unexpected tag received in worker");
      _communicator.recv_empty_message(status.MPI_SOURCE, Tag::REQUEST);
    }
    _free_worker_indices.push(status.MPI_SOURCE);
  }
};

};  // namespace dynampi
