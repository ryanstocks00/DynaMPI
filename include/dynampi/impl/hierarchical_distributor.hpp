/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <ranges>
#include <span>
#include <stack>
#include <type_traits>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/base_distributor.hpp"
#include "dynampi/utilities/assert.hpp"

namespace dynampi {

template <typename TaskT, typename ResultT, typename... Options>
class HierarchicalMPIWorkDistributor : public BaseMPIWorkDistributor<TaskT, ResultT, Options...> {
  using Base = BaseMPIWorkDistributor<TaskT, ResultT, Options...>;

 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    std::optional<size_t> message_batch_size = std::nullopt;
    int max_workers_per_coordinator = 2;
  };

  static constexpr bool prioritize_tasks = Base::prioritize_tasks;
  static const bool ordered = false;

 private:
  typename Base::QueueT _unallocated_task_queue;
  std::vector<ResultT> _results;

  struct TaskRequest {
    int worker_rank;
    std::optional<int> num_tasks_requested = std::nullopt;
  };
  std::stack<TaskRequest, std::vector<TaskRequest>> _free_worker_indices;

  size_t _tasks_sent_to_child = 0;
  size_t _results_received_from_child = 0;
  size_t _results_sent_to_parent = 0;
  size_t _tasks_received_from_parent = 0;

  bool _finalized = false;
  bool _done = false;

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();

  using MPICommunicator = dynampi::MPICommunicator<track_statistics<statistics_mode>>;
  MPICommunicator _communicator;
  std::function<ResultT(TaskT)> _worker_function;
  Config _config;

  inline int parent_rank() const {
    int rank = _communicator.rank();
    int virtual_rank = rank == _config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    if (virtual_rank == 0) return -1;  // Root has no parent
    int virtual_parent = (virtual_rank - 1) / _config.max_workers_per_coordinator;
    int parent_rank =
        virtual_parent == 0 ? _config.manager_rank : worker_for_idx(virtual_parent - 1);
    return parent_rank;
  }

  inline int total_num_children(int rank) const {
    int virtual_rank = rank == _config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    int num_children = 0;
    for (int i = 0; i < _config.max_workers_per_coordinator; ++i) {
      int child = virtual_rank * _config.max_workers_per_coordinator + i + 1;
      if (child >= _communicator.size()) break;  // No more children
      num_children += total_num_children(worker_for_idx(child - 1));
    }
    return num_children;
  }

  inline int child_rank(int rank, int child_index) const {
    DYNAMPI_ASSERT_LT(child_index, _config.max_workers_per_coordinator,
                      "Child index must be less than max workers per coordinator");
    int virtual_rank = rank == _config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    int virtual_child = virtual_rank * _config.max_workers_per_coordinator + child_index + 1;
    if (virtual_child >= _communicator.size()) {
      return -1;  // No child exists
    }
    int child_rank = worker_for_idx(virtual_child - 1);
    return child_rank;
  }

  inline int num_direct_children() const {
    int rank = _communicator.rank();
    int num_children = 0;
    for (int i = 0; i < _config.max_workers_per_coordinator; ++i) {
      if (child_rank(rank, i) != -1) {
        num_children++;
      }
    }
    return num_children;
  }

  bool is_leaf_worker() const {
    int rank = _communicator.rank();
    return child_rank(rank, 0) == -1;
  }

  enum Tag : int {
    TASK = 0,
    DONE = 1,
    RESULT = 2,
    REQUEST = 3,
    ERROR = 4,
    TASK_BATCH = 5,
    RESULT_BATCH = 6,
    REQUEST_BATCH = 7
  };

  struct Statistics {
    const CommStatistics& comm_statistics;
  };

  using StatisticsT =
      std::conditional_t<statistics_mode == StatisticsMode::Detailed, Statistics, std::monostate>;

  StatisticsT _statistics;

  static StatisticsT create_statistics(const MPICommunicator& comm) {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{comm.get_statistics()};
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
    if (_config.auto_run_workers && _communicator.rank() != _config.manager_rank) {
      std::cout << "Auto-running worker on rank " << _communicator.rank() << std::endl;
      run_worker();
    } else {
      std::cout << "Not auto-running worker on rank " << _communicator.rank() << std::endl;
    }
  }

  const StatisticsT& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    DYNAMPI_ASSERT(is_root_manager(), "Only the manager can access statistics");
    return _statistics;
  }

  void run_worker() {
    DYNAMPI_ASSERT(_communicator.rank() != _config.manager_rank,
                   "Worker cannot run on the manager rank");
    if (is_leaf_worker()) {
      _communicator.send(nullptr, parent_rank(), Tag::REQUEST);
    } else {
      int num_children = num_direct_children();
      _communicator.send(num_children, parent_rank(), Tag::REQUEST_BATCH);
    }
    if (is_leaf_worker()) {
      std::cout << "Leaf worker " << _communicator.rank() << " started with parent rank "
                << parent_rank() << std::endl;
      while (!_done) {
        receive_from_anyone();
      }
    } else {
      std::cout << "Coordinator worker " << _communicator.rank()
                << " started with children ranks: ";
      for (int i = 0; i < _config.max_workers_per_coordinator; ++i) {
        int child = child_rank(_communicator.rank(), i);
        if (child == -1) {
          break;
        }
        std::cout << child << " ";
      }
      std::cout << std::endl;
      bool returned_empty_results = false;
      (void)returned_empty_results;  // Suppress unused variable warning
      while (!_done) {
        while (!_done && _unallocated_task_queue.empty()) {
          receive_from_anyone();
        }
        while (!_unallocated_task_queue.empty()) {
          allocate_task_to_child();
        }
        while (_tasks_sent_to_child > _results_received_from_child) {
          receive_from_anyone();
        }
        if (_done) {
          std::cout << "Coordinator worker " << _communicator.rank()
                    << " received done signal, finalizing." << std::endl;
          break;
        }
        if (_results.empty()) returned_empty_results = true;
        return_results_and_request_next_batch_from_manager();
      }
      for (int i = 0; i < _config.max_workers_per_coordinator; ++i) {
        int child = child_rank(_communicator.rank(), i);
        if (child != -1) {
          _communicator.send(nullptr, child, Tag::DONE);
        } else {
          break;
        }
      }
    }
  }

  void return_results_and_request_next_batch_from_manager() {
    DYNAMPI_ASSERT(!is_leaf_worker(), "Leaf workers should not return results directly");
    DYNAMPI_ASSERT_NE(_communicator.rank(), _config.manager_rank,
                      "Manager should not request tasks from itself");
    std::vector<ResultT> results = _results;
    std::cout << "Worker " << _communicator.rank() << " sending batch of " << results.size()
              << " results to manager." << std::endl;
    _results.clear();
    _communicator.send(results, parent_rank(), Tag::RESULT_BATCH);
    _results_sent_to_parent += results.size();
  }

  bool is_root_manager() const { return _communicator.rank() == _config.manager_rank; }

  size_t remaining_tasks_count() const {
    DYNAMPI_ASSERT_EQ(_communicator.rank(), _config.manager_rank,
                      "Only the manager can check remaining tasks");
    return _unallocated_task_queue.size();
  }

  void insert_task(TaskT task)
    requires(!prioritize_tasks)
  {
    DYNAMPI_ASSERT_EQ(_communicator.rank(), _config.manager_rank,
                      "Only the manager can distribute tasks");
    _unallocated_task_queue.push_back(task);
    _tasks_received_from_parent++;
  }
  void insert_task(const TaskT& task, double priority)
    requires(prioritize_tasks)
  {
    DYNAMPI_ASSERT_EQ(_communicator.rank(), _config.manager_rank,
                      "Only the manager can distribute tasks");
    _unallocated_task_queue.emplace(priority, task);
    _tasks_received_from_parent++;
  }

  template <typename Range>
    requires std::ranges::input_range<Range> && (!prioritize_tasks)
  void insert_tasks(const Range& tasks) {
    DYNAMPI_ASSERT_EQ(_communicator.rank(), _config.manager_rank,
                      "Only the manager can distribute tasks");
    std::copy(std::ranges::begin(tasks), std::ranges::end(tasks),
              std::back_inserter(_unallocated_task_queue));
    _tasks_received_from_parent +=
        std::distance(std::ranges::begin(tasks), std::ranges::end(tasks));
  }
  void insert_tasks(const std::vector<TaskT>& tasks)
    requires(!prioritize_tasks)
  {
    insert_tasks(std::span<const TaskT>(tasks));
  }

  void allocate_task_to_child() {
    if (_communicator.size() > 1) {
      if (_free_worker_indices.empty()) {
        // If no free workers, wait for a result to be received
        receive_from_anyone();
      }
      TaskRequest request = _free_worker_indices.top();
      int worker = request.worker_rank;
      _free_worker_indices.pop();
      if (request.num_tasks_requested.has_value()) {
        std::cout << "Allocating batch " << std::endl;
        std::vector<TaskT> tasks;
        tasks.reserve(request.num_tasks_requested.value());
        for (int i = 0; i < request.num_tasks_requested; ++i) {
          if (_unallocated_task_queue.empty()) {
            break;  // No more tasks to allocate
          }
          tasks.push_back(get_next_task_to_send());
        }
        _communicator.send(tasks, worker, Tag::TASK_BATCH);
        _tasks_sent_to_child += tasks.size();
      } else {
        std::cout << "Allocating single " << std::endl;
        const TaskT task = get_next_task_to_send();
        _communicator.send(task, worker, Tag::TASK);
        _tasks_sent_to_child++;
      }
    } else {
      // If there's only one process, we just run the worker function directly
      const TaskT task = get_next_task_to_send();
      _results.emplace_back(_worker_function(task));
    }
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() {
    DYNAMPI_ASSERT_EQ(_communicator.rank(), _config.manager_rank,
                      "Only the manager can finish remaining tasks");
    while (!_unallocated_task_queue.empty()) {
      allocate_task_to_child();
    }
    while (_free_worker_indices.size() < static_cast<size_t>(num_direct_children())) {
      receive_from_anyone();
    }
    _results_sent_to_parent = _results.size();
    std::cout << "Results sent to parent : " << _results_sent_to_parent
              << ", results received from child: " << _results_received_from_child
              << ", tasks sent to child: " << _tasks_sent_to_child
              << ", tasks received from parent: " << _tasks_received_from_parent << std::endl;
    DYNAMPI_ASSERT_EQ(_results_received_from_child, _tasks_sent_to_child,
                      "All tasks should have been processed by workers before finalizing");
    DYNAMPI_ASSERT_EQ(_results_sent_to_parent, _tasks_received_from_parent,
                      "All results should have been sent to the parent before finalizing");
    if (_communicator.size() > 1)
      DYNAMPI_ASSERT_EQ(_results_sent_to_parent, _results_received_from_child,
                        "Manager should not send results to itself");
    if (is_root_manager() && _communicator.size() > 1)
      DYNAMPI_ASSERT_EQ(_results.size(), _results_received_from_child,
                        "Results size should match tasks sent before finalizing");
    // TODO(Change this so we don't return the same tasks multiple times)
    return _results;
  }

  void finalize() {
    DYNAMPI_ASSERT(!_finalized, "Work distribution already finalized");
    if (is_root_manager()) {
      send_done_to_workers();
      _finalized = true;
    }
  }

  ~HierarchicalMPIWorkDistributor() {
    if (!_finalized) {
      finalize();
    }
    DYNAMPI_ASSERT_EQ(_results_received_from_child, _tasks_sent_to_child,
                      "All tasks should have been processed by workers before finalizing");
    DYNAMPI_ASSERT_EQ(_results_sent_to_parent, _tasks_received_from_parent,
                      "All results should have been sent to the parent before finalizing");
    if (is_leaf_worker())
      DYNAMPI_ASSERT_EQ(_results_received_from_child, 0,
                        "Leaf workers should not receive results from children");
    else if (_communicator.size() > 1)
      DYNAMPI_ASSERT_EQ(_results_received_from_child, _results_sent_to_parent,
                        "Results received from children should match results sent to parent");
  }

 private:
  TaskT get_next_task_to_send() {
    DYNAMPI_ASSERT(is_root_manager() || !is_leaf_worker(),
                   "Leaf workers should not send tasks directly");
    DYNAMPI_ASSERT(!_unallocated_task_queue.empty(), "There should be tasks available to send");
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
    DYNAMPI_ASSERT_EQ(_communicator.rank(), _config.manager_rank,
                      "Only the manager can send done messages to workers");
    DYNAMPI_ASSERT_EQ(_free_worker_indices.size(), static_cast<size_t>(num_direct_children()),
                      "All workers should be free before finalizing");
    for (int i = 0; i < _config.max_workers_per_coordinator; ++i) {
      int child = child_rank(_communicator.rank(), i);
      if (child == -1) {
        break;
      }
      _communicator.send(nullptr, child, Tag::DONE);
    }
  }

  int idx_for_worker(int worker_rank) const {
    DYNAMPI_ASSERT_NE(worker_rank, _config.manager_rank,
                      "Manager rank should not be used as a worker rank");
    if (worker_rank < _config.manager_rank) {
      return worker_rank;
    } else {
      return worker_rank - 1;
    }
  }

  int worker_for_idx(int idx) const { return (idx < _config.manager_rank) ? idx : (idx + 1); }

  using result_mpi_type = MPI_Type<ResultT>;
  using task_mpi_type = MPI_Type<TaskT>;

  void receive_result_from(MPI_Status status) {
    _results.push_back(ResultT{});
    if (result_mpi_type::resize_required) {
      DYNAMPI_UNIMPLEMENTED(
          "Dynamic resizing of results is not supported in hierarchical distribution");
      // int count;
      // DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, result_mpi_type::value, &count));
      // result_mpi_type::resize(_results.back(), count);
    }
    _communicator.recv(_results.back(), status.MPI_SOURCE, Tag::RESULT);
    _results_received_from_child++;
    _free_worker_indices.push(TaskRequest{.worker_rank = status.MPI_SOURCE});
  }

  void receive_result_batch_from(MPI_Status status) {
    using message_type = MPI_Type<std::vector<ResultT>>;
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
    std::cout << "Worker " << _communicator.rank() << " received batch of " << count
              << " results from worker." << std::endl;
    std::vector<ResultT> results;
    message_type::resize(results, count);
    _communicator.recv(results, status.MPI_SOURCE, Tag::RESULT_BATCH);
    _free_worker_indices.push({.worker_rank = status.MPI_SOURCE,
                               .num_tasks_requested = static_cast<int>(results.size())});
    std::copy(results.begin(), results.end(), std::back_inserter(_results));
    _results_received_from_child += results.size();
  }

  void receive_execute_return_task_from(MPI_Status status) {
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_mpi_type::value, &count));
    TaskT message;
    task_mpi_type::resize(message, count);
    _communicator.recv(message, status.MPI_SOURCE, Tag::TASK);
    _tasks_received_from_parent++;
    ResultT result = _worker_function(message);
    _communicator.send(result, status.MPI_SOURCE, Tag::RESULT);
    _results_sent_to_parent++;
  }

  void receive_task_batch_from(MPI_Status status) {
    if constexpr (prioritize_tasks) {
      DYNAMPI_UNIMPLEMENTED("Prioritized hierarchical distribution");
    } else {
      using message_type = MPI_Type<std::vector<TaskT>>;
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
      std::vector<TaskT> tasks;
      message_type::resize(tasks, count);
      _communicator.recv(tasks, parent_rank(), Tag::TASK_BATCH);
      _tasks_received_from_parent += tasks.size();
      std::cout << "Worker " << _communicator.rank() << " received batch of " << tasks.size()
                << " tasks from manager." << std::endl;
      for (const auto& task : tasks) {
        _unallocated_task_queue.push_back(task);
      }
    }
  }

  void receive_from_anyone() {
    DYNAMPI_ASSERT_GT(_communicator.size(), 1,
                      "There should be at least one worker to receive results from");
    MPI_Status status = _communicator.probe();
    std::cout << "Rank " << _communicator.rank() << " received message with tag " << status.MPI_TAG
              << " from source " << status.MPI_SOURCE << std::endl;
    switch (status.MPI_TAG) {
      case Tag::TASK: {
        return receive_execute_return_task_from(status);
      }
      case Tag::TASK_BATCH: {
        return receive_task_batch_from(status);
      }
      case Tag::RESULT: {
        return receive_result_from(status);
      }
      case Tag::RESULT_BATCH: {
        return receive_result_batch_from(status);
      }
      case Tag::REQUEST: {
        _communicator.recv_empty_message(status.MPI_SOURCE, Tag::REQUEST);
        _free_worker_indices.push(TaskRequest{.worker_rank = status.MPI_SOURCE});
        return;
      }
      case Tag::REQUEST_BATCH: {
        int request_count;
        _communicator.recv(request_count, status.MPI_SOURCE, Tag::REQUEST_BATCH);
        _free_worker_indices.push(
            TaskRequest{.worker_rank = status.MPI_SOURCE, .num_tasks_requested = request_count});
        return;
      }
      case Tag::DONE: {
        _communicator.recv_empty_message(status.MPI_SOURCE, Tag::DONE);
        _done = true;
        return;
      }
    }
  }
};

};  // namespace dynampi
