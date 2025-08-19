/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <queue>
#include <ranges>
#include <span>
#include <stack>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <variant>
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

struct prioritize_tasks_t {
  using type = bool;
  static constexpr bool value = false;
};

struct enable_prioritization : public prioritize_tasks_t {
  static constexpr bool value = true;
};

struct track_statistics_t {
  using type = bool;
  static constexpr bool value = false;
};

struct enable_statistics : public track_statistics_t {
  static constexpr bool value = true;
};

template <typename Option, typename... Options>
struct option_value {
  static constexpr typename Option::type value = Option::value;
};

// Specialization: if Head is derived from Option → use Head::value
template <typename Option, typename Head, typename... Tail>
struct option_value<Option, Head, Tail...> {
  static constexpr typename Option::type value =
      std::is_base_of_v<Option, Head> ? Head::value : option_value<Option, Tail...>::value;
};

// Convenience function
template <typename Option, typename... Options>
consteval typename Option::type get_option_value() {
  return option_value<Option, Options...>::value;
}

template <typename TaskT, typename ResultT, typename... Options>
class NaiveMPIWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
  };

  struct Statistics {
    std::vector<size_t> worker_task_counts;
    size_t total_bytes_sent = 0;
    size_t total_bytes_received = 0;
    size_t total_messages_sent = 0;
    size_t total_messages_received = 0;

    template <typename T>
    void send_message(const T& message) {
      using message_type = MPI_Type<T>;
      if constexpr (!std::is_same_v<T, std::nullptr_t>) {
        int count = message_type::count(message);
        int size;
        MPI_Type_size(message_type::value, &size);
        total_bytes_sent += count * size;
      }
      total_messages_sent++;
    }

    template <typename T>
    void receive_message(const T& message) {
      using message_type = MPI_Type<T>;
      if constexpr (!std::is_same_v<T, std::nullptr_t>) {
        int count = message_type::count(message);
        int size;
        MPI_Type_size(message_type::value, &size);
        total_bytes_received += count * size;
      }
      total_messages_received++;
    }

    double average_send_size() const {
      if (total_messages_sent == 0) return 0.0;
      return static_cast<double>(total_bytes_sent) / total_messages_sent;
    }

    double average_receive_size() const {
      if (total_messages_received == 0) return 0.0;
      return static_cast<double>(total_bytes_received) / total_messages_received;
    }
  };

 private:
  static constexpr bool prioritize_tasks = get_option_value<prioritize_tasks_t, Options...>();
  using QueueT = std::conditional_t<prioritize_tasks, std::priority_queue<std::pair<double, TaskT>>,
                                    std::deque<TaskT>>;

  static constexpr bool track_statistics = get_option_value<track_statistics_t, Options...>();
  using StatisticsT = std::conditional_t<track_statistics, Statistics, std::monostate>;

  QueueT _unallocated_task_queue;
  std::vector<int64_t> _worker_current_task_indices;
  std::vector<ResultT> _results;
  std::stack<int, std::vector<int>> _free_worker_indices;

  size_t _tasks_sent = 0;
  size_t _results_received = 0;
  bool _finalized = false;

  MPICommunicator _communicator;
  std::function<ResultT(TaskT)> _worker_function;
  Config _config;

  enum Tag : int { TASK = 0, DONE = 1, RESULT = 2, REQUEST = 3, ERROR = 4 };

  StatisticsT _statistics;

 public:
  NaiveMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                          Config runtime_config = Config{})
      : _communicator(runtime_config.comm, MPICommunicator::Owned),
        _worker_function(worker_function),
        _config(runtime_config) {
    if (is_manager()) _worker_current_task_indices.resize(_communicator.size() - 1, -1);
    if (_config.auto_run_workers && _communicator.rank() != _config.manager_rank) {
      run_worker();
    }
    if constexpr (track_statistics) {
      if (is_manager()) _statistics.worker_task_counts.resize(_communicator.size(), 0);
    }
  }

  const StatisticsT& get_statistics() const
    requires(track_statistics)
  {
    assert(is_manager() && "Only the manager can access statistics");
    return _statistics;
  }

  void run_worker() {
    assert(_communicator.rank() != _config.manager_rank && "Worker cannot run on the manager rank");
    using task_type = MPI_Type<TaskT>;
    using result_type = MPI_Type<ResultT>;
    if constexpr (track_statistics) {
      _statistics.send_message(nullptr);
    }
    MPI_Send(nullptr, 0, task_type::value, _config.manager_rank, Tag::REQUEST, _communicator.get());
    while (true) {
      MPI_Status status;
      DYNAMPI_MPI_CHECK(MPI_Probe, (MPI_ANY_SOURCE, MPI_ANY_TAG, _communicator.get(), &status));
      if (status.MPI_TAG == Tag::DONE) {
        if constexpr (track_statistics) {
          _statistics.receive_message(nullptr);
        }
        MPI_Recv(nullptr, 0, task_type::value, _config.manager_rank, Tag::DONE, _communicator.get(),
                 &status);
        break;
      }
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_type::value, &count));
      TaskT message;
      task_type::resize(message, count);
      if constexpr (track_statistics) {
        _statistics.receive_message(message);
      }
      MPI_Recv(task_type::ptr(message), 1, task_type::value, _config.manager_rank, Tag::TASK,
               _communicator.get(), &status);
      _tasks_sent++;
      ResultT result = _worker_function(message);
      if constexpr (track_statistics) {
        _statistics.send_message(result);
      }
      MPI_Send(result_type::ptr(result), result_type::count(result), result_type::value,
               _config.manager_rank, Tag::RESULT, _communicator.get());
      _results_received++;
    }
  }

  bool is_manager() const { return _communicator.rank() == _config.manager_rank; }

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
    std::ranges::copy(tasks, std::back_inserter(_unallocated_task_queue));
  }
  void insert_tasks(const std::vector<TaskT>& tasks)
    requires(!prioritize_tasks)
  {
    insert_tasks(std::span<const TaskT>(tasks));
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() {
    using task_type = MPI_Type<TaskT>;
    assert(_communicator.rank() == _config.manager_rank && "Only the manager can distribute tasks");
    while (!_unallocated_task_queue.empty()) {
      const TaskT task = get_next_task_to_send();
      if (_communicator.size() > 1) {
        if (_free_worker_indices.empty()) {
          // If no free workers, wait for a result to be received
          receive_from_any_worker();
        }
        int worker = _free_worker_indices.top();
        _free_worker_indices.pop();
        _worker_current_task_indices[idx_for_worker(worker)] = _tasks_sent;
        if constexpr (track_statistics) {
          _statistics.worker_task_counts[worker]++;
          _statistics.send_message(task);
        }
        DYNAMPI_MPI_CHECK(MPI_Send, (task_type::ptr(task), task_type::count(task), task_type::value,
                                     worker, Tag::TASK, _communicator.get()));
      } else {
        // If there's only one process, we just run the worker function directly
        _results.emplace_back(_worker_function(task));
        _results_received++;
      }
      _tasks_sent++;
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
    if (is_manager()) {
      send_done_to_workers();
      _finalized = true;
    }
  }

  ~NaiveMPIWorkDistributor() {
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
    if constexpr (std::is_same_v<QueueT, std::deque<TaskT>>) {
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
      using task_type = MPI_Type<TaskT>;
      if constexpr (track_statistics) {
        _statistics.send_message(nullptr);
      }
      DYNAMPI_MPI_CHECK(MPI_Send, (nullptr, 0, task_type::value, worker_for_idx(i), Tag::DONE,
                                   _communicator.get()));
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
      if constexpr (track_statistics) {
        _statistics.receive_message(_results[task_idx]);
      }
      DYNAMPI_MPI_CHECK(MPI_Recv, (result_type::ptr(_results[task_idx]), count, result_type::value,
                                   status.MPI_SOURCE, Tag::RESULT, _communicator.get(), &status));
      _results_received++;
    } else {
      assert(status.MPI_TAG == Tag::REQUEST && "Unexpected tag received in worker");
      if constexpr (track_statistics) {
        _statistics.receive_message(nullptr);
      }
      DYNAMPI_MPI_CHECK(MPI_Recv, (nullptr, 0, result_type::value, status.MPI_SOURCE, Tag::REQUEST,
                                   _communicator.get(), &status));
    }
    _free_worker_indices.push(status.MPI_SOURCE);
  }
};

template <typename ResultT>
std::optional<std::vector<ResultT>> mpi_manager_worker_distribution(
    size_t n_tasks, std::function<ResultT(size_t)> worker_function, MPI_Comm comm = MPI_COMM_WORLD,
    int manager_rank = 0) {
  NaiveMPIWorkDistributor<size_t, ResultT> distributor(
      worker_function, {.comm = comm, .manager_rank = manager_rank});
  if (distributor.is_manager()) {
    for (size_t i = 0; i < n_tasks; ++i) {
      distributor.insert_task(i);
    }
    return distributor.finish_remaining_tasks();
  }
  return {};
}

template <typename TaskT, typename ResultT, typename... Options>
using MPIDynamicWorkDistributor = NaiveMPIWorkDistributor<TaskT, ResultT, Options...>;

}  // namespace dynampi
