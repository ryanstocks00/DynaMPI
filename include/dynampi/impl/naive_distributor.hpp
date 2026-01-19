/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <optional>
#include <queue>
#include <ranges>
#include <stack>
#include <type_traits>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/base_distributor.hpp"

namespace dynampi {

template <typename TaskT, typename ResultT, typename... Options>
class NaiveMPIWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    bool return_new_results_only = true;
  };
  struct RunConfig {
    std::optional<size_t> min_tasks = std::nullopt;
    std::optional<size_t> max_tasks = std::nullopt;
    std::optional<double> max_seconds = std::nullopt;
  };
  static const bool ordered = true;

 private:
  static constexpr bool prioritize_tasks = get_option_value<prioritize_tasks_t, Options...>();
  using QueueT = std::conditional_t<prioritize_tasks, std::priority_queue<std::pair<double, TaskT>>,
                                    std::deque<TaskT>>;

  QueueT m_unallocated_task_queue;
  std::vector<int64_t> m_worker_current_task_indices;
  std::vector<ResultT> m_results;
  std::stack<int, std::vector<int>> m_free_worker_indices;

  size_t m_tasks_sent = 0;
  size_t m_results_received = 0;
  bool m_finalized = false;
  size_t m_results_returned = 0;
  size_t m_tasks_sent_in_current_run = 0;
  bool m_use_local_result_indices = false;

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();

  using MPICommunicator = dynampi::MPICommunicator<track_statistics<statistics_mode>>;
  MPICommunicator m_communicator;
  std::function<ResultT(TaskT)> m_worker_function;
  Config m_config;

  enum Tag : int { TASK = 0, DONE = 1, RESULT = 2, REQUEST = 3, ERROR = 4 };

  struct Statistics {
    const CommStatistics& comm_statistics;
    std::vector<size_t> worker_task_counts;
  };

  using StatisticsT =
      std::conditional_t<statistics_mode == StatisticsMode::Detailed, Statistics, std::monostate>;

  StatisticsT m_statistics;

  static StatisticsT create_statistics(const MPICommunicator& comm) {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{comm.get_statistics(), {}};
    } else {
      return {};
    }
  }

 public:
  explicit NaiveMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                   Config runtime_config = Config{})
      : m_communicator(runtime_config.comm, MPICommunicator::Duplicate),
        m_worker_function(worker_function),
        m_config(runtime_config),
        m_statistics{create_statistics(m_communicator)} {
    if (is_root_manager()) m_worker_current_task_indices.resize(m_communicator.size() - 1, -1);
    if (m_config.auto_run_workers && m_communicator.rank() != m_config.manager_rank) {
      run_worker();
    }
    if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
      if (is_root_manager()) m_statistics.worker_task_counts.resize(m_communicator.size(), 0);
    }
  }

  const StatisticsT& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    assert(is_root_manager() && "Only the manager can access statistics");
    return m_statistics;
  }

  void run_worker() {
    assert(m_communicator.rank() != m_config.manager_rank &&
           "Worker cannot run on the manager rank");
    using task_type = MPI_Type<TaskT>;
    m_communicator.send(nullptr, m_config.manager_rank, Tag::REQUEST);
    while (true) {
      MPI_Status status = m_communicator.probe();
      if (status.MPI_TAG == Tag::DONE) {
        m_communicator.recv_empty_message(m_config.manager_rank, Tag::DONE);
        break;
      }
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_type::value, &count));
      TaskT message;
      task_type::resize(message, count);
      m_communicator.recv(message, m_config.manager_rank, Tag::TASK);
      m_tasks_sent++;
      ResultT result = m_worker_function(message);
      m_communicator.send(result, m_config.manager_rank, Tag::RESULT);
      m_results_received++;
    }
  }

  bool is_root_manager() const { return m_communicator.rank() == m_config.manager_rank; }

  size_t remaining_tasks_count() const {
    assert(m_communicator.rank() == m_config.manager_rank &&
           "Only the manager can check remaining tasks");
    return m_unallocated_task_queue.size();
  }

  void insert_task(TaskT task)
    requires(!prioritize_tasks)
  {
    assert(m_communicator.rank() == m_config.manager_rank &&
           "Only the manager can distribute tasks");
    m_unallocated_task_queue.push_back(task);
  }
  void insert_task(const TaskT& task, double priority)
    requires(prioritize_tasks)
  {
    assert(m_communicator.rank() == m_config.manager_rank &&
           "Only the manager can distribute tasks");
    m_unallocated_task_queue.emplace(priority, task);
  }

  template <typename Range>
    requires std::ranges::input_range<Range> && (!prioritize_tasks)
  void insert_tasks(const Range& tasks) {
    assert(m_communicator.rank() == m_config.manager_rank &&
           "Only the manager can distribute tasks");
    std::copy(std::ranges::begin(tasks), std::ranges::end(tasks),
              std::back_inserter(m_unallocated_task_queue));
  }
  void insert_tasks(const std::vector<TaskT>& tasks)
    requires(!prioritize_tasks)
  {
    insert_tasks(std::span<const TaskT>(tasks));
  }

  void get_task_and_allocate() {
    const TaskT task = get_next_task_to_send();
    if (m_communicator.size() > 1) {
      if (m_free_worker_indices.empty()) {
        // If no free workers, wait for a result to be received
        receive_from_any_worker();
      }
      int worker = m_free_worker_indices.top();
      m_free_worker_indices.pop();
      int64_t task_idx = m_use_local_result_indices
                             ? static_cast<int64_t>(m_tasks_sent_in_current_run)
                             : static_cast<int64_t>(m_tasks_sent);
      m_worker_current_task_indices[idx_for_worker(worker)] = task_idx;
      if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
        m_statistics.worker_task_counts[worker]++;
      }
      m_communicator.send(task, worker, Tag::TASK);
    } else {
      // If there's only one process, we just run the worker function directly
      m_results.emplace_back(m_worker_function(task));
      m_results_received++;
    }
    if (m_use_local_result_indices) {
      m_tasks_sent_in_current_run++;
    }
    m_tasks_sent++;
  }

  [[nodiscard]] std::vector<ResultT> run_tasks(const RunConfig& config = RunConfig{}) {
    assert(m_communicator.rank() == m_config.manager_rank &&
           "Only the manager can distribute tasks");
    const auto start_time = std::chrono::steady_clock::now();
    const size_t tasks_sent_before = m_tasks_sent;
    const size_t results_received_before = m_results_received;
    size_t tasks_sent_this_call = 0;
    if (m_config.return_new_results_only) {
      m_results.clear();
      m_results_returned = 0;
      m_tasks_sent_in_current_run = 0;
      m_use_local_result_indices = true;
    } else {
      m_use_local_result_indices = false;
    }
    auto should_stop = [&](double elapsed_s) {
      if (config.max_tasks && tasks_sent_this_call >= config.max_tasks.value()) {
        return true;
      }
      if (config.max_seconds && elapsed_s >= config.max_seconds.value()) {
        if (!config.min_tasks || tasks_sent_this_call >= config.min_tasks.value()) {
          return true;
        }
      }
      return false;
    };
    while (!m_unallocated_task_queue.empty()) {
      const double elapsed_s =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
      if (should_stop(elapsed_s)) {
        break;
      }
      get_task_and_allocate();
      tasks_sent_this_call++;
    }
    while (m_free_worker_indices.size() + 1 < static_cast<size_t>(m_communicator.size())) {
      receive_from_any_worker();
    }
    const size_t tasks_sent_delta = m_tasks_sent - tasks_sent_before;
    const size_t results_received_delta = m_results_received - results_received_before;
    assert(results_received_delta == tasks_sent_delta && "Not all tasks were processed by workers");
    if (m_use_local_result_indices) {
      assert(m_results.size() == tasks_sent_delta && "Results size should match tasks sent");
    } else {
      assert(m_results.size() == m_tasks_sent && "Results size should match tasks sent");
    }
    m_use_local_result_indices = false;
    if (m_config.return_new_results_only) {
      m_results_returned = 0;
      return std::move(m_results);
    }
    return m_results;
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() { return run_tasks(); }

  void finalize() {
    assert(!m_finalized && "Work distribution already finalized");
    if (is_root_manager()) {
      send_done_to_workers();
      m_finalized = true;
    }
  }

  ~NaiveMPIWorkDistributor() {
    if (!m_finalized) {
      finalize();
    }
    assert(m_tasks_sent == m_results_received && "Not all tasks were processed by workers");
  }

 private:
  TaskT get_next_task_to_send() {
    assert(m_communicator.rank() == m_config.manager_rank && "Only the manager can get next task");
    assert(!m_unallocated_task_queue.empty() && "There should be tasks available to send");
    TaskT task;
    if constexpr (std::is_same_v<QueueT, std::deque<TaskT>>) {
      task = m_unallocated_task_queue.front();
      m_unallocated_task_queue.pop_front();
    } else {
      task = m_unallocated_task_queue.top().second;
      m_unallocated_task_queue.pop();
    }
    return task;
  }

  void send_done_to_workers() {
    assert(m_communicator.rank() == m_config.manager_rank &&
           "Only the manager can finalize the work distribution");
    assert(m_free_worker_indices.size() + 1 == static_cast<size_t>(m_communicator.size()) &&
           "All workers should be free before finalizing");
    for (int i = 0; i < m_communicator.size() - 1; i++) {
      m_communicator.send(nullptr, worker_for_idx(i), Tag::DONE);
    }
  }

  int idx_for_worker(int worker_rank) const {
    assert(worker_rank != m_config.manager_rank &&
           "Manager rank should not be used as a worker rank");
    if (worker_rank < m_config.manager_rank) {
      return worker_rank;
    } else {
      return worker_rank - 1;
    }
  }

  int worker_for_idx(int idx) const { return (idx < m_config.manager_rank) ? idx : (idx + 1); }

  void receive_from_any_worker() {
    assert(m_communicator.rank() == m_config.manager_rank &&
           "Only the manager can receive results and send tasks");
    assert(m_communicator.size() > 1 &&
           "There should be at least one worker to receive results from");
    using result_type = MPI_Type<ResultT>;
    MPI_Status status = m_communicator.probe(MPI_ANY_SOURCE, MPI_ANY_TAG);
    if (status.MPI_TAG == Tag::RESULT) {
      int64_t task_idx = m_worker_current_task_indices[status.MPI_SOURCE -
                                                       (status.MPI_SOURCE > m_config.manager_rank)];
      m_worker_current_task_indices[status.MPI_SOURCE -
                                    (status.MPI_SOURCE > m_config.manager_rank)] = -1;
      assert(task_idx >= 0 && "Task index should be valid");
      if (static_cast<uint64_t>(task_idx) >= m_results.size()) {
        m_results.resize(task_idx + 1);
      }
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, result_type::value, &count));
      result_type::resize(m_results[task_idx], count);
      m_communicator.recv(m_results[task_idx], status.MPI_SOURCE, Tag::RESULT);
      m_results_received++;
    } else {
      assert(status.MPI_TAG == Tag::REQUEST && "Unexpected tag received in worker");
      m_communicator.recv_empty_message(status.MPI_SOURCE, Tag::REQUEST);
    }
    m_free_worker_indices.push(status.MPI_SOURCE);
  }
};

};  // namespace dynampi
