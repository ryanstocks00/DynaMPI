/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
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
#include <stdexcept>
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
    int max_workers_per_coordinator = 10;
    int batch_size_multiplier = 1000;
  };

  static constexpr bool prioritize_tasks = Base::prioritize_tasks;
  static const bool ordered = false;

 private:
  typename Base::QueueT m_unallocated_task_queue;
  std::vector<ResultT> m_results;

  struct TaskRequest {
    int worker_rank;
    std::optional<int> num_tasks_requested = std::nullopt;
  };
  std::stack<TaskRequest, std::vector<TaskRequest>> m_free_worker_indices;

  size_t m_tasks_sent_to_child = 0;
  size_t m_results_received_from_child = 0;
  size_t m_results_sent_to_parent = 0;
  size_t m_tasks_received_from_parent = 0;
  size_t m_tasks_executed = 0;

  bool m_finalized = false;
  bool m_done = false;
  std::optional<std::string> m_stored_error;  // Deferred error to throw after cleanup

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();

  using MPICommunicator = dynampi::MPICommunicator<track_statistics<statistics_mode>>;
  MPICommunicator m_communicator;
  std::function<ResultT(TaskT)> m_worker_function;
  Config m_config;

  inline int parent_rank() const {
    int rank = m_communicator.rank();
    int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    if (virtual_rank == 0) return -1;  // Root has no parent
    int virtual_parent = (virtual_rank - 1) / m_config.max_workers_per_coordinator;
    int parent_rank =
        virtual_parent == 0 ? m_config.manager_rank : worker_for_idx(virtual_parent - 1);
    return parent_rank;
  }

  inline int total_num_children(int rank) const {
    int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    int num_children = 0;
    for (int i = 0; i < m_config.max_workers_per_coordinator; ++i) {
      int child = virtual_rank * m_config.max_workers_per_coordinator + i + 1;
      if (child >= m_communicator.size()) break;  // No more children
      num_children += total_num_children(worker_for_idx(child - 1));
    }
    return num_children;
  }

  inline int child_rank(int rank, int child_index) const {
    DYNAMPI_ASSERT_LT(child_index, m_config.max_workers_per_coordinator,
                      "Child index must be less than max workers per coordinator");
    int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    int virtual_child = virtual_rank * m_config.max_workers_per_coordinator + child_index + 1;
    if (virtual_child >= m_communicator.size()) {
      return -1;  // No child exists
    }
    int child_rank = worker_for_idx(virtual_child - 1);
    return child_rank;
  }

  inline int num_direct_children() const {
    int rank = m_communicator.rank();
    int num_children = 0;
    for (int i = 0; i < m_config.max_workers_per_coordinator; ++i) {
      if (child_rank(rank, i) != -1) {
        num_children++;
      }
    }
    return num_children;
  }

  bool is_leaf_worker() const {
    int rank = m_communicator.rank();
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
    std::optional<std::vector<size_t>> worker_task_counts = {};
  };

  using StatisticsT =
      std::conditional_t<statistics_mode == StatisticsMode::Detailed, Statistics, std::monostate>;

  StatisticsT _statistics;

  static StatisticsT create_statistics(const MPICommunicator& comm) {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{.comm_statistics = comm.get_statistics()};
    } else {
      return {};
    }
  }

 public:
  explicit HierarchicalMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                          Config runtime_config = Config{})
      : m_communicator(runtime_config.comm, MPICommunicator::Duplicate),
        m_worker_function(worker_function),
        m_config(runtime_config),
        _statistics{create_statistics(m_communicator)} {
    if (m_config.auto_run_workers && m_communicator.rank() != m_config.manager_rank) {
      run_worker();
    }
  }

  const StatisticsT& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    DYNAMPI_ASSERT(is_root_manager(), "Only the manager can access statistics");
    return _statistics;
  }

  void run_worker() {
    DYNAMPI_ASSERT(m_communicator.rank() != m_config.manager_rank,
                   "Worker cannot run on the manager rank");
    if (is_leaf_worker()) {
      m_communicator.send(nullptr, parent_rank(), Tag::REQUEST);
    } else {
      int num_children = num_direct_children();
      m_communicator.send(num_children * m_config.batch_size_multiplier, parent_rank(),
                          Tag::REQUEST_BATCH);
    }
    if (is_leaf_worker()) {
      while (!m_done) {
        receive_from_anyone();
      }
    } else {
      while (!m_done) {
        while (!m_done && m_unallocated_task_queue.empty()) {
          receive_from_anyone();
        }
        size_t num_tasks_should_be_received = m_unallocated_task_queue.size();
        while (!m_stored_error && !m_unallocated_task_queue.empty()) {
          allocate_task_to_child();
        }
        while (!m_stored_error && m_tasks_sent_to_child > m_results_received_from_child) {
          receive_from_anyone();
        }
        if (m_done) {
          break;
        }
        (void)num_tasks_should_be_received;
        if (!m_stored_error) {
          DYNAMPI_ASSERT_EQ(m_results.size(), num_tasks_should_be_received);
        }
        return_results_and_request_next_batch_from_manager();
      }
      for (int i = 0; i < m_config.max_workers_per_coordinator; ++i) {
        int child = child_rank(m_communicator.rank(), i);
        if (child != -1) {
          m_communicator.send(nullptr, child, Tag::DONE);
        } else {
          break;
        }
      }
    }
  }

  void return_results_and_request_next_batch_from_manager() {
    DYNAMPI_ASSERT(!is_leaf_worker(), "Leaf workers should not return results directly");
    DYNAMPI_ASSERT_NE(m_communicator.rank(), m_config.manager_rank,
                      "Manager should not request tasks from itself");
    // If there was an error from a child, propagate it up instead of results
    std::cout << "Returning results and requesting next batch from manager on rank "
              << m_communicator.rank() << std::endl;
    if (m_stored_error) {
      std::cerr << "Error propagating up from rank " << m_communicator.rank() << " to parent "
                << parent_rank() << ": " << *m_stored_error << std::endl;
      m_communicator.send(*m_stored_error, parent_rank(), Tag::ERROR);
      m_results.clear();
      return;
    }
    std::vector<ResultT> results = m_results;
    m_results.clear();
    m_communicator.send(results, parent_rank(), Tag::RESULT_BATCH);
    m_results_sent_to_parent += results.size();
  }

  bool is_root_manager() const { return m_communicator.rank() == m_config.manager_rank; }

  size_t remaining_tasks_count() const {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can check remaining tasks");
    return m_unallocated_task_queue.size();
  }

  void insert_task(TaskT task)
    requires(!prioritize_tasks)
  {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can distribute tasks");
    m_unallocated_task_queue.push_back(task);
    m_tasks_received_from_parent++;
  }
  void insert_task(const TaskT& task, double priority)
    requires(prioritize_tasks)
  {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can distribute tasks");
    m_unallocated_task_queue.emplace(priority, task);
    m_tasks_received_from_parent++;
  }

  template <typename Range>
    requires std::ranges::input_range<Range> && (!prioritize_tasks)
  void insert_tasks(const Range& tasks) {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can distribute tasks");
    std::copy(std::ranges::begin(tasks), std::ranges::end(tasks),
              std::back_inserter(m_unallocated_task_queue));
    m_tasks_received_from_parent +=
        std::distance(std::ranges::begin(tasks), std::ranges::end(tasks));
  }
  void insert_tasks(const std::vector<TaskT>& tasks)
    requires(!prioritize_tasks)
  {
    insert_tasks(std::span<const TaskT>(tasks));
  }

  void allocate_task_to_child() {
    if (m_communicator.size() > 1) {
      while (m_free_worker_indices.empty()) {
        // If no free workers, wait for a result to be received
        receive_from_anyone();
      }
      if (m_stored_error) {
        return;  // Don't send more tasks after an error
      }
      TaskRequest request = m_free_worker_indices.top();
      int worker = request.worker_rank;
      m_free_worker_indices.pop();
      if (request.num_tasks_requested.has_value()) {
        std::vector<TaskT> tasks;
        int num_tasks = request.num_tasks_requested.value();
        tasks.reserve(num_tasks);
        for (int i = 0; i < num_tasks; ++i) {
          if (m_unallocated_task_queue.empty()) {
            break;  // No more tasks to allocate
          }
          tasks.push_back(get_next_task_to_send());
        }
        m_communicator.send(tasks, worker, Tag::TASK_BATCH);
        m_tasks_sent_to_child += tasks.size();
      } else {
        const TaskT task = get_next_task_to_send();
        m_communicator.send(task, worker, Tag::TASK);
        m_tasks_sent_to_child++;
      }
    } else {
      // If there's only one process, we just run the worker function directly
      const TaskT task = get_next_task_to_send();
      m_results.emplace_back(m_worker_function(task));
      m_tasks_executed++;
    }
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can finish remaining tasks");
    // Distribute tasks until queue is empty or we get an error
    while (!m_unallocated_task_queue.empty() && !m_stored_error) {
      allocate_task_to_child();
    }
    // Wait for all tasks to be processed (results received == tasks sent)
    while (!m_stored_error && m_results_received_from_child < m_tasks_sent_to_child) {
      receive_from_anyone();
    }
    // Ensure all workers are free before finalizing (they may have sent multiple batches)
    while (!m_stored_error &&
           m_free_worker_indices.size() < static_cast<size_t>(num_direct_children())) {
      receive_from_anyone();
    }
    // If there was an error, send DONE to workers and throw
    if (m_stored_error) {
      finalize();
      throw std::runtime_error(*m_stored_error);
    }
    m_results_sent_to_parent = m_results.size();
    DYNAMPI_ASSERT_EQ(m_results_received_from_child, m_tasks_sent_to_child,
                      "All tasks should have been processed by workers before finalizing");
    DYNAMPI_ASSERT_EQ(m_results_sent_to_parent, m_tasks_received_from_parent,
                      "All results should have been sent to the parent before finalizing");
    if (m_communicator.size() > 1)
      DYNAMPI_ASSERT_EQ(m_results_sent_to_parent, m_results_received_from_child + m_tasks_executed,
                        "Manager should not send results to itself");
    if (is_root_manager() && m_communicator.size() > 1)
      DYNAMPI_ASSERT_EQ(m_results.size(), m_results_received_from_child,
                        "Results size should match tasks sent before finalizing");
    // All workers are free and all results received - ready for finalize() to be called
    // (either explicitly by caller or by destructor)
    // TODO(Change this so we don't return the same tasks multiple times)
    return m_results;
  }

  void finalize() {
    DYNAMPI_ASSERT(!m_finalized, "Work distribution already finalized");
    if (is_root_manager()) {
      send_done_to_workers();
    }
    m_finalized = true;  // Set for all managers immediately to prevent double-finalize
    if constexpr (statistics_mode != StatisticsMode::None) {
      if (is_root_manager()) {
        _statistics.worker_task_counts = std::vector<size_t>(m_communicator.size(), 0);
      }
      m_communicator.gather(m_tasks_executed,
                            _statistics.worker_task_counts.has_value()
                                ? &_statistics.worker_task_counts.value()
                                : nullptr,
                            m_config.manager_rank);
    }
  }

  ~HierarchicalMPIWorkDistributor() {
    if (!m_finalized) {
      finalize();
    }
    if (!m_stored_error) {
      DYNAMPI_ASSERT_EQ(m_results_received_from_child, m_tasks_sent_to_child,
                        "All tasks should have been processed by workers before finalizing");
      DYNAMPI_ASSERT_EQ(m_results_sent_to_parent, m_tasks_received_from_parent,
                        "All results should have been sent to the parent before finalizing");
    }
    if (is_leaf_worker())
      DYNAMPI_ASSERT_EQ(m_results_received_from_child, 0,
                        "Leaf workers should not receive results from children");
    else if (!m_stored_error && m_communicator.size() > 1)
      DYNAMPI_ASSERT_EQ(m_results_received_from_child + m_tasks_executed, m_results_sent_to_parent,
                        "Results received from children should match results sent to parent");
  }

 private:
  TaskT get_next_task_to_send() {
    DYNAMPI_ASSERT(is_root_manager() || !is_leaf_worker(),
                   "Leaf workers should not send tasks directly");
    DYNAMPI_ASSERT(!m_unallocated_task_queue.empty(), "There should be tasks available to send");
    TaskT task;
    if constexpr (std::is_same_v<decltype(m_unallocated_task_queue), std::deque<TaskT>>) {
      task = m_unallocated_task_queue.front();
      m_unallocated_task_queue.pop_front();
    } else {
      task = m_unallocated_task_queue.top().second;
      m_unallocated_task_queue.pop();
    }
    return task;
  }

  void send_done_to_workers() {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can send done messages to workers");
    DYNAMPI_ASSERT_EQ(m_free_worker_indices.size(), static_cast<size_t>(num_direct_children()),
                      "All workers should be free or errored before finalizing");
    for (int i = 0; i < m_config.max_workers_per_coordinator; ++i) {
      int child = child_rank(m_communicator.rank(), i);
      if (child == -1) {
        break;
      }
      m_communicator.send(nullptr, child, Tag::DONE);
    }
  }

  int idx_for_worker(int worker_rank) const {
    DYNAMPI_ASSERT_NE(worker_rank, m_config.manager_rank,
                      "Manager rank should not be used as a worker rank");
    if (worker_rank < m_config.manager_rank) {
      return worker_rank;
    } else {
      return worker_rank - 1;
    }
  }

  int worker_for_idx(int idx) const { return (idx < m_config.manager_rank) ? idx : (idx + 1); }

  using result_mpi_type = MPI_Type<ResultT>;
  using task_mpi_type = MPI_Type<TaskT>;

  void receive_result_from(MPI_Status status) {
    m_results.push_back(ResultT{});
    if (result_mpi_type::resize_required) {
      DYNAMPI_UNIMPLEMENTED(  // LCOV_EXCL_LINE
          "Dynamic resizing of results is not supported in hierarchical distribution");
      // int count;
      // DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, result_mpi_type::value, &count));
      // result_mpi_type::resize(_results.back(), count);
    }
    m_communicator.recv(m_results.back(), status.MPI_SOURCE, Tag::RESULT);
    m_results_received_from_child++;
    m_free_worker_indices.push(TaskRequest{.worker_rank = status.MPI_SOURCE});
  }

  void receive_result_batch_from(MPI_Status status) {
    using message_type = MPI_Type<std::vector<ResultT>>;
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
    std::vector<ResultT> results;
    message_type::resize(results, count);
    m_communicator.recv(results, status.MPI_SOURCE, Tag::RESULT_BATCH);
    m_free_worker_indices.push({.worker_rank = status.MPI_SOURCE,
                                .num_tasks_requested = static_cast<int>(results.size())});
    std::copy(results.begin(), results.end(), std::back_inserter(m_results));
    m_results_received_from_child += results.size();
  }

  void receive_execute_return_task_from(MPI_Status status) {
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_mpi_type::value, &count));
    TaskT message;
    task_mpi_type::resize(message, count);
    m_communicator.recv(message, status.MPI_SOURCE, Tag::TASK);
    m_tasks_received_from_parent++;
    try {
      ResultT result = m_worker_function(message);
      m_tasks_executed++;
      m_communicator.send(result, status.MPI_SOURCE, Tag::RESULT);
      m_results_sent_to_parent++;
    } catch (const std::exception& e) {
      std::string error_message = e.what();
      std::cerr << "Error executing task on rank " << m_communicator.rank() << std::endl;
      m_communicator.send(error_message, status.MPI_SOURCE, Tag::ERROR);
      m_stored_error = error_message;
    }
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
      m_communicator.recv(tasks, status.MPI_SOURCE, Tag::TASK_BATCH);
      m_tasks_received_from_parent += tasks.size();
      for (const auto& task : tasks) {
        m_unallocated_task_queue.push_back(task);
      }
    }
  }

  void receive_request_from(MPI_Status status) {
    m_communicator.recv_empty_message(status.MPI_SOURCE, Tag::REQUEST);
    m_free_worker_indices.push(TaskRequest{.worker_rank = status.MPI_SOURCE});
  }

  void receive_request_batch_from(MPI_Status status) {
    int request_count;
    m_communicator.recv(request_count, status.MPI_SOURCE, Tag::REQUEST_BATCH);
    m_free_worker_indices.push(
        TaskRequest{.worker_rank = status.MPI_SOURCE, .num_tasks_requested = request_count});
  }

  void receive_done_from(MPI_Status status) {
    m_communicator.recv_empty_message(status.MPI_SOURCE, Tag::DONE);
    m_done = true;
  }

  void receive_error_from(MPI_Status status) {
    // String requires resizing, so get count from status first
    using string_mpi_type = MPI_Type<std::string>;
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, string_mpi_type::value, &count));
    std::string error_message;
    string_mpi_type::resize(error_message, count);
    m_communicator.recv(error_message, status.MPI_SOURCE, Tag::ERROR);
    // Log the error
    std::cerr << "Error received on rank " << m_communicator.rank() << " from source "
              << status.MPI_SOURCE << ": " << error_message << std::endl;
    // Store error to throw later (after cleanup), continue normal flow to avoid deadlock
    if (!m_stored_error) {
      m_stored_error =
          "MPI error from source " + std::to_string(status.MPI_SOURCE) + ": " + error_message;
    }
    m_free_worker_indices.push(TaskRequest{.worker_rank = status.MPI_SOURCE});
  }

  void receive_from_anyone() {
    DYNAMPI_ASSERT_GT(m_communicator.size(), 1,
                      "There should be at least one worker to receive results from");
    MPI_Status status = m_communicator.probe();
    // Assert that the tag is a valid Tag enum value before casting
    DYNAMPI_ASSERT(status.MPI_TAG >= static_cast<int>(Tag::TASK) &&
                       status.MPI_TAG <= static_cast<int>(Tag::REQUEST_BATCH),
                   "Received invalid MPI tag: " + std::to_string(status.MPI_TAG));
    Tag tag = static_cast<Tag>(status.MPI_TAG);
    switch (tag) {
      case Tag::TASK:
        return receive_execute_return_task_from(status);
      case Tag::TASK_BATCH:
        return receive_task_batch_from(status);
      case Tag::RESULT:
        return receive_result_from(status);
      case Tag::RESULT_BATCH:
        return receive_result_batch_from(status);
      case Tag::REQUEST:
        return receive_request_from(status);
      case Tag::REQUEST_BATCH:
        return receive_request_batch_from(status);
      case Tag::DONE:
        return receive_done_from(status);
      case Tag::ERROR:
        return receive_error_from(status);
    }
  }
};

};  // namespace dynampi
