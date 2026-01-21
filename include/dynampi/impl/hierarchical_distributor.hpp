/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <stack>
#include <type_traits>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/base_distributor.hpp"
#include "dynampi/utilities/assert.hpp"
#include "dynampi/utilities/timer.hpp"

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
    std::optional<int> max_workers_per_coordinator = std::nullopt;
    int batch_size_multiplier = 2;
  };

  struct RunConfig {
    // Stop once we have at least this many results ready to return.
    size_t target_num_tasks = std::numeric_limits<size_t>::max();

    // If false, strictly clips the return vector to `target_num_tasks`.
    // Excess results are buffered for the next call.
    // MOVED: Placed before max_seconds to prevent implicit bool->double conversion bugs.
    bool allow_more_than_target_tasks = true;

    // Stop if this much time has passed.
    std::optional<double> max_seconds = std::nullopt;
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
  size_t m_results_returned = 0;

  bool m_finalized = false;
  bool m_done = false;

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();

  using MPICommunicator = dynampi::MPICommunicator<track_statistics<statistics_mode>>;
  MPICommunicator m_communicator;
  std::function<ResultT(TaskT)> m_worker_function;
  Config m_config;

  inline int max_workers_per_coordinator() const {
    const int default_value = std::max(2, static_cast<int>(std::sqrt(m_communicator.size())));
    const int configured = m_config.max_workers_per_coordinator.value_or(default_value);
    return std::max(1, configured);
  }

  inline int parent_rank() const {
    int rank = m_communicator.rank();
    int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    if (virtual_rank == 0) return -1;  // Root has no parent
    int virtual_parent = (virtual_rank - 1) / max_workers_per_coordinator();
    int parent_rank =
        virtual_parent == 0 ? m_config.manager_rank : worker_for_idx(virtual_parent - 1);
    return parent_rank;
  }

  inline int total_num_children(int rank) const {
    int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    int num_children = 0;
    int max_children = max_workers_per_coordinator();
    for (int i = 0; i < max_children; ++i) {
      int child = virtual_rank * max_children + i + 1;
      if (child >= m_communicator.size()) break;  // No more children
      num_children += 1 + total_num_children(worker_for_idx(child - 1));
    }
    return num_children;
  }

  inline int child_rank(int rank, int child_index) const {
    int max_children = max_workers_per_coordinator();
    DYNAMPI_ASSERT_LT(child_index, max_children,
                      "Child index must be less than max workers per coordinator");
    int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    int virtual_child = virtual_rank * max_children + child_index + 1;
    if (virtual_child >= m_communicator.size()) {
      return -1;  // No child exists
    }
    int child_rank = worker_for_idx(virtual_child - 1);
    return child_rank;
  }

  inline int num_direct_children() const {
    int rank = m_communicator.rank();
    int num_children = 0;
    int max_children = max_workers_per_coordinator();
    for (int i = 0; i < max_children; ++i) {
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
    TASK_BATCH = 4,
    RESULT_BATCH = 5,
    REQUEST_BATCH = 6
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
      while (!m_done) {
        receive_from_anyone();
      }
    } else {
      int num_children = num_direct_children();
      m_communicator.send(num_children * m_config.batch_size_multiplier, parent_rank(),
                          Tag::REQUEST_BATCH);
      while (!m_done) {
        while (!m_done && m_unallocated_task_queue.empty()) {
          receive_from_anyone();
        }

        size_t num_tasks_should_be_received = m_unallocated_task_queue.size();

        while (!m_unallocated_task_queue.empty()) {
          if (m_done) break;
          // Intermediate nodes must respect worker availability
          if (m_free_worker_indices.empty()) {
            receive_from_anyone();
          } else {
            allocate_task_to_child();
          }
        }

        while (m_tasks_sent_to_child > m_results_received_from_child) {
          receive_from_anyone();
        }

        if (m_done) break;

        (void)num_tasks_should_be_received;
        DYNAMPI_ASSERT_EQ(m_results.size(), num_tasks_should_be_received);
        return_results_and_request_next_batch_from_manager();
      }
      send_done_to_children_when_free();
    }
  }

  void return_results_and_request_next_batch_from_manager() {
    DYNAMPI_ASSERT(!is_leaf_worker(), "Leaf workers should not return results directly");
    DYNAMPI_ASSERT_NE(m_communicator.rank(), m_config.manager_rank,
                      "Manager should not request tasks from itself");
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
      DYNAMPI_ASSERT(!m_free_worker_indices.empty(), "Cannot allocate task with no free workers");

      TaskRequest request = m_free_worker_indices.top();
      int worker = request.worker_rank;
      m_free_worker_indices.pop();
      if (request.num_tasks_requested.has_value()) {
        std::vector<TaskT> tasks;
        int num_tasks = request.num_tasks_requested.value();

        const int actual_num_tasks =
            std::min<int>(num_tasks, static_cast<int>(m_unallocated_task_queue.size()));
        tasks.reserve(actual_num_tasks);
        if constexpr (std::is_same_v<decltype(m_unallocated_task_queue), std::deque<TaskT>>) {
          tasks.assign(m_unallocated_task_queue.begin(),
                       m_unallocated_task_queue.begin() + actual_num_tasks);
          m_unallocated_task_queue.erase(m_unallocated_task_queue.begin(),
                                         m_unallocated_task_queue.begin() + actual_num_tasks);
        } else {
          for (int i = 0; i < actual_num_tasks; ++i) {
            tasks.push_back(std::move(m_unallocated_task_queue.top().second));
            m_unallocated_task_queue.pop();
          }
        }
        m_communicator.send(tasks, worker, Tag::TASK_BATCH);
        m_tasks_sent_to_child += tasks.size();
      } else {
        const TaskT task = get_next_task_to_send();
        m_communicator.send(task, worker, Tag::TASK);
        m_tasks_sent_to_child++;
      }
    } else {
      const TaskT task = get_next_task_to_send();
      m_results.emplace_back(m_worker_function(task));
      m_tasks_executed++;
    }
  }

  [[nodiscard]] std::vector<ResultT> run_tasks(const RunConfig& config = RunConfig{}) {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can finish remaining tasks");
    Timer timer;

    while (true) {
      // A. Target reached
      if (m_results.size() >= config.target_num_tasks) {
        break;
      }

      // B. Time limit
      if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) {
        break;
      }

      // C. Exhaustion
      size_t active_tasks = m_tasks_sent_to_child - m_results_received_from_child;
      if (m_unallocated_task_queue.empty() && active_tasks == 0) {
        break;
      }

      bool tasks_available = !m_unallocated_task_queue.empty();
      bool workers_available = !m_free_worker_indices.empty();
      bool is_single_proc = (m_communicator.size() == 1);

      if (tasks_available && (is_single_proc || workers_available)) {
        allocate_task_to_child();
      } else if (active_tasks > 0 || (tasks_available && !workers_available)) {
        receive_from_anyone();
      }
    }

    // --- Return Logic ---
    std::vector<ResultT> batch;

    size_t available = m_results.size();
    size_t count_to_return = available;

    if (!config.allow_more_than_target_tasks) {
      count_to_return = std::min(available, config.target_num_tasks);
    }

    batch.reserve(count_to_return);
    auto end_it = m_results.begin() + count_to_return;
    std::move(m_results.begin(), end_it, std::back_inserter(batch));
    m_results.erase(m_results.begin(), end_it);

    m_results_sent_to_parent += batch.size();
    return batch;
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() {
    RunConfig cfg;
    cfg.target_num_tasks = std::numeric_limits<size_t>::max();
    return run_tasks(cfg);
  }

  void finalize() {
    DYNAMPI_ASSERT(!m_finalized, "Work distribution already finalized");
    if (is_root_manager()) {
      send_done_to_children_when_free();
    }
    m_finalized = true;
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
    DYNAMPI_ASSERT_EQ(m_results_received_from_child, m_tasks_sent_to_child,
                      "All tasks should have been processed by workers before finalizing");
    DYNAMPI_ASSERT_EQ(m_results_sent_to_parent, m_tasks_received_from_parent,
                      "All results should have been sent to the parent before finalizing");
    if (is_leaf_worker())
      DYNAMPI_ASSERT_EQ(m_results_received_from_child, 0,
                        "Leaf workers should not receive results from children");
    else if (m_communicator.size() > 1)
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

  void send_done_to_children_when_free() {
    const int direct_children = num_direct_children();
    int done_sent_count = 0;
    while (done_sent_count < direct_children) {
      if (m_free_worker_indices.empty()) {
        receive_from_anyone();
        continue;
      }
      const int child = m_free_worker_indices.top().worker_rank;
      m_free_worker_indices.pop();
      m_communicator.send(nullptr, child, Tag::DONE);
      done_sent_count++;
    }
  }

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
    ResultT result = m_worker_function(message);
    m_tasks_executed++;
    m_communicator.send(result, status.MPI_SOURCE, Tag::RESULT);
    m_results_sent_to_parent++;
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
    }
  }
};

};  // namespace dynampi
