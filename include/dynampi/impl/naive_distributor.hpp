/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <stack>
#include <type_traits>
#include <variant>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/base_distributor.hpp"
#include "dynampi/utilities/timer.hpp"

namespace dynampi {

template <typename TaskT, typename ResultT, typename... Options>
class NaiveMPIWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
  };

  struct RunConfig {
    // Stop once we have at least this many contiguous results ready to return.
    size_t target_num_tasks = std::numeric_limits<size_t>::max();

    // If false, strictly clips the return vector to `target_num_tasks`.
    // Excess results remain in the internal buffer for the next call.
    bool allow_more_than_target_tasks = true;

    // Stop if this much time has passed.
    std::optional<double> max_seconds = std::nullopt;
  };

  static const bool ordered = true;

 private:
  static constexpr bool prioritize_tasks = get_option_value<prioritize_tasks_t, Options...>();
  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();

  using QueueT = std::conditional_t<prioritize_tasks, std::priority_queue<std::pair<double, TaskT>>,
                                    std::deque<TaskT>>;
  using MPICommunicator = dynampi::MPICommunicator<track_statistics<statistics_mode>>;

  // --- Member Variables ---
  Config m_config;
  MPICommunicator m_communicator;
  std::function<ResultT(TaskT)> m_worker_function;

  QueueT m_unallocated_task_queue;

  // State tracking
  std::vector<int64_t> m_worker_current_task_indices;  // Maps worker_idx -> task_id
  std::stack<int> m_free_worker_ranks;

  // Transient Storage:
  // We use a map to handle out-of-order completions efficiently.
  // Items are removed from here as soon as they become contiguous and ready to return.
  std::map<int64_t, ResultT> m_pending_results;

  // Counters
  size_t m_tasks_sent = 0;       // Total tasks ever sent (acts as the unique ID for the next task)
  size_t m_next_return_idx = 0;  // The unique ID of the next result the user expects
  bool m_finalized = false;

  enum Tag : int { TASK = 0, DONE = 1, RESULT = 2, REQUEST = 3, ERROR = 4 };

 public:
  struct Statistics {
    const CommStatistics& comm_statistics;
    std::vector<size_t> worker_task_counts;
  };

  using StatisticsT =
      std::conditional_t<statistics_mode == StatisticsMode::Detailed, Statistics, std::monostate>;

 private:
  StatisticsT m_statistics;

 public:
  explicit NaiveMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                   Config runtime_config = Config{})
      : m_config(runtime_config),
        m_communicator(runtime_config.comm, MPICommunicator::Duplicate),
        m_worker_function(worker_function),
        m_statistics{create_statistics(m_communicator)} {
    if (is_root_manager()) {
      m_worker_current_task_indices.resize(num_workers(), -1);
    }

    if (m_config.auto_run_workers && !is_root_manager()) {
      run_worker();
    }

    if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
      if (is_root_manager()) m_statistics.worker_task_counts.resize(m_communicator.size(), 0);
    }
  }

  ~NaiveMPIWorkDistributor() {
    if (!m_finalized) finalize();
  }

  // --- Main Interface ---

  [[nodiscard]] std::vector<ResultT> run_tasks(RunConfig config = RunConfig{}) {
    assert(is_root_manager() && "Only the manager can distribute tasks");

    Timer timer;

    // We loop until one of the exit conditions is met.
    while (true) {
      // --- 1. Check Exit Conditions ---

      // A. Have we collected enough contiguous results?
      if (count_contiguous_ready_results() >= config.target_num_tasks) {
        break;
      }

      // B. Time limit check
      if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) {
        break;
      }

      // C. Total exhaustion check
      if (m_unallocated_task_queue.empty() && active_worker_count() == 0) {
        break;
      }

      // --- 2. Action Logic (Send vs Receive) ---

      // Priority: Keep workers busy
      if (!m_unallocated_task_queue.empty() && !m_free_worker_ranks.empty()) {
        send_next_task_to_worker(m_free_worker_ranks.top());
        m_free_worker_ranks.pop();
      } else {
        // Single process mode fallback
        if (num_workers() == 0 && !m_unallocated_task_queue.empty()) {
          run_task_locally();
        }
        // Standard MPI wait
        else if (active_worker_count() > 0) {
          process_incoming_message(true);  // Blocking wait
        }
      }
    }

    // --- 3. Return Logic ---
    size_t limit = std::numeric_limits<size_t>::max();
    if (!config.allow_more_than_target_tasks) {
      limit = config.target_num_tasks;
    }

    return collect_available_results(limit);
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() {
    RunConfig cfg;
    cfg.target_num_tasks = std::numeric_limits<size_t>::max();
    return run_tasks(cfg);
  }

  void finalize() {
    assert(!m_finalized && "Work distribution already finalized");
    if (is_root_manager()) {
      wait_for_all_active_workers();
      broadcast_done();
      m_finalized = true;
    }
  }

  // --- Public Accessors ---

  bool is_root_manager() const { return m_communicator.rank() == m_config.manager_rank; }

  size_t remaining_tasks_count() const {
    assert(is_root_manager() && "Only the manager can check remaining tasks");
    return m_unallocated_task_queue.size();
  }

  const StatisticsT& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    assert(is_root_manager() && "Only the manager can access statistics");
    return m_statistics;
  }

  // --- Task Insertion ---

  void insert_task(TaskT task)
    requires(!prioritize_tasks)
  {
    assert(is_root_manager());
    m_unallocated_task_queue.push_back(std::move(task));
  }

  void insert_task(const TaskT& task, double priority)
    requires(prioritize_tasks)
  {
    assert(is_root_manager());
    m_unallocated_task_queue.emplace(priority, task);
  }

  void insert_tasks(const std::vector<TaskT>& tasks)
    requires(!prioritize_tasks)
  {
    assert(is_root_manager());
    for (const auto& t : tasks) m_unallocated_task_queue.push_back(t);
  }

  // --- Worker Logic ---

  void run_worker() {
    assert(!is_root_manager());
    using task_type = MPI_Type<TaskT>;

    // Handshake
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

      ResultT result = m_worker_function(std::move(message));

      m_communicator.send(result, m_config.manager_rank, Tag::RESULT);
    }
  }

 private:
  // --- Helpers ---

  int num_workers() const { return m_communicator.size() - 1; }

  size_t active_worker_count() const {
    return static_cast<size_t>(num_workers()) - m_free_worker_ranks.size();
  }

  int rank_to_worker_idx(int rank) const {
    return (rank < m_config.manager_rank) ? rank : (rank - 1);
  }

  int worker_idx_to_rank(int idx) const { return (idx < m_config.manager_rank) ? idx : (idx + 1); }

  TaskT pop_next_task() {
    TaskT task;
    if constexpr (prioritize_tasks) {
      task = m_unallocated_task_queue.top().second;
      m_unallocated_task_queue.pop();
    } else {
      task = std::move(m_unallocated_task_queue.front());
      m_unallocated_task_queue.pop_front();
    }
    return task;
  }

  void run_task_locally() {
    TaskT task = pop_next_task();
    // Store result directly in map
    m_pending_results[static_cast<int64_t>(m_tasks_sent)] = m_worker_function(std::move(task));
    m_tasks_sent++;
  }

  void send_next_task_to_worker(int worker_rank) {
    TaskT task = pop_next_task();
    int64_t task_id = static_cast<int64_t>(m_tasks_sent);

    m_worker_current_task_indices[rank_to_worker_idx(worker_rank)] = task_id;
    if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
      m_statistics.worker_task_counts[rank_to_worker_idx(worker_rank)]++;
    }

    m_communicator.send(task, worker_rank, Tag::TASK);
    m_tasks_sent++;
  }

  void process_incoming_message(bool blocking) {
    if (!blocking) return;  // iprobe not available in wrapper

    MPI_Status status = m_communicator.probe(MPI_ANY_SOURCE, MPI_ANY_TAG);
    int source = status.MPI_SOURCE;

    if (status.MPI_TAG == Tag::RESULT) {
      handle_result_message(source, status);
    } else if (status.MPI_TAG == Tag::REQUEST) {
      m_communicator.recv_empty_message(source, Tag::REQUEST);
    } else {
      assert(false && "Unexpected tag received");
    }
    m_free_worker_ranks.push(source);
  }

  void handle_result_message(int source, MPI_Status& probe_status) {
    int worker_idx = rank_to_worker_idx(source);
    int64_t task_id = m_worker_current_task_indices[worker_idx];
    m_worker_current_task_indices[worker_idx] = -1;

    using result_type = MPI_Type<ResultT>;
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&probe_status, result_type::value, &count));

    ResultT result_data;
    result_type::resize(result_data, count);
    m_communicator.recv(result_data, source, Tag::RESULT);

    // Store in map
    m_pending_results[task_id] = std::move(result_data);
  }

  void wait_for_all_active_workers() {
    while (m_free_worker_ranks.size() < static_cast<size_t>(num_workers())) {
      process_incoming_message(true);
    }
  }

  // Efficiently counts how many contiguous results are available
  // starting from m_next_return_idx without expensive vector scans.
  size_t count_contiguous_ready_results() const {
    size_t count = 0;
    auto it = m_pending_results.find(static_cast<int64_t>(m_next_return_idx));

    while (it != m_pending_results.end()) {
      // Verify continuity: Key must match (start + count)
      if (it->first != static_cast<int64_t>(m_next_return_idx + count)) {
        break;
      }
      count++;
      it++;
    }
    return count;
  }

  std::vector<ResultT> collect_available_results(size_t limit) {
    std::vector<ResultT> batch;

    // Extract contiguous results from the map up to the limit.
    // Logic: Look for m_next_return_idx. If found, move it to batch,
    // erase from map (saving memory), increment index.
    while (batch.size() < limit) {
      auto it = m_pending_results.find(static_cast<int64_t>(m_next_return_idx));
      if (it == m_pending_results.end()) {
        break;  // Gap detected or no more results
      }

      batch.push_back(std::move(it->second));
      m_pending_results.erase(it);  // Remove from storage
      m_next_return_idx++;
    }

    return batch;
  }

  void broadcast_done() {
    for (int i = 0; i < num_workers(); i++) {
      m_communicator.send(nullptr, worker_idx_to_rank(i), Tag::DONE);
    }
  }

  static StatisticsT create_statistics(const MPICommunicator& comm) {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{comm.get_statistics(), {}};
    } else {
      return {};
    }
  }
};

}  // namespace dynampi
