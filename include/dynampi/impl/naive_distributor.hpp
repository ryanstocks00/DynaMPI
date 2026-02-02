/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
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

    // If set, use pre-posted MPI_IRecv instead of MPI_Probe with the given maximum message size (in
    // elements) If std::nullopt, use MPI_Probe (default behavior)
    std::optional<size_t> prepost_recv_size = std::nullopt;
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
  // We use a vector to store results by task ID, with a bitmap to track validity.
  // Items are marked invalid as soon as they become contiguous and ready to return.
  std::vector<ResultT> m_pending_results;
  std::vector<bool> m_pending_results_valid;

  // Counters
  size_t m_tasks_sent = 0;        // Total tasks ever sent (acts as the unique ID for the next task)
  size_t m_front_result_idx = 0;  // The task ID of the result at the front of the vector (index 0)
  size_t m_known_contiguous_results =
      0;  // Number of contiguous valid results starting from m_front_result_idx
  bool m_finalized = false;

  // Pre-posted receive state (only used if prepost_recv_size is set)
  struct PrePostedReceiveState {
    // Worker state
    TaskT task_buffer;
    MPI_Request task_request = MPI_REQUEST_NULL;
    MPI_Request done_request = MPI_REQUEST_NULL;

    // Manager state
    std::vector<ResultT> result_buffers;  // One per worker
    std::vector<MPI_Request> result_requests;
    std::vector<MPI_Request> request_requests;  // For REQUEST messages (empty messages)

    bool initialized = false;
  };
  PrePostedReceiveState m_preposted_state;

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

    if (m_config.prepost_recv_size.has_value()) {
      initialize_preposted_receives();
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
    if (m_config.prepost_recv_size.has_value()) {
      cleanup_preposted_receives();
    }
  }

  // --- Main Interface ---

  [[nodiscard]] std::vector<ResultT> run_tasks(RunConfig config = RunConfig{}) {
    assert(is_root_manager() && "Only the manager can distribute tasks");

    Timer timer;

    // We loop until one of the exit conditions is met.
    while (true) {
      // --- 1. Check Exit Conditions ---

      // A. Have we collected enough contiguous results?
      if (m_known_contiguous_results >= config.target_num_tasks) {
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
      broadcast_done();
    }
    m_finalized = true;
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

    if (m_config.prepost_recv_size.has_value()) {
      run_worker_preposted();
    } else {
      run_worker_probe();
    }
  }

  void run_worker_probe() {
    using task_type = MPI_Type<TaskT>;

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

  void run_worker_preposted() {
    using task_type = MPI_Type<TaskT>;
    size_t max_size = m_config.prepost_recv_size.value();

    // Ensure task buffer is sized appropriately
    task_type::resize(m_preposted_state.task_buffer, static_cast<int>(max_size));

    while (true) {
      // Wait for either TASK or DONE message
      int index;
      MPI_Status status;
      std::array<MPI_Request, 2> requests = {m_preposted_state.task_request,
                                             m_preposted_state.done_request};
      DYNAMPI_MPI_CHECK(MPI_Waitany, (2, requests.data(), &index, &status));

      if (index == 1) {  // DONE message
        // DONE is an empty message, already received
        // Cancel the pending task request if any
        if (m_preposted_state.task_request != MPI_REQUEST_NULL) {
          MPI_Cancel(&m_preposted_state.task_request);
          MPI_Wait(&m_preposted_state.task_request, MPI_STATUS_IGNORE);
        }
        break;
      }

      // TASK message received
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_type::value, &count));

      // Resize buffer if message is smaller than max
      if (count < static_cast<int>(max_size)) {
        task_type::resize(m_preposted_state.task_buffer, count);
      }

      TaskT message = std::move(m_preposted_state.task_buffer);
      ResultT result = m_worker_function(std::move(message));

      m_communicator.send(result, m_config.manager_rank, Tag::RESULT);

      // Repost receive for next TASK
      task_type::resize(m_preposted_state.task_buffer, static_cast<int>(max_size));
      DYNAMPI_MPI_CHECK(MPI_Irecv,
                        (task_type::ptr(m_preposted_state.task_buffer), static_cast<int>(max_size),
                         task_type::value, m_config.manager_rank, Tag::TASK, m_communicator.get(),
                         &m_preposted_state.task_request));
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
    // Store result directly in vector (using relative indexing)
    int64_t task_id = static_cast<int64_t>(m_tasks_sent);
    ensure_result_capacity(task_id - m_front_result_idx + 1);
    size_t vector_idx = task_id - m_front_result_idx;
    m_pending_results[vector_idx] = m_worker_function(std::move(task));
    m_pending_results_valid[vector_idx] = true;
    m_tasks_sent++;
    update_contiguous_results_count(task_id);
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

    if (m_config.prepost_recv_size.has_value()) {
      process_incoming_message_preposted();
    } else {
      process_incoming_message_probe();
    }
  }

  void process_incoming_message_probe() {
    MPI_Status status = m_communicator.probe(MPI_ANY_SOURCE, MPI_ANY_TAG);
    int source = status.MPI_SOURCE;

    if (status.MPI_TAG == Tag::RESULT) {
      handle_result_message(source, status);
    } else {
      DYNAMPI_ASSERT_EQ(status.MPI_TAG, Tag::REQUEST, "Unexpected tag received");
      m_communicator.recv_empty_message(source, Tag::REQUEST);
    }
    m_free_worker_ranks.push(source);
  }

  void process_incoming_message_preposted() {
    // Wait for any pre-posted receive to complete
    int index;
    MPI_Status status;
    std::vector<MPI_Request> all_requests;
    std::vector<std::pair<int, bool>> request_mapping;  // (worker_idx, is_result)
    all_requests.reserve(m_preposted_state.result_requests.size() +
                         m_preposted_state.request_requests.size());

    // Build request list with mapping
    for (size_t i = 0; i < m_preposted_state.result_requests.size(); i++) {
      if (m_preposted_state.result_requests[i] != MPI_REQUEST_NULL) {
        all_requests.push_back(m_preposted_state.result_requests[i]);
        request_mapping.push_back({static_cast<int>(i), true});
      }
    }
    for (size_t i = 0; i < m_preposted_state.request_requests.size(); i++) {
      if (m_preposted_state.request_requests[i] != MPI_REQUEST_NULL) {
        all_requests.push_back(m_preposted_state.request_requests[i]);
        request_mapping.push_back({static_cast<int>(i), false});
      }
    }

    if (all_requests.empty()) {
      // No pre-posted receives, fall back to probe
      process_incoming_message_probe();
      return;
    }

    DYNAMPI_MPI_CHECK(
        MPI_Waitany, (static_cast<int>(all_requests.size()), all_requests.data(), &index, &status));

    int source = status.MPI_SOURCE;
    auto [worker_idx, is_result] = request_mapping[index];

    if (is_result) {
      // RESULT message
      handle_result_message_preposted(source, worker_idx, status);
      // Repost receive for this worker
      repost_result_receive(worker_idx);
    } else {
      // REQUEST message
      DYNAMPI_ASSERT_EQ(status.MPI_TAG, Tag::REQUEST, "Unexpected tag received");
      // Empty message already received
      m_free_worker_ranks.push(source);
      // Repost receive for this worker
      repost_request_receive(worker_idx);
    }
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

    // Store in vector (using relative indexing)
    size_t vector_idx = task_id - m_front_result_idx;
    ensure_result_capacity(vector_idx + 1);
    m_pending_results[vector_idx] = std::move(result_data);
    m_pending_results_valid[vector_idx] = true;
    update_contiguous_results_count(task_id);
  }

  std::vector<ResultT> collect_available_results(size_t limit) {
    std::vector<ResultT> batch;
    size_t num_results_to_return = std::min(limit, m_known_contiguous_results);
    if (num_results_to_return == 0) {
      return batch;
    }

    batch.reserve(num_results_to_return);
    // Extract from the beginning of the vectors (which contain contiguous results starting from
    // m_front_result_idx)
    batch.insert(batch.end(), std::make_move_iterator(m_pending_results.begin()),
                 std::make_move_iterator(m_pending_results.begin() + num_results_to_return));

    // Erase the collected results from the beginning
    m_pending_results_valid.erase(m_pending_results_valid.begin(),
                                  m_pending_results_valid.begin() + num_results_to_return);
    m_pending_results.erase(m_pending_results.begin(),
                            m_pending_results.begin() + num_results_to_return);

    // Update counters: increment m_front_result_idx to reflect the new starting point,
    // and decrement the contiguous count. The vectors now use relative indexing
    // where index 0 corresponds to task_id = m_front_result_idx.
    m_front_result_idx += num_results_to_return;
    m_known_contiguous_results -= num_results_to_return;

    return batch;
  }

  void broadcast_done() {
    for (int i = 0; i < num_workers(); i++) {
      m_communicator.send(nullptr, worker_idx_to_rank(i), Tag::DONE);
    }
  }

  void ensure_result_capacity(size_t required_size) {
    // required_size is relative to m_front_result_idx
    if (m_pending_results.size() < required_size) {
      m_pending_results.resize(required_size);
      m_pending_results_valid.resize(required_size, false);
    }
  }

  // Updates m_known_contiguous_results when a new result arrives.
  // If the result extends the contiguous sequence, increment and check forward.
  void update_contiguous_results_count(int64_t task_id) {
    int64_t expected_task_id =
        static_cast<int64_t>(m_front_result_idx + m_known_contiguous_results);

    // Only update if this result extends the contiguous sequence
    if (task_id == expected_task_id) {
      // Extend the contiguous sequence forward as far as possible
      // Use relative indexing for vector access
      size_t vector_idx = expected_task_id - m_front_result_idx;
      while (vector_idx < m_pending_results.size() && m_pending_results_valid[vector_idx]) {
        m_known_contiguous_results++;
        vector_idx++;
      }
    }
  }

  static StatisticsT create_statistics(const MPICommunicator& comm) {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{comm.get_statistics(), {}};
    } else {
      return {};
    }
  }

  // --- Pre-posted receive helpers ---

  void initialize_preposted_receives() {
    if (m_preposted_state.initialized) return;
    size_t max_size = m_config.prepost_recv_size.value();

    if (is_root_manager()) {
      // Manager: pre-post receives for RESULT and REQUEST from each worker
      int n_workers = num_workers();
      m_preposted_state.result_buffers.resize(n_workers);
      m_preposted_state.result_requests.resize(n_workers, MPI_REQUEST_NULL);
      m_preposted_state.request_requests.resize(n_workers, MPI_REQUEST_NULL);

      using result_type = MPI_Type<ResultT>;
      for (int i = 0; i < n_workers; i++) {
        int worker_rank = worker_idx_to_rank(i);
        result_type::resize(m_preposted_state.result_buffers[i], static_cast<int>(max_size));
        repost_result_receive(i);
        repost_request_receive(i);
      }
    } else {
      // Worker: pre-post receives for TASK and DONE from manager
      using task_type = MPI_Type<TaskT>;
      task_type::resize(m_preposted_state.task_buffer, static_cast<int>(max_size));
      DYNAMPI_MPI_CHECK(MPI_Irecv,
                        (task_type::ptr(m_preposted_state.task_buffer), static_cast<int>(max_size),
                         task_type::value, m_config.manager_rank, Tag::TASK, m_communicator.get(),
                         &m_preposted_state.task_request));
      DYNAMPI_MPI_CHECK(MPI_Irecv, (nullptr, 0, MPI_BYTE, m_config.manager_rank, Tag::DONE,
                                    m_communicator.get(), &m_preposted_state.done_request));
    }

    m_preposted_state.initialized = true;
  }

  void cleanup_preposted_receives() {
    if (!m_preposted_state.initialized) return;

    // Cancel and wait for all outstanding requests
    if (is_root_manager()) {
      for (auto& req : m_preposted_state.result_requests) {
        if (req != MPI_REQUEST_NULL) {
          MPI_Cancel(&req);
          MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
      }
      for (auto& req : m_preposted_state.request_requests) {
        if (req != MPI_REQUEST_NULL) {
          MPI_Cancel(&req);
          MPI_Wait(&req, MPI_STATUS_IGNORE);
        }
      }
    } else {
      if (m_preposted_state.task_request != MPI_REQUEST_NULL) {
        MPI_Cancel(&m_preposted_state.task_request);
        MPI_Wait(&m_preposted_state.task_request, MPI_STATUS_IGNORE);
      }
      if (m_preposted_state.done_request != MPI_REQUEST_NULL) {
        MPI_Cancel(&m_preposted_state.done_request);
        MPI_Wait(&m_preposted_state.done_request, MPI_STATUS_IGNORE);
      }
    }

    m_preposted_state.initialized = false;
  }

  void repost_result_receive(int worker_idx) {
    int worker_rank = worker_idx_to_rank(worker_idx);
    using result_type = MPI_Type<ResultT>;
    size_t max_size = m_config.prepost_recv_size.value();

    result_type::resize(m_preposted_state.result_buffers[worker_idx], static_cast<int>(max_size));
    DYNAMPI_MPI_CHECK(MPI_Irecv,
                      (result_type::ptr(m_preposted_state.result_buffers[worker_idx]),
                       static_cast<int>(max_size), result_type::value, worker_rank, Tag::RESULT,
                       m_communicator.get(), &m_preposted_state.result_requests[worker_idx]));
  }

  void repost_request_receive(int worker_idx) {
    int worker_rank = worker_idx_to_rank(worker_idx);
    DYNAMPI_MPI_CHECK(MPI_Irecv,
                      (nullptr, 0, MPI_BYTE, worker_rank, Tag::REQUEST, m_communicator.get(),
                       &m_preposted_state.request_requests[worker_idx]));
  }

  void handle_result_message_preposted(int source, int worker_idx, MPI_Status& status) {
    int64_t task_id = m_worker_current_task_indices[worker_idx];
    m_worker_current_task_indices[worker_idx] = -1;

    using result_type = MPI_Type<ResultT>;
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, result_type::value, &count));

    ResultT result_data = std::move(m_preposted_state.result_buffers[worker_idx]);
    size_t max_size = m_config.prepost_recv_size.value();

    // Resize if message is smaller than max
    if (count < static_cast<int>(max_size)) {
      result_type::resize(result_data, count);
    }

    // Store in vector (using relative indexing)
    size_t vector_idx = task_id - m_front_result_idx;
    ensure_result_capacity(vector_idx + 1);
    m_pending_results[vector_idx] = std::move(result_data);
    m_pending_results_valid[vector_idx] = true;
    update_contiguous_results_count(task_id);
  }
};

}  // namespace dynampi
