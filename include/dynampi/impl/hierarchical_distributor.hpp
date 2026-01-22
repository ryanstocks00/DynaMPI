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
#include <thread>
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

    // If true, topology is strictly mapped to physical nodes:
    // Manager <-> Node Coordinators <-> Local Workers
    // Note: Manager is excluded from its node's Local Comm to separate duties.
    bool coordinator_per_node = true;
  };

  struct RunConfig {
    // Stop once we have at least this many results ready to return.
    size_t target_num_tasks = std::numeric_limits<size_t>::max();

    // If false, strictly clips the return vector to `target_num_tasks`.
    // Excess results are buffered for the next call.
    bool allow_more_than_target_tasks = true;

    // Stop if this much time has passed.
    std::optional<double> max_seconds = std::nullopt;
  };

  static constexpr bool prioritize_tasks = Base::prioritize_tasks;
  static const bool ordered = false;

 private:
  typename Base::QueueT m_unallocated_task_queue;
  std::vector<ResultT> m_results;

  enum class CommLayer { Global, Local, Leader };

  struct TaskRequest {
    int worker_rank;
    CommLayer source_layer = CommLayer::Global;  // Which comm did this come from?
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

  MPICommunicator m_communicator;                // Global communicator
  std::optional<MPICommunicator> m_local_comm;   // Intra-node (Shared Memory)
  std::optional<MPICommunicator> m_leader_comm;  // Inter-node (Leaders only)

  std::function<ResultT(TaskT)> m_worker_function;
  Config m_config;

  // Cached parent target to avoid repeated MPI_Group_translate_ranks calls
  mutable std::optional<std::pair<int, CommLayer>> m_cached_parent_target;

  // --- Topology Helper Methods ---

  inline int max_workers_per_coordinator() const {
    const int default_value = std::max(2, static_cast<int>(std::sqrt(m_communicator.size())));
    const int configured = m_config.max_workers_per_coordinator.value_or(default_value);
    return std::max(1, configured);
  }

  // Returns {parent_rank, communicator_layer}
  inline std::pair<int, CommLayer> get_parent_target() const {
    // Return cached value if available
    if (m_cached_parent_target.has_value()) {
      return m_cached_parent_target.value();
    }

    std::pair<int, CommLayer> result;
    if (m_config.coordinator_per_node) {
      if (is_root_manager()) {
        result = {-1, CommLayer::Global};
      } else if (m_local_comm && m_local_comm->rank() > 0) {
        // Case 1: I am a Local Worker (Rank > 0 in Local Comm)
        // Parent is the Node Coordinator (Local Rank 0).
        result = {0, CommLayer::Local};
      } else if (m_leader_comm) {
        // Case 2: I am a Node Coordinator (Local Rank 0).
        // Parent is the Global Manager.
        // With the new topology, Manager is ALWAYS in the leader comm.
        int global_manager = m_config.manager_rank;
        MPI_Group world_group, leader_group;
        MPI_Comm_group(m_communicator.get(), &world_group);
        MPI_Comm_group(m_leader_comm->get(), &leader_group);
        int leader_rank;
        MPI_Group_translate_ranks(world_group, 1, &global_manager, leader_group, &leader_rank);

        DYNAMPI_ASSERT_NE(leader_rank, MPI_UNDEFINED,
                          "Manager must be part of the leader communicator in this topology");
        result = {leader_rank, CommLayer::Leader};
      } else {
        // Should not be reachable if topology initialized correctly
        DYNAMPI_ASSERT(false, "Unreachable topology state in get_parent_target");
        result = {-1, CommLayer::Global};
      }
    } else {
      // Original Logic
      int rank = m_communicator.rank();
      int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
      if (virtual_rank == 0) {
        result = {-1, CommLayer::Global};  // Root
      } else {
        int virtual_parent = (virtual_rank - 1) / max_workers_per_coordinator();
        int parent_rank =
            virtual_parent == 0 ? m_config.manager_rank : worker_for_idx(virtual_parent - 1);
        result = {parent_rank, CommLayer::Global};
      }
    }

    // Cache the result
    m_cached_parent_target = result;
    return result;
  }

  inline int total_num_children(int rank) const {
    if (m_config.coordinator_per_node) {
      DYNAMPI_UNIMPLEMENTED("Recursive child counting not supported/needed in Node topology mode");
      return 0;
    }
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

  // Calculate number of direct children based on active topology
  inline int num_direct_children() const {
    if (m_config.coordinator_per_node) {
      int count = 0;
      // 1. Local Children: Everyone in local comm except me (Rank 0)
      if (m_local_comm && m_local_comm->rank() == 0) {
        count += (m_local_comm->size() - 1);
      }
      // 2. Remote Children: If I am Manager, other Leaders are my children.
      // Note: In this topology, Manager is IN leader comm, but NOT in local comm.
      if (is_root_manager() && m_leader_comm) {
        count += (m_leader_comm->size() - 1);
      }
      return count;
    } else {
      // Original Logic
      int rank = m_communicator.rank();
      int num_children = 0;
      int max_children = max_workers_per_coordinator();
      for (int i = 0; i < max_children; ++i) {
        int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
        int virtual_child = virtual_rank * max_children + i + 1;
        if (virtual_child < m_communicator.size()) {
          num_children++;
        }
      }
      return num_children;
    }
  }

  bool is_leaf_worker() const {
    if (m_config.coordinator_per_node) {
      if (is_root_manager()) return false;

      // If I am NOT in local comm (should only be Manager, handled above), panic?
      // Actually, with this topology, everyone except Manager is in local comm.
      if (!m_local_comm) return true;  // Safety fallback

      // Standard Worker: Rank > 0 in Local Comm
      if (m_local_comm->rank() > 0) return true;

      // Node Coordinator: Rank 0 in Local Comm.
      // Leaf only if single-core node (no children).
      return num_direct_children() == 0;
    } else {
      int rank = m_communicator.rank();
      int max_children = max_workers_per_coordinator();
      int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
      int first_child_virtual = virtual_rank * max_children + 1;
      return first_child_virtual >= m_communicator.size();
    }
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
    // --- Initialize Topology Communicators ---
    if (m_config.coordinator_per_node) {
      // 1. Identify physical nodes via split_by_node
      MPICommunicator node_comm = m_communicator.split_by_node();

      // 2. Create Local Comm: Exclude Manager!
      // If I am Manager, color is Undefined (I don't participate in local worker pool).
      // Everyone else participates.
      int local_color = (m_communicator.rank() == m_config.manager_rank) ? MPI_UNDEFINED : 0;

      auto local_comm_opt = node_comm.split(local_color, m_communicator.rank());
      if (local_comm_opt.has_value()) {
        m_local_comm.emplace(std::move(*local_comm_opt));
      }

      // 3. Create Leader Comm
      // Who joins?
      // A: The Manager (Always)
      // B: The Node Coordinators (Rank 0 of the *Local* Comm)
      bool is_manager = (m_communicator.rank() == m_config.manager_rank);
      bool is_node_coordinator = (m_local_comm && m_local_comm->rank() == 0);

      int leader_color = (is_manager || is_node_coordinator) ? 0 : MPI_UNDEFINED;

      // Key is global rank to maintain global ordering among leaders
      auto leader_comm_opt = m_communicator.split(leader_color, m_communicator.rank());
      if (leader_comm_opt.has_value()) {
        m_leader_comm.emplace(std::move(*leader_comm_opt));
      }
    }

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
      // Leaf workers (usually local ranks > 0) just request from parent
      send_to_parent(nullptr, Tag::REQUEST);
      while (!m_done) {
        receive_from_anyone();
      }
    } else {
      // Intermediate nodes (Node Coordinators)
      int num_children = num_direct_children();
      int prefetch = num_children * m_config.batch_size_multiplier;

      // Initial request to parent (Manager)
      send_to_parent(prefetch, Tag::REQUEST_BATCH);

      while (!m_done) {
        // If we have no tasks to give, wait for tasks from parent
        while (!m_done && m_unallocated_task_queue.empty()) {
          receive_from_anyone();
        }

        size_t num_tasks_should_be_received = m_unallocated_task_queue.size();

        // Process tasks: Give to workers or execute ourselves if needed
        while (!m_unallocated_task_queue.empty()) {
          if (m_done) break;

          if (m_free_worker_indices.empty()) {
            // Must wait for a worker to become free
            receive_from_anyone();
          } else {
            allocate_task_to_child();
          }
        }

        // Wait for results from children
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

    send_to_parent(results, Tag::RESULT_BATCH);
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
      m_free_worker_indices.pop();

      // Determine target and communicator based on request source
      int worker_rank = request.worker_rank;
      CommLayer layer = request.source_layer;

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

        send_to_worker(tasks, worker_rank, Tag::TASK_BATCH, layer);
        m_tasks_sent_to_child += tasks.size();
      } else {
        const TaskT task = get_next_task_to_send();
        send_to_worker(task, worker_rank, Tag::TASK, layer);
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

  // --- Abstract Send Wrappers ---

  template <typename T>
  void send_to_parent(const T& data, Tag tag) {
    auto [target, layer] = get_parent_target();
    DYNAMPI_ASSERT_NE(target, -1, "Root cannot send to parent");

    if (m_config.coordinator_per_node) {
      if (layer == CommLayer::Leader) {
        DYNAMPI_ASSERT(m_leader_comm.has_value(), "Expected leader comm");
        m_leader_comm->send(data, target, tag);
      } else if (layer == CommLayer::Local) {
        DYNAMPI_ASSERT(m_local_comm.has_value(), "Expected local comm");
        m_local_comm->send(data, target, tag);
      } else {
        // Global fallback
        m_communicator.send(data, target, tag);
      }
    } else {
      m_communicator.send(data, target, tag);
    }
  }

  template <typename T>
  void send_to_worker(const T& data, int rank, Tag tag, CommLayer layer) {
    if (m_config.coordinator_per_node) {
      if (layer == CommLayer::Local) {
        DYNAMPI_ASSERT(m_local_comm.has_value(), "Cannot send local without local comm");
        m_local_comm->send(data, rank, tag);
      } else if (layer == CommLayer::Leader) {
        DYNAMPI_ASSERT(m_leader_comm.has_value(), "Cannot send leader without leader comm");
        m_leader_comm->send(data, rank, tag);
      } else {
        m_communicator.send(data, rank, tag);
      }
    } else {
      m_communicator.send(data, rank, tag);
    }
  }

  void send_done_to_children_when_free() {
    const int direct_children = num_direct_children();
    int done_sent_count = 0;
    while (done_sent_count < direct_children) {
      if (m_free_worker_indices.empty()) {
        receive_from_anyone();
        continue;
      }
      TaskRequest request = m_free_worker_indices.top();
      m_free_worker_indices.pop();

      send_to_worker(nullptr, request.worker_rank, Tag::DONE, request.source_layer);
      done_sent_count++;
    }
  }

  using result_mpi_type = MPI_Type<ResultT>;
  using task_mpi_type = MPI_Type<TaskT>;

  void receive_result_from(MPI_Status status, MPICommunicator& source_comm, CommLayer layer) {
    m_results.push_back(ResultT{});
    if (result_mpi_type::resize_required) {
      DYNAMPI_UNIMPLEMENTED(  // LCOV_EXCL_LINE
          "Dynamic resizing of results is not supported in hierarchical distribution");
    }
    source_comm.recv(m_results.back(), status.MPI_SOURCE, Tag::RESULT);
    m_results_received_from_child++;
    m_free_worker_indices.push(
        TaskRequest{.worker_rank = status.MPI_SOURCE, .source_layer = layer});
  }

  void receive_result_batch_from(MPI_Status status, MPICommunicator& source_comm, CommLayer layer) {
    using message_type = MPI_Type<std::vector<ResultT>>;
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
    std::vector<ResultT> results;
    message_type::resize(results, count);
    source_comm.recv(results, status.MPI_SOURCE, Tag::RESULT_BATCH);
    m_free_worker_indices.push({.worker_rank = status.MPI_SOURCE,
                                .source_layer = layer,
                                .num_tasks_requested = static_cast<int>(results.size())});
    std::copy(results.begin(), results.end(), std::back_inserter(m_results));
    m_results_received_from_child += results.size();
  }

  void receive_execute_return_task_from(MPI_Status status, MPICommunicator& source_comm,
                                        [[maybe_unused]] CommLayer layer) {
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_mpi_type::value, &count));
    TaskT message;
    task_mpi_type::resize(message, count);
    source_comm.recv(message, status.MPI_SOURCE, Tag::TASK);
    m_tasks_received_from_parent++;
    ResultT result = m_worker_function(message);
    m_tasks_executed++;
    // Reply on the same comm/layer
    source_comm.send(result, status.MPI_SOURCE, Tag::RESULT);
    m_results_sent_to_parent++;
  }

  void receive_task_batch_from(MPI_Status status, MPICommunicator& source_comm,
                               [[maybe_unused]] CommLayer layer) {
    if constexpr (prioritize_tasks) {
      DYNAMPI_UNIMPLEMENTED("Prioritized hierarchical distribution");
    } else {
      using message_type = MPI_Type<std::vector<TaskT>>;
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
      std::vector<TaskT> tasks;
      message_type::resize(tasks, count);
      source_comm.recv(tasks, status.MPI_SOURCE, Tag::TASK_BATCH);
      m_tasks_received_from_parent += tasks.size();
      for (const auto& task : tasks) {
        m_unallocated_task_queue.push_back(task);
      }
    }
  }

  void receive_request_from(MPI_Status status, MPICommunicator& source_comm, CommLayer layer) {
    source_comm.recv_empty_message(status.MPI_SOURCE, Tag::REQUEST);
    m_free_worker_indices.push(
        TaskRequest{.worker_rank = status.MPI_SOURCE, .source_layer = layer});
  }

  void receive_request_batch_from(MPI_Status status, MPICommunicator& source_comm,
                                  CommLayer layer) {
    int request_count;
    source_comm.recv(request_count, status.MPI_SOURCE, Tag::REQUEST_BATCH);
    m_free_worker_indices.push(TaskRequest{.worker_rank = status.MPI_SOURCE,
                                           .source_layer = layer,
                                           .num_tasks_requested = request_count});
  }

  void receive_done_from(MPI_Status status, MPICommunicator& source_comm,
                         [[maybe_unused]] CommLayer layer) {
    source_comm.recv_empty_message(status.MPI_SOURCE, Tag::DONE);
    m_done = true;
  }

  void receive_from_anyone() {
    DYNAMPI_ASSERT_GT(m_communicator.size(), 1,
                      "There should be at least one worker to receive results from");

    MPI_Status status;
    CommLayer layer = CommLayer::Global;
    MPICommunicator* active_comm = &m_communicator;

    if (m_config.coordinator_per_node) {
      // Poll active communicators non-blocking until one matches
      bool found = false;
      while (!found) {
        if (m_local_comm) {
          auto opt_status = m_local_comm->iprobe();
          if (opt_status.has_value()) {
            status = opt_status.value();
            layer = CommLayer::Local;
            active_comm = &m_local_comm.value();
            found = true;
            break;
          }
        }
        if (m_leader_comm) {
          auto opt_status = m_leader_comm->iprobe();
          if (opt_status.has_value()) {
            status = opt_status.value();
            layer = CommLayer::Leader;
            active_comm = &m_leader_comm.value();
            found = true;
            break;
          }
        }
        std::this_thread::yield();
      }
    } else {
      // Legacy: blocking probe on global
      status = m_communicator.probe();
      layer = CommLayer::Global;
      active_comm = &m_communicator;
    }

    // Assert that the tag is a valid Tag enum value before casting
    DYNAMPI_ASSERT(status.MPI_TAG >= static_cast<int>(Tag::TASK) &&
                       status.MPI_TAG <= static_cast<int>(Tag::REQUEST_BATCH),
                   "Received invalid MPI tag: " + std::to_string(status.MPI_TAG));
    Tag tag = static_cast<Tag>(status.MPI_TAG);
    switch (tag) {
      case Tag::TASK:
        return receive_execute_return_task_from(status, *active_comm, layer);
      case Tag::TASK_BATCH:
        return receive_task_batch_from(status, *active_comm, layer);
      case Tag::RESULT:
        return receive_result_from(status, *active_comm, layer);
      case Tag::RESULT_BATCH:
        return receive_result_batch_from(status, *active_comm, layer);
      case Tag::REQUEST:
        return receive_request_from(status, *active_comm, layer);
      case Tag::REQUEST_BATCH:
        return receive_request_batch_from(status, *active_comm, layer);
      case Tag::DONE:
        return receive_done_from(status, *active_comm, layer);
    }
  }
};

};  // namespace dynampi
