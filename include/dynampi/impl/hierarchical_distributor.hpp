/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <queue>
#include <ranges>
#include <set>
#include <span>
#include <type_traits>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/base_distributor.hpp"
#include "dynampi/impl/hierarchical_topology_detail.hpp"
#include "dynampi/impl/variable_batch.hpp"
#include "dynampi/task_error.hpp"
#include "dynampi/utilities/assert.hpp"
#include "dynampi/utilities/timer.hpp"

namespace dynampi {

template <typename TaskT, typename ResultT, typename... Options>
class HierarchicalWorkDistributor : public BaseWorkDistributor<TaskT, ResultT, Options...> {
  using Base = BaseWorkDistributor<TaskT, ResultT, Options...>;

 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    // Unused; kept for API compatibility with older Config initializers.
    std::optional<size_t> message_batch_size = std::nullopt;
    std::optional<int> max_workers_per_manager = std::nullopt;

    // Tasks requested per subtree leaf. Prefer pipeline_depth for latency
    // hiding because larger batches reduce load-balancing flexibility.
    int batch_size_multiplier = 1;

    // Batches owned concurrently. 1 disables prefetching; 2 double-buffers.
    // Prefetched tasks cannot be reassigned to an idle sibling subtree.
    int pipeline_depth = 2;

    // Root manager -> node managers -> local workers. The root manager is
    // excluded from its node's local worker communicator.
    bool manager_per_node = true;

    // 0 keeps one local group per node; positive values partition it into
    // contiguous groups of at most this size.
    int max_local_group_size = 0;

    // <0 selects fanout automatically, 0 keeps a flat leader layer, and >0
    // recursively caps the number of leader-layer children.
    int max_upper_fanout = -1;

    // Rethrow worker failures at the root; otherwise collect them for
    // take_task_errors().
    bool rethrow_task_errors = true;
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

  struct TaskRequest {
    int worker_rank;
    std::optional<int> num_tasks_requested = std::nullopt;
  };
  static constexpr int kMaxTasksRequested = 1'000'000;  // guard against pathological reserve()
  // FIFO prevents new requests repeatedly jumping older ones.
  std::queue<TaskRequest> m_free_worker_indices;

  size_t m_tasks_sent_to_child = 0;
  size_t m_results_received_from_child = 0;
  size_t m_results_sent_to_parent = 0;
  size_t m_tasks_received_from_parent = 0;
  size_t m_tasks_executed = 0;

  bool m_finalized = false;
  bool m_done = false;

  detail::TaskErrorLog m_task_errors;  // root manager only

  // Batches received during an active round wait here for its boundary.
  bool m_round_active = false;
  std::deque<TaskT> m_prefetched_tasks;
  // Whether the one allowed parent batch request is unanswered.
  bool m_request_outstanding = false;

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();

  using MPICommunicator = dynampi::MPICommunicator<track_statistics<statistics_mode>>;

  MPICommunicator m_communicator;  // Global communicator
  MPIGroup m_world_group;          // Group for the global communicator (for rank translation)
  std::optional<MPIGroup> m_local_group;  // Intra-node group (Shared Memory, excludes manager)

  // A promoted leader owns one group per level and belongs to one parent
  // group. The root manager always owns the top group.
  std::vector<MPIGroup> m_owned_leader_levels;
  std::optional<MPIGroup> m_leader_parent_group;

  // Number of leaf workers fed by this leader.
  int m_subtree_leaf_count = 1;

  std::function<ResultT(TaskT)> m_worker_function;
  Config m_config;

  // Cached parent target to avoid repeated MPI_Group_translate_ranks calls
  mutable std::optional<int> m_cached_parent_target;

  // --- Topology Helper Methods ---

  inline int max_workers_per_manager() const {
    const int default_value = std::max(2, static_cast<int>(std::sqrt(m_communicator.size())));
    const int configured = m_config.max_workers_per_manager.value_or(default_value);
    return std::max(1, configured);
  }

  // Builds the k-ary leader layer. Every rank in each input communicator
  // must reach its matching MPI_Comm_split, including ranks not promoted.
  void setup_leader_hierarchy(bool is_manager, bool is_node_manager) {
    const int local_children =
        (m_local_group && m_local_group->rank() == 0) ? (m_local_group->size() - 1) : 0;
    m_subtree_leaf_count = std::max(1, local_children);

    const int leader_color = (is_manager || is_node_manager) ? 0 : MPI_UNDEFINED;
    auto flat_opt = m_communicator.split(leader_color, m_communicator.rank());
    if (!flat_opt.has_value()) return;
    MPICommunicator flat_comm = std::move(*flat_opt);

    const int manager_count = flat_comm.size() - 1;  // excludes the root manager
    const int effective_fanout =
        detail::resolve_upper_fanout(manager_count, m_config.max_upper_fanout);

    if (manager_count <= effective_fanout) {
      MPIGroup flat_group(flat_comm);
      if (is_manager) {
        m_owned_leader_levels.push_back(std::move(flat_group));
      } else {
        m_leader_parent_group.emplace(std::move(flat_group));
      }
      return;
    }

    // Collective over flat_comm; only node managers join the result.
    auto managers_opt = flat_comm.split(is_manager ? MPI_UNDEFINED : 0, flat_comm.rank());

    bool is_final_round_leader = false;
    if (!is_manager) {
      // optional::emplace works around MPICommunicator's deleted move assignment.
      std::optional<MPICommunicator> round_comm(std::move(*managers_opt));
      while (true) {
        if (round_comm->size() <= effective_fanout) {
          is_final_round_leader = true;
          break;
        }
        const int color = round_comm->rank() / effective_fanout;
        auto group_opt = round_comm->split(color, round_comm->rank());
        MPICommunicator group_comm = std::move(*group_opt);
        const bool is_group_leader = (group_comm.rank() == 0);
        const int group_leaf_count =
            detail::sum_subtree_widths_to_group_leader(m_subtree_leaf_count, group_comm);

        auto leaders_opt =
            round_comm->split(is_group_leader ? 0 : MPI_UNDEFINED, round_comm->rank());

        if (!is_group_leader) {
          m_leader_parent_group.emplace(std::move(group_comm));
          break;
        }
        m_owned_leader_levels.emplace_back(group_comm);
        m_subtree_leaf_count = std::max(1, group_leaf_count);
        round_comm.emplace(std::move(*leaders_opt));
      }
    }

    // Collective over the original flat_comm attaches final leaders to root.
    auto top_opt = flat_comm.split((is_manager || is_final_round_leader) ? 0 : MPI_UNDEFINED,
                                   flat_comm.rank());
    if (top_opt.has_value()) {
      MPIGroup top_group(*top_opt);
      if (is_manager) {
        m_owned_leader_levels.push_back(std::move(top_group));
      } else {
        m_leader_parent_group.emplace(std::move(top_group));
      }
    }
  }

  inline int get_parent_target() const {
    if (m_cached_parent_target.has_value()) {
      return m_cached_parent_target.value();
    }

    int result;
    DYNAMPI_ASSERT(!is_root_manager(), "Root manager should not have a parent");
    if (m_config.manager_per_node) {
      DYNAMPI_ASSERT(m_local_group.has_value() || m_leader_parent_group.has_value(),
                     "Local or leader parent group should be present");
      if (m_local_group && m_local_group->rank() > 0) {
        result = m_local_group->translate_rank(0, m_world_group);
      } else {
        // Root owns a top group; intermediate groups are owned by rank 0.
        DYNAMPI_ASSERT(m_leader_parent_group.has_value(),
                       "Non-manager leader-layer rank must have a parent group");
        int parent_in_group = 0;
        const int manager_in_group =
            m_world_group.translate_rank(m_config.manager_rank, *m_leader_parent_group);
        if (manager_in_group != MPI_UNDEFINED) {
          parent_in_group = manager_in_group;
        }
        result = m_leader_parent_group->translate_rank(parent_in_group, m_world_group);
      }
    } else {
      int virtual_parent = (virtual_rank() - 1) / max_workers_per_manager();
      result = virtual_parent == 0 ? m_config.manager_rank : worker_for_idx(virtual_parent - 1);
    }

    m_cached_parent_target = result;
    return result;
  }

  inline int num_direct_children() const {
    if (m_config.manager_per_node) {
      int count = 0;
      if (m_local_group && m_local_group->rank() == 0) {
        count += (m_local_group->size() - 1);
      }
      for (const auto& level : m_owned_leader_levels) {
        count += (level.size() - 1);
      }
      return count;
    } else {
      int num_children = 0;
      int max_children = max_workers_per_manager();
      for (int i = 0; i < max_children; ++i) {
        int virtual_child = virtual_rank() * max_children + i + 1;
        if (virtual_child < m_communicator.size()) {
          num_children++;
        }
      }
      return num_children;
    }
  }

  // Round size is one task per subtree leaf, not per direct manager child.
  inline int subtree_leaf_count() const {
    return m_config.manager_per_node ? m_subtree_leaf_count : num_direct_children();
  }

  bool is_leaf_worker() const {
    if (m_config.manager_per_node) {
      if (is_root_manager()) return false;

      if (!m_local_group) return true;
      if (m_local_group->rank() > 0) return true;

      return num_direct_children() == 0;
    } else {
      return virtual_rank() * max_workers_per_manager() + 1 >= m_communicator.size();
    }
  }

  enum Tag : int {
    TASK = 0,
    DONE = 1,
    RESULT = 2,
    REQUEST = 3,
    TASK_BATCH = 4,
    RESULT_BATCH = 5,
    REQUEST_BATCH = 6,
    ERROR = 7
  };

  struct Statistics {
    const CommStatistics& comm_statistics;
    std::optional<std::vector<size_t>> worker_task_counts = {};
  };

  using StatisticsT =
      std::conditional_t<statistics_mode != StatisticsMode::None, Statistics, std::monostate>;

  StatisticsT _statistics;

  static StatisticsT create_statistics(const MPICommunicator& comm) {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{.comm_statistics = comm.get_statistics()};
    } else {
      return {};
    }
  }

 public:
  explicit HierarchicalWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                       Config runtime_config = Config{})
      : m_communicator(runtime_config.comm, MPICommunicator::Duplicate),
        m_world_group(m_communicator),
        m_worker_function(worker_function),
        m_config(runtime_config),
        _statistics{create_statistics(m_communicator)} {
    // Managers exchange whole batches as one std::vector<T> message, whose
    // element count is the number of values -- so a non-resizable payload
    // spanning more than one datatype element would send a fraction of each.
    // See check_fixed_size_mpi_type().
    check_fixed_size_mpi_type<TaskT>("task", "HierarchicalWorkDistributor");
    check_fixed_size_mpi_type<ResultT>("result", "HierarchicalWorkDistributor");

    // --- Initialize Topology Groups ---
    if (m_config.manager_per_node) {
      // Identify physical/local domains and exclude the root manager from
      // its node's worker communicator.
      auto local_comm_opt = detail::split_local_worker_communicator(
          m_communicator, m_config.manager_rank, m_config.max_local_group_size);
      if (local_comm_opt.has_value()) {
        // Extract group from the temporary communicator, then let it be freed
        m_local_group.emplace(*local_comm_opt);
      }

      // Build the leader layer: manager + Node Managers (Rank 0 of
      // the *Local* Comm), optionally grouped into a tree -- see
      // setup_leader_hierarchy().
      bool is_manager = (m_communicator.rank() == m_config.manager_rank);
      // Check if we're rank 0 in the local group (node manager)
      bool is_node_manager = false;
      if (m_local_group.has_value()) {
        int my_local_rank = m_local_group->rank();
        is_node_manager = (my_local_rank == 0);
      }

      setup_leader_hierarchy(is_manager, is_node_manager);
    }

    // Ensure every rank finishes topology setup before use, matching
    // HierarchicalLockFreeRMAWorkDistributor's constructor barrier. Without
    // it, a straggler rank still inside setup_leader_hierarchy()'s
    // collectives can miss the manager's first request round, folding that
    // one-time setup latency into the caller's first batch instead of
    // construction.
    DYNAMPI_MPI_CHECK(MPI_Barrier, (static_cast<MPI_Comm>(m_communicator)));

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

  // Tasks that threw, oldest first, removed as they are returned. See
  // Config::rethrow_task_errors.
  [[nodiscard]] std::vector<TaskError> take_task_errors() {
    DYNAMPI_ASSERT(is_root_manager(), "Only the manager collects task errors");
    return m_task_errors.take();
  }

  bool has_task_errors() const {
    DYNAMPI_ASSERT(is_root_manager(), "Only the manager collects task errors");
    return !m_task_errors.empty();
  }

  void run_worker() {
    DYNAMPI_ASSERT(m_communicator.rank() != m_config.manager_rank,
                   "Worker cannot run on the manager rank");
    if (is_leaf_worker()) {
      send_to_parent(nullptr, Tag::REQUEST);
      while (!m_done) {
        receive_from_anyone();
      }
    } else {
      run_intermediate_worker();
    }
  }

  // Keep pipeline_depth batches incomplete in this subtree, but only one
  // request outstanding. The latter prevents unmatched requests at teardown.
  void request_next_batch_if_room() {
    if (m_done || m_request_outstanding) return;
    const int prefetch =
        std::max(1, subtree_leaf_count() * std::max(1, m_config.batch_size_multiplier));
    const int pipeline_depth = std::max(1, m_config.pipeline_depth);
    const size_t uncompleted =
        m_tasks_received_from_parent - m_results_received_from_child - m_tasks_executed;
    if (uncompleted + static_cast<size_t>(prefetch) >
        static_cast<size_t>(pipeline_depth) * static_cast<size_t>(prefetch)) {
      return;
    }
    send_to_parent(prefetch, Tag::REQUEST_BATCH);
    m_request_outstanding = true;
  }

  void send_results_to_parent() {
    DYNAMPI_ASSERT(!is_leaf_worker(), "Leaf workers should not return results directly");
    DYNAMPI_ASSERT_NE(m_communicator.rank(), m_config.manager_rank,
                      "Manager should not request tasks from itself");
    std::vector<ResultT> results = std::move(m_results);
    m_results.clear();

    send_batch(results, get_parent_target(), Tag::RESULT_BATCH);
    m_results_sent_to_parent += results.size();
    request_next_batch_if_room();
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
    m_unallocated_task_queue.push_back(std::move(task));
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
  void insert_tasks(Range&& tasks) {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can distribute tasks");
    size_t inserted = 0;
    for (const auto& task : tasks) {
      m_unallocated_task_queue.push_back(task);
      ++inserted;
    }
    m_tasks_received_from_parent += inserted;
  }
  void insert_tasks(const std::vector<TaskT>& tasks)
    requires(!prioritize_tasks)
  {
    insert_tasks(std::span<const TaskT>(tasks));
  }

  void allocate_task_to_child() {
    if (m_communicator.size() > 1) {
      DYNAMPI_ASSERT(!m_free_worker_indices.empty(), "Cannot allocate task with no free workers");

      TaskRequest request = m_free_worker_indices.front();
      m_free_worker_indices.pop();

      int worker_rank = request.worker_rank;

      if (request.num_tasks_requested.has_value()) {
        const size_t actual_num_tasks = std::min(static_cast<size_t>(*request.num_tasks_requested),
                                                 m_unallocated_task_queue.size());
        std::vector<TaskT> tasks = take_tasks_from_queue(actual_num_tasks);

        send_batch(tasks, worker_rank, Tag::TASK_BATCH);
        m_tasks_sent_to_child += tasks.size();
      } else {
        const TaskT task = get_next_task_to_send();
        m_communicator.send(task, worker_rank, Tag::TASK);
        m_tasks_sent_to_child++;
      }
    } else {
      // Single-rank mode: the manager runs the task on its own stack. Guarded
      // all the same -- this path exists so a workload can be debugged
      // serially, which it cannot be if failures surface differently here.
      const TaskT task = get_next_task_to_send();
      ResultT result;
      auto failure = detail::run_task_guarded(m_worker_function, task, result);
      if (failure) m_task_errors.record(TaskError{m_communicator.rank(), std::move(*failure)});
      m_results.emplace_back(std::move(result));
      m_tasks_executed++;
    }
  }

  [[nodiscard]] std::vector<ResultT> run_tasks(const RunConfig& config = RunConfig{}) {
    DYNAMPI_ASSERT_EQ(m_communicator.rank(), m_config.manager_rank,
                      "Only the manager can finish remaining tasks");
    Timer timer;

    while (true) {
      if (m_results.size() >= config.target_num_tasks) break;
      if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;

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
    // Thrown before draining, so results collected so far stay buffered for
    // whoever catches this and calls again.
    m_task_errors.rethrow_first_if(m_config.rethrow_task_errors);

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

  ~HierarchicalWorkDistributor() {
    if (!m_finalized) {
      finalize();
    }
    m_task_errors.warn_if_unreported("HierarchicalWorkDistributor");
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
  void release_prefetched_tasks() {
    if constexpr (!prioritize_tasks) {
      if (m_prefetched_tasks.empty()) return;
      for (auto& task : m_prefetched_tasks) {
        m_unallocated_task_queue.push_back(std::move(task));
      }
      m_prefetched_tasks.clear();
    }
  }

  void wait_for_work_or_flush() {
    // Flush results while waiting; the parent may need them to declare
    // exhaustion before sending DONE.
    while (!m_done && m_unallocated_task_queue.empty()) {
      receive_from_anyone();
      if (!m_results.empty()) send_results_to_parent();
    }
  }

  void distribute_current_round() {
    m_round_active = true;
    // Finish tasks already in this round even if DONE arrives.
    while (!m_unallocated_task_queue.empty()) {
      if (m_free_worker_indices.empty()) {
        receive_from_anyone();
      } else {
        allocate_task_to_child();
      }
    }
    m_round_active = false;
  }

  void drain_inflight_results_at_shutdown() {
    // A next-batch request can precede the previous results; drain them.
    while (m_tasks_sent_to_child > m_results_received_from_child) {
      receive_from_anyone();  // LCOV_EXCL_LINE -- requires a shutdown/in-flight-result race
    }
    if (!m_results.empty()) {
      send_results_to_parent();  // LCOV_EXCL_LINE -- same race as the drain above
    }
  }

  void run_intermediate_worker() {
    request_next_batch_if_room();
    while (!m_done) {
      release_prefetched_tasks();
      wait_for_work_or_flush();
      if (m_done) break;
      distribute_current_round();
      if (!m_results.empty()) send_results_to_parent();
      if (m_done) break;
    }
    send_done_to_children_when_free();
    drain_inflight_results_at_shutdown();
  }

  std::vector<TaskT> take_tasks_from_queue(size_t count) {
    DYNAMPI_ASSERT_LE(count, m_unallocated_task_queue.size(),
                      "Cannot take more tasks than are available");
    std::vector<TaskT> tasks;
    tasks.reserve(count);
    if constexpr (std::is_same_v<decltype(m_unallocated_task_queue), std::deque<TaskT>>) {
      auto end = m_unallocated_task_queue.begin() + static_cast<ptrdiff_t>(count);
      tasks.assign(m_unallocated_task_queue.begin(), end);
      m_unallocated_task_queue.erase(m_unallocated_task_queue.begin(), end);
    } else {
      for (size_t i = 0; i < count; ++i) {
        tasks.push_back(m_unallocated_task_queue.top().second);
        m_unallocated_task_queue.pop();
      }
    }
    return tasks;
  }

  TaskT get_next_task_to_send() {
    DYNAMPI_ASSERT(is_root_manager() || !is_leaf_worker(),
                   "Leaf workers should not send tasks directly");
    DYNAMPI_ASSERT(!m_unallocated_task_queue.empty(), "There should be tasks available to send");
    auto tasks = take_tasks_from_queue(1);
    return std::move(tasks.front());
  }

  int idx_for_worker(int worker_rank) const {
    DYNAMPI_ASSERT_NE(worker_rank, m_config.manager_rank,
                      "Manager rank should not be used as a worker rank");
    return (worker_rank < m_config.manager_rank) ? worker_rank : (worker_rank - 1);
  }

  int worker_for_idx(int idx) const { return (idx < m_config.manager_rank) ? idx : (idx + 1); }

  int virtual_rank() const {
    const int rank = m_communicator.rank();
    return rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
  }

  template <typename T>
  void send_to_parent(const T& data, Tag tag) {
    const int target = get_parent_target();
    DYNAMPI_ASSERT_NE(target, -1, "Root cannot send to parent");

    m_communicator.send(data, target, tag);
  }

  template <typename T>
  void send_batch(const std::vector<T>& values, int rank, Tag tag) {
    if constexpr (MPI_Type<T>::resize_required) {
      m_communicator.send(detail::pack_variable_batch(values), rank, tag);
    } else {
      m_communicator.send(values, rank, tag);
    }
  }

  template <typename T>
  std::vector<T> receive_batch(MPI_Status status, Tag tag) {
    const int source = status.MPI_SOURCE;
    if constexpr (MPI_Type<T>::resize_required) {
      using message_type = MPI_Type<std::vector<std::byte>>;
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
      std::vector<std::byte> packed;
      message_type::resize(packed, count);
      m_communicator.recv(packed, source, tag);
      return detail::unpack_variable_batch<T>(packed);
    } else {
      using message_type = MPI_Type<std::vector<T>>;
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
      std::vector<T> values;
      message_type::resize(values, count);
      m_communicator.recv(values, source, tag);
      return values;
    }
  }

  void send_done_to_children_when_free() {
    const int direct_children = num_direct_children();
    // Deduplicate defensively; normally each child has one outstanding request.
    std::set<int> notified;
    while (static_cast<int>(notified.size()) < direct_children) {
      if (m_free_worker_indices.empty()) {
        receive_from_anyone();
        continue;
      }
      TaskRequest request = m_free_worker_indices.front();
      m_free_worker_indices.pop();

      if (notified.contains(request.worker_rank)) {
        continue;  // LCOV_EXCL_LINE -- defensive; see comment above
      }
      m_communicator.send(nullptr, request.worker_rank, Tag::DONE);
      notified.insert(request.worker_rank);
    }
  }

  using result_mpi_type = MPI_Type<ResultT>;
  using task_mpi_type = MPI_Type<TaskT>;

  void receive_result_from(MPI_Status status) {
    int world_source = status.MPI_SOURCE;
    if constexpr (result_mpi_type::resize_required) {
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, result_mpi_type::value, &count));
      ResultT result{};
      result_mpi_type::resize(result, count);
      m_communicator.recv(result, world_source, Tag::RESULT);
      m_results.push_back(std::move(result));
    } else {
      m_results.push_back(ResultT{});
      m_communicator.recv(m_results.back(), world_source, Tag::RESULT);
    }
    m_results_received_from_child++;
    m_free_worker_indices.push(TaskRequest{.worker_rank = world_source});
  }

  void receive_result_batch_from(MPI_Status status) {
    std::vector<ResultT> results = receive_batch<ResultT>(status, Tag::RESULT_BATCH);
    std::move(results.begin(), results.end(), std::back_inserter(m_results));
    m_results_received_from_child += results.size();
  }

  void receive_execute_return_task_from(MPI_Status status) {
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_mpi_type::value, &count));
    TaskT message;
    task_mpi_type::resize(message, count);
    int world_source = status.MPI_SOURCE;
    m_communicator.recv(message, world_source, Tag::TASK);
    m_tasks_received_from_parent++;
    ResultT result;
    auto failure = detail::run_task_guarded(m_worker_function, message, result);
    m_tasks_executed++;
    // ERROR is advisory; RESULT still carries completion accounting.
    if (failure) {
      m_communicator.send(
          detail::encode_task_error(TaskError{m_communicator.rank(), std::move(*failure)}),
          world_source, Tag::ERROR);
    }
    m_communicator.send(result, world_source, Tag::RESULT);
    m_results_sent_to_parent++;
  }

  // Failures are forwarded independently of their placeholder result.
  void receive_error_from(MPI_Status status) {
    int world_source = status.MPI_SOURCE;
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, MPI_CHAR, &count));
    std::string payload;
    MPI_Type<std::string>::resize(payload, count);
    m_communicator.recv(payload, world_source, Tag::ERROR);

    if (is_root_manager()) {
      m_task_errors.record(detail::decode_task_error(payload));
    } else {
      send_to_parent(payload, Tag::ERROR);
    }
  }

  void receive_task_batch_from(MPI_Status status) {
    if constexpr (prioritize_tasks) {
      DYNAMPI_UNIMPLEMENTED("Prioritized hierarchical distribution");
    } else {
      std::vector<TaskT> tasks = receive_batch<TaskT>(status, Tag::TASK_BATCH);
      m_tasks_received_from_parent += tasks.size();
      m_request_outstanding = false;
      request_next_batch_if_room();
      auto& target = m_round_active ? m_prefetched_tasks : m_unallocated_task_queue;
      for (auto& task : tasks) {
        target.push_back(std::move(task));
      }
    }
  }

  void receive_request_from(MPI_Status status) {
    int world_source = status.MPI_SOURCE;
    m_communicator.recv_empty_message(world_source, Tag::REQUEST);
    m_free_worker_indices.push(TaskRequest{.worker_rank = world_source});
  }

  void receive_request_batch_from(MPI_Status status) {
    int world_source = status.MPI_SOURCE;
    int request_count;
    m_communicator.recv(request_count, world_source, Tag::REQUEST_BATCH);
    DYNAMPI_ASSERT_GT(request_count, 0, "Invalid request count");
    DYNAMPI_ASSERT_LE(request_count, kMaxTasksRequested, "Request count exceeds maximum allowed");
    m_free_worker_indices.push(
        TaskRequest{.worker_rank = world_source, .num_tasks_requested = request_count});
  }

  void receive_done_from(MPI_Status status) {
    int world_source = status.MPI_SOURCE;
    m_communicator.recv_empty_message(world_source, Tag::DONE);
    m_done = true;
  }

  void receive_from_anyone() {
    DYNAMPI_ASSERT_GT(m_communicator.size(), 1,
                      "There should be at least one worker to receive results from");

    // All protocol messages use world ranks on the global communicator.
    MPI_Status status = m_communicator.probe();

    DYNAMPI_ASSERT(status.MPI_TAG >= static_cast<int>(Tag::TASK) &&
                       status.MPI_TAG <= static_cast<int>(Tag::ERROR),
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
