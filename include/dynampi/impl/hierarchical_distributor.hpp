/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <functional>
#include <iostream>
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
    std::optional<size_t> message_batch_size = std::nullopt;
    std::optional<int> max_workers_per_manager = std::nullopt;

    // Scales the batch a manager requests from its parent, which is
    // sized to the leaf workers in its subtree (see subtree_leaf_count()) --
    // 1 asks for exactly one task per leaf below it. Raising this trades
    // coarser load balancing for fewer round trips, the same way
    // pipeline_depth does; prefer pipeline_depth, which buys the same
    // latency hiding without widening the unit that can be stranded on one
    // manager.
    int batch_size_multiplier = 1;

    // Batches a manager keeps in the pipeline at once, including the one
    // being distributed. 1 disables prefetching; 2 is double-buffering, so
    // children never wait a full round trip between batches. Higher values
    // buy more slack against latency variance at the cost of coarser load
    // balancing -- tasks committed that far ahead cannot be reassigned to an
    // idle sibling.
    int pipeline_depth = 2;

    // If true, topology is strictly mapped to physical nodes:
    // Root Manager <-> Node Managers <-> Local Workers
    // Note: Manager is excluded from its node's Local Comm to separate duties.
    bool manager_per_node = true;

    // Only meaningful when manager_per_node is true. 0 (default) keeps one
    // local group per shared-memory node. A positive value partitions large
    // nodes into smaller contiguous groups -- same knob as
    // HierarchicalLockFreeRMAWorkDistributor::Config::max_local_group_size
    // -- so a single-node CI job can still synthesize multiple managers
    // and exercise max_upper_fanout grouping.
    int max_local_group_size = 0;

    // manager_per_node only. <0 (default, auto): derive a fanout from the
    // node manager count -- see setup_leader_hierarchy() for the formula.
    // 0: disabled, a flat two-level tree. >0: caps direct leader-layer
    // children per rank, grouping node managers recursively into
    // intermediate leaders once they exceed it.
    int max_upper_fanout = -1;

    // If true (default), run_tasks()/finish_remaining_tasks() throw
    // dynampi::TaskFailure on the root manager once a task has thrown. Set
    // false to recover instead: distribution runs to completion and the
    // failures are available from take_task_errors().
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

  enum class CommLayer { Global, Local, Leader };

  struct TaskRequest {
    int worker_rank;
    CommLayer source_layer = CommLayer::Global;  // Which comm did this come from?
    std::optional<int> num_tasks_requested = std::nullopt;
  };
  static constexpr int kMaxTasksRequested = 1'000'000;  // guard against pathological reserve()
  // FIFO, not LIFO: allocate_task_or_batch() drains this whenever new work
  // arrives, and a stack would serve whichever request arrived most
  // recently first, letting it jump every request already queued ahead of
  // it. Harmless once there's enough work to eventually serve everyone
  // fully regardless of order, but visible at low tasks_per_worker as
  // latency (not lost tasks) piling onto whichever request loses that race
  // repeatedly -- it can wait behind an unbounded number of later arrivals
  // instead of just the ones actually ahead of it.
  std::queue<TaskRequest> m_free_worker_indices;

  size_t m_tasks_sent_to_child = 0;
  size_t m_results_received_from_child = 0;
  size_t m_results_sent_to_parent = 0;
  size_t m_tasks_received_from_parent = 0;
  size_t m_tasks_executed = 0;
  size_t m_results_returned = 0;

  bool m_finalized = false;
  bool m_done = false;

  detail::TaskErrorLog m_task_errors;  // root manager only

  // Pipelining (see run_worker()'s intermediate-manager branch):
  // tasks that arrive as a reply to a proactive next-batch request sent
  // while the current round is still active are quarantined here and only
  // released into m_unallocated_task_queue at the next round boundary, so
  // the current round's distribution loop can never overshoot into the
  // next round's tasks. Replies stay in arrival order (FIFO) regardless of
  // which of possibly several outstanding requests they answer.
  bool m_round_active = false;
  std::deque<TaskT> m_prefetched_tasks;
  // Requests sent to our parent (Tag::REQUEST_BATCH) whose reply --
  // Tag::TASK_BATCH into m_prefetched_tasks/m_unallocated_task_queue, or
  // Tag::DONE -- hasn't arrived yet. Kept below Config::pipeline_depth (see
  // top_up_pipeline()).
  int m_outstanding_requests = 0;

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();

  using MPICommunicator = dynampi::MPICommunicator<track_statistics<statistics_mode>>;

  MPICommunicator m_communicator;  // Global communicator
  MPIGroup m_world_group;          // Group for the global communicator (for rank translation)
  std::optional<MPIGroup> m_local_group;  // Intra-node group (Shared Memory, excludes manager)

  // Leader-layer topology (root manager + node managers), built by
  // setup_leader_hierarchy(). A rank owns one group per promotion round it
  // leads (the manager always owns the top round), and every non-manager
  // leader-layer rank has exactly one parent group. That group's owner is
  // the manager when it is a member -- looked up by manager_rank, since
  // flat/top groups are keyed by world rank and the manager need not be
  // group rank 0 -- otherwise group rank 0 by construction. With grouping
  // disabled or unneeded this degenerates to a single flat group.
  std::vector<MPIGroup> m_owned_leader_levels;
  std::optional<MPIGroup> m_leader_parent_group;

  // Leaf workers in this rank's subtree, including its own node's local
  // workers -- the unit a manager requests from its parent (see
  // subtree_leaf_count() and run_worker()). Accumulated by
  // setup_leader_hierarchy() as this rank is promoted, mirroring
  // HierarchicalLockFreeRMAWorkDistributor::setup_upper_chain()'s
  // feed_width. Left at its initial value on ranks that never reach the
  // leader layer (plain local workers), which never request batches.
  int m_subtree_leaf_count = 1;

  std::function<ResultT(TaskT)> m_worker_function;
  Config m_config;

  // Cached parent target to avoid repeated MPI_Group_translate_ranks calls
  mutable std::optional<std::pair<int, CommLayer>> m_cached_parent_target;

  // --- Topology Helper Methods ---

  inline int max_workers_per_manager() const {
    const int default_value = std::max(2, static_cast<int>(std::sqrt(m_communicator.size())));
    const int configured = m_config.max_workers_per_manager.value_or(default_value);
    return std::max(1, configured);
  }

  // Resolves max_upper_fanout to a branching factor; manager_count excludes
  // the root manager. Auto mode matches the RMA class: stay flat below ~32
  // managers, where a fanout sweep found grouped and flat indistinguishable,
  // and above that use the smallest power of 2 >= sqrt(manager_count). The
  // same sweep measured a deeper tree ~6x worse and a shallower one, which
  // concentrates traffic onto too few leaders, worse again.
  inline int resolve_leader_fanout(int manager_count) const {
    if (m_config.max_upper_fanout < 0) {
      if (manager_count <= 32) return std::numeric_limits<int>::max();
      int fanout = 1;
      const double target = std::sqrt(static_cast<double>(manager_count));
      while (fanout < target) fanout *= 2;
      return fanout;
    }
    return m_config.max_upper_fanout > 0 ? m_config.max_upper_fanout
                                         : std::numeric_limits<int>::max();
  }

  // Builds the leader layer (root manager + node managers), optionally
  // grouped into a k-ary tree. The same shape as setup_upper_chain() in the
  // RMA class, but over send/recv, so there is no per-level window to create
  // -- only group membership to record.
  //
  // Every MPI_Comm_split call below is collective over its *input* comm's
  // full membership; the control flow is written so every rank in that
  // membership reaches the matching call, even ranks that stop being
  // promoted early (mirrors the same requirement in setup_upper_chain()).
  void setup_leader_hierarchy(bool is_manager, bool is_node_manager) {
    // A node manager's subtree is initially just its own node's workers;
    // each promotion below multiplies that by the round's group size. The
    // manager has no local group and never requests batches, so its value
    // stays at the 1 floor and is inert.
    const int local_children =
        (m_local_group && m_local_group->rank() == 0) ? (m_local_group->size() - 1) : 0;
    m_subtree_leaf_count = std::max(1, local_children);

    const int leader_color = (is_manager || is_node_manager) ? 0 : MPI_UNDEFINED;
    // Key is global rank to maintain global ordering among leaders.
    auto flat_opt = m_communicator.split(leader_color, m_communicator.rank());
    if (!flat_opt.has_value()) return;  // plain local worker: no leader-layer role at all
    MPICommunicator flat_comm = std::move(*flat_opt);

    const int manager_count = flat_comm.size() - 1;  // excludes the root manager
    const int effective_fanout = resolve_leader_fanout(manager_count);

    if (manager_count <= effective_fanout) {
      // Fits directly under the manager: exactly today's single flat group
      // (also always true when max_upper_fanout is disabled).
      MPIGroup flat_group(flat_comm);
      if (is_manager) {
        m_owned_leader_levels.push_back(std::move(flat_group));
      } else {
        m_leader_parent_group.emplace(std::move(flat_group));
      }
      return;
    }

    // Real grouping needed. Carve "managers only" out of flat_comm --
    // every member of flat_comm (manager included) calls this split
    // together, even though only managers use the result.
    auto managers_opt = flat_comm.split(is_manager ? MPI_UNDEFINED : 0, flat_comm.rank());

    bool is_final_round_leader = false;
    if (!is_manager) {
      // std::optional, not a bare MPICommunicator: MPICommunicator has no
      // move assignment (only move construction), so replacing round_comm
      // each iteration needs emplace()'s in-place construction rather than
      // `round_comm = ...`.
      std::optional<MPICommunicator> round_comm(std::move(*managers_opt));
      while (true) {
        if (round_comm->size() <= effective_fanout) {
          // This round's membership (this rank included) already fits
          // directly under the manager -- stop promoting.
          is_final_round_leader = true;
          break;
        }
        const int color = round_comm->rank() / effective_fanout;
        auto group_opt = round_comm->split(color, round_comm->rank());
        MPICommunicator group_comm = std::move(*group_opt);
        const bool is_group_leader = (group_comm.rank() == 0);
        // Every member of this group fronts a subtree the same size as this
        // rank's own (nodes are uniform), and the leader feeds its own node
        // out of the same queue -- so leading this round multiplies the
        // subtree by the full group size, not by the child count.
        const int child_count = group_comm.size();

        // Collective over round_comm: every member (leader or not) calls
        // this together, before acting on their differing result below.
        auto leaders_opt =
            round_comm->split(is_group_leader ? 0 : MPI_UNDEFINED, round_comm->rank());

        if (!is_group_leader) {
          m_leader_parent_group.emplace(std::move(group_comm));
          break;
        }
        m_owned_leader_levels.emplace_back(group_comm);
        m_subtree_leaf_count = std::max(1, child_count * m_subtree_leaf_count);
        round_comm.emplace(std::move(*leaders_opt));
      }
    }

    // Attach to the manager: every ORIGINAL member of flat_comm (manager +
    // every manager, whether promoted zero, one, or many times) reaches
    // this exact call.
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

  // Returns {parent_rank, communicator_layer}
  inline std::pair<int, CommLayer> get_parent_target() const {
    // Return cached value if available
    if (m_cached_parent_target.has_value()) {
      return m_cached_parent_target.value();
    }

    std::pair<int, CommLayer> result;
    DYNAMPI_ASSERT(!is_root_manager(), "Root manager should not have a parent");
    if (m_config.manager_per_node) {
      DYNAMPI_ASSERT(m_local_group.has_value() || m_leader_parent_group.has_value(),
                     "Local or leader parent group should be present");
      if (m_local_group && m_local_group->rank() > 0) {
        // Case 1: I am a Local Worker (Rank > 0 in Local Group)
        // Parent is the Node Manager (Local Rank 0).
        // Translate local rank 0 to world rank
        int node_manager_world_rank = m_local_group->translate_rank(0, m_world_group);
        result = {node_manager_world_rank, CommLayer::Local};
      } else {
        // Case 2: leader-layer rank. My parent owns m_leader_parent_group
        // -- the root manager if I am in the top round, otherwise a
        // higher-level leader. Flat/top groups include the manager but are
        // keyed by world rank, so it need not be group rank 0; intermediate
        // levels never include it and their owner is group rank 0.
        DYNAMPI_ASSERT(m_leader_parent_group.has_value(),
                       "Non-manager leader-layer rank must have a parent group");
        int parent_in_group = 0;
        const int manager_in_group =
            m_world_group.translate_rank(m_config.manager_rank, *m_leader_parent_group);
        if (manager_in_group != MPI_UNDEFINED) {
          parent_in_group = manager_in_group;
        }
        int parent_world_rank =
            m_leader_parent_group->translate_rank(parent_in_group, m_world_group);
        result = std::make_pair(parent_world_rank, CommLayer::Leader);
      }
    } else {
      // Original Logic
      int rank = m_communicator.rank();
      int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
      int virtual_parent = (virtual_rank - 1) / max_workers_per_manager();
      int parent_rank =
          virtual_parent == 0 ? m_config.manager_rank : worker_for_idx(virtual_parent - 1);
      result = {parent_rank, CommLayer::Global};
    }

    // Cache the result
    m_cached_parent_target = result;
    return result;
  }

  inline int total_num_children(int rank) const {
    if (m_config.manager_per_node) {
      DYNAMPI_UNIMPLEMENTED("Recursive child counting not supported/needed in Node topology mode");
      return 0;
    }
    int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
    int num_children = 0;
    int max_children = max_workers_per_manager();
    for (int i = 0; i < max_children; ++i) {
      int child = virtual_rank * max_children + i + 1;
      if (child >= m_communicator.size()) break;  // No more children
      num_children += 1 + total_num_children(worker_for_idx(child - 1));
    }
    return num_children;
  }

  // Calculate number of direct children based on active topology
  inline int num_direct_children() const {
    if (m_config.manager_per_node) {
      int count = 0;
      // 1. Local Children: Everyone in local group except me (Rank 0)
      if (m_local_group && m_local_group->rank() == 0) {
        count += (m_local_group->size() - 1);
      }
      // 2. Leader-layer children: every group this rank owns -- the manager
      // always owns exactly the top round (even when grouping is
      // disabled/unneeded), and a promoted manager owns one group per
      // round it leads. Each owned group's other members are this rank's
      // direct leader-layer children for that round.
      for (const auto& level : m_owned_leader_levels) {
        count += (level.size() - 1);
      }
      return count;
    } else {
      // Original Logic
      int rank = m_communicator.rank();
      int num_children = 0;
      int max_children = max_workers_per_manager();
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

  // How many tasks one round should carry: the leaf workers this rank is
  // responsible for feeding, not how many messages it sends to do it.
  //
  // These differ above the node managers by a factor of the fanout per
  // level: a leader's children are themselves managers fronting whole nodes,
  // so sizing to the direct child count asks for one task per subtree rather
  // than per worker -- 69 tasks for 384 leaf workers at 2048 nodes, 7 ranks
  // per node. The RMA class scales its claims the same way (feed_width).
  //
  // The rank-order virtual tree (manager_per_node == false) keeps the direct
  // child count; total_num_children() exists there but counts descendants
  // rather than leaves, and no measured configuration runs that topology.
  inline int subtree_leaf_count() const {
    return m_config.manager_per_node ? m_subtree_leaf_count : num_direct_children();
  }

  bool is_leaf_worker() const {
    if (m_config.manager_per_node) {
      if (is_root_manager()) return false;

      // If I am NOT in local group (should only be Manager, handled above), panic?
      // Actually, with this topology, everyone except Manager is in local group.
      if (!m_local_group) return true;  // Safety fallback

      // Standard Worker: Rank > 0 in Local Group
      if (m_local_group->rank() > 0) return true;

      // Node Manager: Rank 0 in Local Comm.
      // Leaf only if single-core node (no children).
      return num_direct_children() == 0;
    } else {
      int rank = m_communicator.rank();
      int max_children = max_workers_per_manager();
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
      // 1. Identify physical nodes via split_by_node, optionally partitioning
      // large nodes into smaller local domains (max_local_group_size).
      MPICommunicator node_comm = m_communicator.split_by_node();
      std::optional<MPICommunicator> local_domain;
      if (m_config.max_local_group_size > 0 && node_comm.size() > m_config.max_local_group_size) {
        const int color = node_comm.rank() / m_config.max_local_group_size;
        auto partition = node_comm.split(color, node_comm.rank());
        local_domain.emplace(std::move(*partition));
      } else {
        local_domain.emplace(std::move(node_comm));
      }

      // 2. Create Local Group: Exclude Manager!
      // If I am Manager, color is Undefined (I don't participate in local worker pool).
      // Everyone else participates.
      int local_color = (m_communicator.rank() == m_config.manager_rank) ? MPI_UNDEFINED : 0;

      auto local_comm_opt = local_domain->split(local_color, m_communicator.rank());
      if (local_comm_opt.has_value()) {
        // Extract group from the temporary communicator, then let it be freed
        m_local_group.emplace(*local_comm_opt);
      }

      // 3. Build the leader layer: manager + Node Managers (Rank 0 of
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
      // Leaf workers (usually local ranks > 0) just request from parent
      send_to_parent(nullptr, Tag::REQUEST);
      while (!m_done) {
        receive_from_anyone();
      }
    } else {
      // Intermediate nodes (Node Managers)
      const int prefetch =
          std::max(1, subtree_leaf_count() * std::max(1, m_config.batch_size_multiplier));
      const int pipeline_depth = std::max(1, m_config.pipeline_depth);

      // Keeps up to `target` batches' worth of requests outstanding with
      // our parent at once (see Config::pipeline_depth), sending only as
      // many new ones as needed to close the gap -- so calling this
      // repeatedly as replies trickle in and reduce m_outstanding_requests
      // never over-sends. No-ops once m_done is known: the manager has
      // stopped listening (finalize() only answers each child once -- see
      // send_done_to_children_when_free()), so a request sent after that
      // would never be received.
      auto top_up_pipeline = [&](int target) {
        while (!m_done && m_outstanding_requests < target) {
          send_to_parent(prefetch, Tag::REQUEST_BATCH);
          m_outstanding_requests++;
        }
      };

      // Prime with one request, not the full pipeline_depth: m_round_active
      // is still false here, so replies route straight into the queue rather
      // than being quarantined, and several of them would merge into one
      // oversized first round. The rest of the depth is built up below,
      // after the flag is set.
      top_up_pipeline(1);

      while (!m_done) {
        // A batch prefetched during the previous round (see below) is
        // already sitting here, ready to go, with no round trip needed.
        // (Prioritized task queues don't support batch prefetching; see
        // receive_task_batch_from.)
        if constexpr (!prioritize_tasks) {
          if (!m_prefetched_tasks.empty()) {
            for (auto& task : m_prefetched_tasks) {
              m_unallocated_task_queue.push_back(std::move(task));
            }
            m_prefetched_tasks.clear();
          }
        }

        // Wait for tasks from the parent, flushing any results that arrive
        // meanwhile. Without that flush this deadlocks: once everything on
        // hand is distributed, this loop is the only place the manager
        // spends time, so stragglers would sit in m_results unsent while the
        // parent's exhaustion check waits to see them.
        while (!m_done && m_unallocated_task_queue.empty()) {
          receive_from_anyone();
          if (!m_results.empty()) {
            send_results_to_parent();
          }
        }
        if (m_done) break;

        // While this round is active, any TASK_BATCH that arrives (a reply
        // to a pipelined request) is quarantined in m_prefetched_tasks
        // rather than merged into m_unallocated_task_queue, so this round's
        // distribution loop can never overshoot past this round's own
        // starting queue size into next round's tasks.
        m_round_active = true;

        // Pipelining: top up outstanding requests to pipeline_depth-1 (the
        // "-1" is this now-active round itself) as soon as we start working
        // it, rather than only after its results have been fully collected
        // and sent upstream. Requests and result returns are independent
        // messages (see receive_request_batch_from / receive_result_batch_from),
        // so replies can already be arriving while we're still distributing
        // or awaiting results for this round. Without this, children sit
        // idle for a full parent round trip between batches.
        top_up_pipeline(pipeline_depth - 1);

        // Deliberately does NOT break on m_done: a pipelined request can be
        // answered with DONE while this round is still being handed out.
        // Quarantining keeps anything new out of the queue, so finishing the
        // round is safe -- and required, since abandoning it would silently
        // drop the tasks in hand.
        while (!m_unallocated_task_queue.empty()) {
          if (m_free_worker_indices.empty()) {
            // Must wait for a worker to become free
            receive_from_anyone();
          } else {
            allocate_task_to_child();
          }
        }

        m_round_active = false;

        // Not gated on the whole round having arrived: results are
        // unordered, so nothing downstream cares which round one came from.
        // Waiting for a round's slowest straggler used to hold back results
        // from children that finished long ago, and those children can pick
        // up next-round work immediately since allocation only needs a free
        // worker. Stragglers are forwarded by a later round's flush or the
        // final drain.
        if (!m_results.empty()) {
          send_results_to_parent();
        }

        // Minimum-progress guarantee: ensure at least one request is
        // outstanding before we loop back to wait for the next round.
        // Without this, pipeline_depth == 1 (top_up_pipeline(pipeline_depth
        // - 1) above is a no-op at 0) would never ask for a next batch at
        // all -- unlike the old implicit protocol this replaced, where
        // sending RESULT_BATCH itself doubled as a request (see
        // receive_result_batch_from), nothing here sends a request except
        // top_up_pipeline. This is also not just a depth==1 special case:
        // even at higher depths, a long round can fully drain
        // m_outstanding_requests to 0 (every previously-sent request
        // already answered and parked in m_prefetched_tasks) before the
        // round finishes, and needs the same top-up to keep the pipeline
        // fed afterward.
        top_up_pipeline(1);
        // Safe to honor m_done now: this round is fully distributed and
        // whatever was ready has been flushed to our parent. Any results
        // still outstanding after the loop (e.g. in-flight RESULT_BATCH
        // while a child already REQUEST_BATCH'd ahead) are drained and
        // flushed after send_done_to_children_when_free() below.
        if (m_done) break;
      }
      send_done_to_children_when_free();
      // REQUEST_BATCH (not RESULT_BATCH) is what marks a child free, so
      // send_done_to_children_when_free() can finish while results for
      // already-dispatched tasks are still in flight -- especially with
      // pipeline_depth > 1, where a child may request its next batch before
      // returning the current one's results. Drain those before the final
      // flush so we don't drop them or trip finalize()'s sent==received
      // invariants.
      while (m_tasks_sent_to_child > m_results_received_from_child) {
        receive_from_anyone();  // LCOV_EXCL_LINE -- requires a shutdown/in-flight-result race
      }
      if (!m_results.empty()) {
        send_results_to_parent();  // LCOV_EXCL_LINE -- same race as the drain above
      }
    }
  }

  void send_results_to_parent() {
    DYNAMPI_ASSERT(!is_leaf_worker(), "Leaf workers should not return results directly");
    DYNAMPI_ASSERT_NE(m_communicator.rank(), m_config.manager_rank,
                      "Manager should not request tasks from itself");
    std::vector<ResultT> results = std::move(m_results);
    m_results.clear();

    if constexpr (result_mpi_type::resize_required) {
      send_to_parent(detail::pack_variable_batch(results), Tag::RESULT_BATCH);
    } else {
      send_to_parent(results, Tag::RESULT_BATCH);
    }
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

        if constexpr (task_mpi_type::resize_required) {
          send_to_worker(detail::pack_variable_batch(tasks), worker_rank, Tag::TASK_BATCH, layer);
        } else {
          send_to_worker(tasks, worker_rank, Tag::TASK_BATCH, layer);
        }
        m_tasks_sent_to_child += tasks.size();
      } else {
        const TaskT task = get_next_task_to_send();
        send_to_worker(task, worker_rank, Tag::TASK, layer);
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

  // --- Helper: Determine which layer a world rank belongs to ---
  CommLayer determine_layer_from_world_rank(int world_rank) const {
    DYNAMPI_ASSERT(m_config.manager_per_node);
    // Check if rank is in local group (and not manager)
    if (m_local_group) {
      int local_rank = m_world_group.translate_rank(world_rank, *m_local_group);
      if (local_rank != MPI_UNDEFINED) {
        return CommLayer::Local;
      }
    }
    // Not a local child, so it's a leader-layer child -- one of this rank's
    // direct children at one of its owned levels (see m_owned_leader_levels;
    // there's no single group spanning every leader-layer rank to verify
    // membership against once multi-level grouping is active).
    DYNAMPI_ASSERT(!m_owned_leader_levels.empty(), "Leader-layer rank should own a level");
    return CommLayer::Leader;
  }

  // --- Abstract Send Wrappers ---

  template <typename T>
  void send_to_parent(const T& data, Tag tag) {
    auto [target, layer] = get_parent_target();
    DYNAMPI_ASSERT_NE(target, -1, "Root cannot send to parent");

    // With groups, target is always a world rank, so use global communicator
    m_communicator.send(data, target, tag);
  }

  template <typename T>
  void send_to_worker(const T& data, int rank, Tag tag, [[maybe_unused]] CommLayer layer) {
    // With groups, rank is stored as world rank in TaskRequest, so use global communicator
    m_communicator.send(data, rank, tag);
  }

  void send_done_to_children_when_free() {
    const int direct_children = num_direct_children();
    // Tracks unique children notified, not requests answered: with
    // pipeline_depth > 1 (see run_worker()) a single child can have several
    // outstanding requests queued here at once, all requesting the same
    // "no more work" answer. Counting requests instead of children would let
    // one over-represented child consume the whole direct_children budget,
    // leaving others never told to stop -- a real deadlock, not a
    // theoretical one, once more than one request per child is possible.
    std::set<int> notified;
    while (static_cast<int>(notified.size()) < direct_children) {
      if (m_free_worker_indices.empty()) {
        receive_from_anyone();
        continue;
      }
      TaskRequest request = m_free_worker_indices.front();
      m_free_worker_indices.pop();

      if (notified.contains(request.worker_rank)) {
        // Already told this child; it doesn't need (or expect) a separate
        // reply per request it happened to have outstanding.
        continue;  // LCOV_EXCL_LINE -- duplicate arrival order is MPI timing-dependent
      }
      send_to_worker(nullptr, request.worker_rank, Tag::DONE, request.source_layer);
      notified.insert(request.worker_rank);
    }
  }

  using result_mpi_type = MPI_Type<ResultT>;
  using task_mpi_type = MPI_Type<TaskT>;

  void receive_result_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                           CommLayer layer) {
    // With groups, always use global communicator and determine layer from source rank
    int world_source = status.MPI_SOURCE;
    if (m_config.manager_per_node) {
      layer = determine_layer_from_world_rank(world_source);
    }
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
    m_free_worker_indices.push(TaskRequest{.worker_rank = world_source, .source_layer = layer});
  }

  void receive_result_batch_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                                 [[maybe_unused]] CommLayer layer) {
    int world_source = status.MPI_SOURCE;
    std::vector<ResultT> results;
    if constexpr (result_mpi_type::resize_required) {
      // See variable_batch.hpp: a batch of variable-length ResultT can't use
      // the generic vector-of-T wire format, so it's packed as raw bytes.
      using message_type = MPI_Type<std::vector<std::byte>>;
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
      std::vector<std::byte> packed;
      message_type::resize(packed, count);
      m_communicator.recv(packed, world_source, Tag::RESULT_BATCH);
      results = detail::unpack_variable_batch<ResultT>(packed);
    } else {
      using message_type = MPI_Type<std::vector<ResultT>>;
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
      message_type::resize(results, count);
      m_communicator.recv(results, world_source, Tag::RESULT_BATCH);
    }
    // Results and next-batch requests are independent messages (see the
    // double-buffering refill in run_worker()), so receiving results here
    // has no side effect on task allocation for this child.
    std::move(results.begin(), results.end(), std::back_inserter(m_results));
    m_results_received_from_child += results.size();
  }

  void receive_execute_return_task_from(MPI_Status status,
                                        [[maybe_unused]] MPICommunicator& source_comm,
                                        [[maybe_unused]] CommLayer layer) {
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, task_mpi_type::value, &count));
    TaskT message;
    task_mpi_type::resize(message, count);
    // With groups, always use global communicator
    int world_source = status.MPI_SOURCE;
    m_communicator.recv(message, world_source, Tag::TASK);
    m_tasks_received_from_parent++;
    ResultT result;
    auto failure = detail::run_task_guarded(m_worker_function, message, result);
    m_tasks_executed++;
    // ERROR replaces this task's RESULT rather than accompanying it: the parent
    // counts one reply per dispatch either way, so the shutdown accounting and
    // the free-child bookkeeping stay exactly as they were.
    // ERROR is a pure notification: it travels alongside the result, never
    // instead of it. Every hop can then forward it upward without deciding
    // whether it also stands in for a completion -- a distinction that is not
    // even locally decidable at an intermediate manager, which forwards a
    // subtree's failure but has already accounted for the placeholder itself.
    if (failure) {
      m_communicator.send(
          detail::encode_task_error(TaskError{m_communicator.rank(), std::move(*failure)}),
          world_source, Tag::ERROR);
    }
    {
      m_communicator.send(result, world_source, Tag::RESULT);
    }
    m_results_sent_to_parent++;
  }

  // A failure reported from somewhere below us -- one of our own leaves, or a
  // subtree forwarding one of its own. Carries no completion: the failing rank
  // sent a placeholder result too, which travels the normal result path.
  void receive_error_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                          [[maybe_unused]] CommLayer layer) {
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

  void receive_task_batch_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                               [[maybe_unused]] CommLayer layer) {
    if constexpr (prioritize_tasks) {
      DYNAMPI_UNIMPLEMENTED("Prioritized hierarchical distribution");
    } else {
      // With groups, always use global communicator
      int world_source = status.MPI_SOURCE;
      std::vector<TaskT> tasks;
      if constexpr (task_mpi_type::resize_required) {
        // See variable_batch.hpp: a batch of variable-length TaskT can't use
        // the generic vector-of-T wire format, so it's packed as raw bytes.
        using message_type = MPI_Type<std::vector<std::byte>>;
        int count;
        DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
        std::vector<std::byte> packed;
        message_type::resize(packed, count);
        m_communicator.recv(packed, world_source, Tag::TASK_BATCH);
        tasks = detail::unpack_variable_batch<TaskT>(packed);
      } else {
        using message_type = MPI_Type<std::vector<TaskT>>;
        int count;
        DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
        message_type::resize(tasks, count);
        m_communicator.recv(tasks, world_source, Tag::TASK_BATCH);
      }
      m_tasks_received_from_parent += tasks.size();
      // This reply fulfills one of our outstanding pipelined requests (see
      // run_worker()'s top_up_pipeline()); only intermediate managers
      // (never leaf workers, which use the unbatched TASK/RESULT protocol)
      // send REQUEST_BATCH, so only they receive TASK_BATCH replies here.
      if (m_outstanding_requests > 0) --m_outstanding_requests;
      // While a round is active, this is the reply to a request for a
      // *future* round: quarantine it so the current round's distribution
      // loop can't overshoot into it (see run_worker()).
      auto& target = m_round_active ? m_prefetched_tasks : m_unallocated_task_queue;
      for (auto& task : tasks) {
        target.push_back(std::move(task));
      }
    }
  }

  void receive_request_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                            CommLayer layer) {
    // With groups, always use global communicator and determine layer from source rank
    int world_source = status.MPI_SOURCE;
    if (m_config.manager_per_node) {
      layer = determine_layer_from_world_rank(world_source);
    }
    m_communicator.recv_empty_message(world_source, Tag::REQUEST);
    m_free_worker_indices.push(TaskRequest{.worker_rank = world_source, .source_layer = layer});
  }

  void receive_request_batch_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                                  CommLayer layer) {
    // With groups, always use global communicator and determine layer from source rank
    int world_source = status.MPI_SOURCE;
    if (m_config.manager_per_node) {
      layer = determine_layer_from_world_rank(world_source);
    }
    int request_count;
    m_communicator.recv(request_count, world_source, Tag::REQUEST_BATCH);
    DYNAMPI_ASSERT_GT(request_count, 0, "Invalid request count");
    DYNAMPI_ASSERT_LE(request_count, kMaxTasksRequested, "Request count exceeds maximum allowed");
    m_free_worker_indices.push(TaskRequest{
        .worker_rank = world_source, .source_layer = layer, .num_tasks_requested = request_count});
  }

  void receive_done_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                         [[maybe_unused]] CommLayer layer) {
    // With groups, always use global communicator
    int world_source = status.MPI_SOURCE;
    m_communicator.recv_empty_message(world_source, Tag::DONE);
    m_done = true;
  }

  void receive_from_anyone() {
    DYNAMPI_ASSERT_GT(m_communicator.size(), 1,
                      "There should be at least one worker to receive results from");

    MPI_Status status{};
    CommLayer layer = CommLayer::Global;

    // All messages now route through the global communicator regardless of topology,
    // so a blocking probe is sufficient. The CommLayer is determined from the source
    // rank in each receive method via determine_layer_from_world_rank.
    status = m_communicator.probe();

    // Assert that the tag is a valid Tag enum value before casting
    DYNAMPI_ASSERT(status.MPI_TAG >= static_cast<int>(Tag::TASK) &&
                       status.MPI_TAG <= static_cast<int>(Tag::ERROR),
                   "Received invalid MPI tag: " + std::to_string(status.MPI_TAG));
    Tag tag = static_cast<Tag>(status.MPI_TAG);
    // Note: receive methods now use global communicator and determine layer from source rank
    switch (tag) {
      case Tag::TASK:
        return receive_execute_return_task_from(status, m_communicator, layer);
      case Tag::TASK_BATCH:
        return receive_task_batch_from(status, m_communicator, layer);
      case Tag::RESULT:
        return receive_result_from(status, m_communicator, layer);
      case Tag::RESULT_BATCH:
        return receive_result_batch_from(status, m_communicator, layer);
      case Tag::REQUEST:
        return receive_request_from(status, m_communicator, layer);
      case Tag::REQUEST_BATCH:
        return receive_request_batch_from(status, m_communicator, layer);
      case Tag::DONE:
        return receive_done_from(status, m_communicator, layer);
      case Tag::ERROR:
        return receive_error_from(status, m_communicator, layer);
    }
  }
};

};  // namespace dynampi
