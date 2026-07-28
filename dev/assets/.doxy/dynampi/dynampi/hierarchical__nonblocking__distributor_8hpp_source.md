

# File hierarchical\_nonblocking\_distributor.hpp

[**File List**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**impl**](dir_87365955bfd9c1869b83a1fbd3fdb221.md) **>** [**hierarchical\_nonblocking\_distributor.hpp**](hierarchical__nonblocking__distributor_8hpp.md)

[Go to the documentation of this file](hierarchical__nonblocking__distributor_8hpp.md)


```C++
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
#include <ranges>
#include <set>
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

// ---------------------------------------------------------------------------
// HierarchicalNonBlockingMPIWorkDistributor
//
// Identical topology, protocol, and pipelining logic to
// HierarchicalMPIWorkDistributor (see that file) -- this is a copy with one
// change: every outgoing message (TASK, TASK_BATCH, RESULT, RESULT_BATCH,
// REQUEST_BATCH) is sent via MPI_Isend instead of a blocking MPI_Send, so a
// coordinator posting a reply to one child never stalls before it can move
// on to serve the next. Each async send's buffer is kept alive in a small
// per-type pool (AsyncSendPool below) until MPI confirms it, reaped
// opportunistically whenever a new send of that type is posted.
//
// Receiving is unchanged (still a blocking MPI_Probe + matching MPI_Recv):
// every call site that waits on it already has nothing else productive to
// do at that exact point (see run_worker()), so converting it to a
// proactively-posted MPI_Irecv would need a byte-count-then-data staging
// protocol for the variable-length TASK_BATCH/RESULT_BATCH messages (sizes
// aren't known until the sender tells you) without a clear benefit here --
// this variant isolates the question of whether the *send* side blocking
// was costing anything.
//
// Empty messages (REQUEST, DONE -- zero payload bytes) skip the pool
// entirely: with no buffer to keep alive, MPI_Isend + MPI_Request_free is a
// safe, genuine fire-and-forget.
// ---------------------------------------------------------------------------
template <typename TaskT, typename ResultT, typename... Options>
class HierarchicalNonBlockingMPIWorkDistributor
    : public BaseMPIWorkDistributor<TaskT, ResultT, Options...> {
  using Base = BaseMPIWorkDistributor<TaskT, ResultT, Options...>;

 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    std::optional<size_t> message_batch_size = std::nullopt;
    std::optional<int> max_workers_per_coordinator = std::nullopt;
    int batch_size_multiplier = 2;

    // How many batches' worth of tasks/requests a coordinator tries to keep
    // in the pipeline at once, including the one currently being
    // distributed to its children. 1 disables prefetching entirely (a
    // coordinator only asks for its next batch after fully finishing the
    // current one). 2 is double-buffering: one batch active, one already
    // requested (or in flight) so children never wait a full round trip
    // between batches. Values above 2 keep additional batches' worth of
    // requests outstanding simultaneously, trading more slack against
    // round-trip latency variance for coarser load-balancing granularity
    // (tasks committed this far ahead can't be reassigned to an idle
    // sibling coordinator).
    int pipeline_depth = 2;

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
  static constexpr int kMaxTasksRequested = 1'000'000;  // guard against pathological reserve()
  std::stack<TaskRequest, std::vector<TaskRequest>> m_free_worker_indices;

  size_t m_tasks_sent_to_child = 0;
  size_t m_results_received_from_child = 0;
  size_t m_results_sent_to_parent = 0;
  size_t m_tasks_received_from_parent = 0;
  size_t m_tasks_executed = 0;
  size_t m_results_returned = 0;

  bool m_finalized = false;
  bool m_done = false;

  // Pipelining (see run_worker()'s intermediate-coordinator branch):
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
  std::optional<MPIGroup>
      m_leader_group;  // Inter-node group (Leaders only: manager + node coordinators)

  std::function<ResultT(TaskT)> m_worker_function;
  Config m_config;

  // Cached parent target to avoid repeated MPI_Group_translate_ranks calls
  mutable std::optional<std::pair<int, CommLayer>> m_cached_parent_target;

  // Keeps a non-blocking send's buffer alive until MPI confirms the send,
  // without ever blocking the poster on that confirmation. post() reaps
  // already-completed entries (via MPI_Test, not Wait) before adding the
  // new one -- an amortized, non-blocking cleanup rather than a separate
  // sweep -- so the pool stays close to "however many of this message type
  // are genuinely in flight right now," not unboundedly growing.
  template <typename T>
  class AsyncSendPool {
   public:
    void post(MPICommunicator& comm, T value, int dest, int tag) {
      reap();
      pending.emplace_back(std::move(value), MPI_Request{});
      auto& [buf, req] = pending.back();
      using mpi_type = MPI_Type<T>;
      DYNAMPI_MPI_CHECK(MPI_Isend, (mpi_type::ptr(buf), mpi_type::count(buf), mpi_type::value,
                                    dest, tag, comm.get(), &req));
    }

    // Must be called before this rank's process moves toward destruction:
    // an Isend's buffer has to stay valid and its request has to be
    // completed (Wait or Test-until-true) before it's safe to let go of
    // either -- see the call sites in run_worker()/finalize().
    void wait_all() {
      for (auto& [buf, req] : pending) {
        DYNAMPI_MPI_CHECK(MPI_Wait, (&req, MPI_STATUS_IGNORE));
      }
      pending.clear();
    }

   private:
    void reap() {
      for (auto it = pending.begin(); it != pending.end();) {
        int flag = 0;
        DYNAMPI_MPI_CHECK(MPI_Test, (&it->second, &flag, MPI_STATUS_IGNORE));
        if (flag) {
          it = pending.erase(it);
        } else {
          ++it;
        }
      }
    }
    std::deque<std::pair<T, MPI_Request>> pending;
  };

  AsyncSendPool<int> m_pending_int_sends;                        // REQUEST_BATCH
  AsyncSendPool<TaskT> m_pending_task_sends;                      // TASK
  AsyncSendPool<ResultT> m_pending_result_sends;                  // RESULT
  AsyncSendPool<std::vector<TaskT>> m_pending_task_batch_sends;   // TASK_BATCH
  AsyncSendPool<std::vector<ResultT>> m_pending_result_batch_sends;  // RESULT_BATCH

  void wait_all_pending_sends() {
    m_pending_int_sends.wait_all();
    m_pending_task_sends.wait_all();
    m_pending_result_sends.wait_all();
    m_pending_task_batch_sends.wait_all();
    m_pending_result_batch_sends.wait_all();
  }

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
    DYNAMPI_ASSERT(!is_root_manager(), "Root manager should not have a parent");
    if (m_config.coordinator_per_node) {
      DYNAMPI_ASSERT(m_local_group.has_value() || m_leader_group.has_value(),
                     "Local or leader group should be present");
      if (m_local_group && m_local_group->rank() > 0) {
        // Case 1: I am a Local Worker (Rank > 0 in Local Group)
        // Parent is the Node Coordinator (Local Rank 0).
        // Translate local rank 0 to world rank
        int node_coord_world_rank = m_local_group->translate_rank(0, m_world_group);
        result = {node_coord_world_rank, CommLayer::Local};
      } else {
        // Case 2: I am a Node Coordinator (Local Rank 0).
        // Parent is the Global Manager.
        // With the new topology, Manager is ALWAYS in the leader group.
        // We need the manager's world rank, which we already have
        int global_manager = m_config.manager_rank;
        result = std::make_pair(global_manager, CommLayer::Leader);
      }
    } else {
      // Original Logic
      int rank = m_communicator.rank();
      int virtual_rank = rank == m_config.manager_rank ? 0 : idx_for_worker(rank) + 1;
      int virtual_parent = (virtual_rank - 1) / max_workers_per_coordinator();
      int parent_rank =
          virtual_parent == 0 ? m_config.manager_rank : worker_for_idx(virtual_parent - 1);
      result = {parent_rank, CommLayer::Global};
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
      // 1. Local Children: Everyone in local group except me (Rank 0)
      if (m_local_group && m_local_group->rank() == 0) {
        count += (m_local_group->size() - 1);
      }
      // 2. Remote Children: If I am Manager, other Leaders are my children.
      // Note: In this topology, Manager is IN leader group, but NOT in local group.
      if (is_root_manager() && m_leader_group) {
        count += (m_leader_group->size() - 1);
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

      // If I am NOT in local group (should only be Manager, handled above), panic?
      // Actually, with this topology, everyone except Manager is in local group.
      if (!m_local_group) return true;  // Safety fallback

      // Standard Worker: Rank > 0 in Local Group
      if (m_local_group->rank() > 0) return true;

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
  explicit HierarchicalNonBlockingMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                          Config runtime_config = Config{})
      : m_communicator(runtime_config.comm, MPICommunicator::Duplicate),
        m_world_group(m_communicator),
        m_worker_function(worker_function),
        m_config(runtime_config),
        _statistics{create_statistics(m_communicator)} {
    // --- Initialize Topology Groups ---
    if (m_config.coordinator_per_node) {
      // 1. Identify physical nodes via split_by_node
      MPICommunicator node_comm = m_communicator.split_by_node();

      // 2. Create Local Group: Exclude Manager!
      // If I am Manager, color is Undefined (I don't participate in local worker pool).
      // Everyone else participates.
      int local_color = (m_communicator.rank() == m_config.manager_rank) ? MPI_UNDEFINED : 0;

      auto local_comm_opt = node_comm.split(local_color, m_communicator.rank());
      if (local_comm_opt.has_value()) {
        // Extract group from the temporary communicator, then let it be freed
        m_local_group.emplace(*local_comm_opt);
      }

      // 3. Create Leader Group
      // Who joins?
      // A: The Manager (Always)
      // B: The Node Coordinators (Rank 0 of the *Local* Comm)
      bool is_manager = (m_communicator.rank() == m_config.manager_rank);
      // Check if we're rank 0 in the local group (node coordinator)
      bool is_node_coordinator = false;
      if (m_local_group.has_value()) {
        int my_local_rank = m_local_group->rank();
        is_node_coordinator = (my_local_rank == 0);
      }

      int leader_color = (is_manager || is_node_coordinator) ? 0 : MPI_UNDEFINED;

      // Key is global rank to maintain global ordering among leaders
      auto leader_comm_opt = m_communicator.split(leader_color, m_communicator.rank());
      if (leader_comm_opt.has_value()) {
        // Extract group from the temporary communicator, then let it be freed
        m_leader_group.emplace(*leader_comm_opt);
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
      wait_all_pending_sends();
    } else {
      // Intermediate nodes (Node Coordinators)
      int num_children = num_direct_children();
      int prefetch = num_children * m_config.batch_size_multiplier;
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

      // Prime with exactly one request, not the full pipeline_depth: while
      // we're waiting for this very first reply, m_round_active is still
      // false, so receive_task_batch_from() has no round to quarantine
      // against yet and routes any reply straight into
      // m_unallocated_task_queue (see below). Priming with more than one
      // would risk several replies landing there before we ever set
      // m_round_active, merging what should be separate rounds into one
      // oversized first batch. Sending only one guarantees at most one
      // reply is pending during this wait; the rest of the pipeline_depth
      // is built up below, entirely after m_round_active is true.
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

        // If we have no tasks to give, wait for tasks from parent
        while (!m_done && m_unallocated_task_queue.empty()) {
          receive_from_anyone();
        }
        if (m_done) break;

        size_t num_tasks_should_be_received = m_unallocated_task_queue.size();
        // While this round is active, any TASK_BATCH that arrives (a reply
        // to a pipelined request) is quarantined in m_prefetched_tasks
        // rather than merged into m_unallocated_task_queue, so this round's
        // distribution loop can never overshoot past
        // num_tasks_should_be_received into next round's tasks.
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

        // Process tasks: Give to workers or execute ourselves if needed.
        // Deliberately does NOT break on m_done: a pipelined request can be
        // answered with DONE (if the manager has nothing left and
        // finalize() runs) while THIS round is still being handed out to
        // our own children. Since m_round_active already quarantines any
        // newly-arriving TASK_BATCH into m_prefetched_tasks, nothing new
        // can leak into m_unallocated_task_queue here -- so finishing it
        // out is always safe, and required: abandoning it would silently
        // drop this round's tasks (DYNAMPI_ASSERT_EQ below compiles to
        // nothing under NDEBUG/Release, so this would fail silently, not
        // loudly).
        while (!m_unallocated_task_queue.empty()) {
          if (m_free_worker_indices.empty()) {
            // Must wait for a worker to become free
            receive_from_anyone();
          } else {
            allocate_task_to_child();
          }
        }

        // Wait for results from children -- also not gated on m_done, for
        // the same reason as above: this round's already-dispatched tasks
        // must be collected before we can honor a DONE observed mid-round.
        while (m_tasks_sent_to_child > m_results_received_from_child) {
          receive_from_anyone();
        }

        m_round_active = false;

        (void)num_tasks_should_be_received;
        DYNAMPI_ASSERT_EQ(m_results.size(), num_tasks_should_be_received);

        send_results_to_parent();

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
        // Safe to honor m_done now: this round is fully distributed,
        // collected, and flushed to our parent.
        if (m_done) break;
      }
      send_done_to_children_when_free();
      wait_all_pending_sends();
    }
  }

  void send_results_to_parent() {
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
      wait_all_pending_sends();
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

  ~HierarchicalNonBlockingMPIWorkDistributor() {
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

  // --- Helper: Determine which layer a world rank belongs to ---
  CommLayer determine_layer_from_world_rank(int world_rank) const {
    DYNAMPI_ASSERT(m_config.coordinator_per_node);
    // Check if rank is in local group (and not manager)
    if (m_local_group) {
      int local_rank = m_world_group.translate_rank(world_rank, *m_local_group);
      if (local_rank != MPI_UNDEFINED) {
        return CommLayer::Local;
      }
    }
    DYNAMPI_ASSERT(m_leader_group.has_value(), "Leader group should be present");
    [[maybe_unused]] int leader_rank = m_world_group.translate_rank(world_rank, *m_leader_group);
    DYNAMPI_ASSERT_NE(leader_rank, MPI_UNDEFINED, "Rank should be in leader group");
    return CommLayer::Leader;
  }

  // --- Abstract Send Wrappers ---

  template <typename T>
  void send_to_parent(const T& data, Tag tag) {
    auto [target, layer] = get_parent_target();
    DYNAMPI_ASSERT_NE(target, -1, "Root cannot send to parent");
    send_async(data, target, tag);
  }

  template <typename T>
  void send_to_worker(const T& data, int rank, Tag tag, [[maybe_unused]] CommLayer layer) {
    send_async(data, rank, tag);
  }

  // Routes to the pool matching T (see AsyncSendPool above), or a genuine
  // fire-and-forget Isend+Request_free for zero-payload messages. If
  // TaskT == ResultT (e.g. both uint32_t, as in the benchmarks), a TaskT
  // send and a ResultT send are the same instantiation and both land in
  // m_pending_task_sends -- harmless: pool identity is only a buffer-
  // lifetime bookkeeping detail, not part of what goes out on the wire
  // (dest/tag/data are whatever the caller passed regardless of which pool
  // holds the buffer).
  template <typename T>
  void send_async(const T& data, int dest, Tag tag) {
    if constexpr (std::is_same_v<T, std::nullptr_t>) {
      using mpi_type = MPI_Type<std::nullptr_t>;
      MPI_Request req;
      DYNAMPI_MPI_CHECK(MPI_Isend, (nullptr, mpi_type::count(nullptr), mpi_type::value, dest,
                                    static_cast<int>(tag), m_communicator.get(), &req));
      DYNAMPI_MPI_CHECK(MPI_Request_free, (&req));
    } else if constexpr (std::is_same_v<T, std::vector<TaskT>>) {
      m_pending_task_batch_sends.post(m_communicator, data, dest, static_cast<int>(tag));
    } else if constexpr (std::is_same_v<T, std::vector<ResultT>>) {
      m_pending_result_batch_sends.post(m_communicator, data, dest, static_cast<int>(tag));
    } else if constexpr (std::is_same_v<T, TaskT>) {
      m_pending_task_sends.post(m_communicator, data, dest, static_cast<int>(tag));
    } else if constexpr (std::is_same_v<T, ResultT>) {
      m_pending_result_sends.post(m_communicator, data, dest, static_cast<int>(tag));
    } else if constexpr (std::is_same_v<T, int>) {
      m_pending_int_sends.post(m_communicator, data, dest, static_cast<int>(tag));
    } else {
      static_assert(!sizeof(T*), "send_async: unsupported type");
    }
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
      TaskRequest request = m_free_worker_indices.top();
      m_free_worker_indices.pop();

      if (notified.contains(request.worker_rank)) {
        // Already told this child; it doesn't need (or expect) a separate
        // reply per request it happened to have outstanding.
        continue;
      }
      send_to_worker(nullptr, request.worker_rank, Tag::DONE, request.source_layer);
      notified.insert(request.worker_rank);
    }
  }

  using result_mpi_type = MPI_Type<ResultT>;
  using task_mpi_type = MPI_Type<TaskT>;

  void receive_result_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                           CommLayer layer) {
    m_results.push_back(ResultT{});
    if (result_mpi_type::resize_required) {
      DYNAMPI_UNIMPLEMENTED(  // LCOV_EXCL_LINE
          "Dynamic resizing of results is not supported in hierarchical distribution");
    }
    // With groups, always use global communicator and determine layer from source rank
    int world_source = status.MPI_SOURCE;
    if (m_config.coordinator_per_node) {
      layer = determine_layer_from_world_rank(world_source);
    }
    m_communicator.recv(m_results.back(), world_source, Tag::RESULT);
    m_results_received_from_child++;
    m_free_worker_indices.push(TaskRequest{.worker_rank = world_source, .source_layer = layer});
  }

  void receive_result_batch_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                                 [[maybe_unused]] CommLayer layer) {
    using message_type = MPI_Type<std::vector<ResultT>>;
    int count;
    DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
    std::vector<ResultT> results;
    message_type::resize(results, count);
    int world_source = status.MPI_SOURCE;
    m_communicator.recv(results, world_source, Tag::RESULT_BATCH);
    // Results and next-batch requests are independent messages (see the
    // double-buffering refill in run_worker()), so receiving results here
    // has no side effect on task allocation for this child.
    std::copy(results.begin(), results.end(), std::back_inserter(m_results));
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
    ResultT result = m_worker_function(message);
    m_tasks_executed++;
    // Reply on the global communicator
    send_async(result, world_source, Tag::RESULT);
    m_results_sent_to_parent++;
  }

  void receive_task_batch_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                               [[maybe_unused]] CommLayer layer) {
    if constexpr (prioritize_tasks) {
      DYNAMPI_UNIMPLEMENTED("Prioritized hierarchical distribution");
    } else {
      using message_type = MPI_Type<std::vector<TaskT>>;
      int count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, message_type::value, &count));
      std::vector<TaskT> tasks;
      message_type::resize(tasks, count);
      // With groups, always use global communicator
      int world_source = status.MPI_SOURCE;
      m_communicator.recv(tasks, world_source, Tag::TASK_BATCH);
      m_tasks_received_from_parent += tasks.size();
      // This reply fulfills one of our outstanding pipelined requests (see
      // run_worker()'s top_up_pipeline()); only intermediate coordinators
      // (never leaf workers, which use the unbatched TASK/RESULT protocol)
      // send REQUEST_BATCH, so only they receive TASK_BATCH replies here.
      if (m_outstanding_requests > 0) --m_outstanding_requests;
      // While a round is active, this is the reply to a request for a
      // *future* round: quarantine it so the current round's distribution
      // loop can't overshoot into it (see run_worker()).
      auto& target = m_round_active ? m_prefetched_tasks : m_unallocated_task_queue;
      for (const auto& task : tasks) {
        target.push_back(task);
      }
    }
  }

  void receive_request_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                            CommLayer layer) {
    // With groups, always use global communicator and determine layer from source rank
    int world_source = status.MPI_SOURCE;
    if (m_config.coordinator_per_node) {
      layer = determine_layer_from_world_rank(world_source);
    }
    m_communicator.recv_empty_message(world_source, Tag::REQUEST);
    m_free_worker_indices.push(TaskRequest{.worker_rank = world_source, .source_layer = layer});
  }

  void receive_request_batch_from(MPI_Status status, [[maybe_unused]] MPICommunicator& source_comm,
                                  CommLayer layer) {
    // With groups, always use global communicator and determine layer from source rank
    int world_source = status.MPI_SOURCE;
    if (m_config.coordinator_per_node) {
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
                       status.MPI_TAG <= static_cast<int>(Tag::REQUEST_BATCH),
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
    }
  }
};

};  // namespace dynampi
```


