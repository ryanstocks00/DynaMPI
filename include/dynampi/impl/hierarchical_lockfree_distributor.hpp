/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_group.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/lockfree_distributor.hpp"  // reuses dynampi::detail byte-packing helpers
#include "dynampi/mpi/mpi_error.hpp"
#include "dynampi/utilities/timer.hpp"

namespace dynampi {

namespace detail {

// ---------------------------------------------------------------------------
// LockFreeLevel
//
// One level of lock-free RMA task claiming plus Gatherv-based result
// collection, scoped to an arbitrary communicator. The rank at `owner_rank`
// (within `comm`) hosts the task table window; every other rank in `comm`
// claims tasks from it in batches via atomic fetch-and-add/CAS (no messages,
// no owner-side receive loop), executes them, and returns results via
// periodic Gatherv rounds triggered by the owner bumping a gather-sequence
// counter -- the same mechanism LockFreeMPIWorkDistributor uses at global
// scope, factored out here so it can be composed at multiple levels of a
// tree topology (see HierarchicalLockFreeMPIWorkDistributor below, which
// instantiates one LockFreeLevel per tree level: manager<->node coordinators,
// and, independently, node coordinator<->its local workers).
//
// Results are unordered: this level only tracks how many results came back,
// not which original task each belongs to.
// ---------------------------------------------------------------------------
template <typename TaskT, typename ResultT>
class LockFreeLevel {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_NULL;
    int owner_rank = 0;
    int max_tasks = 8192;       // capacity of the task table (lifetime total for this level)
    int max_task_count = 256;   // max elements per task (only for resizable TaskT)
    int max_result_count = 256;  // unused by this level directly, kept for API symmetry
  };

  explicit LockFreeLevel(Config config) : m_config(config) { initialize_window(); }

  LockFreeLevel(const LockFreeLevel&) = delete;
  LockFreeLevel& operator=(const LockFreeLevel&) = delete;

  ~LockFreeLevel() {
    if (m_window != MPI_WIN_NULL) {
      DYNAMPI_MPI_CHECK(MPI_Win_unlock_all, (m_window));
      MPI_Barrier(m_config.comm);
      DYNAMPI_MPI_CHECK(MPI_Win_free, (&m_window));
      m_window = MPI_WIN_NULL;
    }
  }

  bool is_owner() const { return comm_rank() == m_config.owner_rank; }
  int comm_rank() const {
    int r = 0;
    MPI_Comm_rank(m_config.comm, &r);
    return r;
  }
  int comm_size() const {
    int s = 0;
    MPI_Comm_size(m_config.comm, &s);
    return s;
  }

  void idle_wait() { detail::rma_wait_idle(m_window); }

  // --- Owner-side API ---

  void publish_tasks(const std::vector<TaskT>& tasks) {
    assert(is_owner());
    for (const auto& task : tasks) publish_task(task);
  }

  void mark_finished() {
    assert(is_owner());
    m_owner_marked_finished = true;
    if (comm_size() > 1) {
      atomic_set(FINISHED_OFF, 1);
      idle_wait();
    }
  }

  bool owner_marked_finished() const {
    assert(is_owner());
    return m_owner_marked_finished;
  }

  // Call only once the owner's own collection loop has fully finished
  // (owner_collected_count() >= owner_published_count(), with nothing else
  // that could ever add to owner_published_count()). Drives one final
  // announce/gather round (its Ibcast payload carries is_final=1) all the
  // way to completion -- see advance_gather_round() -- so every claimant is
  // guaranteed to see it via the same broadcast-driven mechanism it already
  // polls with, rather than a separately-polled flag. finalize()'s drain
  // loop above has already confirmed nothing remains to collect, so this is
  // purely the termination handshake: it must still be a real, completed
  // round (not a fire-and-forget flag write) because a claimant only learns
  // "no more rounds are coming" by observing this exact broadcast.
  void mark_gather_done() {
    assert(is_owner());
    if (comm_size() > 1) {
      assert(m_stage == GatherStage::Idle);
      while (!m_owner_gather_finished) {
        auto results = advance_gather_round(/*start_final=*/true);
        assert(results.empty());
        if (!m_owner_gather_finished) idle_wait();
      }
    } else {
      m_owner_gather_finished = true;
    }
  }
  size_t owner_published_count() const {
    assert(is_owner());
    return static_cast<size_t>(m_total_tasks);
  }
  size_t owner_collected_count() const {
    assert(is_owner());
    return m_owner_collected_count;
  }

  // Owner: advance the round state machine by one non-blocking step and
  // return whatever a completed round just produced (possibly empty --
  // most calls land mid-round and return nothing). Safe to call anytime,
  // including with nothing new pending: an Idle call always starts a new
  // round (see advance_gather_round()), matching the old code's "every call
  // announces a round" behavior, just spread asynchronously over however
  // many calls it takes claimants to catch up instead of blocking here for
  // all of them at once.
  std::vector<ResultT> request_gather() {
    assert(is_owner());
    std::vector<ResultT> results;
    if (comm_size() > 1) {
      results = advance_gather_round();
    } else {
      results = std::move(m_pending_results);
      m_pending_results.clear();
    }
    m_owner_collected_count += results.size();
    return results;
  }

  // Like request_gather(), but backs off briefly if the round came back
  // empty, instead of immediately announcing another one. Without this, an
  // owner polling in a tight loop (run_tasks()/finalize()) floods claimants
  // with back-to-back gather rounds; claimants must service every round
  // they observe (see run_local_worker()/run_leaf_leader_worker()), so a
  // relentless owner starves them of time to actually claim and execute
  // tasks between rounds.
  std::vector<ResultT> request_gather_throttled() {
    const size_t before = owner_collected_count();
    auto results = request_gather();
    if (owner_collected_count() == before) idle_wait();
    return results;
  }

  // --- Claimant-side API ---

  // Attempts to claim up to `want` tasks from the current publish frontier.
  // Returns fewer (possibly zero) if the owner hasn't published that many
  // yet, if another claimant won the race for this slice, or if the level
  // has been marked finished with nothing left.
  std::vector<TaskT> try_claim_batch(int want) {
    assert(!is_owner());
    if (want <= 0) return {};
    const int64_t total = atomic_read(TOTAL_OFF);
    const int64_t head = atomic_read(HEAD_OFF);
    if (head >= total) return {};
    const int64_t claim = std::min<int64_t>(want, total - head);
    const int64_t start = compare_and_swap(HEAD_OFF, head, head + claim);
    if (start != head) return {};  // lost the race; caller retries next call
    std::vector<TaskT> tasks;
    tasks.reserve(static_cast<size_t>(claim));
    for (int64_t i = start; i < start + claim; ++i) tasks.push_back(read_task(i));
    return tasks;
  }

  // True once FINISHED has been observed and no tasks remain to claim. Use
  // this to decide whether to keep trying to claim work -- NOT as a loop
  // exit condition (see gather_fully_done()).
  bool drained() {
    assert(!is_owner());
    if (!m_seen_finished && atomic_read(FINISHED_OFF) != 0) m_seen_finished = true;
    if (!m_seen_finished) return false;
    return atomic_read(HEAD_OFF) >= atomic_read(TOTAL_OFF);
  }

  // True once this claimant has fully processed the owner's final round
  // (its Ibcast announce carried is_final=1, and this rank's own
  // announce/count/data phases for that round have all completed -- see
  // advance_gather_round()). This -- not drained() -- is the only condition
  // under which a claimant may safely stop polling and let its process move
  // toward destruction: MPI_Igatherv is collective, so any round the owner
  // still announces after this claimant stops polling would hang the owner
  // forever waiting for a participant that will never arrive. Unlike the
  // RMA-flag version this replaced, there is nothing to read here -- this
  // rank set the flag itself, locally, the moment it finished that round.
  bool gather_fully_done() {
    assert(!is_owner());
    return m_gather_fully_done;
  }

  // Buffers a completed result for the next gather round this rank joins.
  void stage_result(ResultT result) {
    assert(!is_owner());
    m_pending_results.push_back(std::move(result));
  }

  // True if this rank has staged results not yet flushed to the owner via a
  // gather round. A claimant must not stop polling for gather rounds (and
  // must not let its process move on toward destruction) while this is
  // true: staging and reporting are decoupled (stage_result() only buffers
  // locally; a result is only sent once this rank participates in a round
  // the owner initiates), so exiting with pending results here strands
  // them forever -- the owner has no way to know they exist, and this
  // rank's own process is by then past the point where it could still
  // respond to a gather request. See the fix this guards in
  // run_local_worker() / run_leaf_leader_worker() / run_node_coordinator().
  bool has_pending_results() const {
    assert(!is_owner());
    return !m_pending_results.empty();
  }

  // Call periodically (whenever otherwise idle, and after claiming/staging)
  // to notice and join a gather round the owner initiated, or to advance
  // one this rank already joined. Non-blocking: each call does at most one
  // MPI_Test plus, if that just completed, posting the next phase -- never
  // waits on a peer. Returns true if this call made progress.
  bool maybe_participate_in_gather() {
    assert(!is_owner());
    const GatherStage before = m_stage;
    advance_gather_round();
    return m_stage != before;
  }

 private:
  // Window layout: [head][total][finished] then task slots. This window
  // only carries the lock-free task-claiming state now -- gather-round
  // announcement and completion are no longer RMA-polled at all (see
  // GatherStage below); they travel over an MPI_Ibcast/Igather/Igatherv
  // pipeline instead, which is why FINISHED is the only "done" flag left
  // here. (A GATHER_DONE flag used to live here too; it's now carried as
  // the is_final element of the Ibcast payload in m_bcast_buf, delivered by
  // the same broadcast a claimant already polls to learn a round exists at
  // all, rather than a second thing to separately poll for.)
  static constexpr MPI_Aint HEAD_OFF = 0;
  static constexpr MPI_Aint TOTAL_OFF = 8;
  static constexpr MPI_Aint FINISHED_OFF = 16;
  static constexpr size_t CONTROL_BYTES = 24;
  // Per-task slot: [int64 count][data].
  static constexpr size_t T_COUNT = 0;
  static constexpr size_t T_DATA = 8;

  // Gather-round state machine, driven one non-blocking step per call to
  // advance_gather_round() (see below). Shared by owner and claimant: the
  // Idle and AwaitingAnnounce phases behave differently per role, but
  // AwaitingCounts/AwaitingData are structurally identical (just
  // recvbuf-null-or-not), so both roles drive the same m_stage through the
  // same function.
  enum class GatherStage { Idle, AwaitingAnnounce, AwaitingCounts, AwaitingData };

  Config m_config;
  MPI_Win m_window = MPI_WIN_NULL;
  std::vector<std::byte> m_window_buffer;  // owner: control + task table
  alignas(int64_t) std::byte m_peer_window[sizeof(int64_t)]{};  // non-owner: Win_create placeholder

  size_t m_task_elem = 0;
  size_t m_max_task_count = 1;
  size_t m_task_slot_stride = 0;
  size_t m_task_base = 0;

  int64_t m_total_tasks = 0;
  bool m_seen_finished = false;
  bool m_owner_marked_finished = false;
  size_t m_owner_collected_count = 0;
  std::vector<ResultT> m_pending_results;  // claimant: staged, not yet sent; owner (size==1): local store

  GatherStage m_stage = GatherStage::Idle;
  MPI_Request m_bcast_request = MPI_REQUEST_NULL;
  MPI_Request m_gather_request = MPI_REQUEST_NULL;
  MPI_Request m_gatherv_request = MPI_REQUEST_NULL;
  int64_t m_bcast_buf[2] = {0, 0};  // [0]=target task id (informational), [1]=is_final
  int m_exchange_send_count = 0;
  std::vector<std::byte> m_exchange_send_buf;   // this rank's serialized contribution to the in-flight round
  std::vector<int> m_exchange_byte_counts;      // owner only
  std::vector<int> m_exchange_displacements;    // owner only
  std::vector<std::byte> m_exchange_recv_buf;   // owner only
  bool m_gather_fully_done = false;   // claimant only: seen and finished the final round
  bool m_owner_gather_finished = false;  // owner only: final round fully completed

  void initialize_window() {
    m_task_elem = static_cast<size_t>(detail::mpi_type_size_bytes<TaskT>());
    m_max_task_count =
        MPI_Type<TaskT>::resize_required ? static_cast<size_t>(m_config.max_task_count) : 1;
    m_task_slot_stride = detail::round_up_8(T_DATA + m_max_task_count * m_task_elem);

    const size_t capacity = static_cast<size_t>(m_config.max_tasks);
    m_task_base = CONTROL_BYTES;
    const size_t owner_window_bytes = m_task_base + capacity * m_task_slot_stride;

    if (is_owner() && comm_size() > 1) {
      m_window_buffer.resize(owner_window_bytes);
    }
    if (comm_size() == 1) return;  // no peers: nothing to expose over RMA

    void* base = nullptr;
    MPI_Aint bsize = 0;
    if (is_owner()) {
      base = m_window_buffer.data();
      bsize = static_cast<MPI_Aint>(m_window_buffer.size());
    } else {
      base = m_peer_window;
      bsize = static_cast<MPI_Aint>(sizeof(m_peer_window));
    }
    DYNAMPI_MPI_CHECK(MPI_Win_create, (base, bsize, 1, MPI_INFO_NULL, m_config.comm, &m_window));
    DYNAMPI_MPI_CHECK(MPI_Win_lock_all, (MPI_MODE_NOCHECK, m_window));
  }

  MPI_Aint task_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_task_base + static_cast<size_t>(index) * m_task_slot_stride);
  }

  void flush(int rank) { DYNAMPI_MPI_CHECK(MPI_Win_flush, (rank, m_window)); }

  int64_t atomic_read(MPI_Aint offset) {
    int64_t in = 0, out;
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op,
                      (&in, &out, MPI_INT64_T, m_config.owner_rank, offset, MPI_NO_OP, m_window));
    flush(m_config.owner_rank);
    return out;
  }
  void atomic_set(MPI_Aint offset, int64_t value) {
    int64_t in = value, out;
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op, (&in, &out, MPI_INT64_T, m_config.owner_rank, offset,
                                         MPI_REPLACE, m_window));
    flush(m_config.owner_rank);
  }
  int64_t compare_and_swap(MPI_Aint offset, int64_t expected, int64_t desired) {
    int64_t comp = expected, des = desired, out;
    DYNAMPI_MPI_CHECK(MPI_Compare_and_swap,
                      (&des, &comp, &out, MPI_INT64_T, m_config.owner_rank, offset, m_window));
    flush(m_config.owner_rank);
    return out;
  }
  // MPI_Put/MPI_Get take a plain `int` count, but n (a byte length derived
  // from task/result-table capacities that can legitimately reach into the
  // hundreds of millions of slots) can exceed INT_MAX. static_cast<int>(n)
  // on an over-INT_MAX n silently wraps -- see the identical fix (and its
  // fuller rationale) in hierarchical_async_put_lockfree_distributor.hpp's
  // put_bytes()/get_bytes(). Chunk into INT_MAX-bounded pieces so no single
  // MPI_Put/MPI_Get call is ever handed a count that doesn't fit in `int`.
  static constexpr size_t kMaxRmaChunkBytes = static_cast<size_t>(std::numeric_limits<int>::max());

  void put_bytes(const void* src, size_t n, MPI_Aint offset) {
    if (n == 0) return;
    const auto* bytes = static_cast<const std::byte*>(src);
    size_t done = 0;
    while (done < n) {
      const size_t chunk = std::min(kMaxRmaChunkBytes, n - done);
      DYNAMPI_MPI_CHECK(MPI_Put, (bytes + done, static_cast<int>(chunk), MPI_BYTE,
                                  m_config.owner_rank, offset + static_cast<MPI_Aint>(done),
                                  static_cast<int>(chunk), MPI_BYTE, m_window));
      done += chunk;
    }
    flush(m_config.owner_rank);
  }
  void get_bytes(void* dst, size_t n, MPI_Aint offset) {
    if (n == 0) return;
    auto* bytes = static_cast<std::byte*>(dst);
    size_t done = 0;
    while (done < n) {
      const size_t chunk = std::min(kMaxRmaChunkBytes, n - done);
      DYNAMPI_MPI_CHECK(MPI_Get, (bytes + done, static_cast<int>(chunk), MPI_BYTE,
                                  m_config.owner_rank, offset + static_cast<MPI_Aint>(done),
                                  static_cast<int>(chunk), MPI_BYTE, m_window));
      done += chunk;
    }
    flush(m_config.owner_rank);
  }

  void publish_task(const TaskT& task) {
    const int64_t index = m_total_tasks;
    assert(static_cast<size_t>(index) < static_cast<size_t>(m_config.max_tasks) &&
           "LockFreeLevel: exceeded max_tasks capacity");
    if (comm_size() > 1) {
      const int count = MPI_Type<TaskT>::count(task);
      assert(static_cast<size_t>(count) <= m_max_task_count &&
             "LockFreeLevel: task exceeds max_task_count");
      const size_t data_bytes = static_cast<size_t>(count) * m_task_elem;
      std::vector<std::byte> buffer(T_DATA + data_bytes);
      detail::write_i64(buffer.data(), buffer.size(), T_COUNT, count);
      if (count > 0) {
        detail::write_bytes(buffer.data(), buffer.size(), T_DATA, MPI_Type<TaskT>::ptr(task),
                            data_bytes);
      }
      put_bytes(buffer.data(), buffer.size(), task_slot(index));
      m_total_tasks++;
      atomic_set(TOTAL_OFF, m_total_tasks);  // publish to claimants
    } else {
      m_total_tasks++;
    }
  }

  TaskT read_task(int64_t index) {
    int64_t count = 0;
    get_bytes(&count, 8, task_slot(index) + static_cast<MPI_Aint>(T_COUNT));
    TaskT task{};
    if constexpr (MPI_Type<TaskT>::resize_required) MPI_Type<TaskT>::resize(task, static_cast<int>(count));
    get_bytes(MPI_Type<TaskT>::ptr(task), static_cast<size_t>(count) * m_task_elem,
              task_slot(index) + static_cast<MPI_Aint>(T_DATA));
    return task;
  }

  void serialize_pending_results(std::vector<std::byte>& out) const {
    const int elem = detail::mpi_type_size_bytes<ResultT>();
    out.clear();
    for (auto& result : m_pending_results) {
      const int count = MPI_Type<ResultT>::count(result);
      assert(count >= 0);
      const size_t data_bytes =
          count > 0 ? static_cast<size_t>(count) * static_cast<size_t>(elem) : size_t{0};
      const size_t offset = out.size();
      out.resize(offset + 8 + data_bytes);
      detail::write_i64(out.data(), out.size(), offset, count);
      if (data_bytes > 0) {
        detail::write_bytes(out.data(), out.size(), offset + 8, MPI_Type<ResultT>::ptr(result),
                            data_bytes);
      }
    }
  }

  static std::vector<ResultT> deserialize_results(const std::vector<std::byte>& in) {
    const int elem = detail::mpi_type_size_bytes<ResultT>();
    std::vector<ResultT> output;
    size_t pos = 0;
    while (pos < in.size()) {
      assert(pos + 8 <= in.size());
      const int64_t count = detail::read_i64(in.data(), in.size(), pos);
      pos += 8;
      ResultT result{};
      if constexpr (MPI_Type<ResultT>::resize_required)
        MPI_Type<ResultT>::resize(result, static_cast<int>(count));
      assert(count >= 0);
      const size_t data_bytes =
          count > 0 ? static_cast<size_t>(count) * static_cast<size_t>(elem) : size_t{0};
      detail::read_result_bytes(in.data(), in.size(), pos, result, data_bytes);
      pos += data_bytes;
      output.push_back(std::move(result));
    }
    return output;
  }

  // Phase 1->2: snapshot this rank's staged results (claimant only -- the
  // owner never stages any, see stage_result()'s assert) and post the
  // byte-count Igather every rank must participate in before displacements
  // for the data phase can be computed.
  void post_count_gather() {
    const bool owner = is_owner();
    if (!owner) {
      // Snapshot now: m_exchange_send_buf must stay valid and unmodified
      // until post_data_gatherv()'s Igatherv completes. Anything staged
      // after this point accumulates fresh into m_pending_results for
      // whichever round comes after this one.
      serialize_pending_results(m_exchange_send_buf);
      m_pending_results.clear();
    }
    m_exchange_send_count = static_cast<int>(m_exchange_send_buf.size());
    if (owner) m_exchange_byte_counts.resize(static_cast<size_t>(comm_size()));
    DYNAMPI_MPI_CHECK(MPI_Igather,
                      (&m_exchange_send_count, 1, MPI_INT,
                       owner ? m_exchange_byte_counts.data() : nullptr, 1, MPI_INT,
                       m_config.owner_rank, m_config.comm, &m_gather_request));
  }

  // Phase 2->3: owner now knows every rank's byte count, so it can size its
  // receive buffer and displacements; post the data Igatherv every rank
  // must also participate in, sending its snapshot from post_count_gather().
  void post_data_gatherv() {
    const bool owner = is_owner();
    if (owner) {
      const int size = comm_size();
      m_exchange_displacements.resize(static_cast<size_t>(size));
      int total = 0;
      for (int r = 0; r < size; ++r) {
        m_exchange_displacements[static_cast<size_t>(r)] = total;
        total += m_exchange_byte_counts[static_cast<size_t>(r)];
      }
      m_exchange_recv_buf.resize(static_cast<size_t>(total));
    }
    DYNAMPI_MPI_CHECK(
        MPI_Igatherv, (m_exchange_send_buf.data(), m_exchange_send_count, MPI_BYTE,
                      owner ? m_exchange_recv_buf.data() : nullptr,
                      owner ? m_exchange_byte_counts.data() : nullptr,
                      owner ? m_exchange_displacements.data() : nullptr, MPI_BYTE,
                      m_config.owner_rank, m_config.comm, &m_gatherv_request));
  }

  // Advances this rank's participation in the current (or next) gather
  // round by exactly one non-blocking step: one MPI_Test (or, only when
  // starting a round, one Ibcast post) per call, never a wait on a peer.
  // Returns any results the owner just finished harvesting -- always empty
  // except on the call whose MPI_Test observes the data phase complete.
  //
  // `start_final` (owner only): when true and currently Idle, the round
  // this starts carries is_final=1 in its Ibcast payload, telling every
  // claimant this is the last round they'll ever need to join (see
  // mark_gather_done() and gather_fully_done()). Claimants ignore the
  // parameter entirely -- they only ever react to what's broadcast to them.
  std::vector<ResultT> advance_gather_round(bool start_final = false) {
    switch (m_stage) {
      case GatherStage::Idle: {
        if (is_owner()) {
          m_bcast_buf[0] = m_total_tasks;
          m_bcast_buf[1] = start_final ? 1 : 0;
          DYNAMPI_MPI_CHECK(MPI_Ibcast, (m_bcast_buf, 2, MPI_INT64_T, m_config.owner_rank,
                                         m_config.comm, &m_bcast_request));
          // Root's own local completion of a broadcast doesn't depend on
          // any claimant -- it's this rank's own send, expected to finish
          // promptly regardless of how far behind a straggler claimant is
          // (unlike the old barrier, this is not a whole-comm rendezvous).
          // Reaping it inline here, rather than adding a fourth polled
          // phase, keeps the state machine to the two genuinely
          // peer-dependent stages below.
          DYNAMPI_MPI_CHECK(MPI_Wait, (&m_bcast_request, MPI_STATUS_IGNORE));
          post_count_gather();
          m_stage = GatherStage::AwaitingCounts;
        } else {
          if (m_gather_fully_done) return {};
          DYNAMPI_MPI_CHECK(MPI_Ibcast, (m_bcast_buf, 2, MPI_INT64_T, m_config.owner_rank,
                                         m_config.comm, &m_bcast_request));
          m_stage = GatherStage::AwaitingAnnounce;
        }
        return {};
      }
      case GatherStage::AwaitingAnnounce: {
        assert(!is_owner());
        int done = 0;
        DYNAMPI_MPI_CHECK(MPI_Test, (&m_bcast_request, &done, MPI_STATUS_IGNORE));
        if (!done) return {};
        post_count_gather();
        m_stage = GatherStage::AwaitingCounts;
        return {};
      }
      case GatherStage::AwaitingCounts: {
        int done = 0;
        DYNAMPI_MPI_CHECK(MPI_Test, (&m_gather_request, &done, MPI_STATUS_IGNORE));
        if (!done) return {};
        post_data_gatherv();
        m_stage = GatherStage::AwaitingData;
        return {};
      }
      case GatherStage::AwaitingData: {
        int done = 0;
        DYNAMPI_MPI_CHECK(MPI_Test, (&m_gatherv_request, &done, MPI_STATUS_IGNORE));
        if (!done) return {};
        std::vector<ResultT> results = is_owner() ? deserialize_results(m_exchange_recv_buf)
                                                   : std::vector<ResultT>{};
        const bool was_final = (m_bcast_buf[1] != 0);
        m_exchange_send_buf.clear();
        m_stage = GatherStage::Idle;
        if (was_final) {
          if (is_owner()) {
            m_owner_gather_finished = true;
          } else {
            m_gather_fully_done = true;
          }
        }
        return results;
      }
    }
    return {};
  }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// HierarchicalLockFreeMPIWorkDistributor
//
// Combines HierarchicalMPIWorkDistributor's node-aware tree topology
// (manager <-> per-node coordinators <-> per-node local workers) with
// LockFreeMPIWorkDistributor's one-sided RMA task dispatch, applied
// independently at each level of the tree:
//
//   - Leader level: the manager's task table window is claimed, in
//     batches, only by node coordinators (fan-in bounded by node count,
//     not total rank count).
//   - Local level: each node coordinator hosts its own task table window
//     that only its own node's workers claim from (fan-in bounded by
//     ranks-per-node).
//
// This avoids both bottlenecks of the two distributors it combines: the
// flat LockFreeMPIWorkDistributor's single window fielding RMA traffic from
// every rank in the job, and HierarchicalMPIWorkDistributor's blocking
// message-passing receive loop at each coordinator.
//
// Results are unordered (see LockFreeLevel); use a different distributor if
// task-index-ordered output is required. Task prioritization and detailed
// statistics are not supported (matching LockFreeMPIWorkDistributor).
// ---------------------------------------------------------------------------
template <typename TaskT, typename ResultT, typename... Options>
class HierarchicalLockFreeMPIWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    int leader_batch_multiplier = 2;  // coordinators claim (local peers)*multiplier tasks at once
    int local_batch_size = 1;         // local workers claim this many tasks at once from their coordinator
    int max_tasks = 8192;             // leader-level task table capacity (lifetime total)
    int max_local_tasks = 8192;       // per-node local task table capacity (lifetime total, per node)
    int max_task_count = 256;
    int max_result_count = 256;
  };

  struct RunConfig {
    size_t target_num_tasks = std::numeric_limits<size_t>::max();
    bool allow_more_than_target_tasks = true;
    std::optional<double> max_seconds = std::nullopt;
  };

  static const bool ordered = false;

  explicit HierarchicalLockFreeMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                                   Config config = {})
      : m_config(config),
        m_world_comm(config.comm, MPICommunicator<>::Duplicate),
        m_worker_function(std::move(worker_function)) {
    setup_topology();
    setup_levels();

    if (m_config.auto_run_workers && !is_root_manager()) run_worker();
  }

  ~HierarchicalLockFreeMPIWorkDistributor() {
    if (!m_finalized) finalize();
  }

  bool is_root_manager() const { return m_world_comm.rank() == m_config.manager_rank; }

  size_t remaining_tasks_count() const {
    assert(is_root_manager());
    if (m_solo) return m_local_task_store.size() - m_local_collected_count;
    return m_leader_level->owner_published_count() - m_leader_level->owner_collected_count() -
           m_results.size();
  }

  void insert_task(TaskT task) {
    assert(is_root_manager());
    insert_tasks(std::vector<TaskT>{std::move(task)});
  }

  void insert_tasks(const std::vector<TaskT>& tasks) {
    assert(is_root_manager());
    if (m_solo) {
      m_local_task_store.insert(m_local_task_store.end(), tasks.begin(), tasks.end());
    } else {
      m_leader_level->publish_tasks(tasks);
    }
  }

  [[nodiscard]] std::vector<ResultT> run_tasks(const RunConfig& config = RunConfig{}) {
    assert(is_root_manager());
    Timer timer;

    if (m_solo) {
      // No other ranks at all: run everything inline. (m_leader_level would
      // otherwise still get constructed for the manager -- see setup_levels
      // -- so this must be gated on true world size, not level presence.)
      while (m_local_collected_count < m_local_task_store.size()) {
        if (m_results.size() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        m_results.push_back(m_worker_function(m_local_task_store[m_local_collected_count]));
        m_local_collected_count++;
      }
    } else {
      while (true) {
        if (m_results.size() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        if (m_leader_level->owner_collected_count() >= m_leader_level->owner_published_count()) break;
        auto results = m_leader_level->request_gather_throttled();
        m_results.insert(m_results.end(), std::make_move_iterator(results.begin()),
                         std::make_move_iterator(results.end()));
      }
    }

    return drain_results(config.allow_more_than_target_tasks
                             ? std::numeric_limits<size_t>::max()
                             : config.target_num_tasks);
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() { return run_tasks({}); }

  void finalize() {
    assert(!m_finalized);
    if (is_root_manager()) {
      if (m_solo) {
        while (m_local_collected_count < m_local_task_store.size()) {
          m_results.push_back(m_worker_function(m_local_task_store[m_local_collected_count]));
          m_local_collected_count++;
        }
      } else {
        while (m_leader_level->owner_collected_count() < m_leader_level->owner_published_count()) {
          auto results = m_leader_level->request_gather_throttled();
          m_results.insert(m_results.end(), std::make_move_iterator(results.begin()),
                           std::make_move_iterator(results.end()));
        }
        m_leader_level->mark_finished();
        // Strictly after mark_finished(), and only now that the drain loop
        // above confirms nothing more will ever be collected: see the
        // window-layout comment on GATHER_DONE_OFF in LockFreeLevel.
        m_leader_level->mark_gather_done();
      }
    }
    m_finalized = true;
  }

  void run_worker() {
    assert(!is_root_manager());
    if (m_local_level && m_local_comm->size() > 1) {
      if (m_leader_level) {
        run_node_coordinator();
      } else {
        run_local_worker();
      }
    } else if (m_leader_level) {
      // Sole rank on this node besides being a leader: no local peers, so
      // this rank claims and executes directly at the leader level.
      run_leaf_leader_worker();
    }
  }

 private:
  Config m_config;
  MPICommunicator<> m_world_comm;
  std::function<ResultT(TaskT)> m_worker_function;
  const bool m_solo = (m_world_comm.size() == 1);  // true world size 1: no levels are usable

  std::optional<MPICommunicator<>> m_local_comm;   // this node's coordinator + local workers
  std::optional<MPICommunicator<>> m_leader_comm;  // manager + all node coordinators

  std::optional<detail::LockFreeLevel<TaskT, ResultT>> m_leader_level;
  std::optional<detail::LockFreeLevel<TaskT, ResultT>> m_local_level;

  bool m_finalized = false;
  std::vector<ResultT> m_results;         // manager only: results ready to return
  size_t m_returned_count = 0;            // manager only: results already handed to the caller

  std::vector<TaskT> m_local_task_store;  // manager only, solo-world fallback
  size_t m_local_collected_count = 0;     // manager only, solo-world fallback

  void setup_topology() {
    if (m_solo) return;  // nothing to split; LockFreeLevel can't run tasks itself (see m_solo)

    MPICommunicator<> node_comm = m_world_comm.split_by_node();

    const int local_color = (m_world_comm.rank() == m_config.manager_rank) ? MPI_UNDEFINED : 0;
    auto local_opt = node_comm.split(local_color, m_world_comm.rank());
    if (local_opt.has_value()) m_local_comm.emplace(std::move(*local_opt));

    const bool is_manager = (m_world_comm.rank() == m_config.manager_rank);
    bool is_node_coordinator = false;
    if (m_local_comm.has_value()) is_node_coordinator = (m_local_comm->rank() == 0);

    const int leader_color = (is_manager || is_node_coordinator) ? 0 : MPI_UNDEFINED;
    auto leader_opt = m_world_comm.split(leader_color, m_world_comm.rank());
    if (leader_opt.has_value()) m_leader_comm.emplace(std::move(*leader_opt));
  }

  void setup_levels() {
    if (m_leader_comm.has_value()) {
      MPIGroup world_group(m_world_comm);
      MPIGroup leader_group(*m_leader_comm);
      const int leader_owner_rank = world_group.translate_rank(m_config.manager_rank, leader_group);
      typename detail::LockFreeLevel<TaskT, ResultT>::Config leader_cfg;
      leader_cfg.comm = m_leader_comm->get();
      leader_cfg.owner_rank = leader_owner_rank;
      leader_cfg.max_tasks = m_config.max_tasks;
      leader_cfg.max_task_count = m_config.max_task_count;
      m_leader_level.emplace(leader_cfg);
    }

    if (m_local_comm.has_value()) {
      typename detail::LockFreeLevel<TaskT, ResultT>::Config local_cfg;
      local_cfg.comm = m_local_comm->get();
      local_cfg.owner_rank = 0;  // node coordinator is always local rank 0
      local_cfg.max_tasks = m_config.max_local_tasks;
      local_cfg.max_task_count = m_config.max_task_count;
      m_local_level.emplace(local_cfg);
    }
  }

  // A node coordinator: bridges the leader level (claiming batches from the
  // manager) and the local level (its own window, published to for its
  // local workers to claim from). This rank plays both roles at once: owner
  // of the local level, and claimant of the leader level -- so its exit
  // condition is the AND of both sides' termination signals (see
  // LockFreeLevel::gather_fully_done() for why "no more tasks" is not
  // enough on its own).
  void run_node_coordinator() {
    const int local_peers = std::max(0, m_local_comm->size() - 1);
    const int leader_claim_size = std::max(1, local_peers * m_config.leader_batch_multiplier);
    bool local_gather_done_marked = false;

    while (true) {
      m_leader_level->maybe_participate_in_gather();

      if (!m_local_level->owner_marked_finished()) {
        auto tasks = m_leader_level->try_claim_batch(leader_claim_size);
        if (!tasks.empty()) {
          m_local_level->publish_tasks(tasks);
        } else if (m_leader_level->drained()) {
          m_local_level->mark_finished();
        }
      }

      // Stop initiating local-level rounds once this coordinator's own
      // collection there is confirmed complete -- there's nothing left to
      // gather, and continuing would just needlessly force every local
      // worker still polling into another synchronization for no reason.
      if (!local_gather_done_marked) {
        auto local_results = m_local_level->request_gather_throttled();
        for (auto& result : local_results) m_leader_level->stage_result(std::move(result));

        if (m_local_level->owner_marked_finished() &&
            m_local_level->owner_collected_count() >= m_local_level->owner_published_count()) {
          // Local level: this coordinator (as its owner) has now collected
          // everything it ever will. Local workers waiting on
          // gather_fully_done() can now safely exit.
          m_local_level->mark_gather_done();
          local_gather_done_marked = true;
        }
      }

      // Leader level: this coordinator (as a claimant) may only exit once
      // the manager has announced it will never call another gather round
      // -- and, symmetrically to the local level above, only once any of
      // this coordinator's own still-unflushed leader-level results have
      // gone out (which they always will before the manager can possibly
      // reach that point: it can't see everything collected while this
      // coordinator still has something staged that hasn't reached it).
      if (local_gather_done_marked && m_leader_level->gather_fully_done() &&
          !m_leader_level->has_pending_results()) {
        break;
      }

      m_leader_level->idle_wait();
    }
  }

  // A plain local worker: claims individual tasks from its node coordinator.
  void run_local_worker() {
    while (true) {
      // Deliberately ignore the return value here (unlike a naive "if
      // pending round: continue" short-circuit). If the coordinator is
      // issuing gather rounds back-to-back (e.g. every iteration of
      // run_node_coordinator(), which calls request_gather()
      // unconditionally), skipping straight back to the top of this loop
      // whenever a round was just run starves this rank of ever reaching
      // try_claim_batch() again -- a real livelock, not merely a
      // theoretical one: reproduced with 2 ranks (see repro_hlf.cpp in the
      // session that found this). Always fall through to attempt a claim.
      m_local_level->maybe_participate_in_gather();
      if (!m_local_level->drained()) {
        auto tasks = m_local_level->try_claim_batch(m_config.local_batch_size);
        if (!tasks.empty()) {
          for (auto& task : tasks) m_local_level->stage_result(m_worker_function(std::move(task)));
          continue;
        }
      }
      // gather_fully_done() -- not drained() -- gates exit: see the
      // window-layout comment on GATHER_DONE_OFF in LockFreeLevel. Confirmed
      // via gdb backtraces on hung 8/9-rank runs that drained() alone lets
      // this rank exit and stop polling while the coordinator is still
      // waiting on other stragglers and will announce more rounds --
      // MPI_Gather is collective, so this rank not showing up for one of
      // those hangs the coordinator forever.
      if (m_local_level->gather_fully_done()) break;
      m_local_level->idle_wait();
    }
  }

  // A leader-level rank with no local peers (alone on its node): claims and
  // executes tasks directly at the leader level, no local level involved.
  void run_leaf_leader_worker() {
    while (true) {
      // See run_local_worker(): must not skip claiming just because a
      // gather round was serviced this iteration, or a manager hammering
      // request_gather() in a tight loop starves this rank forever.
      m_leader_level->maybe_participate_in_gather();
      if (!m_leader_level->drained()) {
        auto tasks = m_leader_level->try_claim_batch(m_config.leader_batch_multiplier);
        if (!tasks.empty()) {
          for (auto& task : tasks) m_leader_level->stage_result(m_worker_function(std::move(task)));
          continue;
        }
      }
      // See run_local_worker(): gather_fully_done() (not drained()) gates
      // exit -- same bug, same fix, at the leader level instead of local.
      if (m_leader_level->gather_fully_done()) break;
      m_leader_level->idle_wait();
    }
  }

  std::vector<ResultT> drain_results(size_t limit) {
    const size_t count = std::min(limit, m_results.size());
    std::vector<ResultT> output;
    output.reserve(count);
    for (size_t i = 0; i < count; ++i) output.push_back(std::move(m_results[i]));
    m_results.erase(m_results.begin(), m_results.begin() + static_cast<ptrdiff_t>(count));
    m_returned_count += count;
    return output;
  }
};

}  // namespace dynampi
