/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <thread>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_group.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/rma_detail.hpp"
#include "dynampi/mpi/mpi_error.hpp"
#include "dynampi/task_error.hpp"
#include "dynampi/utilities/timer.hpp"

namespace dynampi {

namespace detail {

// ---------------------------------------------------------------------------
// LockFreeRMALevel
//
// One level of LockFreeRMAWorkDistributor's protocol (fetch-and-add claiming,
// one-sided Put of results plus a completion log, no collectives on the hot
// path), scoped to an arbitrary communicator so it can be composed per tree
// level: root manager <-> node managers, and node manager <-> its local
// workers.
//
// Publishing, claiming, writing results and harvesting are each purely
// one-sided, with no requirement that claimants participate together, so
// composing the protocol hierarchically reintroduces no synchronization.
//
// The API is split into non-blocking steps (try_claim() /
// write_result_range()) rather than one claim-compute-write loop: a manager
// claiming here does not compute the results, it republishes into a
// different level and writes them back once they return, so it must
// interleave the two independently. A leaf worker just calls both
// back-to-back.
// ---------------------------------------------------------------------------
template <typename TaskT, typename ResultT>
class LockFreeRMALevel {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_NULL;
    int owner_rank = 0;
    int max_tasks = 8192;
    int max_task_count = 256;
    int max_result_count = 256;
  };

  struct ClaimedRange {
    int64_t start =
        -1;  // -1: nothing claimable right now (not necessarily drained -- see drained())
    std::vector<TaskT> tasks;
  };

  explicit LockFreeRMALevel(Config config, int claim_width = 1)
      : m_config(config),
        m_claim_width(std::max(1, claim_width)),
        m_comm(m_config.comm, MPICommunicator<>::Reference),
        m_errors_seen(static_cast<size_t>(kMaxRecordedErrors)) {
    initialize_window();
  }

  LockFreeRMALevel(const LockFreeRMALevel&) = delete;
  LockFreeRMALevel& operator=(const LockFreeRMALevel&) = delete;

  ~LockFreeRMALevel() {
    if (m_window != MPI_WIN_NULL) {
      DYNAMPI_MPI_CHECK(MPI_Win_unlock_all, (m_window));
      MPI_Barrier(m_config.comm);
      DYNAMPI_MPI_CHECK(MPI_Win_free, (&m_window));
      m_window = MPI_WIN_NULL;
    }
  }

  bool is_owner() const { return m_comm.rank() == m_config.owner_rank; }
  int comm_rank() const { return m_comm.rank(); }
  int comm_size() const { return m_comm.size(); }
  int claim_width() const { return m_claim_width; }

  void idle_wait() {
    if (m_window == MPI_WIN_NULL) {
      // Size-1 local path: no RMA window (Open MPI rejects Win_create on
      // singleton communicators). Still yield so a tight owner/claimant
      // spin doesn't burn a core.
      std::this_thread::yield();
      return;
    }
    detail::rma_wait_idle(m_window, m_comm.get());
  }

  // --- Owner-side API ---

  // Publishes tasks.size() tasks in two round trips total regardless of how
  // many: one bulk Put, then one atomic_set(TOTAL_OFF) -- see
  // LockFreeRMAWorkDistributor::publish_tasks() for the full
  // rationale (this is the same fix applied there).
  void publish_tasks(const std::vector<TaskT>& tasks) {
    assert(is_owner());
    if (tasks.empty()) return;
    const int64_t start = m_total_tasks;
    detail::check_task_capacity(start, tasks.size(), m_config.max_tasks, "LockFreeRMALevel");
    std::vector<std::byte> buffer(tasks.size() * m_task_slot_stride);
    for (size_t i = 0; i < tasks.size(); ++i) {
      const TaskT& task = tasks[i];
      const int count = MPI_Type<TaskT>::count(task);
      assert(static_cast<size_t>(count) <= m_max_task_count &&
             "LockFreeRMALevel: task exceeds max_task_count");
      const size_t data_bytes = static_cast<size_t>(count) * m_task_elem;
      const size_t off = i * m_task_slot_stride;
      detail::write_i64(buffer.data(), buffer.size(), off + T_COUNT, count);
      if (data_bytes > 0) {
        detail::write_bytes(buffer.data(), buffer.size(), off + T_DATA, MPI_Type<TaskT>::ptr(task),
                            data_bytes);
      }
    }
    if (local_only()) {
      detail::write_bytes(m_window_buffer.data(), m_window_buffer.size(),
                          static_cast<size_t>(task_slot(start)), buffer.data(), buffer.size());
      m_total_tasks += static_cast<int64_t>(tasks.size());
      local_store_i64(TOTAL_OFF, m_total_tasks);
      return;
    }
    post_put_bytes(buffer.data(), buffer.size(), task_slot(start));
    m_total_tasks += static_cast<int64_t>(tasks.size());
    int64_t total_out = 0;
    post_fetch_and_op(m_total_tasks, total_out, TOTAL_OFF, MPI_REPLACE);
    flush_remote();
  }

  void mark_finished() {
    assert(is_owner());
    m_owner_marked_finished = true;
    if (local_only()) {
      local_store_i64(FINISHED_OFF, 1);
    } else {
      int64_t finished_out = 0;
      post_fetch_and_op(static_cast<int64_t>(1), finished_out, FINISHED_OFF, MPI_REPLACE);
      flush_remote();
    }
    idle_wait();
  }

  bool owner_marked_finished() const {
    assert(is_owner());
    return m_owner_marked_finished;
  }

  size_t owner_published_count() const {
    assert(is_owner());
    return static_cast<size_t>(m_total_tasks);
  }
  size_t owner_collected_count() const {
    assert(is_owner());
    return m_owner_collected_count;
  }

  // Returns the newly-confirmed contiguous prefix of results, possibly
  // empty, in at most three round trips -- see
  // LockFreeRMAWorkDistributor::harvest_ready_results(). Factored out here so
  // it is not tied to one result vector: the top-level manager appends the
  // batch to its own buffer, a node manager routes it into a relay queue.
  std::vector<ResultT> harvest_ready_results() {
    assert(is_owner());
    const int64_t head_now = atomic_read(HEAD_OFF);
    const int64_t frontier = static_cast<int64_t>(m_owner_collected_count);
    if (head_now <= frontier) return {};

    const size_t scan_count = static_cast<size_t>(head_now - frontier);
    std::vector<std::byte> log_buf(scan_count * LOG_ENTRY_BYTES);
    get_bytes_local(log_buf.data(), log_buf.size(), log_slot(frontier));

    int64_t confirmed_end = frontier;
    bool saw_error = false;
    while (confirmed_end < head_now) {
      const size_t off = static_cast<size_t>(confirmed_end - frontier) * LOG_ENTRY_BYTES;
      const int64_t entry = detail::read_i64(log_buf.data(), log_buf.size(), off);
      if (entry == 0) break;  // gap: not yet written; stop the contiguous prefix here
      if (entry < 0) saw_error = true;
      confirmed_end += entry < 0 ? -entry : entry;
    }
    // A negated length means "this range produced at least one error". The scan
    // reads every entry anyway, so the flag costs nothing until it fires.
    if (saw_error) harvest_task_errors();
    if (confirmed_end <= frontier) return {};

    const int64_t n = confirmed_end - frontier;
    std::vector<std::byte> result_buf(static_cast<size_t>(n) * m_result_slot_stride);
    get_bytes_local(result_buf.data(), result_buf.size(), result_slot(frontier));

    std::vector<ResultT> output;
    output.reserve(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      const size_t off = static_cast<size_t>(i) * m_result_slot_stride;
      const int64_t count = detail::read_i64(result_buf.data(), result_buf.size(), off + R_COUNT);
      ResultT result{};
      if constexpr (MPI_Type<ResultT>::resize_required)
        MPI_Type<ResultT>::resize(result, static_cast<int>(count));
      const size_t data_bytes = count > 0 ? static_cast<size_t>(count) * m_result_elem : size_t{0};
      detail::read_result_bytes(result_buf.data(), result_buf.size(), off + R_DATA, result,
                                data_bytes);
      output.push_back(std::move(result));
    }
    m_owner_collected_count = static_cast<size_t>(confirmed_end);
    return output;
  }

  // Like harvest_ready_results(), but backs off briefly if nothing new was
  // found, instead of being called back-to-back in a tight owner loop -- a
  // politeness backoff against pure CPU/RMA-poll churn rather than a
  // correctness requirement (there's no gather round to flood).
  std::vector<ResultT> harvest_ready_results_throttled() {
    const size_t before = owner_collected_count();
    auto results = harvest_ready_results();
    if (owner_collected_count() == before) idle_wait();
    return results;
  }

  // --- Claimant-side API ---
  //
  // NOTE: unlike the owner-side API, these do NOT assert !is_owner(). A
  // promoted group leader owns this level and must also claim from it, since
  // its own subtree needs feeding. Every RMA helper targets owner_rank
  // unconditionally, so a self-targeted call is just a loopback of the
  // remote path -- the same reason an owner reads its own window through
  // the RMA API rather than a memory load.

  // Attempts one non-blocking claim step (same state machine as
  // LockFreeRMAWorkDistributor::run_worker(), exposed here as a
  // discrete callable rather than an internal loop -- see the class
  // comment). Returns start=-1 if nothing is claimable right this instant
  // (could be a genuine claim miss, a wait for TOTAL_OFF to catch up to an
  // already-reserved range, or true exhaustion -- see drained() for the
  // latter).
  ClaimedRange try_claim() {
    if (m_pending_start != -1) {
      const int64_t total = atomic_read(TOTAL_OFF);
      const int64_t ready = std::min(m_pending_end, total);
      if (ready > m_pending_start) {
        const int64_t start = m_pending_start;
        const int64_t count = ready - m_pending_start;
        m_pending_start = ready;
        if (m_pending_start >= m_pending_end || m_seen_finished) m_pending_start = -1;
        return ClaimedRange{start, read_task_batch(start, count)};
      }
      if (m_pending_start >= m_pending_end || m_seen_finished) m_pending_start = -1;
      return ClaimedRange{};
    }

    // Deliberately NOT gated on !m_seen_finished when re-checking
    // cached_total below the threshold -- see drained()'s exit-check
    // symmetric handling and LockFreeRMAWorkDistributor::run_worker()'s
    // identical comment: once finished_seen is true, a stale cached_total
    // must still be correctable by drained()'s own fresh read, not frozen.
    if (m_cached_head >= m_cached_total && !m_seen_finished)
      m_cached_total = atomic_read(TOTAL_OFF);

    if (m_cached_head < m_cached_total) {
      const int64_t claim = std::min<int64_t>(m_claim_width, m_cached_total - m_cached_head);
      const int64_t start = fetch_add(HEAD_OFF, claim);
      const int64_t end = start + claim;
      m_cached_head = end;

      const int64_t total = (end <= m_cached_total) ? m_cached_total : atomic_read(TOTAL_OFF);
      // Clamped into [start, end]: total can land below start (this whole
      // claim landed beyond what's published yet), not just below end. See
      // LockFreeRMAWorkDistributor::run_worker()'s comment on the
      // exact bug this guards -- an unclamped `min(end, total)` here could
      // produce a pending_start below this claim's own start, an index this
      // rank never actually owns, corrupting another claimant's range.
      const int64_t ready = std::clamp(total, start, end);
      if (ready < end) {
        m_pending_start = ready;
        m_pending_end = end;
      }
      if (ready > start) {
        return ClaimedRange{start, read_task_batch(start, ready - start)};
      }
    }
    return ClaimedRange{};
  }

  // Refreshes the cached finished-observation; call periodically (try_claim()
  // does not do this itself, since a caller managing multiple levels --
  // see run_node_manager() -- may want to control polling cadence
  // explicitly rather than pay a FINISHED_OFF read on every try_claim()).
  bool check_finished() {
    if (!m_seen_finished && atomic_read(FINISHED_OFF) != 0) m_seen_finished = true;
    return m_seen_finished;
  }

  // True once nothing more will ever be claimable: no pending remainder,
  // FINISHED_OFF observed, and cached_head caught up to a fresh read of
  // TOTAL_OFF -- fresh because cached_total can lag the point where
  // publishing actually stopped.
  bool drained() {
    // Refreshed BEFORE the pending-remainder early return, not after. A
    // claimant holding a remainder would otherwise return false here every
    // time without reaching check_finished(), leaving m_seen_finished stuck
    // false -- and try_claim()'s pending branch needs that flag to give up
    // on a remainder that will never resolve, so the claimant deadlocks.
    check_finished();
    if (m_pending_start != -1) return false;
    if (!m_seen_finished) return false;
    const int64_t final_total = atomic_read(TOTAL_OFF);
    if (m_cached_head >= final_total) return true;
    m_cached_total = final_total;  // real unclaimed work remains; try_claim() will pick it up
    return false;
  }

  // Claimant-side. Publishes one failed task into the error table, preserving
  // the rank that actually ran it -- a manager relaying a subtree's failure
  // upward reports the original rank, not its own. Must complete before the log
  // entry that advertises it; every RMA helper here flushes its target before
  // returning, so calling this before write_result_range() is enough.
  void report_task_error(const TaskError& error) {
    const int64_t slot = fetch_add(ERROR_COUNT_OFF, 1);
    if (slot >= kMaxRecordedErrors) return;
    // Message first, then the ready word, each flushed separately: fetch_add
    // hands out the slot before anything is in it, so the count alone never
    // means a slot can be read. The ready word is rank+1 so that zero stays the
    // untouched sentinel for rank 0.
    std::vector<std::byte> message_bytes(kMaxTaskErrorMessage, std::byte{0});
    const size_t bytes = std::min(error.message.size(), kMaxTaskErrorMessage - 1);
    if (bytes > 0) {
      detail::write_bytes(message_bytes.data(), message_bytes.size(), 0, error.message.data(),
                          bytes);
    }
    const int64_t ready = static_cast<int64_t>(error.worker_rank) + 1;
    if (local_only()) {
      // LCOV_EXCL_START -- only reachable when this level's own owned group
      // is a singleton (an odd manager count under max_upper_fanout
      // leaves one such leftover group; see setup_upper_chain()) AND a task
      // fails there specifically, rather than in the far more common case
      // of a genuinely remote claimant.
      detail::write_bytes(m_window_buffer.data(), m_window_buffer.size(),
                          static_cast<size_t>(error_slot(slot)) + E_DATA, message_bytes.data(),
                          message_bytes.size());
      local_store_i64(error_slot(slot) + static_cast<MPI_Aint>(E_RANK), ready);
      return;
      // LCOV_EXCL_STOP
    }
    post_put_bytes(message_bytes.data(), message_bytes.size(),
                   error_slot(slot) + static_cast<MPI_Aint>(E_DATA));
    flush_remote();
    post_put_bytes(&ready, sizeof(ready), error_slot(slot) + static_cast<MPI_Aint>(E_RANK));
    flush_remote();
  }

  // Owner-side. Everything harvested since the last call, oldest first.
  std::vector<TaskError> take_errors() {
    std::vector<TaskError> taken;
    taken.swap(m_owner_errors);
    return taken;
  }

  // Writes results.size() results starting at task index `start`:
  // Put data -> flush -> Put completion-log flag -> flush. The log write is a
  // plain Put (not an atomic): the claim index is exclusively owned.
  void write_result_range(int64_t start, const std::vector<ResultT>& results,
                          bool contains_error = false) {
    const int64_t count = static_cast<int64_t>(results.size());
    assert(count > 0);
    std::vector<std::byte> buffer(static_cast<size_t>(count) * m_result_slot_stride);
    for (int64_t i = 0; i < count; ++i) {
      const ResultT& result = results[static_cast<size_t>(i)];
      const int elem_count = MPI_Type<ResultT>::count(result);
      assert(elem_count >= 0);
      const size_t data_bytes =
          elem_count > 0 ? static_cast<size_t>(elem_count) * m_result_elem : size_t{0};
      const size_t off = static_cast<size_t>(i) * m_result_slot_stride;
      detail::write_i64(buffer.data(), buffer.size(), off + R_COUNT, elem_count);
      if (data_bytes > 0) {
        detail::write_bytes(buffer.data(), buffer.size(), off + R_DATA,
                            MPI_Type<ResultT>::ptr(result), data_bytes);
      }
    }
    const int64_t entry = contains_error ? -count : count;
    if (local_only()) {
      detail::write_bytes(m_window_buffer.data(), m_window_buffer.size(),
                          static_cast<size_t>(result_slot(start)), buffer.data(), buffer.size());
      local_store_i64(log_slot(start), entry);
      return;
    }
    post_put_bytes(buffer.data(), buffer.size(), result_slot(start));
    flush_remote();
    post_put_bytes(&entry, sizeof(entry), log_slot(start));
    flush_remote();
  }

 private:
  // Window layout: [head][total][finished] then the task table, the result
  // table, and the completion log, each sized for max_tasks entries -- same
  // layout as LockFreeRMAWorkDistributor.
  static constexpr MPI_Aint HEAD_OFF = 0;
  static constexpr MPI_Aint TOTAL_OFF = 8;
  static constexpr MPI_Aint FINISHED_OFF = 16;
  static constexpr MPI_Aint ERROR_COUNT_OFF = 24;
  static constexpr size_t CONTROL_BYTES = 32;

  // Failed tasks report into a small fixed table -- same layout and rationale
  // as LockFreeRMAWorkDistributor's.
  static constexpr int64_t kMaxRecordedErrors = 16;
  static constexpr size_t E_RANK = 0;
  static constexpr size_t E_DATA = 8;
  static constexpr size_t ERROR_SLOT_BYTES = E_DATA + kMaxTaskErrorMessage;

  static constexpr size_t T_COUNT = 0;
  static constexpr size_t T_DATA = 8;
  static constexpr size_t R_COUNT = 0;
  static constexpr size_t R_DATA = 8;
  static constexpr size_t LOG_ENTRY_BYTES = 8;

  Config m_config;
  int m_claim_width;
  MPICommunicator<> m_comm;
  MPI_Win m_window = MPI_WIN_NULL;
  std::vector<std::byte> m_window_buffer;
  alignas(int64_t) std::byte m_peer_window[sizeof(int64_t)]{};

  size_t m_task_elem = 0;
  size_t m_result_elem = 0;
  size_t m_max_task_count = 1;
  size_t m_max_result_count = 1;
  size_t m_task_slot_stride = 0;
  size_t m_result_slot_stride = 0;
  size_t m_task_base = 0;
  size_t m_result_base = 0;
  size_t m_log_base = 0;
  size_t m_error_base = 0;

  int64_t m_total_tasks = 0;
  bool m_owner_marked_finished = false;
  size_t m_owner_collected_count = 0;

  // Claimant-only state (see try_claim()/drained()).
  int64_t m_cached_head = 0;
  int64_t m_cached_total = 0;
  int64_t m_pending_start = -1;
  int64_t m_pending_end = -1;
  bool m_seen_finished = false;
  std::vector<bool> m_errors_seen;        // owner-side: error slots already consumed
  std::vector<TaskError> m_owner_errors;  // owner-side, drained by take_errors()

  void initialize_window() {
    // Same fixed-width slot requirement as the flat distributor's window.
    check_fixed_size_mpi_type<TaskT>("task", "LockFreeRMALevel");
    check_fixed_size_mpi_type<ResultT>("result", "LockFreeRMALevel");

    m_task_elem = static_cast<size_t>(detail::mpi_type_size_bytes<TaskT>());
    m_result_elem = static_cast<size_t>(detail::mpi_type_size_bytes<ResultT>());
    // Same per-value element sizing as the flat distributor's window.
    m_max_task_count = MPI_Type<TaskT>::resize_required
                           ? static_cast<size_t>(m_config.max_task_count)
                           : static_cast<size_t>(mpi_elements_per_value<TaskT>());
    m_max_result_count = MPI_Type<ResultT>::resize_required
                             ? static_cast<size_t>(m_config.max_result_count)
                             : static_cast<size_t>(mpi_elements_per_value<ResultT>());

    m_task_slot_stride = detail::round_up_8(T_DATA + m_max_task_count * m_task_elem);
    m_result_slot_stride = detail::round_up_8(R_DATA + m_max_result_count * m_result_elem);

    const size_t capacity = static_cast<size_t>(m_config.max_tasks);
    m_task_base = CONTROL_BYTES;
    m_result_base = m_task_base + capacity * m_task_slot_stride;
    m_log_base = m_result_base + capacity * m_result_slot_stride;
    m_error_base = m_log_base + capacity * LOG_ENTRY_BYTES;
    const size_t owner_window_bytes =
        m_error_base + static_cast<size_t>(kMaxRecordedErrors) * ERROR_SLOT_BYTES;

    // Size-1 communicators deliberately skip MPI_Win_create: Open MPI and
    // others reject it with MPI_ERR_WIN. A solo group leader still
    // self-claims from its size-1 level via the local helpers below -- same
    // process, so plain loads/stores are correct.
    if (is_owner()) {
      m_window_buffer.resize(owner_window_bytes);
    }
    if (comm_size() == 1) return;

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
  MPI_Aint result_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_result_base + static_cast<size_t>(index) * m_result_slot_stride);
  }
  MPI_Aint log_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_log_base + static_cast<size_t>(index) * LOG_ENTRY_BYTES);
  }
  MPI_Aint error_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_error_base + static_cast<size_t>(index) * ERROR_SLOT_BYTES);
  }

  void flush_remote() { DYNAMPI_MPI_CHECK(MPI_Win_flush, (m_config.owner_rank, m_window)); }
  void flush_local() { DYNAMPI_MPI_CHECK(MPI_Win_flush_local, (m_config.owner_rank, m_window)); }

  // Size-1 levels have no MPI window; owner buffer is the sole storage.
  bool local_only() const { return m_window == MPI_WIN_NULL; }

  int64_t local_load_i64(MPI_Aint offset) const {
    return detail::read_i64(m_window_buffer.data(), m_window_buffer.size(),
                            static_cast<size_t>(offset));
  }
  void local_store_i64(MPI_Aint offset, int64_t value) {
    detail::write_i64(m_window_buffer.data(), m_window_buffer.size(), static_cast<size_t>(offset),
                      value);
  }

  void post_fetch_and_op(int64_t in, int64_t& out, MPI_Aint offset, MPI_Op op) {
    m_comm.fetch_and_op(in, out, m_config.owner_rank, offset, op, m_window);
  }
  void post_put_bytes(const void* src, size_t n, MPI_Aint offset) {
    m_comm.put_bytes(src, n, m_config.owner_rank, offset, m_window);
  }
  void post_get_bytes(void* dst, size_t n, MPI_Aint offset) {
    m_comm.get_bytes(dst, n, m_config.owner_rank, offset, m_window);
  }

  int64_t atomic_read(MPI_Aint offset) {
    if (local_only()) return local_load_i64(offset);
    int64_t in = 0, out;
    post_fetch_and_op(in, out, offset, MPI_NO_OP);
    flush_local();
    return out;
  }
  int64_t fetch_add(MPI_Aint offset, int64_t increment) {
    if (local_only()) {
      const int64_t out = local_load_i64(offset);
      local_store_i64(offset, out + increment);
      return out;
    }
    int64_t in = increment, out;
    post_fetch_and_op(in, out, offset, MPI_SUM);
    flush_local();
    return out;
  }

  void get_bytes_local(void* dst, size_t n, MPI_Aint offset) {
    if (local_only()) {
      detail::read_bytes(dst, n, m_window_buffer.data(), m_window_buffer.size(),
                         static_cast<size_t>(offset), n);
      return;
    }
    post_get_bytes(dst, n, offset);
    flush_local();
  }

  // Owner-side. Reads whatever error records appeared since the last call.
  void harvest_task_errors() {
    const int64_t claimed = std::min(atomic_read(ERROR_COUNT_OFF), kMaxRecordedErrors);
    if (claimed <= 0) return;  // LCOV_EXCL_LINE -- only reachable on a spurious flag
    // Takes ready slots anywhere in the table, not a contiguous frontier:
    // slots are claimed in fetch_add order but completed out of order, so an
    // in-flight slot is skipped now and picked up later. Safe because a
    // claimant flushes its record before the log entry that brings the
    // manager here.
    std::vector<std::byte> buf(static_cast<size_t>(claimed) * ERROR_SLOT_BYTES);
    get_bytes_local(buf.data(), buf.size(), error_slot(0));
    for (int64_t i = 0; i < claimed; ++i) {
      if (m_errors_seen[static_cast<size_t>(i)]) continue;
      const size_t off = static_cast<size_t>(i) * ERROR_SLOT_BYTES;
      const int64_t ready = detail::read_i64(buf.data(), buf.size(), off + E_RANK);
      if (ready == 0) continue;  // still in flight; its own flag will bring us back
      TaskError error;
      error.worker_rank = static_cast<int>(ready - 1);
      const char* text = reinterpret_cast<const char*>(buf.data() + off + E_DATA);
      error.message.assign(text, ::strnlen(text, kMaxTaskErrorMessage - 1));
      m_owner_errors.push_back(std::move(error));
      m_errors_seen[static_cast<size_t>(i)] = true;
    }
  }

  std::vector<TaskT> read_task_batch(int64_t index, int64_t count) {
    const size_t bytes = static_cast<size_t>(count) * m_task_slot_stride;
    std::vector<std::byte> buf(bytes);
    get_bytes_local(buf.data(), bytes, task_slot(index));

    std::vector<TaskT> tasks;
    tasks.reserve(static_cast<size_t>(count));
    for (int64_t i = 0; i < count; ++i) {
      const size_t slot_offset = static_cast<size_t>(i) * m_task_slot_stride;
      const int64_t elem_count = detail::read_i64(buf.data(), buf.size(), slot_offset + T_COUNT);
      TaskT task{};
      if constexpr (MPI_Type<TaskT>::resize_required)
        MPI_Type<TaskT>::resize(task, static_cast<int>(elem_count));
      const size_t data_bytes = static_cast<size_t>(elem_count) * m_task_elem;
      detail::read_result_bytes(buf.data(), buf.size(), slot_offset + T_DATA, task, data_bytes);
      tasks.push_back(std::move(task));
    }
    return tasks;
  }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// HierarchicalLockFreeRMAWorkDistributor
//
// Combines HierarchicalWorkDistributor's node-aware tree topology
// (root manager <-> per-node managers <-> per-node local workers) with
// LockFreeRMAWorkDistributor's one-sided, collective-free RMA
// protocol (fetch-and-add claiming, batched Put-based result return via a
// completion log), applied independently at each level of the tree.
//
// Motivation: the flat class is collective-free but funnels every claim,
// publish and harvest through one manager-owned window, which plateaued
// around 2.1-2.26M tasks/s from 32 nodes on, while the two-sided tree kept
// climbing to 5.2M/s at 128 nodes. One window per level gets both: no
// collectives and no single-window ceiling.
//
// A node manager is owner of its local level and claimant of the leader
// level at once. It never computes a claim itself -- it republishes it into
// its local level and relays results up once confirmed. Local harvesting
// returns whatever contiguous prefix is ready, which can span several
// leader-level claims or partly cover one, so the manager keeps a FIFO of
// {leader_start, local_start, local_end} relay entries (m_pending_relays)
// and slices harvested batches against those boundaries. See
// run_node_manager().
//
// Results ARE ordered: every level's completion log positionally maps a
// result back to the index its claim was assigned, and a level's harvest only
// ever advances past a contiguous prefix of confirmed entries (see
// LockFreeRMALevel::harvest_ready_results()) -- true recursively at each hop
// of the relay chain, so the final vector at the root manager matches
// submission order. Same cost as any ordered distributor: one slow task holds
// back every result behind it, all the way up the tree. Task prioritization
// and detailed statistics are not supported.
// ---------------------------------------------------------------------------
template <typename TaskT, typename ResultT, typename... Options>
class HierarchicalLockFreeRMAWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    int max_tasks = 8192;        // leader-level task table capacity (lifetime total)
    int max_local_tasks = 8192;  // per-node local task table capacity (lifetime total, per node)
    int max_task_count = 256;
    int max_result_count = 256;
    // 0 keeps one local group per shared-memory node. A positive value
    // partitions large nodes into smaller contiguous groups, reducing
    // contention on each local RMA window and making the upper hierarchy
    // useful on machines with many ranks per node.
    int max_local_group_size = 0;
    // <0 (default, "auto"): derive a fanout from the manager count -- see
    // setup_upper_chain() for the formula. 0: disabled, a two-level tree
    // where the root manager talks directly to every node manager. >0: caps
    // direct claimants per upper-level window, grouping node managers
    // recursively into intermediate levels once they exceed it.
    int max_upper_fanout = -1;

    // Caps how many rounds ahead (at the parent's own claim granularity) a
    // relay hop may claim from its parent before backing off, so the
    // pipeline stays fed without unbounded relay latency -- see the
    // backpressure comment at its one call site (step_bridge_hop) for the
    // regression this guards against. Exposed here (rather than left a
    // local constexpr) so it can be swept experimentally against
    // HierarchicalWorkDistributor::Config::pipeline_depth, which plays a
    // comparable role for the two-sided class.
    //
    // A 128-node/rpn=9 sweep (test_load_balancing, tasks_per_worker 0-20)
    // found 2 matched or beat every other tested value (1, 4, 8, 16, 32) at
    // nearly every batch size, by up to 18% in the middle of the range.
    // Values much above this regress for the same reason the cap exists at
    // all: claiming further ahead of a still-FIFO relay buffer adds
    // ordering latency, not useful slack.
    int max_pending_rounds = 2;

    // If true (default), run_tasks()/finish_remaining_tasks() throw
    // dynampi::TaskFailure on the root manager once a task has thrown. Set
    // false to recover instead: distribution runs to completion and the
    // failures are available from take_task_errors().
    bool rethrow_task_errors = true;
  };

  struct RunConfig {
    size_t target_num_tasks = std::numeric_limits<size_t>::max();
    bool allow_more_than_target_tasks = true;
    std::optional<double> max_seconds = std::nullopt;
  };

  static const bool ordered = true;

  explicit HierarchicalLockFreeRMAWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                                  Config config = {})
      : m_config(config),
        m_world_comm(config.comm, MPICommunicator<>::Duplicate),
        m_worker_function(std::move(worker_function)) {
    setup_topology();
    setup_levels();

    // setup_topology()/setup_levels() run MPI_Comm_split chains whose cost
    // grows with rank and level count, and the manager's constructor returns
    // as soon as its own finish. Without this barrier a caller that starts
    // timing there burns the window on ranks still in setup -- measured at
    // ~700-900ms of dead time with zero results relayed, flat across task
    // durations, so a one-time setup cost rather than a per-task one.
    DYNAMPI_MPI_CHECK(MPI_Barrier, (m_world_comm.get()));

    if (m_config.auto_run_workers && !is_root_manager()) run_worker();
  }

  // Tasks that threw, oldest first, removed as they are returned. See
  // Config::rethrow_task_errors.
  [[nodiscard]] std::vector<TaskError> take_task_errors() {
    assert(is_root_manager());
    return m_task_errors.take();
  }

  bool has_task_errors() const {
    assert(is_root_manager());
    return !m_task_errors.empty();
  }

  ~HierarchicalLockFreeRMAWorkDistributor() {
    if (!m_finalized) finalize();
    m_task_errors.warn_if_unreported("HierarchicalLockFreeRMAWorkDistributor");
    if (!m_solo) {
      // Level teardown is collective only over that level's communicator,
      // so without a world-wide rendezvous a rank in fewer levels can start
      // constructing the next distributor while managers are still freeing
      // subgroup windows -- MS-MPI can stall or abort on the overlap.
      // Destroy levels explicitly while m_world_comm is alive, keeping their
      // top-to-bottom order, then hold every rank until all windows are gone.
      m_parent_level.reset();
      m_owned_upper_levels.clear();
      m_local_level.reset();
      m_upper_comms.clear();
      m_local_comm.reset();
      DYNAMPI_MPI_CHECK(MPI_Barrier, (m_world_comm.get()));
    }
  }

  bool is_root_manager() const { return m_world_comm.rank() == m_config.manager_rank; }

  size_t remaining_tasks_count() const {
    assert(is_root_manager());
    if (m_solo) return m_local_task_store.size() - m_local_collected_count;
    const auto& top =
        m_owned_upper_levels.front();  // manager always owns exactly one top-of-chain level
    return top.owner_published_count() - top.owner_collected_count() - m_results.size();
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
      m_owned_upper_levels.front().publish_tasks(tasks);  // manager's top-of-chain level
    }
  }

  [[nodiscard]] std::vector<ResultT> run_tasks(const RunConfig& config = RunConfig{}) {
    assert(is_root_manager());
    Timer timer;

    if (m_solo) {
      while (m_local_collected_count < m_local_task_store.size()) {
        if (m_results.size() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        run_one_solo_task();
      }
    } else {
      auto& top = m_owned_upper_levels.front();  // manager's top-of-chain level
      while (true) {
        if (m_results.size() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        if (top.owner_collected_count() >= top.owner_published_count()) break;
        auto results = top.harvest_ready_results_throttled();
        collect_level_errors(top);
        m_results.insert(m_results.end(), std::make_move_iterator(results.begin()),
                         std::make_move_iterator(results.end()));
      }
    }

    // Thrown before draining, so results collected so far stay buffered for
    // whoever catches this and calls again.
    m_task_errors.rethrow_first_if(m_config.rethrow_task_errors);

    return drain_results(config.allow_more_than_target_tasks ? std::numeric_limits<size_t>::max()
                                                             : config.target_num_tasks);
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() { return run_tasks({}); }

  // One non-looping harvest snapshot -- see
  // LockFreeRMAWorkDistributor::gather_once() for the full
  // rationale (this mirrors it exactly, against the leader level instead
  // of a single flat window).
  [[nodiscard]] std::vector<ResultT> gather_once() {
    assert(is_root_manager());
    if (m_solo) {
      while (m_local_collected_count < m_local_task_store.size()) {
        run_one_solo_task();
      }
    } else {
      auto& top = m_owned_upper_levels.front();  // manager's top-of-chain level
      auto results = top.harvest_ready_results();
      collect_level_errors(top);
      m_results.insert(m_results.end(), std::make_move_iterator(results.begin()),
                       std::make_move_iterator(results.end()));
    }
    return drain_results(std::numeric_limits<size_t>::max());
  }

  void finalize() {
    assert(!m_finalized);
    if (is_root_manager()) {
      if (m_solo) {
        while (m_local_collected_count < m_local_task_store.size()) {
          run_one_solo_task();
        }
      } else {
        auto& top = m_owned_upper_levels.front();  // manager's top-of-chain level
        while (top.owner_collected_count() < top.owner_published_count()) {
          auto results = top.harvest_ready_results_throttled();
          collect_level_errors(top);
          m_results.insert(m_results.end(), std::make_move_iterator(results.begin()),
                           std::make_move_iterator(results.end()));
        }
        // Only the manager's own level needs an explicit mark_finished()
        // call: every level below it in the chain learns "finished" by
        // observing ITS OWN parent drain (see run_bridge_chain()'s
        // leader_drained handling), cascading down automatically.
        top.mark_finished();
      }
    }
    m_finalized = true;
  }

  void run_worker() {
    assert(!is_root_manager());
    const bool has_upper_chain = m_parent_level.has_value() || !m_owned_upper_levels.empty();
    if (m_local_level && m_local_comm->size() > 1) {
      if (has_upper_chain) {
        run_node_manager();
      } else {
        run_local_worker();
      }
    } else if (has_upper_chain) {
      run_leaf_leader_worker();
    }
  }

 private:
  Config m_config;
  MPICommunicator<> m_world_comm;
  std::function<ResultT(TaskT)> m_worker_function;
  const bool m_solo = (m_world_comm.size() == 1);

  std::optional<MPICommunicator<>> m_local_comm;
  std::optional<detail::LockFreeRMALevel<TaskT, ResultT>> m_local_level;

  // Upper hierarchy (everything above the node-local level) -- see
  // setup_upper_chain() for how this is built.
  //
  // m_upper_comms keeps every comm this rank uses up there alive for the
  // distributor's lifetime (LockFreeRMALevel only stores a raw MPI_Comm
  // handle, it doesn't own one).
  std::vector<MPICommunicator<>> m_upper_comms;

  // Levels this rank owns in the upper chain, top (nearest the manager) to
  // bottom. The manager always has exactly one; a promoted node manager has
  // one per level it leads, each also self-claimed; a non-promoted manager
  // or leaf leader-worker has none.
  //
  // std::deque, not vector: LockFreeRMALevel owns an MPI_Win exposing its own
  // member's address and is neither copyable nor movable, so growth must
  // never relocate existing elements.
  std::deque<detail::LockFreeRMALevel<TaskT, ResultT>> m_owned_upper_levels;

  // The one level this rank is a pure claimant of (never owns) -- its
  // immediate parent in the tree. Unset only for the manager (which owns
  // the top of the chain instead, in m_owned_upper_levels) and for a rank
  // with no upper-chain participation at all (a plain local worker).
  std::optional<detail::LockFreeRMALevel<TaskT, ResultT>> m_parent_level;

  bool m_finalized = false;
  detail::TaskErrorLog m_task_errors;  // root manager only
  std::vector<ResultT> m_results;
  size_t m_returned_count = 0;

  std::vector<TaskT> m_local_task_store;
  size_t m_local_collected_count = 0;

  void setup_topology() {
    if (m_solo) return;

    MPICommunicator<> node_comm = m_world_comm.split_by_node();
    std::optional<MPICommunicator<>> local_domain;
    if (m_config.max_local_group_size > 0 && node_comm.size() > m_config.max_local_group_size) {
      const int color = node_comm.rank() / m_config.max_local_group_size;
      auto partition = node_comm.split(color, node_comm.rank());
      local_domain.emplace(std::move(*partition));
    } else {
      local_domain.emplace(std::move(node_comm));
    }

    const int local_color = (m_world_comm.rank() == m_config.manager_rank) ? MPI_UNDEFINED : 0;
    auto local_opt = local_domain->split(local_color, m_world_comm.rank());
    if (local_opt.has_value()) m_local_comm.emplace(std::move(*local_opt));

    const bool is_manager = (m_world_comm.rank() == m_config.manager_rank);
    bool is_node_manager = false;
    if (m_local_comm.has_value()) is_node_manager = (m_local_comm->rank() == 0);

    const int leader_color = (is_manager || is_node_manager) ? 0 : MPI_UNDEFINED;
    auto leader_opt = m_world_comm.split(leader_color, m_world_comm.rank());
    if (!leader_opt.has_value()) return;  // plain local worker: no upper-chain role at all

    setup_upper_chain(std::move(*leader_opt), is_manager);
  }

  // Appends one LockFreeRMALevel this rank OWNS (and, per the class comment on
  // LockFreeRMALevel's claimant-side API, will also claim from -- its own
  // subtree still needs feeding) to m_owned_upper_levels.
  void emplace_owned_upper_level(MPICommunicator<> comm, int owner_rank, int claim_width) {
    m_upper_comms.push_back(std::move(comm));
    typename detail::LockFreeRMALevel<TaskT, ResultT>::Config cfg;
    cfg.comm = m_upper_comms.back().get();
    cfg.owner_rank = owner_rank;
    cfg.max_tasks = m_config.max_tasks;
    cfg.max_task_count = m_config.max_task_count;
    cfg.max_result_count = m_config.max_result_count;
    m_owned_upper_levels.emplace_back(cfg, claim_width);
  }

  // Sets m_parent_level: the one level this rank is a pure claimant of
  // (never owns) -- its immediate parent in the tree.
  void emplace_parent_level(MPICommunicator<> comm, int owner_rank, int claim_width) {
    m_upper_comms.push_back(std::move(comm));
    typename detail::LockFreeRMALevel<TaskT, ResultT>::Config cfg;
    cfg.comm = m_upper_comms.back().get();
    cfg.owner_rank = owner_rank;
    cfg.max_tasks = m_config.max_tasks;
    cfg.max_task_count = m_config.max_task_count;
    cfg.max_result_count = m_config.max_result_count;
    m_parent_level.emplace(cfg, claim_width);
  }

  // Builds this rank's view of the upper hierarchy (everything above the
  // node-local level): a chain of LockFreeRMALevel windows from the manager
  // down to this rank's immediate parent, filling in m_upper_comms (kept
  // alive for the LockFreeRMALevel objects' lifetime, since LockFreeRMALevel only
  // stores a raw MPI_Comm handle) / m_owned_upper_levels / m_parent_level.
  //
  // `flat_comm` is the root manager + every node manager. If grouping is
  // disabled or the manager count already fits under max_upper_fanout, that
  // comm IS the whole chain: one level, owned by the manager, claimed
  // directly by every node manager.
  //
  // Otherwise the node managers are grouped into a k-ary tree: repeatedly
  // split the current round into groups of at most max_upper_fanout, promote
  // each group's rank 0 to own that group's level, and feed the promoted set
  // into the next round, until a round fits under the manager. A promoted
  // rank both owns its group's level and claims from it, since its own
  // subtree still needs feeding -- safe because self-targeted RMA is
  // mechanically no different from any other claimant.
  //
  // Every MPI_Comm_split call below is collective over its *input* comm's
  // full membership; the control flow is written so every rank in that
  // membership reaches the matching call, even ranks that stop being
  // promoted early (see the inline notes at each split).
  void setup_upper_chain(MPICommunicator<> flat_comm, bool is_manager) {
    MPIGroup world_group(m_world_comm);
    const int local_children = m_local_comm.has_value() ? std::max(0, m_local_comm->size() - 1) : 0;
    // A leaf executes one task per claim. Each non-leaf claims enough tasks
    // to feed one task to every child in its immediate subtree.
    const int leaf_claim_width = std::max(1, local_children);

    const int manager_count = flat_comm.size() - 1;
    // Auto mode: stay flat below ~32 managers, where a sweep at 128 nodes
    // found grouped and flat indistinguishable. Above that use the smallest
    // power of 2 >= sqrt(manager_count). The same sweep measured branching
    // factor 8 at ~6x worse than 16 or 32, and 64 -- which concentrates all
    // traffic onto two leaders -- worse again.
    const int effective_fanout = [&] {
      if (m_config.max_upper_fanout < 0) {
        if (manager_count <= 32) return std::numeric_limits<int>::max();
        int fanout = 1;
        const double target = std::sqrt(static_cast<double>(manager_count));
        while (fanout < target) fanout *= 2;
        return fanout;
      }
      return m_config.max_upper_fanout > 0 ? m_config.max_upper_fanout
                                           : std::numeric_limits<int>::max();
    }();

    if (manager_count <= effective_fanout) {
      // Fits directly under the manager: exactly today's single flat level
      // (also always true when max_upper_fanout is disabled).
      MPIGroup flat_group(flat_comm);
      const int owner_rank = world_group.translate_rank(m_config.manager_rank, flat_group);
      if (flat_comm.rank() == owner_rank) {
        emplace_owned_upper_level(std::move(flat_comm), owner_rank, leaf_claim_width);
      } else {
        emplace_parent_level(std::move(flat_comm), owner_rank, leaf_claim_width);
      }
      return;
    }

    // Real grouping needed. Carve "managers only" out of flat_comm --
    // every member of flat_comm (manager included) calls this split
    // together, even though only managers use the result.
    auto managers_opt = flat_comm.split(is_manager ? MPI_UNDEFINED : 0, flat_comm.rank());

    bool is_final_round_leader = false;
    int feed_width = leaf_claim_width;
    if (!is_manager) {
      // std::optional, not a bare MPICommunicator: MPICommunicator has no
      // move assignment (only move construction -- see its deleted
      // operator=), so replacing round_comm each iteration needs
      // emplace()'s in-place construction rather than `round_comm = ...`.
      std::optional<MPICommunicator<>> round_comm(std::move(*managers_opt));
      while (true) {
        if (round_comm->size() <= effective_fanout) {
          // This round's membership (this rank included) already fits
          // directly under the manager -- stop promoting; round_comm itself
          // was only ever bookkeeping (the manager isn't part of it), so it
          // just falls out of scope here.
          is_final_round_leader = true;
          break;
        }
        const int color = round_comm->rank() / effective_fanout;
        auto group_opt = round_comm->split(color, round_comm->rank());
        MPICommunicator<> group_comm = std::move(*group_opt);
        const bool is_group_leader = (group_comm.rank() == 0);
        const int child_count = group_comm.size();

        // Collective over round_comm: every member (leader or not) calls
        // this together, before acting on their differing result below.
        auto leaders_opt =
            round_comm->split(is_group_leader ? 0 : MPI_UNDEFINED, round_comm->rank());

        if (!is_group_leader) {
          emplace_parent_level(std::move(group_comm), 0, feed_width);
          break;
        }
        emplace_owned_upper_level(std::move(group_comm), 0, feed_width);
        feed_width = std::max(1, child_count * feed_width);
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
      const int owner_rank = world_group.translate_rank(m_config.manager_rank, top_group);
      if (is_manager) {
        emplace_owned_upper_level(std::move(*top_opt), owner_rank, feed_width);
      } else {
        emplace_parent_level(std::move(*top_opt), owner_rank, feed_width);
      }
    }
  }

  void setup_levels() {
    // Size-1 local_comm means this rank is alone on its node (manager excluded
    // from the local split). Leaf leader-workers claim/compute against the
    // upper chain directly and never use a local_level, so skip constructing
    // one -- including avoiding a size-1 LockFreeRMALevel that some MPIs cannot
    // Win_create for.
    if (m_local_comm.has_value() && m_local_comm->size() > 1) {
      typename detail::LockFreeRMALevel<TaskT, ResultT>::Config local_cfg;
      local_cfg.comm = m_local_comm->get();
      local_cfg.owner_rank = 0;
      local_cfg.max_tasks = m_config.max_local_tasks;
      local_cfg.max_task_count = m_config.max_task_count;
      local_cfg.max_result_count = m_config.max_result_count;
      m_local_level.emplace(local_cfg);
    }
  }

  // One (parent, child) bridge: claim from `parent`, republish into `child`
  // (a different level this rank owns), relay confirmed results back up. A
  // rank promoted through several grouping rounds runs several of these
  // chained in one loop.
  //
  // Harvesting `child` can return a batch spanning several parent-level
  // claims, or part of one, so pending_relays + relay_buffer map child-side
  // result positions back to the parent range they belong to. Relays are
  // contiguous and gap-free by construction -- a claim is always republished
  // with its relay entry queued in the same step -- so a prefix of
  // relay_buffer always aligns with the front relay's boundary.
  struct PendingRelay {
    int64_t parent_start;
    int64_t child_len;
  };
  struct BridgeHop {
    BridgeHop(detail::LockFreeRMALevel<TaskT, ResultT>* p,
              detail::LockFreeRMALevel<TaskT, ResultT>* c)
        : parent(p), child(c) {}
    detail::LockFreeRMALevel<TaskT, ResultT>* parent;
    detail::LockFreeRMALevel<TaskT, ResultT>* child;
    std::deque<PendingRelay> pending_relays;
    std::vector<ResultT> relay_buffer;  // child results collected but not yet fully relayed upward
    // Set once a failure from below has been republished into `parent`'s error
    // table, and cleared by the next write_result_range() that carries the flag
    // up. The records are already in place by then; the flag only tells the
    // level above to go and read them.
    bool relay_error_pending = false;
    int64_t pending_task_count =
        0;  // sum of child_len across pending_relays -- see step_bridge_hop()
    bool finish_marked = false;
  };

  // One non-blocking step of a single hop. Returns whether it accomplished
  // anything against `parent`, which the caller uses per hop to decide
  // whether that hop's own parent-level polling backs off -- scoped per hop
  // and excluding child-harvest activity, because a bridging rank has no
  // compute time between parent polls and otherwise spins as fast as RMA
  // round trips allow.
  bool step_bridge_hop(BridgeHop& hop) {
    bool made_parent_progress = false;

    // Read once per call and reuse for both the claim-guard and (via the
    // caller's bridge_hop_done()) the exit check -- LockFreeRMALevel::drained()
    // is not a pure query (it does fresh RMA reads and can flip from false
    // to true between two calls made moments apart), so calling it
    // separately for those two decisions was a real, confirmed bug: if it
    // flipped true *between* them, the exit check could fire having never
    // gone through the "else" branch below at all -- the only place that
    // calls hop.child->mark_finished(). That could let this rank exit (and
    // reach its destructor's teardown barrier) without ever telling its
    // child's claimants to stop, hanging them forever. Reusing one cached
    // value and gating exit on hop.finish_marked itself (not a re-derived
    // drained() call) makes "mark_finished() happens before this hop can
    // possibly be done" true by construction, not by hoping two nearby
    // calls agree.
    const bool parent_drained = hop.parent->drained();

    // 1. Claim from parent and republish into child, queueing a relay entry
    // recording where this batch's results must eventually be written back.
    //
    // Gated by backpressure. Because pending_relays flushes strictly FIFO,
    // claiming far ahead adds ordering latency rather than useful slack: a
    // caught-up relay at the back cannot flush until everything ahead of it
    // does. Uncapped, a manager claimed every iteration the parent wasn't
    // drained and raced far past what its child could drain -- one observed
    // manager held 670 relay entries (~9,300 tasks) after a second. Capping
    // at a few rounds of the parent's own claim granularity keeps the
    // pipeline fed without unbounded relay latency.
    const int64_t max_pending_rounds = m_config.max_pending_rounds;
    const int64_t pending_cap =
        static_cast<int64_t>(hop.parent->claim_width()) * max_pending_rounds;
    if (!parent_drained && hop.pending_task_count < pending_cap) {
      auto claimed = hop.parent->try_claim();
      if (claimed.start != -1) {
        const int64_t child_len = static_cast<int64_t>(claimed.tasks.size());
        hop.child->publish_tasks(claimed.tasks);
        hop.pending_relays.push_back(PendingRelay{claimed.start, child_len});
        hop.pending_task_count += child_len;
        made_parent_progress = true;
      }
    } else if (parent_drained && !hop.finish_marked) {
      // Parent is fully drained: nothing more will ever be published to
      // child either (every task this rank will ever see from parent has
      // already been claimed and republished above).
      hop.child->mark_finished();
      hop.finish_marked = true;
      made_parent_progress = true;
    }

    // 2. Harvest whatever's ready in child and append to the relay buffer.
    // Deliberately NOT counted towards made_parent_progress: this is
    // same-owner-rank RMA against child (this rank's own window, no
    // upstream contention), and it succeeds almost every iteration once
    // child's claimants are active -- counting it here is exactly what
    // previously masked parent-level polling from ever backing off.
    auto child_results = hop.child->harvest_ready_results();
    // Failures travel with the results they belong to: harvest_ready_results()
    // has already read the child's error table by the time it returns a range
    // containing one, so republishing here keeps a failure exactly one hop
    // behind its own placeholder result all the way up to the root.
    for (auto& error : hop.child->take_errors()) {
      hop.parent->report_task_error(error);
      hop.relay_error_pending = true;
    }
    if (!child_results.empty()) {
      hop.relay_buffer.insert(hop.relay_buffer.end(),
                              std::make_move_iterator(child_results.begin()),
                              std::make_move_iterator(child_results.end()));
    }

    // 3. Flush whatever contiguous prefix of the front relay is ready now as
    // its own partial write_result_range(), rather than waiting for the whole
    // claimed batch. All-or-nothing flushing was a severe bottleneck: with
    // claim_width scaled to feed every local child at once, nothing relayed
    // upward until the slowest of ~100 tasks finished, costing 5-10x once
    // task duration passed ~1ms.
    //
    // Order across entries is still strict FIFO -- the parent harvests by
    // contiguous-prefix scan, so relay N+1 cannot be confirmed before N is
    // fully covered. Only covering N in a single write is relaxed: successive
    // partial calls are indistinguishable to that scan from one whole-range
    // call.
    size_t consumed = 0;
    while (!hop.pending_relays.empty() && consumed < hop.relay_buffer.size()) {
      auto& front = hop.pending_relays.front();
      const size_t available = hop.relay_buffer.size() - consumed;
      const size_t chunk = std::min(available, static_cast<size_t>(front.child_len));
      std::vector<ResultT> slice(
          std::make_move_iterator(hop.relay_buffer.begin() + static_cast<ptrdiff_t>(consumed)),
          std::make_move_iterator(hop.relay_buffer.begin() +
                                  static_cast<ptrdiff_t>(consumed + chunk)));
      hop.parent->write_result_range(front.parent_start, slice, hop.relay_error_pending);
      hop.relay_error_pending = false;
      consumed += chunk;
      hop.pending_task_count -= static_cast<int64_t>(chunk);
      front.parent_start += static_cast<int64_t>(chunk);
      front.child_len -= static_cast<int64_t>(chunk);
      made_parent_progress = true;
      if (front.child_len == 0) hop.pending_relays.pop_front();
    }
    if (consumed > 0) {
      hop.relay_buffer.erase(hop.relay_buffer.begin(),
                             hop.relay_buffer.begin() + static_cast<ptrdiff_t>(consumed));
    }
    return made_parent_progress;
  }

  // A hop is done once child's claimants have been told to stop
  // (finish_marked, not a re-derived parent->drained() -- see
  // step_bridge_hop()'s comment) and every claimed task has been relayed
  // back upward.
  static bool bridge_hop_done(const BridgeHop& hop) {
    return hop.finish_marked && hop.pending_relays.empty();
  }

  // Builds the chain of bridge hops this rank participates in, from its
  // immediate parent down to (but not including) `terminal_extra` (pass
  // m_local_level for a node manager with real local peers, or nullptr
  // for a leaf leader-worker, which claims+computes directly against the
  // last upper level instead -- see run_leaf_leader_worker()). Most ranks
  // (never promoted -- see setup_upper_chain()) get exactly one hop here,
  // identical in shape to this class's original single-level design.
  std::vector<BridgeHop> build_upper_hops() {
    std::vector<detail::LockFreeRMALevel<TaskT, ResultT>*> chain;
    chain.push_back(&*m_parent_level);
    for (auto& level : m_owned_upper_levels) chain.push_back(&level);
    std::vector<BridgeHop> hops;
    for (size_t i = 0; i + 1 < chain.size(); ++i) hops.push_back(BridgeHop{chain[i], chain[i + 1]});
    return hops;
  }

  // The bottom-most level in this rank's own upper chain: its last owned
  // group level if it was promoted at least once, otherwise its immediate
  // parent. This is what a terminal action (feeding a real local_level in
  // run_node_manager(), or claiming+computing directly in
  // run_leaf_leader_worker()) attaches to.
  detail::LockFreeRMALevel<TaskT, ResultT>& last_upper_level() {
    return m_owned_upper_levels.empty() ? *m_parent_level : m_owned_upper_levels.back();
  }

  // A node manager: bridges its parent level (claiming tasks
  // from the manager or an intermediate group level -- see
  // setup_upper_chain()) down to its local level (its own window, published
  // to for its local workers to claim from), and, if promoted to lead one
  // or more groups, one or more intermediate hops in between. Unlike a
  // plain worker, this rank never computes a task itself -- see
  // step_bridge_hop() and the class comment for the full rationale.
  void run_node_manager() {
    std::vector<BridgeHop> hops = build_upper_hops();
    hops.push_back(BridgeHop{&last_upper_level(), &*m_local_level});

    while (true) {
      bool any_progress = false;
      bool all_done = true;
      for (auto& hop : hops) {
        if (step_bridge_hop(hop)) any_progress = true;
        if (!bridge_hop_done(hop)) all_done = false;
      }
      if (all_done) break;

      // See step_bridge_hop()'s comment: only back off when NONE of this
      // rank's hops made progress. Drive progress on the terminal child,
      // where local workers' result Puts are in flight. This distinction is
      // required by MS-MPI: probing only the upper parent communicator can
      // leave the local window's completion traffic stalled indefinitely.
      if (!any_progress) hops.back().child->idle_wait();
    }
  }

  // A plain local worker: claims tasks from its node manager, computes
  // them inline, writes results back -- the same claim/write pattern
  // LockFreeRMAWorkDistributor::run_worker() uses against its one
  // flat window, here against the local level instead.
  void run_local_worker() {
    while (true) {
      // Cache once per iteration -- see step_bridge_hop()'s comment on why
      // calling LockFreeRMALevel::drained() separately for the claim guard and
      // the exit check is a real bug class (it does fresh RMA reads and is
      // not guaranteed to agree with itself moments apart), not just a
      // style preference.
      const bool is_drained = m_local_level->drained();
      if (!is_drained) {
        auto claimed = m_local_level->try_claim();
        if (claimed.start != -1) {
          std::vector<ResultT> results;
          results.reserve(claimed.tasks.size());
          bool failed = false;
          for (auto& task : claimed.tasks) {
            ResultT result;
            auto failure = detail::run_task_guarded(m_worker_function, std::move(task), result);
            if (failure) {
              m_local_level->report_task_error(TaskError{m_world_comm.rank(), std::move(*failure)});
              failed = true;
            }
            results.push_back(std::move(result));
          }
          m_local_level->write_result_range(claimed.start, results, failed);
          continue;
        }
      }
      if (is_drained) break;
      m_local_level->idle_wait();
    }
  }

  // A leaf leader-worker (no local peers -- alone on its node): bridges
  // whatever upper hops this rank is promoted through (usually none), then
  // claims and executes tasks directly against the bottom of its own chain
  // (its parent level, or its own last owned group level if promoted),
  // instead of publishing further down to a local_level it doesn't have.
  void run_leaf_leader_worker() {
    std::vector<BridgeHop> hops = build_upper_hops();
    detail::LockFreeRMALevel<TaskT, ResultT>& terminal = last_upper_level();

    while (true) {
      bool any_progress = false;
      for (auto& hop : hops) {
        if (step_bridge_hop(hop)) any_progress = true;
      }

      const bool is_drained = terminal.drained();
      if (!is_drained) {
        auto claimed = terminal.try_claim();
        if (claimed.start != -1) {
          std::vector<ResultT> results;
          results.reserve(claimed.tasks.size());
          bool failed = false;
          for (auto& task : claimed.tasks) {
            ResultT result;
            auto failure = detail::run_task_guarded(m_worker_function, std::move(task), result);
            if (failure) {
              terminal.report_task_error(TaskError{m_world_comm.rank(), std::move(*failure)});
              failed = true;
            }
            results.push_back(std::move(result));
          }
          terminal.write_result_range(claimed.start, results, failed);
          any_progress = true;
        }
      }

      bool all_hops_done = true;
      for (auto& hop : hops) {
        if (!bridge_hop_done(hop)) {
          all_hops_done = false;
          break;
        }
      }
      if (all_hops_done && is_drained) break;

      if (!any_progress) terminal.idle_wait();
    }
  }

  // Solo (size-1 communicator) execution, guarded like every other path so a
  // workload behaves the same when debugged without MPI.
  void run_one_solo_task() {
    ResultT result;
    auto failure = detail::run_task_guarded(m_worker_function,
                                            m_local_task_store[m_local_collected_count], result);
    if (failure) m_task_errors.record(TaskError{m_world_comm.rank(), std::move(*failure)});
    m_results.push_back(std::move(result));
    m_local_collected_count++;
  }

  void collect_level_errors(detail::LockFreeRMALevel<TaskT, ResultT>& level) {
    for (auto& error : level.take_errors()) m_task_errors.record(std::move(error));
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
