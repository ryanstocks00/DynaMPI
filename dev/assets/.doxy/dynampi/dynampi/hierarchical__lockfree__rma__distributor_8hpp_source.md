

# File hierarchical\_lockfree\_rma\_distributor.hpp

[**File List**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**impl**](dir_87365955bfd9c1869b83a1fbd3fdb221.md) **>** [**hierarchical\_lockfree\_rma\_distributor.hpp**](hierarchical__lockfree__rma__distributor_8hpp.md)

[Go to the documentation of this file](hierarchical__lockfree__rma__distributor_8hpp.md)


```C++
/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <thread>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_group.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/hierarchical_topology_detail.hpp"
#include "dynampi/impl/rma_detail.hpp"
#include "dynampi/mpi/mpi_error.hpp"
#include "dynampi/task_error.hpp"
#include "dynampi/utilities/timer.hpp"

namespace dynampi {

namespace detail {

// One composable level of fetch-and-add claims, task/result storage, and
// completion logging. Its non-blocking steps let managers relay between levels.
template <typename TaskT, typename ResultT>
class LockFreeRMALevel {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_NULL;
    int owner_rank = 0;
    // Ring capacity; harvested slots are reused modulo max_tasks.
    int max_tasks = 8192;
    int max_task_count = 256;
    int max_result_count = 256;
  };

  struct ClaimedRange {
    int64_t start = -1;  // Nothing currently claimable; drained() determines final exhaustion.
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
  int comm_size() const { return m_comm.size(); }
  int claim_width() const { return m_claim_width; }

  void idle_wait() {
    if (m_window == MPI_WIN_NULL) {
      // Singleton levels have no MPI window.
      std::this_thread::yield();
      return;
    }
    detail::rma_wait_idle(m_window, m_comm.get());
  }

  // Capacity must be checked before an irreversible parent claim.
  int64_t owner_available_capacity() const {
    assert(is_owner());
    const int64_t in_flight = m_total_tasks - static_cast<int64_t>(m_owner_collected_count);
    return static_cast<int64_t>(m_config.max_tasks) - in_flight;
  }

  // Publishes all tasks or returns false if ring space is temporarily
  // insufficient. A batch larger than the ring is an error.
  bool publish_tasks(const std::vector<TaskT>& tasks) {
    assert(is_owner());
    if (tasks.empty()) return true;
    const int64_t count = static_cast<int64_t>(tasks.size());
    detail::check_task_capacity(0, tasks.size(), m_config.max_tasks, "LockFreeRMALevel");
    if (count > owner_available_capacity()) return false;

    const int64_t start = m_total_tasks;
    std::vector<std::byte> buffer(tasks.size() * m_task_slot_stride);
    for (size_t i = 0; i < tasks.size(); ++i) {
      const TaskT& task = tasks[i];
      const int elem_count = MPI_Type<TaskT>::count(task);
      assert(static_cast<size_t>(elem_count) <= m_max_task_count &&
             "LockFreeRMALevel: task exceeds max_task_count");
      const size_t data_bytes = static_cast<size_t>(elem_count) * m_task_elem;
      const size_t off = i * m_task_slot_stride;
      detail::write_i64(buffer.data(), buffer.size(), off + T_COUNT, elem_count);
      if (data_bytes > 0) {
        detail::write_bytes(buffer.data(), buffer.size(), off + T_DATA, MPI_Type<TaskT>::ptr(task),
                            data_bytes);
      }
    }
    put_ring_bytes(buffer.data(), start, count, m_task_slot_stride,
                   [this](int64_t i) { return task_slot(i); });
    // Clear stale completion entries before reusing ring slots.
    std::vector<std::byte> log_clear(static_cast<size_t>(count) * LOG_ENTRY_BYTES, std::byte{0});
    put_ring_bytes(log_clear.data(), start, count, LOG_ENTRY_BYTES,
                   [this](int64_t i) { return log_slot(i); });

    if (local_only()) {
      m_total_tasks += count;
      local_store_i64(TOTAL_OFF, m_total_tasks);
      return true;
    }
    // Land the payload and log-clear Puts before the counter that makes them
    // claimable moves; see the same barrier in LockFreeRMAWorkDistributor.
    // The local_only path above needs none: it writes the window directly.
    flush_remote();
    m_total_tasks += count;
    int64_t total_out = 0;
    post_fetch_and_op(m_total_tasks, total_out, TOTAL_OFF, MPI_REPLACE);
    flush_remote();
    return true;
  }

  void mark_finished() {
    assert(is_owner());
    if (local_only()) {
      local_store_i64(FINISHED_OFF, 1);
    } else {
      int64_t finished_out = 0;
      post_fetch_and_op(static_cast<int64_t>(1), finished_out, FINISHED_OFF, MPI_REPLACE);
      flush_remote();
    }
    idle_wait();
  }

  size_t owner_published_count() const {
    assert(is_owner());
    return static_cast<size_t>(m_total_tasks);
  }
  size_t owner_collected_count() const {
    assert(is_owner());
    return m_owner_collected_count;
  }

  // Completion estimate not blocked by lower-index gaps. It is only a pacing
  // hint; ordered collection controls result access and ring reuse.
  int64_t owner_unordered_completed_estimate() const {
    assert(is_owner());
    return m_owner_unordered_completed;
  }

  // Returns the newly completed contiguous result prefix.
  std::vector<ResultT> harvest_ready_results() {
    assert(is_owner());
    const int64_t head_now = atomic_read(HEAD_OFF);
    // Claims may outrun publication; never scan reused slots beyond this
    // owner's currently published range.
    const int64_t head_now_capped = std::min(head_now, m_total_tasks);

    const int64_t frontier = static_cast<int64_t>(m_owner_collected_count);
    if (head_now_capped <= frontier) return {};

    const int64_t scan_count = head_now_capped - frontier;
    std::vector<std::byte> log_buf(static_cast<size_t>(scan_count) * LOG_ENTRY_BYTES);
    get_ring_bytes_local(log_buf.data(), frontier, scan_count, LOG_ENTRY_BYTES,
                         [this](int64_t i) { return log_slot(i); });

    int64_t confirmed_end = frontier;
    bool saw_error = false;
    while (confirmed_end < head_now_capped) {
      const size_t off = static_cast<size_t>(confirmed_end - frontier) * LOG_ENTRY_BYTES;
      const int64_t entry = detail::read_i64(log_buf.data(), log_buf.size(), off);
      if (entry == 0) break;  // gap: not yet written; stop the contiguous prefix here
      if (entry < 0) saw_error = true;
      confirmed_end += entry < 0 ? -entry : entry;
    }
    // A negative run length indicates at least one task error.
    if (saw_error) harvest_task_errors();

    // Continue past ordering gaps to estimate total completed work.
    int64_t unordered_total = confirmed_end - frontier;
    for (int64_t pos = confirmed_end; pos < head_now_capped;) {
      const size_t off = static_cast<size_t>(pos - frontier) * LOG_ENTRY_BYTES;
      const int64_t entry = detail::read_i64(log_buf.data(), log_buf.size(), off);
      if (entry == 0) {
        ++pos;
        continue;
      }
      const int64_t run = entry < 0 ? -entry : entry;
      unordered_total += run;
      pos += run;
    }
    m_owner_unordered_completed = frontier + unordered_total;

    if (confirmed_end <= frontier) return {};

    const int64_t n = confirmed_end - frontier;
    std::vector<std::byte> result_buf(static_cast<size_t>(n) * m_result_slot_stride);
    get_ring_bytes_local(result_buf.data(), frontier, n, m_result_slot_stride,
                         [this](int64_t i) { return result_slot(i); });

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

  // Harvest once, yielding if no result became available.
  std::vector<ResultT> harvest_ready_results_throttled() {
    const size_t before = owner_collected_count();
    auto results = harvest_ready_results();
    if (owner_collected_count() == before) idle_wait();
    return results;
  }

  // Attempts one non-blocking claim. Owners may self-claim when they also
  // lead a child subtree. start=-1 means no work is currently available.
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

    // A stale total must remain refreshable after FINISHED is observed.
    if (m_cached_head >= m_cached_total && !m_seen_finished)
      m_cached_total = atomic_read(TOTAL_OFF);

    if (m_cached_head < m_cached_total) {
      const int64_t claim = std::min<int64_t>(m_claim_width, m_cached_total - m_cached_head);
      const int64_t start = fetch_add(HEAD_OFF, claim);
      const int64_t end = start + claim;
      m_cached_head = end;

      const int64_t total = (end <= m_cached_total) ? m_cached_total : atomic_read(TOTAL_OFF);
      // Publication may trail the entire claim, so clamp to its owned range.
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

  // Refreshes FINISHED without charging every try_claim() for the RMA read.
  bool check_finished() {
    if (!m_seen_finished && atomic_read(FINISHED_OFF) != 0) m_seen_finished = true;
    return m_seen_finished;
  }

  // True when FINISHED is observed and no published or pending work remains.
  bool drained() {
    // Refresh before the pending check so an unpublished remainder can expire.
    check_finished();
    if (m_pending_start != -1) return false;
    if (!m_seen_finished) return false;
    const int64_t final_total = atomic_read(TOTAL_OFF);
    if (m_cached_head >= final_total) return true;
    m_cached_total = final_total;  // real unclaimed work remains; try_claim() will pick it up
    return false;
  }

  // Publishes a worker failure before its result range advertises completion.
  void report_task_error(const TaskError& error) {
    const int64_t slot = fetch_add(ERROR_COUNT_OFF, 1);
    if (slot >= kMaxRecordedErrors) return;
    // Publish payload before readiness; rank+1 preserves zero as the sentinel.
    std::vector<std::byte> message_bytes(kMaxTaskErrorMessage, std::byte{0});
    const size_t bytes = std::min(error.message.size(), kMaxTaskErrorMessage - 1);
    if (bytes > 0) {
      detail::write_bytes(message_bytes.data(), message_bytes.size(), 0, error.message.data(),
                          bytes);
    }
    const int64_t ready = static_cast<int64_t>(error.worker_rank) + 1;
    if (local_only()) {
      // LCOV_EXCL_START -- singleton owned upper group
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

  std::vector<TaskError> take_errors() {
    std::vector<TaskError> taken;
    taken.swap(m_owner_errors);
    return taken;
  }

  // Writes result data before publishing its completion-log entry.
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
    // One log entry covers the range; only result data can cross a ring lap.
    put_ring_bytes(buffer.data(), start, count, m_result_slot_stride,
                   [this](int64_t i) { return result_slot(i); });
    if (local_only()) {
      local_store_i64(log_slot(start), entry);
      return;
    }
    flush_remote();
    post_put_bytes(&entry, sizeof(entry), log_slot(start));
    flush_remote();
  }

 private:
  // Window layout: controls, task ring, result ring, completion log, errors.
  static constexpr MPI_Aint HEAD_OFF = 0;
  static constexpr MPI_Aint TOTAL_OFF = 8;
  static constexpr MPI_Aint FINISHED_OFF = 16;
  static constexpr MPI_Aint ERROR_COUNT_OFF = 24;
  static constexpr size_t CONTROL_BYTES = 32;

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
  size_t m_owner_collected_count = 0;
  int64_t m_owner_unordered_completed = 0;

  // Claimant-only state (see try_claim()/drained()).
  int64_t m_cached_head = 0;
  int64_t m_cached_total = 0;
  int64_t m_pending_start = -1;
  int64_t m_pending_end = -1;
  bool m_seen_finished = false;
  std::vector<bool> m_errors_seen;        // owner-side: error slots already consumed
  std::vector<TaskError> m_owner_errors;  // owner-side, drained by take_errors()

  void initialize_window() {
    check_fixed_size_mpi_type<TaskT>("task", "LockFreeRMALevel");
    check_fixed_size_mpi_type<ResultT>("result", "LockFreeRMALevel");

    m_task_elem = static_cast<size_t>(detail::mpi_type_size_bytes<TaskT>());
    m_result_elem = static_cast<size_t>(detail::mpi_type_size_bytes<ResultT>());
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

    // Some MPI implementations reject windows on singleton communicators.
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

  // publish_tasks() prevents live indices from colliding modulo max_tasks.
  int64_t ring_slot(int64_t index) const {
    return index % static_cast<int64_t>(m_config.max_tasks);
  }
  MPI_Aint task_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_task_base +
                                 static_cast<size_t>(ring_slot(index)) * m_task_slot_stride);
  }
  MPI_Aint result_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_result_base +
                                 static_cast<size_t>(ring_slot(index)) * m_result_slot_stride);
  }
  MPI_Aint log_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_log_base +
                                 static_cast<size_t>(ring_slot(index)) * LOG_ENTRY_BYTES);
  }
  // Error slots are fixed and not ring-indexed.
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

  // Issues one local or remote chunk; callers flush after all chunks.
  void put_bytes_owner(const void* src, size_t n, MPI_Aint offset) {
    if (local_only()) {
      detail::write_bytes(m_window_buffer.data(), m_window_buffer.size(),
                          static_cast<size_t>(offset), src, n);
      return;
    }
    post_put_bytes(src, n, offset);
  }

  // Splits a write when its logical range crosses the ring boundary.
  template <typename SlotFn>
  void put_ring_bytes(const void* src, int64_t start, int64_t count, size_t stride,
                      SlotFn&& slot_fn) {
    const int64_t ring = static_cast<int64_t>(m_config.max_tasks);
    const int64_t first_count = std::min(count, ring - ring_slot(start));
    const auto* bytes = static_cast<const std::byte*>(src);
    put_bytes_owner(bytes, static_cast<size_t>(first_count) * stride, slot_fn(start));
    if (first_count < count) {
      put_bytes_owner(bytes + static_cast<size_t>(first_count) * stride,
                      static_cast<size_t>(count - first_count) * stride,
                      slot_fn(start + first_count));
    }
  }

  // Read counterpart of put_ring_bytes().
  template <typename SlotFn>
  void get_ring_bytes_local(void* dst, int64_t start, int64_t count, size_t stride,
                            SlotFn&& slot_fn) {
    const int64_t ring = static_cast<int64_t>(m_config.max_tasks);
    const int64_t first_count = std::min(count, ring - ring_slot(start));
    get_bytes_local(dst, static_cast<size_t>(first_count) * stride, slot_fn(start));
    if (first_count < count) {
      get_bytes_local(static_cast<std::byte*>(dst) + static_cast<size_t>(first_count) * stride,
                      static_cast<size_t>(count - first_count) * stride,
                      slot_fn(start + first_count));
    }
  }

  void harvest_task_errors() {
    const int64_t claimed = std::min(atomic_read(ERROR_COUNT_OFF), kMaxRecordedErrors);
    if (claimed <= 0) return;  // LCOV_EXCL_LINE -- only reachable on a spurious flag
    // Read ready slots independently because writers complete out of order.
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
    get_ring_bytes_local(buf.data(), index, count, m_task_slot_stride,
                         [this](int64_t i) { return task_slot(i); });

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

// Node-aware tree of one-sided RMA levels. Managers relay claimed tasks down
// and contiguous result prefixes up. Ordering is preserved across every level;
// task prioritization and detailed statistics are unsupported.
template <typename TaskT, typename ResultT, typename... Options>
class HierarchicalLockFreeRMAWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    int max_tasks = 8192;        // Upper-level ring capacity
    int max_local_tasks = 8192;  // Per-node ring capacity
    int max_task_count = 256;
    int max_result_count = 256;
    // 0 keeps one local group per node; positive values partition it into
    // contiguous groups of at most this size.
    int max_local_group_size = 0;
    // <0 selects fanout automatically, 0 keeps a flat upper layer, and >0
    // recursively caps direct upper-level claimants.
    int max_upper_fanout = -1;

    // Parent claim rounds allowed ahead of child completion.
    int max_pending_rounds = 2;

    // Rethrow worker failures at the root; otherwise collect them for
    // take_task_errors().
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

    // Ensure every rank finishes communicator and window setup before use.
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
      // Destroy every subgroup window before any rank starts another instance.
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
    const auto& top = top_level();
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
      return;
    }
    auto& top = top_level();
    // This blocking API harvests until enough ring space is available.
    while (!top.publish_tasks(tasks)) {
      harvest_top_level(true);
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
      auto& top = top_level();
      while (true) {
        if (m_results.size() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        if (top.owner_collected_count() >= top.owner_published_count()) break;
        harvest_top_level(true);
      }
    }

    // Thrown before draining, so results collected so far stay buffered for
    // whoever catches this and calls again.
    m_task_errors.rethrow_first_if(m_config.rethrow_task_errors);

    return drain_results(config.allow_more_than_target_tasks ? std::numeric_limits<size_t>::max()
                                                             : config.target_num_tasks);
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() { return run_tasks({}); }

  // One non-looping harvest snapshot.
  [[nodiscard]] std::vector<ResultT> gather_once() {
    assert(is_root_manager());
    if (m_solo) {
      while (m_local_collected_count < m_local_task_store.size()) {
        run_one_solo_task();
      }
    } else {
      harvest_top_level(false);
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
        auto& top = top_level();
        while (top.owner_collected_count() < top.owner_published_count()) {
          harvest_top_level(true);
        }
        // Each lower level is marked finished when its own parent drains.
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
  using RMALevel = detail::LockFreeRMALevel<TaskT, ResultT>;

  Config m_config;
  MPICommunicator<> m_world_comm;
  std::function<ResultT(TaskT)> m_worker_function;
  const bool m_solo = (m_world_comm.size() == 1);

  std::optional<MPICommunicator<>> m_local_comm;
  std::optional<detail::LockFreeRMALevel<TaskT, ResultT>> m_local_level;

  // Owns communicators referenced by upper-level RMA objects.
  std::vector<MPICommunicator<>> m_upper_comms;

  // Top-to-bottom levels owned by this rank. A deque keeps non-movable RMA
  // levels at stable addresses.
  std::deque<detail::LockFreeRMALevel<TaskT, ResultT>> m_owned_upper_levels;

  // Immediate parent level, claimed but not owned by this rank.
  std::optional<detail::LockFreeRMALevel<TaskT, ResultT>> m_parent_level;

  bool m_finalized = false;
  detail::TaskErrorLog m_task_errors;  // root manager only
  std::vector<ResultT> m_results;

  std::vector<TaskT> m_local_task_store;
  size_t m_local_collected_count = 0;

  RMALevel& top_level() {
    assert(!m_owned_upper_levels.empty());
    return m_owned_upper_levels.front();
  }

  const RMALevel& top_level() const {
    assert(!m_owned_upper_levels.empty());
    return m_owned_upper_levels.front();
  }

  void harvest_top_level(bool throttled) {
    auto& top = top_level();
    auto results = throttled ? top.harvest_ready_results_throttled() : top.harvest_ready_results();
    collect_level_errors(top);
    m_results.insert(m_results.end(), std::make_move_iterator(results.begin()),
                     std::make_move_iterator(results.end()));
  }

  void setup_topology() {
    if (m_solo) return;

    auto local_opt = detail::split_local_worker_communicator(m_world_comm, m_config.manager_rank,
                                                             m_config.max_local_group_size);
    if (local_opt.has_value()) m_local_comm.emplace(std::move(*local_opt));

    const bool is_manager = (m_world_comm.rank() == m_config.manager_rank);
    bool is_node_manager = false;
    if (m_local_comm.has_value()) is_node_manager = (m_local_comm->rank() == 0);

    const int leader_color = (is_manager || is_node_manager) ? 0 : MPI_UNDEFINED;
    auto leader_opt = m_world_comm.split(leader_color, m_world_comm.rank());
    if (!leader_opt.has_value()) return;  // plain local worker: no upper-chain role at all

    setup_upper_chain(std::move(*leader_opt), is_manager);
  }

  enum class UpperLevelRole { Owned, Parent };

  void emplace_upper_level(MPICommunicator<> comm, int owner_rank, int claim_width,
                           UpperLevelRole role) {
    m_upper_comms.push_back(std::move(comm));
    typename detail::LockFreeRMALevel<TaskT, ResultT>::Config cfg;
    cfg.comm = m_upper_comms.back().get();
    cfg.owner_rank = owner_rank;
    cfg.max_tasks = m_config.max_tasks;
    cfg.max_task_count = m_config.max_task_count;
    cfg.max_result_count = m_config.max_result_count;
    if (role == UpperLevelRole::Owned) {
      m_owned_upper_levels.emplace_back(cfg, claim_width);
    } else {
      m_parent_level.emplace(cfg, claim_width);
    }
  }

  // Builds the k-ary upper RMA chain. Every rank in each input communicator
  // must reach its matching MPI_Comm_split, including ranks not promoted.
  void setup_upper_chain(MPICommunicator<> flat_comm, bool is_manager) {
    MPIGroup world_group(m_world_comm);
    const int local_children = m_local_comm.has_value() ? std::max(0, m_local_comm->size() - 1) : 0;
    // A claim must fit the child ring into which it is republished.
    const int leaf_claim_width = std::clamp(local_children, 1, m_config.max_local_tasks);

    const int manager_count = flat_comm.size() - 1;
    const int effective_fanout =
        detail::resolve_upper_fanout(manager_count, m_config.max_upper_fanout);

    if (manager_count <= effective_fanout) {
      MPIGroup flat_group(flat_comm);
      const int owner_rank = world_group.translate_rank(m_config.manager_rank, flat_group);
      if (flat_comm.rank() == owner_rank) {
        emplace_upper_level(std::move(flat_comm), owner_rank, leaf_claim_width,
                            UpperLevelRole::Owned);
      } else {
        emplace_upper_level(std::move(flat_comm), owner_rank, leaf_claim_width,
                            UpperLevelRole::Parent);
      }
      return;
    }

    // Collective over flat_comm; only node managers join the result.
    auto managers_opt = flat_comm.split(is_manager ? MPI_UNDEFINED : 0, flat_comm.rank());

    bool is_final_round_leader = false;
    int feed_width = leaf_claim_width;
    if (!is_manager) {
      // optional::emplace works around MPICommunicator's deleted move assignment.
      std::optional<MPICommunicator<>> round_comm(std::move(*managers_opt));
      while (true) {
        if (round_comm->size() <= effective_fanout) {
          is_final_round_leader = true;
          break;
        }
        const int color = round_comm->rank() / effective_fanout;
        auto group_opt = round_comm->split(color, round_comm->rank());
        MPICommunicator<> group_comm = std::move(*group_opt);
        const bool is_group_leader = (group_comm.rank() == 0);
        const int group_feed_width =
            detail::sum_subtree_widths_to_group_leader(feed_width, group_comm);

        auto leaders_opt =
            round_comm->split(is_group_leader ? 0 : MPI_UNDEFINED, round_comm->rank());

        if (!is_group_leader) {
          emplace_upper_level(std::move(group_comm), 0, feed_width, UpperLevelRole::Parent);
          break;
        }
        emplace_upper_level(std::move(group_comm), 0, feed_width, UpperLevelRole::Owned);
        // Upper levels share max_tasks as their ring capacity.
        feed_width = std::clamp(group_feed_width, 1, m_config.max_tasks);
        round_comm.emplace(std::move(*leaders_opt));
      }
    }

    // Collective over the original flat_comm attaches final leaders to root.
    auto top_opt = flat_comm.split((is_manager || is_final_round_leader) ? 0 : MPI_UNDEFINED,
                                   flat_comm.rank());
    if (top_opt.has_value()) {
      MPIGroup top_group(*top_opt);
      const int owner_rank = world_group.translate_rank(m_config.manager_rank, top_group);
      if (is_manager) {
        emplace_upper_level(std::move(*top_opt), owner_rank, feed_width, UpperLevelRole::Owned);
      } else {
        emplace_upper_level(std::move(*top_opt), owner_rank, feed_width, UpperLevelRole::Parent);
      }
    }
  }

  void setup_levels() {
    // Singleton local groups execute directly against the upper chain.
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

  // Maps child result positions back to their claimed parent ranges.
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
    std::vector<ResultT> relay_buffer;
    // Tells the next parent write to read already-published error records.
    bool relay_error_pending = false;
    int64_t pending_task_count = 0;
    int64_t total_claimed = 0;
    bool finish_marked = false;
    // Relay layers from this hop's child down to the leaf workers, inclusive.
    // Each buffers a pipelined round of its own -- see step_bridge_hop().
    int layers = 1;
  };

  // Numbers a completed hop chain bottom-up: the hop feeding the local worker
  // level is layer 1, its feeder layer 2, and so on.
  static void assign_hop_layers(std::vector<BridgeHop>& hops) {
    for (size_t i = 0; i < hops.size(); ++i) {
      hops[i].layers = static_cast<int>(hops.size() - i);
    }
  }

  // One non-blocking relay step. Progress excludes local child harvesting so
  // callers still back off parent polling.
  bool step_bridge_hop(BridgeHop& hop) {
    bool made_parent_progress = false;

    // Cache this RMA query so marking the child finished cannot be skipped by
    // a false-to-true transition between checks.
    const bool parent_drained = hop.parent->drained();

    // Pace claims by unordered child completion so an ordering gap does not
    // starve the subtree. Separately cap the unrelayed backlog.
    // One claimed round in flight plus one pipelined round for every relay
    // layer beneath this hop, so a leaf worker is never committed to more
    // than tree-depth tasks. A flat max_pending_rounds instead gives the
    // upper hops the same budget as the bottom one, where it is already
    // spoken for by the layers below -- the two-sided hierarchy measured
    // 2.97ms per round against 1.00ms once layers were budgeted for.
    const int64_t max_pending_rounds = m_config.max_pending_rounds;
    const int64_t pending_cap = static_cast<int64_t>(hop.parent->claim_width()) *
                                (1 + (max_pending_rounds - 1) * hop.layers);
    const int64_t pending_cap_hard = pending_cap * 4;
    const int64_t uncompleted_claimed =
        hop.total_claimed - hop.child->owner_unordered_completed_estimate();
    const bool claim_paced_ok = uncompleted_claimed < pending_cap;
    const bool backlog_safe = hop.pending_task_count < pending_cap_hard;
    // Parent claims cannot be returned, so reserve child capacity first.
    const bool child_has_room =
        hop.child->owner_available_capacity() >= static_cast<int64_t>(hop.parent->claim_width());
    if (!parent_drained && claim_paced_ok && backlog_safe && child_has_room) {
      auto claimed = hop.parent->try_claim();
      if (claimed.start != -1) {
        const int64_t child_len = static_cast<int64_t>(claimed.tasks.size());
        const bool published = hop.child->publish_tasks(claimed.tasks);
        // Safe because this rank is the child's only publisher.
        assert(published && "LockFreeRMALevel: child ring capacity check raced");
        (void)published;
        hop.pending_relays.push_back(PendingRelay{claimed.start, child_len});
        hop.pending_task_count += child_len;
        hop.total_claimed += child_len;
        made_parent_progress = true;
      }
    } else if (parent_drained && !hop.finish_marked) {
      hop.child->mark_finished();
      hop.finish_marked = true;
      made_parent_progress = true;
    }

    // Child harvesting does not count as parent progress.
    auto child_results = hop.child->harvest_ready_results();
    // Forward errors alongside the harvested range that exposed them.
    for (auto& error : hop.child->take_errors()) {
      hop.parent->report_task_error(error);
      hop.relay_error_pending = true;
    }
    if (!child_results.empty()) {
      hop.relay_buffer.insert(hop.relay_buffer.end(),
                              std::make_move_iterator(child_results.begin()),
                              std::make_move_iterator(child_results.end()));
    }

    // Flush partial prefixes of the oldest relay while preserving FIFO order.
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

  // Completion requires both child shutdown and a fully drained relay.
  static bool bridge_hop_done(const BridgeHop& hop) {
    return hop.finish_marked && hop.pending_relays.empty();
  }

  // Builds this rank's upper relay chain.
  std::vector<BridgeHop> build_upper_hops() {
    std::vector<RMALevel*> chain;
    chain.push_back(&*m_parent_level);
    for (auto& level : m_owned_upper_levels) chain.push_back(&level);
    std::vector<BridgeHop> hops;
    for (size_t i = 0; i + 1 < chain.size(); ++i) hops.push_back(BridgeHop{chain[i], chain[i + 1]});
    return hops;
  }

  RMALevel& last_upper_level() {
    return m_owned_upper_levels.empty() ? *m_parent_level : m_owned_upper_levels.back();
  }

  static bool hops_done(const std::vector<BridgeHop>& hops) {
    for (const auto& hop : hops) {
      if (!bridge_hop_done(hop)) return false;
    }
    return true;
  }

  // Relay the upper chain into the local worker level.
  void run_node_manager() {
    std::vector<BridgeHop> hops = build_upper_hops();
    hops.push_back(BridgeHop{&last_upper_level(), &*m_local_level});
    assign_hop_layers(hops);

    while (true) {
      bool any_progress = false;
      for (auto& hop : hops) {
        if (step_bridge_hop(hop)) any_progress = true;
      }
      if (hops_done(hops)) break;
      if (!any_progress) hops.back().child->idle_wait();
    }
  }

  bool claim_and_execute(RMALevel& level) {
    auto claimed = level.try_claim();
    if (claimed.start == -1) return false;

    std::vector<ResultT> results;
    results.reserve(claimed.tasks.size());
    bool failed = false;
    for (auto& task : claimed.tasks) {
      ResultT result;
      auto failure = detail::run_task_guarded(m_worker_function, std::move(task), result);
      if (failure) {
        level.report_task_error(TaskError{m_world_comm.rank(), std::move(*failure)});
        failed = true;
      }
      results.push_back(std::move(result));
    }
    level.write_result_range(claimed.start, results, failed);
    return true;
  }

  // Claim from the local level and compute inline.
  void run_local_worker() {
    while (true) {
      // Cache this RMA query for a consistent claim and exit decision.
      const bool is_drained = m_local_level->drained();
      if (!is_drained && claim_and_execute(*m_local_level)) continue;
      if (is_drained) break;
      m_local_level->idle_wait();
    }
  }

  // Bridge promoted levels, then compute directly against the terminal one.
  void run_leaf_leader_worker() {
    std::vector<BridgeHop> hops = build_upper_hops();
    assign_hop_layers(hops);
    RMALevel& terminal = last_upper_level();

    while (true) {
      bool any_progress = false;
      for (auto& hop : hops) {
        if (step_bridge_hop(hop)) any_progress = true;
      }

      const bool is_drained = terminal.drained();
      if (!is_drained && claim_and_execute(terminal)) any_progress = true;
      if (hops_done(hops) && is_drained) break;
      if (!any_progress) terminal.idle_wait();
    }
  }

  // Size-one communicator execution.
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
    auto end = m_results.begin() + static_cast<ptrdiff_t>(count);
    std::move(m_results.begin(), end, std::back_inserter(output));
    m_results.erase(m_results.begin(), end);
    return output;
  }
};

}  // namespace dynampi
```


