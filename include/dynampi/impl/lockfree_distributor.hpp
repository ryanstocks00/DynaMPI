/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/base_distributor.hpp"
#include "dynampi/mpi/mpi_error.hpp"
#include "dynampi/utilities/timer.hpp"

namespace dynampi {

namespace detail {

// Byte size of a single element of the MPI datatype backing T (e.g. 4 for int,
// 4 for the element type of std::vector<int>).
template <typename T>
inline int mpi_type_size_bytes() {
  int size = 0;
  DYNAMPI_MPI_CHECK(MPI_Type_size, (MPI_Type<T>::value, &size));
  return size;
}

inline constexpr size_t round_up_8(size_t bytes) { return (bytes + 7) & ~static_cast<size_t>(7); }

inline void write_bytes(std::byte* buffer, size_t buffer_size, size_t offset, const void* src,
                        size_t nbytes) {
  if (nbytes == 0) return;
  // Runtime range gate (not assert-only): GCC 14 -Wstringop-overflow treats an
  // unconstrained size_t length as possibly near SIZE_MAX and false-positives
  // on fortified memcpy into buffer+offset under -Werror. Cap against
  // ptrdiff_t max so the length is proven below the "maximum object size".
  constexpr size_t kMaxObjectSize = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (nbytes > kMaxObjectSize || offset > buffer_size || nbytes > buffer_size - offset) {
    DYNAMPI_FAIL("write_bytes out of range");  // LCOV_EXCL_LINE
  }
  std::memcpy(buffer + offset, src, nbytes);
}

inline int64_t read_i64(const std::byte* buffer, [[maybe_unused]] size_t buffer_size,
                        size_t offset) {
  assert(offset + sizeof(int64_t) <= buffer_size);
  int64_t value{};
  std::memcpy(&value, buffer + offset, sizeof(int64_t));
  return value;
}

inline void write_i64(std::byte* buffer, size_t buffer_size, size_t offset, int64_t value) {
  write_bytes(buffer, buffer_size, offset, &value, sizeof(int64_t));
}

template <typename T>
inline void read_result_bytes(const std::byte* buffer, size_t buffer_size, size_t offset, T& value,
                              size_t data_bytes) {
  if (data_bytes == 0) return;
  constexpr size_t kMaxObjectSize = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (data_bytes > kMaxObjectSize || offset > buffer_size || data_bytes > buffer_size - offset) {
    DYNAMPI_FAIL("read_result_bytes out of range");  // LCOV_EXCL_LINE
  }
  if constexpr (MPI_Type<T>::resize_required) {
    // cppcheck-suppress invalidPointerCast
    std::memcpy(MPI_Type<T>::ptr(value), buffer + offset, data_bytes);
  } else {
    assert(data_bytes == sizeof(T));
    std::memcpy(&value, buffer + offset, sizeof(T));
  }
}

// Passive-target RMA on MS-MPI needs explicit flush progress while spinning.
inline void rma_wait_idle(MPI_Win window) {
#if defined(_WIN32)
  if (window != MPI_WIN_NULL) {
    DYNAMPI_MPI_CHECK(MPI_Win_flush_all, (window));
  }
  std::this_thread::yield();
#else
  (void)window;
  std::this_thread::sleep_for(std::chrono::microseconds(50));
#endif
}

}  // namespace detail

// ---------------------------------------------------------------------------
// MinimalLockFreeMPIWorkDistributor
//
// The simplest possible lock-free distributor: a parallel-for over the index
// range [0, n_tasks). The task *is* its global index, and every rank pulls the
// next index by atomically incrementing a single shared counter in the
// manager's RMA window. Results are gathered once at the end.
//
// This is genuinely lock-free (no manager bottleneck, one shared atomic) and
// deliberately tiny. Use it when the work is an embarrassingly parallel loop
// and the task payload is just the loop index. For arbitrary task payloads,
// priorities, incremental result collection or statistics, use
// LockFreeMPIWorkDistributor (or one of the message-based distributors).
//
// Usage (collective: every rank must call run() with the same n_tasks):
//   MinimalLockFreeMPIWorkDistributor<double> dist([](size_t i){ return f(i); });
//   std::vector<double> results = dist.run(n);  // populated on the manager only
// ---------------------------------------------------------------------------
template <typename ResultT>
class MinimalLockFreeMPIWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
  };

  explicit MinimalLockFreeMPIWorkDistributor(std::function<ResultT(size_t)> worker_function,
                                             Config config = {})
      : m_config(config),
        m_comm(config.comm, MPICommunicator<>::Duplicate),
        m_worker_function(std::move(worker_function)) {
    if (m_comm.size() == 1) {
      // Manager-only: no workers to share the claim counter with.
      return;
    }

    void* base = is_root_manager() ? static_cast<void*>(&m_counter) : m_worker_window;
    MPI_Aint size = static_cast<MPI_Aint>(sizeof(int64_t));
    DYNAMPI_MPI_CHECK(MPI_Win_create, (base, size, 1, MPI_INFO_NULL, m_comm.get(), &m_window));
    DYNAMPI_MPI_CHECK(MPI_Win_lock_all, (MPI_MODE_NOCHECK, m_window));
  }

  ~MinimalLockFreeMPIWorkDistributor() {
    if (m_window != MPI_WIN_NULL) {
      DYNAMPI_MPI_CHECK(MPI_Win_unlock_all, (m_window));
      DYNAMPI_MPI_CHECK(MPI_Win_free, (&m_window));
      m_window = MPI_WIN_NULL;
    }
  }

  bool is_root_manager() const { return m_comm.rank() == m_config.manager_rank; }

  // Collective. Every rank must call with the same n_tasks. Returns the results
  // ordered by task index on the manager, and an empty vector on workers.
  [[nodiscard]] std::vector<ResultT> run(size_t n_tasks) {
    unsigned long long n = n_tasks;
    DYNAMPI_MPI_CHECK(MPI_Bcast,
                      (&n, 1, MPI_UNSIGNED_LONG_LONG, m_config.manager_rank, m_comm.get()));

    if (m_comm.size() == 1) {
      assert(is_root_manager());
      std::vector<ResultT> results;
      results.reserve(static_cast<size_t>(n));
      for (unsigned long long i = 0; i < n; ++i) {
        results.push_back(m_worker_function(static_cast<size_t>(i)));
      }
      return results;
    }

    if (is_root_manager()) set_counter(0);
    DYNAMPI_MPI_CHECK(MPI_Barrier, (m_comm.get()));  // reset visible + synchronized start

    std::vector<std::pair<int64_t, ResultT>> local;
    while (true) {
      int64_t index = fetch_add(1);
      if (index >= static_cast<int64_t>(n)) break;
      local.emplace_back(index, m_worker_function(static_cast<size_t>(index)));
    }

    return gather_sorted(local);
  }

 private:
  Config m_config;
  MPICommunicator<> m_comm;
  std::function<ResultT(size_t)> m_worker_function;
  MPI_Win m_window = MPI_WIN_NULL;
  int64_t m_counter = 0;  // window-exposed claim counter (manager only)
  alignas(int64_t) std::byte m_worker_window[sizeof(int64_t)]{};

  void set_counter(int64_t value) {
    int64_t in = value, out;
    m_comm.fetch_and_op(in, out, m_config.manager_rank, 0, MPI_REPLACE, m_window);
    DYNAMPI_MPI_CHECK(MPI_Win_flush, (m_config.manager_rank, m_window));
  }

  int64_t fetch_add(int64_t increment) {
    int64_t in = increment, out;
    m_comm.fetch_and_op(in, out, m_config.manager_rank, 0, MPI_SUM, m_window);
    DYNAMPI_MPI_CHECK(MPI_Win_flush, (m_config.manager_rank, m_window));
    return out;
  }

  std::vector<ResultT> gather_sorted(std::vector<std::pair<int64_t, ResultT>>& local) {
    const int elem = detail::mpi_type_size_bytes<ResultT>();
    const bool manager = is_root_manager();
    const int size = m_comm.size();

    // Pack: per result [int64 index][int64 count][count * elem bytes].
    std::vector<std::byte> send_buf;
    for (auto& [index, result] : local) {
      const int count = MPI_Type<ResultT>::count(result);
      assert(count >= 0);
      const size_t data_bytes =
          count > 0 ? static_cast<size_t>(count) * static_cast<size_t>(elem) : size_t{0};
      const size_t offset = send_buf.size();
      send_buf.resize(offset + 16 + data_bytes);
      detail::write_i64(send_buf.data(), send_buf.size(), offset, index);
      detail::write_i64(send_buf.data(), send_buf.size(), offset + 8, count);
      if (data_bytes > 0) {
        detail::write_bytes(send_buf.data(), send_buf.size(), offset + 16,
                            MPI_Type<ResultT>::ptr(result), data_bytes);
      }
    }

    const int send_count = static_cast<int>(send_buf.size());
    std::vector<int> byte_counts(manager ? static_cast<size_t>(size) : 0);
    DYNAMPI_MPI_CHECK(MPI_Gather, (&send_count, 1, MPI_INT, manager ? byte_counts.data() : nullptr,
                                   1, MPI_INT, m_config.manager_rank, m_comm.get()));

    std::vector<int> displacements;
    std::vector<std::byte> recv_buf;
    int total_bytes = 0;
    if (manager) {
      displacements.resize(static_cast<size_t>(size));
      for (int r = 0; r < size; ++r) {
        displacements[static_cast<size_t>(r)] = total_bytes;
        total_bytes += byte_counts[static_cast<size_t>(r)];
      }
      recv_buf.resize(static_cast<size_t>(total_bytes));
    }

    DYNAMPI_MPI_CHECK(
        MPI_Gatherv,
        (send_buf.data(), send_count, MPI_BYTE, manager ? recv_buf.data() : nullptr,
         manager ? byte_counts.data() : nullptr, manager ? displacements.data() : nullptr, MPI_BYTE,
         m_config.manager_rank, m_comm.get()));

    std::vector<ResultT> output;
    if (!manager) return output;

    std::vector<std::pair<int64_t, ResultT>> all;
    size_t pos = 0;
    while (pos < static_cast<size_t>(total_bytes)) {
      assert(pos + 16 <= static_cast<size_t>(total_bytes));
      const int64_t index = detail::read_i64(recv_buf.data(), recv_buf.size(), pos);
      const int64_t count = detail::read_i64(recv_buf.data(), recv_buf.size(), pos + 8);
      pos += 16;
      ResultT result{};
      if constexpr (MPI_Type<ResultT>::resize_required)
        MPI_Type<ResultT>::resize(result, static_cast<int>(count));
      assert(count >= 0);
      const size_t data_bytes =
          count > 0 ? static_cast<size_t>(count) * static_cast<size_t>(elem) : size_t{0};
      detail::read_result_bytes(recv_buf.data(), recv_buf.size(), pos, result, data_bytes);
      pos += data_bytes;
      all.emplace_back(index, std::move(result));
    }
    std::sort(all.begin(), all.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    output.reserve(all.size());
    for (auto& [index, result] : all) output.push_back(std::move(result));
    return output;
  }
};

// ---------------------------------------------------------------------------
// LockFreeMPIWorkDistributor
//
// Task dispatch uses passive one-sided RMA on the manager's window (no
// MPI_Send/MPI_Recv on the hot path). Workers claim tasks via compare-and-swap,
// read task payloads from that window, execute, and buffer results locally.
// The manager collects via MPI_Gatherv rounds (same packing as
// MinimalLockFreeMPIWorkDistributor), triggered when workers report progress.
//
// Microsoft MPI always uses MPI_WIN_SEPARATE: a rank cannot observe remote RMA
// updates to its own window. Task dispatch stays on the manager window; results
// never use RMA Put/Get on the manager's own memory.
// ---------------------------------------------------------------------------
template <typename TaskT, typename ResultT, typename... Options>
class LockFreeMPIWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    int max_tasks = 8192;        // capacity of the task/result tables (lifetime total)
    int max_task_count = 256;    // max elements per task (only for resizable TaskT)
    int max_result_count = 256;  // max elements per result (only for resizable ResultT)
    int claim_batch_size = 8;    // tasks claimed per RMA round trip (see run_worker())
  };

  struct RunConfig {
    size_t target_num_tasks = std::numeric_limits<size_t>::max();
    bool allow_more_than_target_tasks = true;
    std::optional<double> max_seconds = std::nullopt;
  };

  static const bool ordered = true;

 private:
  static constexpr bool prioritize_tasks = get_option_value<prioritize_tasks_t, Options...>();
  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();
  using Comm = dynampi::MPICommunicator<track_statistics<statistics_mode>>;

  // Manager window: [head][total][finished][gather_seq] then task slots.
  static constexpr MPI_Aint HEAD_OFF = 0;
  static constexpr MPI_Aint TOTAL_OFF = 8;
  static constexpr MPI_Aint FINISHED_OFF = 16;
  static constexpr MPI_Aint GATHER_SEQ_OFF = 24;
  static constexpr size_t CONTROL_BYTES = 32;

  // Manager per-task slot: [int64 count][data].
  static constexpr size_t T_COUNT = 0;
  static constexpr size_t T_DATA = 8;

 public:
  struct Statistics {
    const CommStatistics& comm_statistics;
    std::vector<size_t> worker_task_counts;
  };
  using StatisticsT =
      std::conditional_t<statistics_mode == StatisticsMode::Detailed, Statistics, std::monostate>;

  explicit LockFreeMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                      Config config = {})
      : m_config(config),
        m_comm(config.comm, Comm::Duplicate),
        m_worker_function(std::move(worker_function)),
        m_statistics{make_statistics(m_comm)} {
    initialize_window();

    if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
      if (is_root_manager())
        m_statistics.worker_task_counts.assign(static_cast<size_t>(m_comm.size()), 0);
    }

    if (m_config.auto_run_workers && !is_root_manager()) run_worker();
  }

  ~LockFreeMPIWorkDistributor() {
    if (!m_finalized) finalize();
    if (m_window != MPI_WIN_NULL) {
      DYNAMPI_MPI_CHECK(MPI_Win_unlock_all, (m_window));
      MPI_Barrier(m_comm.get());
      DYNAMPI_MPI_CHECK(MPI_Win_free, (&m_window));
      m_window = MPI_WIN_NULL;
    }
  }

  // --- Public API ---

  [[nodiscard]] std::vector<ResultT> run_tasks(RunConfig config = {}) {
    assert(is_root_manager());
    Timer timer;

    if (num_workers() == 0) {
      // No workers: the manager runs everything itself.
      while (m_collected_count < static_cast<size_t>(m_total_tasks)) {
        if (available() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        run_one_task_locally();
      }
    } else {
      while (true) {
        if (available() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        if (m_collected_count >= static_cast<size_t>(m_total_tasks)) break;

        try_gather_results();
      }
    }

    size_t limit = config.allow_more_than_target_tasks ? std::numeric_limits<size_t>::max()
                                                       : config.target_num_tasks;
    return drain_results(limit);
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() { return run_tasks({}); }

  // Requests exactly one gather round and returns whatever results have been
  // staged by workers so far -- possibly none, possibly all outstanding
  // tasks, depending on how far workers have gotten. Unlike run_tasks() /
  // finish_remaining_tasks(), this does not loop retrying until every
  // outstanding task is collected: each call to those pays for a
  // MPI_Barrier + MPI_Gather/MPI_Gatherv across the whole communicator on
  // *every* internal retry (see try_gather_results()), which at high rank
  // counts can dominate over actual task execution time for fine-grained
  // tasks. Call this instead when the caller wants to control synchronization
  // frequency directly -- e.g. insert a large batch, let workers churn
  // uninterrupted for a while, then take a single snapshot with one call
  // here rather than one call per outstanding task.
  [[nodiscard]] std::vector<ResultT> gather_once() {
    assert(is_root_manager());
    if (num_workers() == 0) {
      while (m_collected_count < static_cast<size_t>(m_total_tasks)) run_one_task_locally();
    } else {
      request_gather();
    }
    return drain_results(std::numeric_limits<size_t>::max());
  }

  void finalize() {
    assert(!m_finalized);
    if (is_root_manager()) {
      if (num_workers() == 0) {
        while (m_collected_count < static_cast<size_t>(m_total_tasks)) run_one_task_locally();
      } else {
        while (m_collected_count < static_cast<size_t>(m_total_tasks)) {
          try_gather_results();
        }
        atomic_set(FINISHED_OFF, 1);  // tell workers to stop
        detail::rma_wait_idle(m_window);
      }
    }
    m_finalized = true;
  }

  bool is_root_manager() const { return m_comm.rank() == m_config.manager_rank; }

  size_t remaining_tasks_count() const {
    assert(is_root_manager());
    return static_cast<size_t>(m_total_tasks) - m_returned_count;
  }

  const StatisticsT& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    assert(is_root_manager());
    return m_statistics;
  }

  void insert_task(TaskT task)
    requires(!prioritize_tasks)
  {
    assert(is_root_manager());
    publish_task(task);
  }

  // LockFree does not support prioritisation; the priority is ignored.
  void insert_task(const TaskT& task, double)
    requires(prioritize_tasks)
  {
    assert(is_root_manager());
    publish_task(task);
  }

  void insert_tasks(const std::vector<TaskT>& tasks)
    requires(!prioritize_tasks)
  {
    assert(is_root_manager());
    for (const auto& task : tasks) publish_task(task);
  }

  void run_worker() {
    assert(!is_root_manager());

    // Fetch-and-add claiming instead of compare-and-swap: FAA can't lose a
    // race (there's nothing to compare against), so under contention it
    // wastes none of the throughput CAS gives up to failed attempts --
    // rma_atomic_microbench measured CAS success rates as low as 3%
    // (32-way contention) to under 1% (128-way), while FAA always succeeds.
    //
    // Unlike compare_and_swap, FAA can't be told "only add up to what's
    // published" -- it always grants exactly what you ask for, with no way
    // to verify that against TOTAL_OFF atomically. The naive fix (always
    // request claim_batch_size, unconditionally) was tried and made things
    // *worse*, not better: every worker instantly races HEAD far ahead of
    // TOTAL regardless of real publish rate, so nearly every claim lands in
    // the slow wait-and-poll path instead of the fast immediate-process
    // path -- the CAS version's habit of bounding its request to
    // min(claim_batch_size, total-head) *before* claiming was quietly doing
    // most of the work, not the compare-and-swap itself. So this still
    // bounds the requested size the same way CAS did (cached_head/
    // cached_total, refreshed lazily) -- the difference is only that FAA
    // can't *verify* cached_head is still accurate the way CAS's compare
    // does, so a claim can occasionally still straddle the published
    // boundary if another worker advanced HEAD since our last observation
    // of it. That's rare and small (bounded by how stale cached_head is),
    // not the systematic full-batch overshoot of the unconditional version.
    //
    // pending_start/pending_end track the unresolved remainder of such a
    // straddling claim across iterations. The one subtlety that isn't
    // optional: a worker with a pending tail MUST still process whatever
    // *prefix* of its claim is already valid immediately, rather than
    // blocking until the whole range resolves. finalize() can't set
    // FINISHED_OFF until it has collected every result -- including this
    // worker's already-ready prefix -- so blocking on the unready remainder
    // before staging that prefix would deadlock: the manager waiting on
    // results this worker refuses to compute until FINISHED, which the
    // manager won't set until it has them. Processing the ready prefix on
    // the spot (both in the fresh-claim branch and the pending-tail branch
    // below) avoids that entirely -- only the genuinely-not-yet-published
    // remainder ever waits.
    int64_t cached_head = 0;
    int64_t cached_total = 0;
    int64_t pending_start = -1;
    int64_t pending_end = -1;
    bool finished_seen = false;

    auto process_range = [&](int64_t start, int64_t count) {
      std::vector<TaskT> tasks = read_task_batch(start, count);
      for (int64_t i = 0; i < count; ++i) {
        ResultT result = m_worker_function(std::move(tasks[static_cast<size_t>(i)]));
        store_result(start + i, std::move(result));
      }
    };

    while (true) {
      maybe_participate_in_gather();
      bool made_progress = false;

      if (pending_start != -1) {
        const int64_t total = atomic_read(TOTAL_OFF);
        const int64_t ready = std::min(pending_end, total);
        if (ready > pending_start) {
          process_range(pending_start, ready - pending_start);
          pending_start = ready;
          made_progress = true;
        }
        if (pending_start >= pending_end) {
          pending_start = -1;  // fully resolved
        } else if (finished_seen) {
          pending_start =
              -1;  // total is final and short of pending_end: that remainder never existed
        }
      } else {
        // Deliberately NOT gated on !finished_seen: finished_seen only means
        // "TOTAL_OFF won't grow anymore", not "stop claiming what's already
        // known to be available". Gating this whole branch on !finished_seen
        // meant that once finished_seen went true, nothing could ever claim
        // again -- a worker sitting on a real, never-claimed gap between
        // cached_head and a just-learned final_total (see the exit check
        // below, which discovers and records that gap into cached_total)
        // would spin forever re-discovering it with no way to act on it.
        // Confirmed via gdb backtraces on a caught hang in the sibling
        // AsyncPutLockFreeMPIWorkDistributor, which shares this exact
        // claim-loop structure: 15/16 ranks parked in the destructor's
        // teardown barrier while the 16th sat in an idle-wait loop forever.
        if (cached_head >= cached_total && !finished_seen) cached_total = atomic_read(TOTAL_OFF);

        if (cached_head < cached_total) {
          const int64_t claim =
              std::min<int64_t>(m_config.claim_batch_size, cached_total - cached_head);
          const int64_t start = fetch_add(HEAD_OFF, claim);
          const int64_t end = start + claim;
          cached_head =
              end;  // best local estimate; may be stale vs. concurrent claimants, see above

          // Usually ready == end (cached_total was accurate); only re-reads
          // TOTAL_OFF when start turned out higher than expected (another
          // worker claimed since our last observation), which is the rare
          // case this whole scheme is built to tolerate rather than prevent.
          const int64_t total =
              (start + claim <= cached_total) ? cached_total : atomic_read(TOTAL_OFF);
          // Clamped into [start, end], not just capped at end: total can be
          // less than start when cached_total was stale enough that this
          // whole claim landed past what's published (same bug AsyncPut
          // documents in run_worker()).
          const int64_t ready = std::clamp(total, start, end);
          if (ready > start) {
            process_range(start, ready - start);
            made_progress = true;
          }
          if (ready < end) {
            pending_start = ready;
            pending_end = end;
          }
        }
      }

      if (!finished_seen && atomic_read(FINISHED_OFF) != 0) finished_seen = true;

      if (pending_start == -1 && finished_seen) {
        // cached_total can be stale relative to when TOTAL_OFF actually
        // stopped growing (publishing finishes well before FINISHED_OFF
        // becomes visible here), so a fresh read is needed before trusting
        // "nothing left" -- otherwise this could exit while real,
        // never-claimed work still exists.
        const int64_t final_total = atomic_read(TOTAL_OFF);
        if (cached_head >= final_total) break;
        cached_total = final_total;  // real unclaimed work remains; go claim it next iteration
      }
      if (!made_progress) {
        maybe_participate_in_gather();
        detail::rma_wait_idle(m_window);
      }
    }
  }

 private:
  Config m_config;
  Comm m_comm;
  std::function<ResultT(TaskT)> m_worker_function;

  MPI_Win m_window = MPI_WIN_NULL;
  std::vector<std::byte> m_window_buffer;                         // manager: control + task table
  alignas(int64_t) std::byte m_worker_window[sizeof(int64_t)]{};  // workers: Win_create placeholder
  bool m_finalized = false;

  // Layout, computed once in initialize_window().
  size_t m_task_elem = 0;
  size_t m_result_elem = 0;
  size_t m_max_task_count = 1;
  size_t m_max_result_count = 1;
  size_t m_task_slot_stride = 0;
  size_t m_task_base = 0;

  int64_t m_total_tasks = 0;
  int64_t m_gather_seq = 0;
  int64_t m_seen_gather_seq = 0;
  size_t m_collected_count = 0;
  size_t m_returned_count = 0;
  std::vector<ResultT> m_results;
  std::map<int64_t, ResultT> m_staging;
  std::vector<TaskT> m_task_store;
  std::vector<std::pair<int64_t, ResultT>> m_local_results;  // workers only

  StatisticsT m_statistics;

  // --- Setup ---

  int num_workers() const { return m_comm.size() - 1; }

  void initialize_window() {
    m_task_elem = static_cast<size_t>(detail::mpi_type_size_bytes<TaskT>());
    m_result_elem = static_cast<size_t>(detail::mpi_type_size_bytes<ResultT>());
    m_max_task_count =
        MPI_Type<TaskT>::resize_required ? static_cast<size_t>(m_config.max_task_count) : 1;
    m_max_result_count =
        MPI_Type<ResultT>::resize_required ? static_cast<size_t>(m_config.max_result_count) : 1;

    m_task_slot_stride = detail::round_up_8(T_DATA + m_max_task_count * m_task_elem);

    const size_t capacity = static_cast<size_t>(m_config.max_tasks);
    m_task_base = CONTROL_BYTES;
    const size_t manager_window_bytes = m_task_base + capacity * m_task_slot_stride;

    if (is_root_manager() && num_workers() > 0) {
      m_window_buffer.resize(manager_window_bytes);
    }

    if (is_root_manager() && num_workers() == 0) {
      return;
    }

    void* base = nullptr;
    MPI_Aint bsize = 0;
    if (is_root_manager()) {
      base = m_window_buffer.data();
      bsize = static_cast<MPI_Aint>(m_window_buffer.size());
    } else {
      base = m_worker_window;
      bsize = static_cast<MPI_Aint>(sizeof(m_worker_window));
    }
    DYNAMPI_MPI_CHECK(MPI_Win_create, (base, bsize, 1, MPI_INFO_NULL, m_comm.get(), &m_window));
    DYNAMPI_MPI_CHECK(MPI_Win_lock_all, (MPI_MODE_NOCHECK, m_window));
  }

  MPI_Aint task_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_task_base + static_cast<size_t>(index) * m_task_slot_stride);
  }

  size_t available() const { return m_results.size(); }

  // --- RMA primitives ---

  void flush(int rank) { DYNAMPI_MPI_CHECK(MPI_Win_flush, (rank, m_window)); }

  // Remote ranks use Fetch_and_op to read/update the manager's window.
  int64_t atomic_read(MPI_Aint offset) {
    int64_t in = 0, out;
    m_comm.fetch_and_op(in, out, m_config.manager_rank, offset, MPI_NO_OP, m_window);
    flush(m_config.manager_rank);
    return out;
  }

  void atomic_set(MPI_Aint offset, int64_t value) {
    int64_t in = value, out;
    m_comm.fetch_and_op(in, out, m_config.manager_rank, offset, MPI_REPLACE, m_window);
    flush(m_config.manager_rank);
  }

  // Unlike compare_and_swap (see run_worker()'s doc comment for why it was
  // dropped), fetch_add can't lose a race -- there's nothing to compare, it
  // unconditionally reserves `increment` slots starting at whatever HEAD_OFF
  // currently is and hands that starting point back.
  int64_t fetch_add(MPI_Aint offset, int64_t increment) {
    int64_t in = increment, out;
    m_comm.fetch_and_op(in, out, m_config.manager_rank, offset, MPI_SUM, m_window);
    flush(m_config.manager_rank);
    return out;
  }

  void put_bytes(const void* src, size_t n, MPI_Aint offset) {
    m_comm.put_bytes(src, n, m_config.manager_rank, offset, m_window);
    flush(m_config.manager_rank);
  }

  // Workers read the manager's window.
  void get_bytes(void* dst, size_t n, MPI_Aint offset) {
    m_comm.get_bytes(dst, n, m_config.manager_rank, offset, m_window);
    flush(m_config.manager_rank);
  }

  // --- Task / result transfer ---

  void publish_task(const TaskT& task) {
    const int64_t index = m_total_tasks;
    assert(static_cast<size_t>(index) < static_cast<size_t>(m_config.max_tasks) &&
           "LockFree: exceeded max_tasks capacity");

    if (num_workers() == 0) {
      m_task_store.push_back(task);
      m_total_tasks++;
      return;
    }

    const int count = MPI_Type<TaskT>::count(task);
    assert(count >= 0);
    assert(static_cast<size_t>(count) <= m_max_task_count &&
           "LockFree: task exceeds max_task_count");
    const size_t data_bytes = static_cast<size_t>(count) * m_task_elem;

    std::vector<std::byte> buffer(T_DATA + data_bytes);
    detail::write_i64(buffer.data(), buffer.size(), T_COUNT, count);
    if (data_bytes > 0) {
      detail::write_bytes(buffer.data(), buffer.size(), T_DATA, MPI_Type<TaskT>::ptr(task),
                          data_bytes);
    }
    put_bytes(buffer.data(), buffer.size(), task_slot(index));

    m_total_tasks++;
    atomic_set(TOTAL_OFF, m_total_tasks);  // publish to workers
  }

  // Reads `count` contiguous task slots starting at `index` in a single RMA
  // Get spanning the whole range (rather than one Get per task, or two for
  // resizable TaskT), then parses each fixed-stride slot out of that one
  // buffer locally. Transfers some unused padding for resizable TaskT whose
  // actual size is below max_task_count, but for a claimed batch that's a
  // bandwidth cost, not a round-trip cost -- round trips are what dominate
  // here (see run_worker()).
  std::vector<TaskT> read_task_batch(int64_t index, int64_t count) {
    const size_t bytes = static_cast<size_t>(count) * m_task_slot_stride;
    std::vector<std::byte> buf(bytes);
    get_bytes(buf.data(), bytes, task_slot(index));

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

  void store_result(int64_t index, ResultT result) {
    assert(static_cast<size_t>(MPI_Type<ResultT>::count(result)) <= m_max_result_count &&
           "LockFree: result exceeds max_result_count");
    m_local_results.emplace_back(index, std::move(result));
  }

  // --- Result collection via Gatherv (all ranks) ---

  void maybe_participate_in_gather() {
    const int64_t seq = atomic_read(GATHER_SEQ_OFF);
    if (seq == m_seen_gather_seq) return;
    m_seen_gather_seq = seq;
    DYNAMPI_MPI_CHECK(MPI_Barrier, (m_comm.get()));
    exchange_gathered_results();
  }

  void request_gather() {
    atomic_set(GATHER_SEQ_OFF, ++m_gather_seq);
    detail::rma_wait_idle(m_window);
    maybe_participate_in_gather();
  }

  void try_gather_results() {
    const size_t before = m_collected_count;
    request_gather();
    if (m_collected_count == before) detail::rma_wait_idle(m_window);
  }

  void exchange_gathered_results() {
    const int elem = detail::mpi_type_size_bytes<ResultT>();
    const bool manager = is_root_manager();
    const int size = m_comm.size();

    std::vector<std::byte> send_buf;
    for (auto& [index, result] : m_local_results) {
      const int count = MPI_Type<ResultT>::count(result);
      assert(count >= 0);
      const size_t data_bytes =
          count > 0 ? static_cast<size_t>(count) * static_cast<size_t>(elem) : size_t{0};
      const size_t offset = send_buf.size();
      send_buf.resize(offset + 16 + data_bytes);
      detail::write_i64(send_buf.data(), send_buf.size(), offset, index);
      detail::write_i64(send_buf.data(), send_buf.size(), offset + 8, count);
      if (data_bytes > 0) {
        detail::write_bytes(send_buf.data(), send_buf.size(), offset + 16,
                            MPI_Type<ResultT>::ptr(result), data_bytes);
      }
    }
    m_local_results.clear();

    const int send_count = static_cast<int>(send_buf.size());
    std::vector<int> byte_counts(manager ? static_cast<size_t>(size) : 0);
    DYNAMPI_MPI_CHECK(MPI_Gather, (&send_count, 1, MPI_INT, manager ? byte_counts.data() : nullptr,
                                   1, MPI_INT, m_config.manager_rank, m_comm.get()));

    std::vector<int> displacements;
    std::vector<std::byte> recv_buf;
    int total_bytes = 0;
    if (manager) {
      displacements.resize(static_cast<size_t>(size));
      for (int r = 0; r < size; ++r) {
        displacements[static_cast<size_t>(r)] = total_bytes;
        total_bytes += byte_counts[static_cast<size_t>(r)];
      }
      recv_buf.resize(static_cast<size_t>(total_bytes));
    }

    DYNAMPI_MPI_CHECK(
        MPI_Gatherv,
        (send_buf.data(), send_count, MPI_BYTE, manager ? recv_buf.data() : nullptr,
         manager ? byte_counts.data() : nullptr, manager ? displacements.data() : nullptr, MPI_BYTE,
         m_config.manager_rank, m_comm.get()));

    if (!manager) return;

    for (int r = 0; r < size; ++r) {
      if (r == m_config.manager_rank) continue;
      size_t pos = static_cast<size_t>(displacements[static_cast<size_t>(r)]);
      const size_t end = pos + static_cast<size_t>(byte_counts[static_cast<size_t>(r)]);
      size_t result_count = 0;
      while (pos < end) {
        assert(pos + 16 <= end);
        const int64_t index = detail::read_i64(recv_buf.data(), recv_buf.size(), pos);
        const int64_t count = detail::read_i64(recv_buf.data(), recv_buf.size(), pos + 8);
        pos += 16;
        ResultT result{};
        if constexpr (MPI_Type<ResultT>::resize_required)
          MPI_Type<ResultT>::resize(result, static_cast<int>(count));
        assert(count >= 0);
        const size_t data_bytes =
            count > 0 ? static_cast<size_t>(count) * static_cast<size_t>(elem) : size_t{0};
        detail::read_result_bytes(recv_buf.data(), recv_buf.size(), pos, result, data_bytes);
        pos += data_bytes;
        m_staging[index] = std::move(result);
        result_count++;
      }
      if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
        if (static_cast<size_t>(r) < m_statistics.worker_task_counts.size())
          m_statistics.worker_task_counts[static_cast<size_t>(r)] += result_count;
      }
    }

    while (m_staging.contains(static_cast<int64_t>(m_collected_count))) {
      m_results.push_back(std::move(m_staging[static_cast<int64_t>(m_collected_count)]));
      m_staging.erase(static_cast<int64_t>(m_collected_count));
      m_collected_count++;
    }
  }

  void run_one_task_locally() {
    m_results.push_back(m_worker_function(m_task_store[m_collected_count]));
    if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
      if (static_cast<size_t>(m_config.manager_rank) < m_statistics.worker_task_counts.size())
        m_statistics.worker_task_counts[static_cast<size_t>(m_config.manager_rank)]++;
    }
    m_collected_count++;
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

  static StatisticsT make_statistics(const Comm& comm) {
    if constexpr (statistics_mode == StatisticsMode::Detailed) {
      return Statistics{.comm_statistics = comm.get_statistics(), .worker_task_counts = {}};
    } else {
      return {};
    }
  }
};

}  // namespace dynampi
