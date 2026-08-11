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
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/rma_detail.hpp"
#include "dynampi/mpi/mpi_error.hpp"
#include "dynampi/task_error.hpp"
#include "dynampi/utilities/timer.hpp"

namespace dynampi {

// ---------------------------------------------------------------------------
// LockFreeRMAWorkDistributor
//
// Lock-free task claiming via fetch-and-add against HEAD_OFF, bounded by a
// cached read of TOTAL_OFF, one task per claim -- this is a flat distributor.
// Per task (worker):
//   fetch_add(HEAD) + flush_local   -- claim; only need origin completion
//   get(task) + flush_local         -- read payload
//   compute
//   put(result) + flush             -- data durable at manager
//   put(log flag) + flush           -- plain Put (slot is exclusively owned);
//                                      second flush publishes the flag after data
//
// Manager publish: put(tasks) + atomic_set(TOTAL) + flush
// Manager harvest reads use flush_local only.
//
// Results are explicitly NOT ordered: harvested in completion order (which,
// via the completion log's contiguous-prefix scan, is close to submission
// order but not guaranteed).

// ---------------------------------------------------------------------------
// Options currently only recognizes track_statistics<...> (no prioritization support in this
// class).
template <typename TaskT, typename ResultT, typename... Options>
class LockFreeRMAWorkDistributor {
 private:
  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();
  using Comm = dynampi::MPICommunicator<track_statistics<statistics_mode>>;

 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;
    bool auto_run_workers = true;
    int max_tasks = 8192;
    int max_task_count = 256;
    int max_result_count = 256;

    // If true (default), run_tasks()/finish_remaining_tasks() throw
    // dynampi::TaskFailure on the manager once a task has thrown. Set false to
    // recover instead: distribution runs to completion and the failures are
    // available from take_task_errors().
    bool rethrow_task_errors = true;
  };

  struct RunConfig {
    size_t target_num_tasks = std::numeric_limits<size_t>::max();
    bool allow_more_than_target_tasks = true;
    std::optional<double> max_seconds = std::nullopt;
  };

  // Not ordered
  static const bool ordered = false;

  struct Statistics {
    const CommStatistics& comm_statistics;
  };
  using StatisticsT =
      std::conditional_t<statistics_mode != StatisticsMode::None, Statistics, std::monostate>;

  const StatisticsT& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    assert(is_root_manager());
    return m_statistics;
  }

  explicit LockFreeRMAWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                      Config config = {})
      : m_config(config),
        m_comm(config.comm, Comm::Duplicate),
        m_worker_function(std::move(worker_function)),
        m_errors_seen(static_cast<size_t>(kMaxRecordedErrors)),
        m_statistics{make_statistics(m_comm)} {
    initialize_window();
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

  ~LockFreeRMAWorkDistributor() {
    if (!m_finalized) finalize();
    m_task_errors.warn_if_unreported("LockFreeRMAWorkDistributor");
    if (m_window != MPI_WIN_NULL) {
      DYNAMPI_MPI_CHECK(MPI_Win_unlock_all, (m_window));
      MPI_Barrier(m_comm.get());
      DYNAMPI_MPI_CHECK(MPI_Win_free, (&m_window));
      m_window = MPI_WIN_NULL;
    }
  }

  bool is_root_manager() const { return m_comm.rank() == m_config.manager_rank; }
  int num_workers() const { return m_comm.size() - 1; }

  size_t remaining_tasks_count() const {
    assert(is_root_manager());
    return static_cast<size_t>(m_total_tasks) - m_returned_count;
  }

  void insert_task(TaskT task) {
    assert(is_root_manager());
    publish_tasks(std::span<const TaskT>(&task, 1));
  }

  // Priority is accepted but ignored (no prioritization support -- see the
  // Options note above); exists only for template-shape compatibility with
  // generic test/benchmark code exercised uniformly across distributors.
  void insert_task(const TaskT& task, double /*priority*/) {
    assert(is_root_manager());
    publish_tasks(std::span<const TaskT>(&task, 1));
  }

  void insert_tasks(const std::vector<TaskT>& tasks) {
    assert(is_root_manager());
    publish_tasks(std::span<const TaskT>(tasks));
  }

  [[nodiscard]] std::vector<ResultT> run_tasks(RunConfig config = {}) {
    assert(is_root_manager());
    Timer timer;

    if (num_workers() == 0) {
      while (m_collected_count < static_cast<size_t>(m_total_tasks)) {
        if (m_results.size() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        run_one_task_locally();
      }
    } else {
      while (true) {
        if (m_results.size() >= config.target_num_tasks) break;
        if (config.max_seconds && timer.elapsed().count() >= *config.max_seconds) break;
        if (m_collected_count >= static_cast<size_t>(m_total_tasks)) break;
        const size_t before = m_collected_count;
        harvest_ready_results();
        if (m_collected_count == before) detail::rma_wait_idle(m_window, m_comm.get());
      }
    }

    // Thrown before draining, so results collected so far stay buffered for
    // whoever catches this and calls again.
    m_task_errors.rethrow_first_if(m_config.rethrow_task_errors);

    const size_t limit = config.allow_more_than_target_tasks ? std::numeric_limits<size_t>::max()
                                                             : config.target_num_tasks;
    return drain_results(limit);
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() { return run_tasks({}); }

  // Requests exactly one harvest pass and returns whatever's ready --
  // possibly none, possibly everything outstanding, depending on how far
  // workers have gotten. Unlike run_tasks()/finish_remaining_tasks(), this
  // does not loop retrying until every outstanding task is collected: those
  // block until m_collected_count reaches m_total_tasks (or a target/
  // timeout), which is the wrong shape for "take one snapshot and return
  // control" callers -- e.g. a benchmark driver measuring throughput of an
  // uninterrupted worker-side spin, which wants to publish a batch, let
  // workers run completely undisturbed for a fixed window, then harvest
  // once at the end (see LockFreeRMA's benchmark path in
  // strong_scaling_distribution_rate.cpp).
  [[nodiscard]] std::vector<ResultT> gather_once() {
    assert(is_root_manager());
    if (num_workers() == 0) {
      while (m_collected_count < static_cast<size_t>(m_total_tasks)) run_one_task_locally();
    } else {
      harvest_ready_results();
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
          const size_t before = m_collected_count;
          harvest_ready_results();
          if (m_collected_count == before) detail::rma_wait_idle(m_window, m_comm.get());
        }
        atomic_set(FINISHED_OFF, 1);  // tell workers to stop
        detail::rma_wait_idle(m_window, m_comm.get());
      }
    }
    m_finalized = true;
  }

  void run_worker() {
    assert(!is_root_manager());

    // One-task-at-a-time fetch-and-add claim loop, bounded by a cached
    // read of TOTAL_OFF (a fresh atomic_read() only when this rank's own
    // claimed range has caught up to what it last observed as published --
    // cached_total is a monotonic lower bound, never an overestimate, since
    // it's just a stale snapshot of a monotonically-increasing counter).
    //
    // The claim branch is deliberately NOT gated on !finished_seen:
    // finished_seen only means "TOTAL_OFF won't grow anymore", not "stop
    // claiming what's already known to be available" -- gating the whole
    // branch on !finished_seen meant that once finished_seen went true,
    // nothing could ever claim again, even a real, never-claimed gap
    // between cached_head and a just-learned final_total (see the exit
    // check below, which discovers and records that gap into
    // cached_total). Confirmed via gdb backtraces on a caught hang: 15/16
    // ranks parked in the destructor's teardown barrier while the 16th sat
    // in an idle-wait loop forever.
    //
    // Even at claim size 1, HEAD_OFF (claims) can race ahead of TOTAL_OFF
    // (publishes): with many workers each doing their own unconditional
    // fetch_add, aggregate claim rate can outpace publish_tasks()'s rate,
    // so a freshly claimed index can land beyond what's actually published
    // yet. pending_start/pending_end track that unresolved remainder across
    // iterations; a worker with a pending tail must still process whatever
    // prefix is already ready immediately rather than blocking on the rest,
    // since finalize() can't set FINISHED_OFF until it has collected every
    // result -- including this worker's already-ready prefix -- so blocking
    // here would deadlock against a manager waiting on results this worker
    // refuses to compute.
    //
    // No maybe_participate_in_gather() here at all -- there is no gather
    // protocol in this class, a result goes out via one-sided Put the
    // moment its task finishes.
    int64_t cached_head = 0;
    int64_t cached_total = 0;
    int64_t pending_start = -1;
    int64_t pending_end = -1;
    bool finished_seen = false;

    auto process_range = [&](int64_t start, int64_t count) {
      std::vector<TaskT> tasks = read_task_batch(start, count);
      std::vector<ResultT> results;
      results.reserve(static_cast<size_t>(count));
      bool failed = false;
      for (int64_t i = 0; i < count; ++i) {
        ResultT result;
        auto failure = detail::run_task_guarded(m_worker_function,
                                                std::move(tasks[static_cast<size_t>(i)]), result);
        if (failure) {
          report_task_error(*failure);
          failed = true;
        }
        results.push_back(std::move(result));
      }
      write_result_range(start, results, failed);
    };

    while (true) {
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
          pending_start = -1;  // LCOV_EXCL_LINE -- depends on publication racing a batch claim
        } else if (finished_seen) {
          pending_start = -1;
        }
      } else {
        // Deliberately NOT gated on !finished_seen: see the comment on the
        // exit check below -- gating this branch on !finished_seen was the
        // exact bug caught via gdb (15/16 ranks parked in the destructor's
        // teardown barrier, the 16th spinning forever re-discovering a real
        // unclaimed gap it had no way left to act on).
        if (cached_head >= cached_total && !finished_seen) cached_total = atomic_read(TOTAL_OFF);

        if (cached_head < cached_total) {
          const int64_t claim = 1;
          const int64_t start = fetch_add(HEAD_OFF, claim);
          const int64_t end = start + claim;
          cached_head = end;

          const int64_t total = (end <= cached_total) ? cached_total : atomic_read(TOTAL_OFF);
          // Clamped into [start, end], not just capped at end: total can be
          // *less than start* here, not only less than end -- cached_total
          // (used above to size the claim) can be stale enough that this
          // worker's whole claim lands beyond what's actually published yet,
          // not merely straddling the boundary. Before this clamp, an
          // unclamped `ready = std::min(end, total)` could come out below
          // `start`, and the pending_start assignment below would then set
          // pending_start to that sub-start value -- an index this worker
          // never claimed and does not own, since fetch_add already handed
          // it to whichever earlier claimant's range covers it. This was a
          // real, confirmed bug: two ranks ended up both writing results
          // for the same task index (one legitimately, one via this
          // corrupted pending_start), and whichever write landed last in
          // the completion log silently clobbered the other's entry,
          // permanently stalling the manager's harvest (confirmed via
          // instrumented logging on a caught hang -- see git history).
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
        const int64_t final_total = atomic_read(TOTAL_OFF);
        if (cached_head >= final_total) break;
        cached_total = final_total;
      }
      if (!made_progress) detail::rma_wait_idle(m_window, m_comm.get());
    }
  }

 private:
  // Window layout: [head][total][finished] then the task table, the result
  // table, and the completion log, each sized for max_tasks entries. No
  // gather-sequence field -- there is no gather protocol.
  static constexpr MPI_Aint HEAD_OFF = 0;
  static constexpr MPI_Aint TOTAL_OFF = 8;
  static constexpr MPI_Aint FINISHED_OFF = 16;
  static constexpr MPI_Aint ERROR_COUNT_OFF = 24;
  static constexpr size_t CONTROL_BYTES = 32;

  // Failed tasks report into a small fixed table rather than through the result
  // slots, which are sized for ResultT and are routinely far too small to hold
  // a message. Claimants take a slot with fetch_and_add, so the table fills in
  // report order; past kMaxRecordedErrors the count still rises (so nothing is
  // silently lost) but the messages are dropped.
  static constexpr int64_t kMaxRecordedErrors = 16;
  static constexpr size_t E_RANK = 0;
  static constexpr size_t E_DATA = 8;
  static constexpr size_t ERROR_SLOT_BYTES = E_DATA + kMaxTaskErrorMessage;

  // Task slot: [count][data].
  static constexpr size_t T_COUNT = 0;
  static constexpr size_t T_DATA = 8;
  // Result slot: [count][data]. No valid flag here -- validity lives in the
  // completion log instead (see write_result_range), one entry per whole
  // claimed batch rather than one flag per task.
  static constexpr size_t R_COUNT = 0;
  static constexpr size_t R_DATA = 8;
  // Completion log entry: a single int64. 0 = untouched (sentinel; a real
  // entry always has count >= 1, see process_range's `ready > start` guard
  // before it ever calls write_result_range). count > 0 at log index `s`
  // means "the batch claimed at task index s, covering [s, s+count), is
  // fully written to the result table". Keyed by the claim's own start
  // index -- collision-free for free, since fetch_add already guarantees
  // this worker is the sole owner of that start index, so no separate
  // slot-allocation round trip is needed to place this entry.
  static constexpr size_t LOG_ENTRY_BYTES = 8;

  Config m_config;
  Comm m_comm;
  std::function<ResultT(TaskT)> m_worker_function;

  MPI_Win m_window = MPI_WIN_NULL;
  std::vector<std::byte> m_window_buffer;  // manager: control + task table + result table + log
  alignas(int64_t) std::byte m_worker_window[sizeof(int64_t)]{};
  bool m_finalized = false;

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
  // Doubles as the completion-log scan frontier: the manager only ever
  // advances it by exactly the number of newly-confirmed results each
  // harvest, so "task indices below m_collected_count are fully collected"
  // and "the log/result tables are unscanned above it" stay equivalent by
  // construction -- no separate frontier variable needed.
  size_t m_collected_count = 0;
  size_t m_returned_count = 0;
  detail::TaskErrorLog m_task_errors;  // manager only
  std::vector<bool> m_errors_seen;     // manager only: error slots already consumed
  std::vector<ResultT> m_results;      // manager only: harvested, ready to return
  std::vector<TaskT> m_task_store;     // manager only, solo-world fallback
  StatisticsT m_statistics;

  void initialize_window() {
    // Window slots are fixed-width, so a non-resizable payload spanning more
    // than one datatype element would overrun them -- see
    // check_fixed_size_mpi_type().
    check_fixed_size_mpi_type<TaskT>("task", "LockFreeRMA");
    check_fixed_size_mpi_type<ResultT>("result", "LockFreeRMA");

    m_task_elem = static_cast<size_t>(detail::mpi_type_size_bytes<TaskT>());
    m_result_elem = static_cast<size_t>(detail::mpi_type_size_bytes<ResultT>());
    // A fixed-size payload always occupies mpi_elements_per_value<T>() elements
    // -- 1 for a scalar, but 3 for a struct of three doubles -- so slots are
    // sized for that rather than for a single element.
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
    const size_t manager_window_bytes =
        m_error_base + static_cast<size_t>(kMaxRecordedErrors) * ERROR_SLOT_BYTES;

    if (num_workers() == 0) return;  // solo world: no RMA window needed at all

    if (is_root_manager()) {
      m_window_buffer.resize(manager_window_bytes);
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
  MPI_Aint result_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_result_base + static_cast<size_t>(index) * m_result_slot_stride);
  }
  MPI_Aint log_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_log_base + static_cast<size_t>(index) * LOG_ENTRY_BYTES);
  }
  MPI_Aint error_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_error_base + static_cast<size_t>(index) * ERROR_SLOT_BYTES);
  }

  // --- RMA primitives ---
  //
  // Passive-target RMA is nonblocking. flush_local completes at the origin
  // (enough before reading claim/get results); flush_remote completes at the
  // target (needed before a remote rank can observe Puts / atomics).
  int manager_rank() const { return m_config.manager_rank; }

  void flush_remote() { DYNAMPI_MPI_CHECK(MPI_Win_flush, (manager_rank(), m_window)); }
  void flush_local() { DYNAMPI_MPI_CHECK(MPI_Win_flush_local, (manager_rank(), m_window)); }

  void post_fetch_and_op(int64_t in, int64_t& out, MPI_Aint offset, MPI_Op op) {
    m_comm.fetch_and_op(in, out, manager_rank(), offset, op, m_window);
  }
  void post_put_bytes(const void* src, size_t n, MPI_Aint offset) {
    m_comm.put_bytes(src, n, manager_rank(), offset, m_window);
  }
  void post_get_bytes(void* dst, size_t n, MPI_Aint offset) {
    m_comm.get_bytes(dst, n, manager_rank(), offset, m_window);
  }

  int64_t atomic_read(MPI_Aint offset) {
    int64_t in = 0, out;
    post_fetch_and_op(in, out, offset, MPI_NO_OP);
    flush_local();
    return out;
  }
  void atomic_set(MPI_Aint offset, int64_t value) {
    int64_t in = value, out;
    post_fetch_and_op(in, out, offset, MPI_REPLACE);
    flush_remote();
  }
  int64_t fetch_add(MPI_Aint offset, int64_t increment) {
    int64_t in = increment, out;
    post_fetch_and_op(in, out, offset, MPI_SUM);
    flush_local();
    return out;
  }

  void get_bytes_local(void* dst, size_t n, MPI_Aint offset) {
    post_get_bytes(dst, n, offset);
    flush_local();
  }

  // --- Task publish (manager side) / read (claimant side) ---

  // Publishes tasks.size() tasks in two round trips total, regardless of
  // how many: one bulk Put for the whole range's [count][data] slots, then
  // one atomic_set for TOTAL_OFF at the end -- the same batching this class
  // already applies on the claim side (fetch_add) and the result-write side
  // (write_result_range). Before this, insert_task()/insert_tasks() called
  // a per-task publish_task() that did its own Put + atomic_set per task (2
  // round trips per task, unbatched) -- confirmed via measurement to be the
  // actual bottleneck once the claim and result paths were already batched:
  // near-zero-compute-time throughput (isolating publish+claim+write from
  // real work) only reached ~18K tasks/s against rma_atomic_microbench's
  // ~900K/s raw ceiling, and this per-task publish loop is exactly the same
  // shape of problem already fixed twice elsewhere in this file.
  //
  // TOTAL_OFF is only bumped once, after every task's data is durably in
  // place -- unlike the old per-task version, a claimant can no longer
  // observe a bumped TOTAL_OFF for some prefix of this batch while the rest
  // is still being Put; that's fine because TOTAL_OFF is the only signal
  // that gates claiming, and this batch was never claimable mid-flight
  // either way (compare to write_result_range's ordering requirement, which
  // is about a *different* pair of fields).
  void publish_tasks(std::span<const TaskT> tasks) {
    if (tasks.empty()) return;
    const int64_t start = m_total_tasks;
    detail::check_task_capacity(start, tasks.size(), m_config.max_tasks, "LockFreeRMA");
    if (num_workers() == 0) {
      m_task_store.insert(m_task_store.end(), tasks.begin(), tasks.end());
      m_total_tasks += static_cast<int64_t>(tasks.size());
      return;
    }
    std::vector<std::byte> buffer(tasks.size() * m_task_slot_stride);
    for (size_t i = 0; i < tasks.size(); ++i) {
      const TaskT& task = tasks[i];
      const int count = MPI_Type<TaskT>::count(task);
      assert(static_cast<size_t>(count) <= m_max_task_count &&
             "LockFreeRMA: task exceeds max_task_count");
      const size_t data_bytes = static_cast<size_t>(count) * m_task_elem;
      const size_t off = i * m_task_slot_stride;
      detail::write_i64(buffer.data(), buffer.size(), off + T_COUNT, count);
      if (data_bytes > 0) {
        detail::write_bytes(buffer.data(), buffer.size(), off + T_DATA, MPI_Type<TaskT>::ptr(task),
                            data_bytes);
      }
    }
    post_put_bytes(buffer.data(), buffer.size(), task_slot(start));
    m_total_tasks += static_cast<int64_t>(tasks.size());
    atomic_set(TOTAL_OFF, m_total_tasks);
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

  // --- Result write (claimant side) / harvest (manager side) ---

  // Writes results.size() results starting at task index `start`:
  //   Put result data -> flush -> Put completion-log flag -> flush.
  // The log write is a plain Put (not an atomic): fetch_add already gave this
  // worker exclusive ownership of `start`, so no concurrent writer shares the
  // slot. The intervening flush is what makes "data before flag" portable.
  // Publishes one failed task into the error table. Must complete before the
  // log entry that advertises it -- every RMA helper here flushes its target
  // before returning, so calling this before write_result_range() is enough.
  void report_task_error(const std::string& message) {
    const int64_t slot = fetch_add(ERROR_COUNT_OFF, 1);
    if (slot >= kMaxRecordedErrors) return;  // LCOV_EXCL_LINE -- needs >16 concurrent failures
    // Message first, then the ready word, each flushed separately: fetch_add
    // hands out the slot before anything is in it, so the count alone never
    // means a slot can be read. The ready word is rank+1 so that zero stays the
    // untouched sentinel for rank 0.
    std::vector<std::byte> message_bytes(kMaxTaskErrorMessage, std::byte{0});
    const size_t bytes = std::min(message.size(), kMaxTaskErrorMessage - 1);
    if (bytes > 0) {
      detail::write_bytes(message_bytes.data(), message_bytes.size(), 0, message.data(), bytes);
    }
    const int64_t ready = static_cast<int64_t>(m_comm.rank()) + 1;
    post_put_bytes(message_bytes.data(), message_bytes.size(),
                   error_slot(slot) + static_cast<MPI_Aint>(E_DATA));
    flush_remote();
    post_put_bytes(&ready, sizeof(ready), error_slot(slot) + static_cast<MPI_Aint>(E_RANK));
    flush_remote();
  }

  // Owner-side. Reads whatever error records have appeared since the last call.
  // Only ever called after the log scan has seen a negative entry, so it costs
  // nothing on a run with no failures.
  void harvest_task_errors() {
    const int64_t claimed = std::min(atomic_read(ERROR_COUNT_OFF), kMaxRecordedErrors);
    if (claimed <= 0) return;  // LCOV_EXCL_LINE -- only reachable on a spurious flag
    // Reads the whole (at most kMaxRecordedErrors) table and takes only the
    // slots that are ready, rather than a contiguous frontier: slots are
    // claimed in fetch_add order but completed out of order, so a slot still in
    // flight must be skipped now and picked up later, not waited on. That is
    // safe because a claimant flushes its record before writing the completion
    // log entry that brings the manager here, so by the time this runs for a
    // given failure, that failure's own slot is readable.
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
      m_task_errors.record(std::move(error));
      m_errors_seen[static_cast<size_t>(i)] = true;
    }
  }

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
    post_put_bytes(buffer.data(), buffer.size(), result_slot(start));
    flush_remote();
    // A negated length marks "this range produced at least one error"; the
    // manager already reads every log entry, so the flag is free, and it only
    // pays for the error table when something has actually failed.
    const int64_t entry = contains_error ? -count : count;
    post_put_bytes(&entry, sizeof(entry), log_slot(start));
    flush_remote();
  }

  // Owner-only. Three round trips regardless of how many batches turn out
  // to be ready: one to see how far claiming has progressed (HEAD_OFF), one
  // bulk read of the completion log over the unscanned range, one bulk read
  // of the result table over whatever contiguous prefix of that range the
  // log confirms is done. A gap (a still-in-flight or straggling batch)
  // simply stops the contiguous-prefix scan there for this call; already-
  // confirmed entries past a gap are picked up on a later call once the gap
  // fills in, at the cost of re-scanning that stretch of the log -- cheap,
  // since the scan is one bulk Get either way, not a per-entry round trip.
  //
  // Goes through the real RMA API (Fetch_and_op/Get), even though the
  // target is this same rank: MPI only guarantees a window owner sees
  // another rank's completed RMA writes via its *local* loads under
  // MPI_WIN_UNIFIED, not MPI_WIN_SEPARATE, and the RMA API is correct
  // either way. Confirmed necessary via a caught hang earlier: plain local
  // loads on m_window_buffer never observed workers' writes at all.
  //
  // On MS-MPI (always SEPARATE), even these self-targeted Gets need the
  // two-sided progress engine driven between polls -- see detail::rma_wait_idle
  // (MPI_Iprobe). flush_all alone is not enough; without Iprobe the manager
  // harvest loop spins forever seeing HEAD/log as unchanged.
  void harvest_ready_results() {
    assert(is_root_manager());
    int64_t head_in = 0, head_now = 0;
    post_fetch_and_op(head_in, head_now, HEAD_OFF, MPI_NO_OP);
    flush_local();

    const int64_t frontier = static_cast<int64_t>(m_collected_count);
    if (head_now <= frontier) return;

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
    if (saw_error) harvest_task_errors();
    if (confirmed_end <= frontier) return;

    const int64_t n = confirmed_end - frontier;
    std::vector<std::byte> result_buf(static_cast<size_t>(n) * m_result_slot_stride);
    get_bytes_local(result_buf.data(), result_buf.size(), result_slot(frontier));

    m_results.reserve(m_results.size() + static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
      const size_t off = static_cast<size_t>(i) * m_result_slot_stride;
      const int64_t count = detail::read_i64(result_buf.data(), result_buf.size(), off + R_COUNT);
      ResultT result{};
      if constexpr (MPI_Type<ResultT>::resize_required)
        MPI_Type<ResultT>::resize(result, static_cast<int>(count));
      const size_t data_bytes = count > 0 ? static_cast<size_t>(count) * m_result_elem : size_t{0};
      detail::read_result_bytes(result_buf.data(), result_buf.size(), off + R_DATA, result,
                                data_bytes);
      m_results.push_back(std::move(result));
    }
    m_collected_count = static_cast<size_t>(confirmed_end);
  }

  void run_one_task_locally() {
    // Guarded like every other execution path: the solo case exists so a
    // workload can be debugged without MPI, which it cannot be if failures
    // surface differently here.
    ResultT result;
    auto failure =
        detail::run_task_guarded(m_worker_function, m_task_store[m_collected_count], result);
    if (failure) m_task_errors.record(TaskError{m_comm.rank(), std::move(*failure)});
    m_results.push_back(std::move(result));
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
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{.comm_statistics = comm.get_statistics()};
    } else {
      return {};
    }
  }
};

}  // namespace dynampi
