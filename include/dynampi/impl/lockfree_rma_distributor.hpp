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
// Results ARE ordered: each write_result_range() writes into the result slot
// at the claimed index, the same index publish_tasks() assigned in submission
// order, and harvest_ready_results() only ever advances past a *contiguous*
// prefix of the completion log -- a gap at index i blocks every result after
// it from being released, however long ago they finished. So the returned
// vector always matches submission order, same as NaiveWorkDistributor and
// for the same reason (see its class comment): the cost of guaranteeing order
// is that one slow task holds back everything behind it.
//
// The task/result/log tables are a ring of Config::max_tasks slots (index %
// max_tasks), not sized for the run's lifetime total -- publish_tasks()
// blocks until harvesting has freed enough slots for a batch rather than
// failing, so window memory stays bounded no matter how many tasks a run
// publishes over its life. See Config::max_tasks and ring_slot().

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
    // Ring capacity: the task/result/log tables are this many slots, reused
    // via index % max_tasks once a slot's result has been harvested -- not a
    // lifetime cap. publish_tasks() blocks (harvesting to make room) rather
    // than failing if the ring is currently full; it only throws if a single
    // call asks for more than max_tasks tasks at once, which can never fit
    // regardless of how much harvesting happens. Bounds RMA window memory
    // independent of how many tasks a run publishes over its lifetime.
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

  // Ordered -- see the class comment above for why.
  static const bool ordered = true;

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

  // One harvest pass, returning whatever is ready. Unlike run_tasks(), does
  // not retry until everything outstanding is collected -- for callers that
  // want a snapshot and control back, such as a benchmark measuring an
  // undisturbed worker spin over a fixed window.
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
        // Raised before the harvest, not after: FINISHED only means "TOTAL
        // won't grow", already true here, so workers drain out while the
        // last results are collected instead of contending the window
        // throughout. Safe because a worker flushes every result before it
        // can reach its exit check.
        atomic_set(FINISHED_OFF, 1);
        while (m_collected_count < static_cast<size_t>(m_total_tasks)) {
          const size_t before = m_collected_count;
          harvest_ready_results();
          if (m_collected_count == before) detail::rma_wait_idle(m_window, m_comm.get());
        }
        detail::rma_wait_idle(m_window, m_comm.get());
      }
    }
    m_finalized = true;
  }

  void run_worker() {
    assert(!is_root_manager());

    // One index per claim, gated on a cached TOTAL_OFF (a stale snapshot of
    // a monotonic counter, so always a safe lower bound). Claims can still
    // outrun publication, so a claim that lands past TOTAL is held in
    // `pending` until it is published or FINISHED says it never will be.
    // Held across iterations rather than spun on inline so the FINISHED
    // check and idle backoff below keep running meanwhile.
    int64_t cached_head = 0;
    int64_t cached_total = 0;
    int64_t pending = -1;  // claimed but not yet published; -1 = none
    bool finished_seen = false;

    auto process_one = [&](int64_t index) {
      std::vector<TaskT> tasks = read_task_batch(index, 1);
      ResultT result;
      auto failure = detail::run_task_guarded(m_worker_function, std::move(tasks[0]), result);
      if (failure) report_task_error(*failure);
      std::vector<ResultT> results;
      results.push_back(std::move(result));
      write_result_range(index, results, failure.has_value());
    };

    while (true) {
      bool made_progress = false;

      if (pending != -1) {
        if (atomic_read(TOTAL_OFF) > pending) {
          process_one(pending);
          pending = -1;
          made_progress = true;
        } else if (finished_seen) {
          pending = -1;  // past the end of the run; owes no result
        }
      } else {
        // Claiming is NOT gated on !finished_seen: that flag only means
        // TOTAL is final, and a real unclaimed gap can still remain below
        // it (see the exit check below, which records that gap).
        if (cached_head >= cached_total && !finished_seen) cached_total = atomic_read(TOTAL_OFF);

        if (cached_head < cached_total) {
          const int64_t index = fetch_add(HEAD_OFF, 1);
          cached_head = index + 1;

          // cached_total may be stale enough that this index still landed
          // past what is published; re-read unless it already covers it.
          const int64_t total = (index < cached_total) ? cached_total : atomic_read(TOTAL_OFF);
          if (index < total) {
            process_one(index);
            made_progress = true;
          } else {
            pending = index;
          }
        }
      }

      if (!finished_seen && atomic_read(FINISHED_OFF) != 0) finished_seen = true;

      if (pending == -1 && finished_seen) {
        const int64_t final_total = atomic_read(TOTAL_OFF);
        if (cached_head >= final_total) break;
        cached_total = final_total;
      }
      if (!made_progress) detail::rma_wait_idle(m_window, m_comm.get());
    }
  }

 private:
  // Window layout: [head][total][finished] then the task table, the result
  // table, and the completion log, each max_tasks entries wide and reused as
  // a ring (index % max_tasks, see ring_slot()) -- head and total are
  // absolute task indices that grow without bound over the run's lifetime,
  // not ring-relative. No gather-sequence field -- there is no gather
  // protocol.
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
  // Completion log entry: a single int64, 0 = untouched. count > 0 at index
  // `s` means the claim starting at task index s, covering [s, s+count), is
  // fully written to the result table. Keyed by the claim's own start index,
  // which fetch_add already made exclusive -- so no slot-allocation round
  // trip is needed.
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

  // Ring slot for a task index: reused every max_tasks indices. Callers are
  // responsible for never having two live (published, not yet harvested)
  // indices map to the same slot -- see publish_tasks()'s capacity wait.
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
    return static_cast<MPI_Aint>(m_log_base + static_cast<size_t>(ring_slot(index)) * LOG_ENTRY_BYTES);
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

  // Reads `count` consecutive task indices starting at `start` into `dst`,
  // splitting into two Gets if the range crosses a ring lap boundary --
  // slot_fn() (task_slot/result_slot/log_slot) is only contiguous within one
  // lap. Safe to assume at most one wrap: callers here only ever scan a
  // range bounded by max_tasks (see publish_tasks()'s capacity wait, which
  // keeps m_total_tasks - m_collected_count <= max_tasks at all times).
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

  // --- Task publish (manager side) / read (claimant side) ---

  // Two round trips regardless of batch size: one bulk Put for the whole
  // range's [count][data] slots, then one atomic_set of TOTAL_OFF. A
  // per-task publish measured ~18K tasks/s against the ~900K/s raw ceiling
  // in rma_atomic_microbench.
  //
  // TOTAL_OFF is bumped last, once every task's data is in place, so a
  // claimant can never see an index whose payload is still in flight.
  void publish_tasks(std::span<const TaskT> tasks) {
    if (tasks.empty()) return;
    const int64_t count = static_cast<int64_t>(tasks.size());
    // A single call can never publish more than the ring holds, however much
    // harvesting happens -- check with a hypothetical start of 0 purely to
    // test that absolute bound (real indices grow unboundedly with the ring,
    // so checking against the true start would spuriously fail once it
    // first exceeds max_tasks). The capacity wait below handles "fits in the
    // ring, just not free yet."
    detail::check_task_capacity(0, tasks.size(), m_config.max_tasks, "LockFreeRMA");

    // Block until the ring has room for the whole batch. Waiting for a
    // partial batch to free up mid-publish would need a second start index,
    // complicating the single-Put-per-batch design below for no benefit --
    // this only spins when the caller publishes faster than the ring
    // drains, an unusual calling pattern to begin with.
    while (count > static_cast<int64_t>(m_config.max_tasks) -
                       (m_total_tasks - static_cast<int64_t>(m_collected_count))) {
      if (num_workers() == 0) {
        run_one_task_locally();
      } else {
        const size_t before = m_collected_count;
        harvest_ready_results();
        if (m_collected_count == before) detail::rma_wait_idle(m_window, m_comm.get());
      }
    }

    const int64_t start = m_total_tasks;
    if (num_workers() == 0) {
      m_task_store.insert(m_task_store.end(), tasks.begin(), tasks.end());
      m_total_tasks += count;
      return;
    }
    std::vector<std::byte> buffer(tasks.size() * m_task_slot_stride);
    for (size_t i = 0; i < tasks.size(); ++i) {
      const TaskT& task = tasks[i];
      const int elem_count = MPI_Type<TaskT>::count(task);
      assert(static_cast<size_t>(elem_count) <= m_max_task_count &&
             "LockFreeRMA: task exceeds max_task_count");
      const size_t data_bytes = static_cast<size_t>(elem_count) * m_task_elem;
      const size_t off = i * m_task_slot_stride;
      detail::write_i64(buffer.data(), buffer.size(), off + T_COUNT, elem_count);
      if (data_bytes > 0) {
        detail::write_bytes(buffer.data(), buffer.size(), off + T_DATA, MPI_Type<TaskT>::ptr(task),
                            data_bytes);
      }
    }
    // The ring can wrap mid-batch (task_slot() is only contiguous within one
    // lap), so a batch crossing the boundary needs two Puts instead of one --
    // both flush together with the atomic_set below, so this costs no extra
    // round trip in the common (non-wrapping) case and only one extra Put
    // issuance in the wrapping one.
    const int64_t ring = static_cast<int64_t>(m_config.max_tasks);
    const int64_t first_count = std::min(count, ring - ring_slot(start));
    post_put_bytes(buffer.data(), static_cast<size_t>(first_count) * m_task_slot_stride,
                   task_slot(start));
    if (first_count < count) {
      post_put_bytes(buffer.data() + static_cast<size_t>(first_count) * m_task_slot_stride,
                     static_cast<size_t>(count - first_count) * m_task_slot_stride,
                     task_slot(start + first_count));
    }
    // Reused slots (every index past the first lap) still hold whatever
    // completion-log entry their previous occupant's write_result_range()
    // left behind. Left uncleared, harvest_ready_results()'s scan would read
    // that stale non-zero entry and treat the brand new index as already
    // complete before this lap's worker has even claimed it -- confirmed as
    // a real bug, not a theoretical one (a stress test with a small ring
    // reproduced exactly this: index i's "result" coming back as index
    // (i - max_tasks)'s, the value the harvester picked up from the
    // not-yet-cleared log slot's leftover entry). Zeroing here, ahead of the
    // TOTAL_OFF bump that makes these indices claimable, guarantees the log
    // entry a claimant's write_result_range() later sets is the only
    // non-zero value the harvester can ever see for this lap.
    std::vector<std::byte> log_clear(static_cast<size_t>(count) * LOG_ENTRY_BYTES, std::byte{0});
    post_put_bytes(log_clear.data(), static_cast<size_t>(first_count) * LOG_ENTRY_BYTES,
                   log_slot(start));
    if (first_count < count) {
      post_put_bytes(log_clear.data() + static_cast<size_t>(first_count) * LOG_ENTRY_BYTES,
                     static_cast<size_t>(count - first_count) * LOG_ENTRY_BYTES,
                     log_slot(start + first_count));
    }
    m_total_tasks += count;
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
    if (slot >= kMaxRecordedErrors) return;
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

  // Owner-only. Three round trips however many claims are ready: read
  // HEAD_OFF, bulk-read the log over the unscanned range, bulk-read the
  // results over whatever contiguous prefix the log confirms. A gap ends
  // the scan for this pass and is picked up on the next one.
  //
  // Uses the RMA API even though the target is this rank: under
  // MPI_WIN_SEPARATE a window owner is not guaranteed to see remote writes
  // through local loads. On MS-MPI these self-targeted Gets also need the
  // two-sided progress engine driven between polls (detail::rma_wait_idle);
  // flush_all alone leaves the loop spinning on an unchanging HEAD.
  void harvest_ready_results() {
    assert(is_root_manager());
    int64_t head_in = 0, head_now = 0;
    post_fetch_and_op(head_in, head_now, HEAD_OFF, MPI_NO_OP);
    flush_local();

    // A claimant can fetch_add HEAD_OFF for an index it hasn't actually been
    // published yet -- run_worker()'s "pending" path deliberately lets
    // claims outrun publication so a claimant doesn't have to wait for a
    // fresh TOTAL_OFF read before claiming. Before the ring, scanning that
    // far ahead was harmless (an unpublished slot's log entry was always
    // still zero, since nothing had ever touched it); with the ring, that
    // same slot may hold a stale non-zero entry left over from a prior lap
    // that this index hasn't reached yet -- confirmed as a real bug (a
    // small-ring stress test misread stale entries this way, several laps
    // behind the true completion). Capping at m_total_tasks means the scan
    // never reads a slot this rank hasn't itself published in the current
    // lap, so it can never see anything but that lap's own entry.
    const int64_t head_now_capped = std::min(head_now, m_total_tasks);

    const int64_t frontier = static_cast<int64_t>(m_collected_count);
    if (head_now_capped <= frontier) return;

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
    if (saw_error) harvest_task_errors();
    if (confirmed_end <= frontier) return;

    const int64_t n = confirmed_end - frontier;
    std::vector<std::byte> result_buf(static_cast<size_t>(n) * m_result_slot_stride);
    get_ring_bytes_local(result_buf.data(), frontier, n, m_result_slot_stride,
                         [this](int64_t i) { return result_slot(i); });

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
