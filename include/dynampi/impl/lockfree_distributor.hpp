/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/impl/base_distributor.hpp"
#include "dynampi/mpi/mpi_error.hpp"

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
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op,
                      (&in, &out, MPI_INT64_T, m_config.manager_rank, 0, MPI_REPLACE, m_window));
    DYNAMPI_MPI_CHECK(MPI_Win_flush, (m_config.manager_rank, m_window));
  }

  int64_t fetch_add(int64_t increment) {
    int64_t in = increment, out;
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op,
                      (&in, &out, MPI_INT64_T, m_config.manager_rank, 0, MPI_SUM, m_window));
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
      int64_t i64 = index, c64 = count;
      std::memcpy(send_buf.data() + offset, &i64, sizeof(i64));
      std::memcpy(send_buf.data() + offset + 8, &c64, sizeof(c64));
      if (data_bytes > 0) {
        std::memcpy(send_buf.data() + offset + 16, MPI_Type<ResultT>::ptr(result), data_bytes);
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
      int64_t index, count;
      std::memcpy(&index, recv_buf.data() + pos, sizeof(index));
      std::memcpy(&count, recv_buf.data() + pos + 8, sizeof(count));
      pos += 16;
      ResultT result{};
      if constexpr (MPI_Type<ResultT>::resize_required)
        MPI_Type<ResultT>::resize(result, static_cast<int>(count));
      assert(count >= 0);
      const size_t data_bytes =
          count > 0 ? static_cast<size_t>(count) * static_cast<size_t>(elem) : size_t{0};
      if (data_bytes > 0) {
        std::memcpy(MPI_Type<ResultT>::ptr(result), recv_buf.data() + pos, data_bytes);
      }
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
// A full manager-worker distributor that uses passive one-sided RMA only — no
// collective communication on the hot path, so it cannot deadlock from
// mismatched collective counts. Workers claim a task by atomically advancing a
// shared counter (compare-and-swap), read the real task payload from the
// manager's window, run it, and write the result back into a per-task result
// slot, publishing it with an atomic "ready" flag. The manager polls those
// slots and collects results in task order.
//
// Capacity (max_tasks, max_task_count, max_result_count) is fixed at
// construction; inserting more than max_tasks over the distributor's lifetime,
// or a task/result larger than the configured element count, asserts.
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

  // Control region (4 atomic int64 counters at the start of the manager window):
  //   head     — next task index, advanced by workers via compare-and-swap
  //   total    — number of tasks published by the manager
  //   done     — number of completed tasks
  //   finished — shutdown flag set by the manager
  static constexpr MPI_Aint HEAD_OFF = 0;
  static constexpr MPI_Aint TOTAL_OFF = 8;
  static constexpr MPI_Aint DONE_OFF = 16;
  static constexpr MPI_Aint FINISHED_OFF = 24;
  static constexpr size_t CONTROL_BYTES = 32;

  // Per-result-slot layout: [int64 ready][int64 worker_rank][int64 count][data].
  static constexpr size_t R_READY = 0;
  static constexpr size_t R_RANK = 8;
  static constexpr size_t R_COUNT = 16;
  static constexpr size_t R_DATA = 24;
  // Per-task-slot layout: [int64 count][data].
  static constexpr size_t T_COUNT = 0;
  static constexpr size_t T_DATA = 8;

 public:
  struct Statistics {
    CommStatistics comm_statistics;
    std::vector<size_t> worker_task_counts;
  };
  using StatisticsT =
      std::conditional_t<statistics_mode == StatisticsMode::Detailed, Statistics, std::monostate>;

  explicit LockFreeMPIWorkDistributor(std::function<ResultT(TaskT)> worker_function,
                                      Config config = {})
      : m_config(config),
        m_comm(config.comm, Comm::Duplicate),
        m_worker_function(std::move(worker_function)),
        m_statistics{make_statistics()} {
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

        poll_results();
        detail::rma_wait_idle(m_window);
      }
    }

    size_t limit = config.allow_more_than_target_tasks ? std::numeric_limits<size_t>::max()
                                                       : config.target_num_tasks;
    return drain_results(limit);
  }

  [[nodiscard]] std::vector<ResultT> finish_remaining_tasks() { return run_tasks({}); }

  void finalize() {
    assert(!m_finalized);
    if (is_root_manager()) {
      if (num_workers() == 0) {
        while (m_collected_count < static_cast<size_t>(m_total_tasks)) run_one_task_locally();
      } else {
        while (m_collected_count < static_cast<size_t>(m_total_tasks)) {
          poll_results();
          detail::rma_wait_idle(m_window);
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
    const int my_rank = m_comm.rank();

    while (true) {
      const int64_t total = atomic_read(TOTAL_OFF);
      const int64_t head = atomic_read(HEAD_OFF);

      if (head < total) {
        // Claim this index with a compare-and-swap so no index is ever skipped.
        if (compare_and_swap(HEAD_OFF, head, head + 1) != head) continue;
        const int64_t index = head;

        TaskT task = read_task(index);
        ResultT result = m_worker_function(std::move(task));
        publish_result(index, my_rank, result);
      } else if (atomic_read(FINISHED_OFF) != 0) {
        break;
      } else {
        detail::rma_wait_idle(m_window);
      }
    }
  }

 private:
  Config m_config;
  Comm m_comm;
  std::function<ResultT(TaskT)> m_worker_function;

  MPI_Win m_window = MPI_WIN_NULL;
  std::vector<std::byte> m_window_buffer;  // only allocated on the manager (with workers)
  bool m_finalized = false;

  // Layout, computed once in initialize_window().
  size_t m_task_elem = 0;
  size_t m_result_elem = 0;
  size_t m_max_task_count = 1;
  size_t m_max_result_count = 1;
  size_t m_task_slot_stride = 0;
  size_t m_result_slot_stride = 0;
  size_t m_task_base = 0;
  size_t m_result_base = 0;

  int64_t m_total_tasks = 0;        // tasks published so far
  size_t m_collected_count = 0;     // results pulled into m_results (in order) so far
  size_t m_returned_count = 0;      // results handed back to the caller so far
  std::vector<ResultT> m_results;   // contiguous, in task order, not yet returned
  std::vector<TaskT> m_task_store;  // only used when there are no workers

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
    m_result_slot_stride = detail::round_up_8(R_DATA + m_max_result_count * m_result_elem);

    const size_t capacity = static_cast<size_t>(m_config.max_tasks);
    m_task_base = CONTROL_BYTES;
    m_result_base = m_task_base + capacity * m_task_slot_stride;
    const size_t window_bytes = m_result_base + capacity * m_result_slot_stride;

    const bool need_buffer = is_root_manager() && num_workers() > 0;
    if (need_buffer) {
      m_window_buffer.resize(window_bytes);
    } else if (!is_root_manager()) {
      // MS-MPI deadlocks on zero-length passive-target windows during lock_all.
      m_window_buffer.assign(1, std::byte{0});
    }

    if (is_root_manager() && num_workers() == 0) {
      // Manager-only (no workers): tasks run locally; MPI rejects zero-size windows.
      return;
    }

    void* base = m_window_buffer.empty() ? nullptr : m_window_buffer.data();
    const MPI_Aint bsize = static_cast<MPI_Aint>(m_window_buffer.size());
    DYNAMPI_MPI_CHECK(MPI_Win_create, (base, bsize, 1, MPI_INFO_NULL, m_comm.get(), &m_window));
    DYNAMPI_MPI_CHECK(MPI_Win_lock_all, (MPI_MODE_NOCHECK, m_window));
  }

  MPI_Aint task_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_task_base + static_cast<size_t>(index) * m_task_slot_stride);
  }
  MPI_Aint result_slot(int64_t index) const {
    return static_cast<MPI_Aint>(m_result_base + static_cast<size_t>(index) * m_result_slot_stride);
  }

  size_t available() const { return m_results.size(); }

  // --- RMA primitives (all target the manager) ---

  void flush() { DYNAMPI_MPI_CHECK(MPI_Win_flush, (m_config.manager_rank, m_window)); }

  int64_t atomic_fetch_add(MPI_Aint offset, int64_t increment) {
    int64_t in = increment, out;
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op,
                      (&in, &out, MPI_INT64_T, m_config.manager_rank, offset, MPI_SUM, m_window));
    flush();
    return out;
  }

  int64_t atomic_read(MPI_Aint offset) {
    int64_t in = 0, out;
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op,
                      (&in, &out, MPI_INT64_T, m_config.manager_rank, offset, MPI_NO_OP, m_window));
    flush();
    return out;
  }

  void atomic_set(MPI_Aint offset, int64_t value) {
    int64_t in = value, out;
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op, (&in, &out, MPI_INT64_T, m_config.manager_rank, offset,
                                         MPI_REPLACE, m_window));
    flush();
  }

  int64_t compare_and_swap(MPI_Aint offset, int64_t expected, int64_t desired) {
    int64_t comp = expected, des = desired, out;
    DYNAMPI_MPI_CHECK(MPI_Compare_and_swap,
                      (&des, &comp, &out, MPI_INT64_T, m_config.manager_rank, offset, m_window));
    flush();
    return out;
  }

  void put_bytes(const void* src, size_t n, MPI_Aint offset) {
    DYNAMPI_MPI_CHECK(MPI_Put, (src, static_cast<int>(n), MPI_BYTE, m_config.manager_rank, offset,
                                static_cast<int>(n), MPI_BYTE, m_window));
    flush();
  }

  void get_bytes(void* dst, size_t n, MPI_Aint offset) {
    if (n == 0) return;
    DYNAMPI_MPI_CHECK(MPI_Get, (dst, static_cast<int>(n), MPI_BYTE, m_config.manager_rank, offset,
                                static_cast<int>(n), MPI_BYTE, m_window));
    flush();
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
    assert(static_cast<size_t>(count) <= m_max_task_count &&
           "LockFree: task exceeds max_task_count");
    const size_t data_bytes = static_cast<size_t>(count) * m_task_elem;

    std::vector<std::byte> buffer(T_DATA + data_bytes);
    int64_t count64 = count;
    std::memcpy(buffer.data() + T_COUNT, &count64, sizeof(count64));
    if (count > 0) std::memcpy(buffer.data() + T_DATA, MPI_Type<TaskT>::ptr(task), data_bytes);
    put_bytes(buffer.data(), buffer.size(), task_slot(index));

    if constexpr (statistics_mode != StatisticsMode::None) {
      m_statistics.comm_statistics.bytes_sent += data_bytes;
      m_statistics.comm_statistics.send_count++;
    }

    m_total_tasks++;
    atomic_set(TOTAL_OFF, m_total_tasks);  // publish to workers
  }

  TaskT read_task(int64_t index) {
    int64_t count = 0;
    get_bytes(&count, 8, task_slot(index) + static_cast<MPI_Aint>(T_COUNT));
    TaskT task{};
    if constexpr (MPI_Type<TaskT>::resize_required)
      MPI_Type<TaskT>::resize(task, static_cast<int>(count));
    get_bytes(MPI_Type<TaskT>::ptr(task), static_cast<size_t>(count) * m_task_elem,
              task_slot(index) + static_cast<MPI_Aint>(T_DATA));
    return task;
  }

  void publish_result(int64_t index, int worker_rank, const ResultT& result) {
    const int count = MPI_Type<ResultT>::count(result);
    assert(static_cast<size_t>(count) <= m_max_result_count &&
           "LockFree: result exceeds max_result_count");
    const size_t data_bytes = static_cast<size_t>(count) * m_result_elem;

    // [int64 worker_rank][int64 count][data], written before the ready flag.
    std::vector<std::byte> buffer(16 + data_bytes);
    int64_t rank64 = worker_rank, count64 = count;
    std::memcpy(buffer.data(), &rank64, sizeof(rank64));
    std::memcpy(buffer.data() + 8, &count64, sizeof(count64));
    if (count > 0) std::memcpy(buffer.data() + 16, MPI_Type<ResultT>::ptr(result), data_bytes);
    put_bytes(buffer.data(), buffer.size(), result_slot(index) + static_cast<MPI_Aint>(R_RANK));

    atomic_set(result_slot(index) + static_cast<MPI_Aint>(R_READY), 1);  // publish
    atomic_fetch_add(DONE_OFF, 1);
  }

  // --- Result collection (manager) ---

  void poll_results() {
    while (m_collected_count < static_cast<size_t>(m_total_tasks)) {
      const MPI_Aint slot = result_slot(static_cast<int64_t>(m_collected_count));
      if (atomic_read(slot + static_cast<MPI_Aint>(R_READY)) == 0) break;

      int64_t header[2];  // [worker_rank][count]
      get_bytes(header, 16, slot + static_cast<MPI_Aint>(R_RANK));
      const int64_t worker_rank = header[0];
      const int64_t count = header[1];

      ResultT result{};
      if constexpr (MPI_Type<ResultT>::resize_required)
        MPI_Type<ResultT>::resize(result, static_cast<int>(count));
      get_bytes(MPI_Type<ResultT>::ptr(result), static_cast<size_t>(count) * m_result_elem,
                slot + static_cast<MPI_Aint>(R_DATA));
      m_results.push_back(std::move(result));

      if constexpr (statistics_mode != StatisticsMode::None) {
        m_statistics.comm_statistics.bytes_received += static_cast<size_t>(count) * m_result_elem;
        m_statistics.comm_statistics.recv_count++;
      }
      if constexpr (statistics_mode >= StatisticsMode::Aggregated) {
        if (static_cast<size_t>(worker_rank) < m_statistics.worker_task_counts.size())
          m_statistics.worker_task_counts[static_cast<size_t>(worker_rank)]++;
      }

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

  static StatisticsT make_statistics() {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{};
    } else {
      return {};
    }
  }
};

}  // namespace dynampi
