

# File lockfree\_distributor.hpp

[**File List**](files.md) **>** [**dynampi**](dir_f8db417ebd5c3d89eea80c24e3fd4369.md) **>** [**impl**](dir_87365955bfd9c1869b83a1fbd3fdb221.md) **>** [**lockfree\_distributor.hpp**](lockfree__distributor_8hpp.md)

[Go to the documentation of this file](lockfree__distributor_8hpp.md)


```C++
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

inline void write_bytes(std::byte* buffer, [[maybe_unused]] size_t buffer_size, size_t offset,
                        const void* src, size_t nbytes) {
  if (nbytes == 0) return;
  assert(offset <= buffer_size && nbytes <= buffer_size - offset);
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
inline void read_result_bytes(const std::byte* buffer, [[maybe_unused]] size_t buffer_size,
                              size_t offset, T& value, size_t data_bytes) {
  if (data_bytes == 0) return;
  assert(offset <= buffer_size && data_bytes <= buffer_size - offset);
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

        try_gather_results();
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

    while (true) {
      maybe_participate_in_gather();

      const int64_t total = atomic_read(TOTAL_OFF);
      const int64_t head = atomic_read(HEAD_OFF);

      if (head < total) {
        // Claim this index with a compare-and-swap so no index is ever skipped.
        if (compare_and_swap(HEAD_OFF, head, head + 1) != head) continue;
        const int64_t index = head;

        TaskT task = read_task(index);
        ResultT result = m_worker_function(std::move(task));
        store_result(index, std::move(result));
      } else if (atomic_read(FINISHED_OFF) != 0) {
        break;
      } else {
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
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op,
                      (&in, &out, MPI_INT64_T, m_config.manager_rank, offset, MPI_NO_OP, m_window));
    flush(m_config.manager_rank);
    return out;
  }

  void atomic_set(MPI_Aint offset, int64_t value) {
    int64_t in = value, out;
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op, (&in, &out, MPI_INT64_T, m_config.manager_rank, offset,
                                         MPI_REPLACE, m_window));
    flush(m_config.manager_rank);
  }

  int64_t compare_and_swap(MPI_Aint offset, int64_t expected, int64_t desired) {
    int64_t comp = expected, des = desired, out;
    DYNAMPI_MPI_CHECK(MPI_Compare_and_swap,
                      (&des, &comp, &out, MPI_INT64_T, m_config.manager_rank, offset, m_window));
    flush(m_config.manager_rank);
    return out;
  }

  void put_bytes(const void* src, size_t n, MPI_Aint offset) {
    DYNAMPI_MPI_CHECK(MPI_Put, (src, static_cast<int>(n), MPI_BYTE, m_config.manager_rank, offset,
                                static_cast<int>(n), MPI_BYTE, m_window));
    flush(m_config.manager_rank);
  }

  // Workers read the manager's window.
  void get_bytes(void* dst, size_t n, MPI_Aint offset) {
    if (n == 0) return;
    DYNAMPI_MPI_CHECK(MPI_Get, (dst, static_cast<int>(n), MPI_BYTE, m_config.manager_rank, offset,
                                static_cast<int>(n), MPI_BYTE, m_window));
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
    assert(static_cast<size_t>(count) <= m_max_task_count &&
           "LockFree: task exceeds max_task_count");
    const size_t data_bytes = static_cast<size_t>(count) * m_task_elem;

    std::vector<std::byte> buffer(T_DATA + data_bytes);
    detail::write_i64(buffer.data(), buffer.size(), T_COUNT, count);
    if (count > 0) {
      detail::write_bytes(buffer.data(), buffer.size(), T_DATA, MPI_Type<TaskT>::ptr(task),
                          data_bytes);
    }
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

        if constexpr (statistics_mode != StatisticsMode::None) {
          m_statistics.comm_statistics.bytes_received += data_bytes;
          m_statistics.comm_statistics.recv_count++;
        }
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

  static StatisticsT make_statistics() {
    if constexpr (statistics_mode != StatisticsMode::None) {
      return Statistics{};
    } else {
      return {};
    }
  }
};

}  // namespace dynampi
```


