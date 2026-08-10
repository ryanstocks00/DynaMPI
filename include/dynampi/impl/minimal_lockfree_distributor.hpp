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
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../mpi/mpi_communicator.hpp"
#include "../mpi/mpi_types.hpp"
#include "dynampi/mpi/mpi_error.hpp"
#include "dynampi/task_error.hpp"

namespace dynampi {

namespace detail {

inline void check_task_capacity(int64_t start, size_t count, int max_tasks,
                                const char* distributor_name) {
  const bool invalid = start < 0 || max_tasks < 0 ||
                       static_cast<uint64_t>(start) > static_cast<uint64_t>(max_tasks) ||
                       count > static_cast<uint64_t>(max_tasks) - static_cast<uint64_t>(start);
  if (invalid) {
    throw std::length_error(std::string(distributor_name) + ": exceeded max_tasks capacity");
  }
}

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
  const auto* in = static_cast<const std::byte*>(src);
  std::copy_n(in, nbytes, buffer + offset);
}

// Bounds-checked copy out of a sized source buffer into a sized destination.
// Uses std::copy_n rather than memcpy: Codacy/Flawfinder flags every memcpy as
// CWE-120 regardless of prior range checks or clamped lengths.
inline void read_bytes(void* dst, size_t dst_capacity, const std::byte* buffer, size_t buffer_size,
                       size_t offset, size_t nbytes) {
  if (nbytes == 0) return;
  constexpr size_t kMaxObjectSize = static_cast<size_t>(std::numeric_limits<std::ptrdiff_t>::max());
  if (nbytes > kMaxObjectSize || nbytes > dst_capacity || offset > buffer_size ||
      nbytes > buffer_size - offset) {
    DYNAMPI_FAIL("read_bytes out of range");  // LCOV_EXCL_LINE
  }
  const size_t copy_n = std::min(nbytes, dst_capacity);
  std::copy_n(buffer + offset, copy_n, static_cast<std::byte*>(dst));
}

inline int64_t read_i64(const std::byte* buffer, size_t buffer_size, size_t offset) {
  int64_t value{};
  read_bytes(&value, sizeof(value), buffer, buffer_size, offset, sizeof(int64_t));
  return value;
}

inline void write_i64(std::byte* buffer, size_t buffer_size, size_t offset, int64_t value) {
  write_bytes(buffer, buffer_size, offset, &value, sizeof(int64_t));
}

template <typename T>
inline void read_result_bytes(const std::byte* buffer, size_t buffer_size, size_t offset, T& value,
                              size_t data_bytes) {
  if (data_bytes == 0) return;
  if constexpr (MPI_Type<T>::resize_required) {
    read_bytes(MPI_Type<T>::ptr(value), data_bytes, buffer, buffer_size, offset, data_bytes);
  } else {
    assert(data_bytes == sizeof(T));
    read_bytes(&value, sizeof(T), buffer, buffer_size, offset, data_bytes);
  }
}

// Drive progress while a rank is spinning on one-sided completion.
//
// Under MPI_WIN_SEPARATE (MS-MPI always; some other stacks too), the
// two-sided progress engine must run while ranks wait on remote RMA state.
// Every LockFreeRMA primitive already flushes its target before returning,
// so another MPI_Win_flush_all here only makes idle ranks contend with useful
// traffic. MPI_Iprobe drives progress without adding that RMA contention.
inline void rma_wait_idle(MPI_Win /*window*/, MPI_Comm comm) {
  int flag = 0;
  DYNAMPI_MPI_CHECK(MPI_Iprobe, (MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE));
  // A yield alone can immediately reschedule oversubscribed ranks. With
  // MS-MPI that lets idle workers continuously flood the passive-target
  // window with synchronized polls, starving in-flight task/result RMA
  // indefinitely. Stagger ranks so they do not wake as a thundering herd.
#if defined(_WIN32)
  int rank = 0;
  DYNAMPI_MPI_CHECK(MPI_Comm_rank, (comm, &rank));
  std::this_thread::sleep_for(std::chrono::microseconds(100 + (rank % 32) * 100));
#else
  std::this_thread::sleep_for(std::chrono::microseconds(50));
#endif
}

}  // namespace detail

// ---------------------------------------------------------------------------
// MinimalLockFreeWorkDistributor
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
// LockFreeRMAWorkDistributor, one of the Hierarchical* distributors,
// or one of the message-based distributors.
//
// Usage (collective: every rank must call run() with the same n_tasks):
//   MinimalLockFreeWorkDistributor<double> dist([](size_t i){ return f(i); });
//   std::vector<double> results = dist.run(n);  // populated on the manager only
// ---------------------------------------------------------------------------
template <typename ResultT>
class MinimalLockFreeWorkDistributor {
 public:
  struct Config {
    MPI_Comm comm = MPI_COMM_WORLD;
    int manager_rank = 0;

    // If true (default), run() throws dynampi::TaskFailure on the manager once
    // a task has thrown -- after every collective in the run has completed, so
    // the other ranks are never left waiting on one that the manager skipped.
    // Set false to recover instead, via take_task_errors().
    bool rethrow_task_errors = true;
  };

  explicit MinimalLockFreeWorkDistributor(std::function<ResultT(size_t)> worker_function,
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
  }  // LCOV_EXCL_LINE -- GCC attributes this closing brace inconsistently for MPI constructors

  ~MinimalLockFreeWorkDistributor() {
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
        ResultT result;
        auto failure = detail::run_task_guarded(m_worker_function, static_cast<size_t>(i), result);
        if (failure) m_task_errors.record(TaskError{m_comm.rank(), std::move(*failure)});
        results.push_back(std::move(result));
      }
      m_task_errors.rethrow_first_if(m_config.rethrow_task_errors);
      return results;
    }

    if (is_root_manager()) set_counter(0);
    DYNAMPI_MPI_CHECK(MPI_Barrier, (m_comm.get()));  // reset visible + synchronized start

    std::vector<std::pair<int64_t, ResultT>> local;
    std::vector<TaskError> local_errors;
    while (true) {
      int64_t index = fetch_add(1);
      if (index >= static_cast<int64_t>(n)) break;
      // A failed task still contributes its (default-constructed) slot, so the
      // gathered output stays one result per index and the surviving results
      // keep their positions.
      ResultT result;
      auto failure =
          detail::run_task_guarded(m_worker_function, static_cast<size_t>(index), result);
      if (failure) local_errors.push_back(TaskError{m_comm.rank(), std::move(*failure)});
      local.emplace_back(index, std::move(result));
    }

    auto results = gather_sorted(local);
    // Collective, and deliberately after the result gather: every rank reaches
    // both, so the manager can throw below without stranding anyone.
    gather_task_errors(local_errors);
    m_task_errors.rethrow_first_if(m_config.rethrow_task_errors);
    return results;
  }

  // Tasks that threw, oldest first, removed as they are returned. Manager only;
  // empty on every other rank. See Config::rethrow_task_errors.
  [[nodiscard]] std::vector<TaskError> take_task_errors() { return m_task_errors.take(); }

 private:
  Config m_config;
  MPICommunicator<> m_comm;
  std::function<ResultT(size_t)> m_worker_function;
  MPI_Win m_window = MPI_WIN_NULL;
  int64_t m_counter = 0;               // window-exposed claim counter (manager only)
  detail::TaskErrorLog m_task_errors;  // manager only
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

  // Collects every rank's failures onto the manager. Records are packed as
  // [int64 rank][int64 length][chars] and moved with the same
  // Gather-then-Gatherv pair as the results; it runs once per run(), on a path
  // that is already collective, so the cost is irrelevant next to the gather it
  // follows.
  void gather_task_errors(const std::vector<TaskError>& local_errors) {
    const bool manager = is_root_manager();
    const int size = m_comm.size();

    std::vector<std::byte> send_buf;
    for (const auto& error : local_errors) {
      const size_t length = std::min(error.message.size(), kMaxTaskErrorMessage);
      const size_t offset = send_buf.size();
      send_buf.resize(offset + 16 + length);
      detail::write_i64(send_buf.data(), send_buf.size(), offset, error.worker_rank);
      detail::write_i64(send_buf.data(), send_buf.size(), offset + 8, static_cast<int64_t>(length));
      if (length > 0) {
        detail::write_bytes(send_buf.data(), send_buf.size(), offset + 16, error.message.data(),
                            length);
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

    if (!manager) return;
    size_t pos = 0;
    while (pos + 16 <= static_cast<size_t>(total_bytes)) {
      TaskError error;
      error.worker_rank = static_cast<int>(detail::read_i64(recv_buf.data(), recv_buf.size(), pos));
      const auto length =
          static_cast<size_t>(detail::read_i64(recv_buf.data(), recv_buf.size(), pos + 8));
      pos += 16;
      error.message.assign(reinterpret_cast<const char*>(recv_buf.data() + pos), length);
      pos += length;
      m_task_errors.record(std::move(error));
    }
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

}  // namespace dynampi
