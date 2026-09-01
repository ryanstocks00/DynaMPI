/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Byte-packing, capacity-checking and idle-wait helpers shared by the
// passive-target RMA distributors (LockFreeRMAWorkDistributor and
// HierarchicalLockFreeRMAWorkDistributor). Both lay their windows out as flat
// byte buffers with 8-byte-aligned records, so the pack/unpack and bounds
// logic is identical between them and lives here rather than in either one.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

// Declared directly; <windows.h>'s min/max/ERROR macros collide with the
// distributor headers that include this one.
#if defined(_WIN32) && defined(_MSC_VER)
extern "C" __declspec(dllimport) unsigned int __stdcall timeBeginPeriod(unsigned int);
#pragma comment(lib, "winmm")
#endif

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

#if defined(__GNUC__)
[[gnu::noinline]]
#endif
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

inline void backoff_sleep(std::chrono::microseconds duration) {
#if defined(_WIN32) && defined(_MSC_VER)
  // MSVC's sleep_for rounds any wait up to the global timer tick (~15.6 ms by
  // default), so every rma_wait_idle backoff was ~300x too long: the MS-MPI
  // ctest run took ~350 s. Raise the resolution to 1 ms once. It can't go
  // lower: below ~1 ms, whether sleeping or spinning, the idle poll loop
  // starves MS-MPI's passive-target progress and livelocks at >= 4 ranks.
  [[maybe_unused]] static const auto timer_res = ::timeBeginPeriod(1);
#endif
  std::this_thread::sleep_for(duration);
}

// Drive progress while a rank spins on one-sided completion. Under
// MPI_WIN_SEPARATE (MS-MPI always) the two-sided engine must run for remote RMA
// state to advance; primitives already flush their target, so MPI_Iprobe drives
// progress here without the extra contention of another MPI_Win_flush_all.
inline void rma_wait_idle(MPI_Win /*window*/, MPI_Comm comm) {
  int flag = 0;
  DYNAMPI_MPI_CHECK(MPI_Iprobe, (MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE));
  backoff_sleep(std::chrono::microseconds(50));
}

}  // namespace detail

}  // namespace dynampi
