/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

#include "dynampi/mpi/mpi_types.hpp"

namespace dynampi::detail {

// MPI_Type<std::vector<T>> (see mpi_types.hpp) assumes T is a fixed-size
// value: it ships the outer vector as `count() = size() * elements_per_T`
// contiguous datatype elements, with elements_per_T a single compile-time
// constant. That model has no way to describe a *batch of variable-length T*
// (e.g. TaskT/ResultT = std::vector<std::byte>, one serialized message per
// task): each element can have a different byte length, and MPI_Type<T>::ptr
// for such a T doesn't even point at contiguous element storage across the
// batch (it's a vector of separately heap-allocated std::vectors). Using the
// generic path for that case silently sends/receives raw std::vector control
// blocks (pointer/size/capacity) as if they were payload bytes -- garbage
// across process boundaries, and exactly the kind of corruption that shows
// up later as a bad-free when the receiving vector's destructor runs on
// those bogus pointers.
//
// So the hierarchical distributors' TASK_BATCH/RESULT_BATCH transfer (see
// hierarchical_distributor.hpp / hierarchical_nonblocking_distributor.hpp)
// packs such batches manually instead, as a flat std::vector<std::byte>
// (which *is* representable via the fixed-size path) with a length-prefixed
// layout:
//   [uint64_t n_items]
//   [uint64_t byte_len_0] .. [uint64_t byte_len_{n-1}]
//   [bytes of item 0] .. [bytes of item {n-1}]

template <typename T>
std::vector<std::byte> pack_variable_batch(const std::vector<T>& items) {
  using ItemType = MPI_Type<T>;
  int elem_bytes = 0;
  MPI_Type_size(ItemType::value, &elem_bytes);

  const uint64_t n_items = items.size();
  std::vector<uint64_t> byte_lens(n_items);
  uint64_t total_bytes = 0;
  for (uint64_t i = 0; i < n_items; ++i) {
    const uint64_t len =
        static_cast<uint64_t>(ItemType::count(items[i])) * static_cast<uint64_t>(elem_bytes);
    byte_lens[i] = len;
    total_bytes += len;
  }

  std::vector<std::byte> buf(sizeof(uint64_t) * (1 + n_items) + total_bytes);
  size_t offset = 0;
  std::memcpy(buf.data() + offset, &n_items, sizeof(uint64_t));
  offset += sizeof(uint64_t);
  if (n_items > 0) {
    std::memcpy(buf.data() + offset, byte_lens.data(), sizeof(uint64_t) * n_items);
  }
  offset += sizeof(uint64_t) * n_items;
  for (uint64_t i = 0; i < n_items; ++i) {
    if (byte_lens[i] > 0) {
      std::memcpy(buf.data() + offset, ItemType::ptr(items[i]), byte_lens[i]);
    }
    offset += byte_lens[i];
  }
  return buf;
}

template <typename T>
std::vector<T> unpack_variable_batch(const std::vector<std::byte>& buf) {
  using ItemType = MPI_Type<T>;
  int elem_bytes = 0;
  MPI_Type_size(ItemType::value, &elem_bytes);

  size_t offset = 0;
  uint64_t n_items = 0;
  std::memcpy(&n_items, buf.data() + offset, sizeof(uint64_t));
  offset += sizeof(uint64_t);

  std::vector<uint64_t> byte_lens(n_items);
  if (n_items > 0) {
    std::memcpy(byte_lens.data(), buf.data() + offset, sizeof(uint64_t) * n_items);
  }
  offset += sizeof(uint64_t) * n_items;

  std::vector<T> items(n_items);
  for (uint64_t i = 0; i < n_items; ++i) {
    const uint64_t len = byte_lens[i];
    const int count_in_elements =
        elem_bytes > 0 ? static_cast<int>(len / static_cast<uint64_t>(elem_bytes)) : 0;
    ItemType::resize(items[i], count_in_elements);
    if (len > 0) {
      std::memcpy(ItemType::ptr(items[i]), buf.data() + offset, len);
    }
    offset += len;
  }
  return items;
}

}  // namespace dynampi::detail
