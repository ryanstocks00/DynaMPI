/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>
#include <variant>

#include "dynampi/mpi/mpi_group.hpp"
#include "dynampi/mpi/mpi_types.hpp"
#include "dynampi/utilities/assert.hpp"
#include "dynampi/utilities/template_options.hpp"
#include "mpi_error.hpp"

namespace dynampi {

enum class StatisticsMode {
  None,
  Aggregated,
  Detailed,
};

struct track_statistics_t {
  static constexpr StatisticsMode value = StatisticsMode::None;
};

template <StatisticsMode Mode = StatisticsMode::Detailed>
struct track_statistics : public track_statistics_t {
  static constexpr StatisticsMode value = Mode;
};

struct CommStatistics {
  int send_count = 0;
  int recv_count = 0;
  int collective_count = 0;
  int atomic_count = 0;
  size_t bytes_sent = 0;
  size_t bytes_received = 0;
  size_t atomic_bytes = 0;
  double send_time = 0.0;
  double recv_time = 0.0;

  void reset() {
    send_count = 0;
    recv_count = 0;
    collective_count = 0;
    atomic_count = 0;
    bytes_sent = 0;
    bytes_received = 0;
    atomic_bytes = 0;
    send_time = 0.0;
    recv_time = 0.0;
  }

  double average_send_size() const {
    if (send_count == 0) return 0.0;
    return static_cast<double>(bytes_sent) / send_count;
  }

  double average_receive_size() const {
    if (recv_count == 0) return 0.0;
    return static_cast<double>(bytes_received) / recv_count;
  }
};

template <typename... Options>
class MPICommunicator {
 public:
  enum Ownership {
    Reference,  // The communicator is not owned by this class and should not be freed.
    Move,       // The communicator is moved into this class and will be freed in the destructor.
    Duplicate,  // The communicator is duplicated by this class and will be freed in the destructor.
  };

 private:
  MPI_Comm m_comm;
  Ownership m_ownership;

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();
  using StatisticsT =
      std::conditional_t<statistics_mode != StatisticsMode::None, CommStatistics, std::monostate>;

  StatisticsT _statistics;

 public:
  MPICommunicator(MPI_Comm comm, Ownership ownership = Duplicate)
      : m_comm(comm), m_ownership(ownership) {
    if (m_ownership == Duplicate) {
      DYNAMPI_MPI_CHECK(MPI_Comm_dup, (comm, &m_comm));
    }
  }

  MPICommunicator(const MPICommunicator& other) = delete;
  MPICommunicator& operator=(const MPICommunicator& other) = delete;
  MPICommunicator(MPICommunicator&& other) noexcept
      : m_comm(other.m_comm),
        m_ownership(other.m_ownership),
        _statistics(std::move(other._statistics)) {
    other.m_comm = MPI_COMM_NULL;
    other.m_ownership = Reference;
  }
  MPICommunicator& operator=(MPICommunicator&& other) = delete;

  ~MPICommunicator() {
    if (m_ownership != Reference) {
      MPI_Comm_free(&m_comm);
    }
  }

  MPICommunicator split_by_node() const {
    MPI_Comm node_comm;
    DYNAMPI_MPI_CHECK(MPI_Comm_split_type,
                      (m_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm));
    return MPICommunicator(node_comm, Move);
  }

  std::optional<MPICommunicator> split(int color, int key = 0) const {
    MPI_Comm new_comm;
    DYNAMPI_MPI_CHECK(MPI_Comm_split, (m_comm, color, key, &new_comm));
    if (new_comm == MPI_COMM_NULL) {
      return std::nullopt;
    }
    assert(color != MPI_UNDEFINED && "Undefined color should not result in a valid communicator");
    return MPICommunicator(new_comm, Move);
  }

  operator MPI_Comm() const { return m_comm; }

  const CommStatistics& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    return _statistics;
  }

  int rank() const {
    int rank;
    DYNAMPI_MPI_CHECK(MPI_Comm_rank, (m_comm, &rank));
    return rank;
  }

  int size() const {
    int size;
    DYNAMPI_MPI_CHECK(MPI_Comm_size, (m_comm, &size));
    return size;
  }

  template <typename T>
  inline void send(const T& data, int dest, int tag = 0) {
    using mpi_type = MPI_Type<T>;
    DYNAMPI_MPI_CHECK(
        MPI_Send, (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value, dest, tag, m_comm));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.send_count++;
      int size;
      MPI_Type_size(mpi_type::value, &size);
      _statistics.bytes_sent += mpi_type::count(data) * size;
    }
  }

  // Non-blocking counterpart of send(): posts MPI_Isend and records the same
  // statistics immediately (byte counts are known at post time). Caller owns
  // the request and must keep `data` alive until the request completes.
  template <typename T>
  inline void isend(const T& data, int dest, int tag, MPI_Request* request) {
    using mpi_type = MPI_Type<T>;
    DYNAMPI_MPI_CHECK(MPI_Isend, (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value, dest,
                                  tag, m_comm, request));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.send_count++;
      int size;
      MPI_Type_size(mpi_type::value, &size);
      _statistics.bytes_sent += mpi_type::count(data) * size;
    }
  }

  inline MPI_Status probe(int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) {
    MPI_Status status;
    DYNAMPI_MPI_CHECK(MPI_Probe, (source, tag, m_comm, &status));
    return status;
  }

  inline std::optional<MPI_Status> iprobe(int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) {
    MPI_Status status;
    int flag;
    DYNAMPI_MPI_CHECK(MPI_Iprobe, (source, tag, m_comm, &flag, &status));
    if (flag) {
      return status;
    }
    return std::nullopt;
  }

  template <typename T>
  inline void recv(T& data, int source, int tag = 0) {
    using mpi_type = MPI_Type<T>;
    MPI_Status status;
    DYNAMPI_MPI_CHECK(MPI_Recv, (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value,
                                 source, tag, m_comm, &status));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.recv_count++;
      int actual_count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, mpi_type::value, &actual_count));
      int size;
      MPI_Type_size(mpi_type::value, &size);
      _statistics.bytes_received += actual_count * size;
    }
  }

  // Receive with MPI_ANY_SOURCE/MPI_ANY_TAG and return status
  template <typename T>
  inline MPI_Status recv_any(T& data, int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) {
    using mpi_type = MPI_Type<T>;
    MPI_Status status;
    DYNAMPI_MPI_CHECK(MPI_Recv, (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value,
                                 source, tag, m_comm, &status));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.recv_count++;
      int actual_count;
      DYNAMPI_MPI_CHECK(MPI_Get_count, (&status, mpi_type::value, &actual_count));
      int size;
      MPI_Type_size(mpi_type::value, &size);
      _statistics.bytes_received += actual_count * size;
    }
    return status;
  }

  template <typename T>
  inline void broadcast(T& data, int root = 0) {
    using mpi_type = MPI_Type<T>;
    if constexpr (mpi_type::resize_required) {
      int size = mpi_type::count(data);
      broadcast(size, root);
      if (rank() != root) {
        mpi_type::resize(data, size);
      }
    }
    DYNAMPI_MPI_CHECK(MPI_Bcast,
                      (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value, root, m_comm));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.collective_count++;
    }
  }

  inline void recv_empty_message(int source, int tag = 0) {
    using mpi_type = MPI_Type<std::nullptr_t>;
    DYNAMPI_MPI_CHECK(MPI_Recv, (nullptr, mpi_type::count(nullptr), mpi_type::value, source, tag,
                                 m_comm, MPI_STATUS_IGNORE));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.recv_count++;
    }
  }

  /// Non-blocking 0-byte send (Tag::DONE / Tag::REQUEST). Same statistics as
  /// send_empty; caller owns the request (may MPI_Request_free immediately).
  inline void isend_empty(int dest, int tag, MPI_Request* request) {
    using mpi_type = MPI_Type<std::nullptr_t>;
    DYNAMPI_MPI_CHECK(MPI_Isend, (nullptr, mpi_type::count(nullptr), mpi_type::value, dest, tag,
                                  m_comm, request));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.send_count++;
    }
  }

  /// Sends 0 elements of type T (same type as recv buffer) so that recv_any(T&) can receive any
  /// worker message (REQUEST or RESULT) into a single buffer type.
  template <typename T>
  inline void send_empty(int dest, int tag = 0) {
    using mpi_type = MPI_Type<T>;
    DYNAMPI_MPI_CHECK(MPI_Send, (nullptr, 0, mpi_type::value, dest, tag, m_comm));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.send_count++;
    }
  }

  /// Receives 0 elements of type T. Use when the sender used send_empty<T>.
  template <typename T>
  inline void recv_empty(int source, int tag = 0) {
    using mpi_type = MPI_Type<T>;
    DYNAMPI_MPI_CHECK(MPI_Recv,
                      (nullptr, 0, mpi_type::value, source, tag, m_comm, MPI_STATUS_IGNORE));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.recv_count++;
    }
  }

  template <typename T>
  inline void gather(const T& data, std::vector<T>* result, int root = 0) {
    DYNAMPI_ASSERT_EQ(result != nullptr, root == rank(),
                      "Gather result must be provided only on the root rank");
    using mpi_type = MPI_Type<T>;
    DYNAMPI_MPI_CHECK(MPI_Gather, (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value,
                                   result == nullptr ? nullptr : result->data(),
                                   mpi_type::count(data), mpi_type::value, root, m_comm));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.collective_count++;
    }
  }

  // --- One-sided RMA (window is owned by the caller) ---
  //
  // Displacements are byte offsets. The target rank's window must have
  // been created with disp_unit == 1 when using put_bytes/get_bytes.
  //
  // put_bytes/get_bytes chunk transfers larger than INT_MAX so MPI's plain
  // `int` count argument never silently wraps. Statistics count one logical
  // transfer (send_count/recv_count += 1, bytes = total n), not one entry
  // per chunk.

  static constexpr size_t kMaxRmaChunkBytes = static_cast<size_t>(std::numeric_limits<int>::max());

  inline void put_bytes(const void* src, size_t n, int target_rank, MPI_Aint target_disp,
                        MPI_Win win) {
    if (n == 0) return;
    const auto* bytes = static_cast<const std::byte*>(src);
    size_t done = 0;
    while (done < n) {
      const size_t chunk = std::min(kMaxRmaChunkBytes, n - done);
      DYNAMPI_MPI_CHECK(MPI_Put, (bytes + done, static_cast<int>(chunk), MPI_BYTE, target_rank,
                                  target_disp + static_cast<MPI_Aint>(done),
                                  static_cast<int>(chunk), MPI_BYTE, win));
      done += chunk;
    }
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.send_count++;
      _statistics.bytes_sent += n;
    }
  }

  inline void get_bytes(void* dst, size_t n, int target_rank, MPI_Aint target_disp, MPI_Win win) {
    if (n == 0) return;
    auto* bytes = static_cast<std::byte*>(dst);
    size_t done = 0;
    while (done < n) {
      const size_t chunk = std::min(kMaxRmaChunkBytes, n - done);
      DYNAMPI_MPI_CHECK(MPI_Get, (bytes + done, static_cast<int>(chunk), MPI_BYTE, target_rank,
                                  target_disp + static_cast<MPI_Aint>(done),
                                  static_cast<int>(chunk), MPI_BYTE, win));
      done += chunk;
    }
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.recv_count++;
      _statistics.bytes_received += n;
    }
  }

  // RMA atomic: counted separately from point-to-point send/recv.
  template <typename T>
  inline void fetch_and_op(T& origin, T& result, int target_rank, MPI_Aint target_disp, MPI_Op op,
                           MPI_Win win) {
    using mpi_type = MPI_Type<T>;
    static_assert(!mpi_type::resize_required, "fetch_and_op requires a fixed-size type");
    DYNAMPI_MPI_CHECK(MPI_Fetch_and_op, (mpi_type::ptr(origin), mpi_type::ptr(result),
                                         mpi_type::value, target_rank, target_disp, op, win));
    if constexpr (statistics_mode != StatisticsMode::None) {
      int size = 0;
      MPI_Type_size(mpi_type::value, &size);
      const size_t bytes = static_cast<size_t>(mpi_type::count(origin)) * static_cast<size_t>(size);
      _statistics.atomic_count++;
      _statistics.atomic_bytes += bytes;
    }
  }

  [[nodiscard]] MPI_Comm get() const { return m_comm; }

  // Get the group associated with this communicator
  [[nodiscard]] MPIGroup get_group() const { return MPIGroup(*this); }
};

}  // namespace dynampi
