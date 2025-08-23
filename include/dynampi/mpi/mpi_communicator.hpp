/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#include <optional>
#include <variant>

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
  size_t bytes_sent = 0;
  size_t bytes_received = 0;
  double send_time = 0.0;
  double recv_time = 0.0;

  void reset() {
    send_count = 0;
    recv_count = 0;
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
  MPI_Comm _comm;
  Ownership _ownership;

  static constexpr StatisticsMode statistics_mode =
      get_option_value<track_statistics_t, Options...>();
  using StatisticsT =
      std::conditional_t<statistics_mode != StatisticsMode::None, CommStatistics, std::monostate>;

  StatisticsT _statistics;

 public:
  MPICommunicator(MPI_Comm comm, Ownership ownership = Duplicate)
      : _comm(comm), _ownership(ownership) {
    if (_ownership == Duplicate) {
      DYNAMPI_MPI_CHECK(MPI_Comm_dup, (comm, &_comm));
    }
  }

  MPICommunicator(const MPICommunicator& other) = delete;
  MPICommunicator& operator=(const MPICommunicator& other) = delete;
  MPICommunicator(MPICommunicator&& other) noexcept
      : _comm(other._comm),
        _ownership(other._ownership),
        _statistics(std::move(other._statistics)) {
    other._comm = MPI_COMM_NULL;
    other._ownership = Reference;
  }
  MPICommunicator& operator=(MPICommunicator&& other) = delete;

  ~MPICommunicator() {
    if (_ownership != Reference) {
      MPI_Comm_free(&_comm);
    }
  }

  MPICommunicator split_by_node() const {
    MPI_Comm node_comm;
    DYNAMPI_MPI_CHECK(MPI_Comm_split_type,
                      (_comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm));
    return MPICommunicator(node_comm, Move);
  }

  std::optional<MPICommunicator> split(int color, int key = 0) const {
    MPI_Comm new_comm;
    DYNAMPI_MPI_CHECK(MPI_Comm_split, (_comm, color, key, &new_comm));
    if (new_comm == MPI_COMM_NULL) {
      return std::nullopt;
    }
    assert(color != MPI_UNDEFINED && "Undefined color should not result in a valid communicator");
    return MPICommunicator(new_comm, Move);
  }

  operator MPI_Comm() const { return _comm; }

  const CommStatistics& get_statistics() const
    requires(statistics_mode != StatisticsMode::None)
  {
    return _statistics;
  }

  int rank() const {
    int rank;
    DYNAMPI_MPI_CHECK(MPI_Comm_rank, (_comm, &rank));
    return rank;
  }

  int size() const {
    int size;
    DYNAMPI_MPI_CHECK(MPI_Comm_size, (_comm, &size));
    return size;
  }

  template <typename T>
  inline void send(const T& data, int dest, int tag = 0) {
    using mpi_type = MPI_Type<T>;
    DYNAMPI_MPI_CHECK(
        MPI_Send, (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value, dest, tag, _comm));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.send_count++;
      int size;
      MPI_Type_size(mpi_type::value, &size);
      _statistics.bytes_sent += mpi_type::count(data) * size;
    }
  }

  inline MPI_Status probe(int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) {
    MPI_Status status;
    DYNAMPI_MPI_CHECK(MPI_Probe, (source, tag, _comm, &status));
    return status;
  }

  template <typename T>
  inline void recv(T& data, int source, int tag = 0) {
    using mpi_type = MPI_Type<T>;
    DYNAMPI_MPI_CHECK(MPI_Recv, (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value,
                                 source, tag, _comm, MPI_STATUS_IGNORE));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.recv_count++;
      int size;
      MPI_Type_size(mpi_type::value, &size);
      _statistics.bytes_received += mpi_type::count(data) * size;
    }
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
                      (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value, root, _comm));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.collective_count++;
    }
  }

  inline void recv_empty_message(int source, int tag = 0) {
    using mpi_type = MPI_Type<std::nullptr_t>;
    DYNAMPI_MPI_CHECK(MPI_Recv, (nullptr, mpi_type::count(nullptr), mpi_type::value, source, tag,
                                 _comm, MPI_STATUS_IGNORE));
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
                                   mpi_type::count(data), mpi_type::value, root, _comm));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.collective_count++;
    }
  }

  [[nodiscard]] MPI_Comm get() const { return _comm; }
};

}  // namespace dynampi
