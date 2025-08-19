/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#include <variant>

#include "dynampi/mpi/mpi_types.hpp"
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
    NotOwned,  // The communicator is not owned by this class and should not be freed.
    Owned,     // The communicator is owned by this class and will be freed in the destructor.
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
  MPICommunicator(MPI_Comm comm, Ownership ownership = NotOwned)
      : _comm(comm), _ownership(ownership) {
    if (_ownership == Owned) {
      DYNAMPI_MPI_CHECK(MPI_Comm_dup, (comm, &_comm));
    }
  }

  ~MPICommunicator() {
    if (_ownership == Owned) {
      MPI_Comm_free(&_comm);
    }
  }

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

  inline void recv_empty_message(int source, int tag = 0) {
    using mpi_type = MPI_Type<std::nullptr_t>;
    DYNAMPI_MPI_CHECK(MPI_Recv, (nullptr, mpi_type::count(nullptr), mpi_type::value, source, tag,
                                 _comm, MPI_STATUS_IGNORE));
    if constexpr (statistics_mode != StatisticsMode::None) {
      _statistics.recv_count++;
    }
  }

  [[nodiscard]] MPI_Comm get() const { return _comm; }
};

}  // namespace dynampi
