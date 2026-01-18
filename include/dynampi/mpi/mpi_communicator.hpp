/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include <optional>
#include <variant>

#include "dynampi/mpi/mpi_types.hpp"
#include "dynampi/utilities/assert.hpp"
#include "dynampi/utilities/debug_log.hpp"
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

  // Explicitly free the communicator (collective operation - all ranks must call together)
  void free() {
    int rank = -1;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && m_comm != MPI_COMM_NULL) {
      MPI_Comm_rank(m_comm, &rank);
    }
    if (mpi_initialized) {
      dynampi::get_debug_log() << "[MPI_COMM] free(): ENTRY rank=" << rank << " ownership="
                               << (m_ownership == Reference ? "Reference"
                                   : m_ownership == Move    ? "Move"
                                                            : "Duplicate")
                               << std::endl;
    }
    if (m_ownership != Reference && m_comm != MPI_COMM_NULL) {
      if (mpi_initialized) {
        if (mpi_initialized) {
          dynampi::get_debug_log()
              << "[MPI_COMM] free(): Calling MPI_Comm_free on rank " << rank << std::endl;
          dynampi::get_debug_log().flush();
        }
        DYNAMPI_MPI_CHECK(MPI_Comm_free, (&m_comm));
        m_comm = MPI_COMM_NULL;   // Mark as freed
        m_ownership = Reference;  // Mark as no longer owned
        if (mpi_initialized) {
          dynampi::get_debug_log()
              << "[MPI_COMM] free(): MPI_Comm_free completed on rank " << rank << std::endl;
        }
      }
    }
    if (mpi_initialized) {
      dynampi::get_debug_log() << "[MPI_COMM] free(): EXIT rank=" << rank << std::endl;
    }
  }

  ~MPICommunicator() {
    int rank = -1;
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized && m_comm != MPI_COMM_NULL) {
      // Only try to get rank if MPI is still initialized and communicator is valid
      MPI_Comm_rank(m_comm, &rank);
    }
    if (mpi_initialized) {
      dynampi::get_debug_log() << "[MPI_COMM] ~MPICommunicator(): ENTRY rank=" << rank
                               << " ownership="
                               << (m_ownership == Reference ? "Reference"
                                   : m_ownership == Move    ? "Move"
                                                            : "Duplicate")
                               << " comm_null=" << (m_comm == MPI_COMM_NULL) << std::endl;
    }
    // Only free in destructor if not already freed via free()
    // This allows explicit synchronization in finalize() before destruction
    if (m_ownership != Reference && m_comm != MPI_COMM_NULL) {
      // MPI_Comm_free is a collective operation - all ranks must call it together
      // If we reach here, free() was not called, so we try to free it
      // This is a fallback but may deadlock if ranks reach destructor at different times
      if (mpi_initialized) {
        if (mpi_initialized) {
          dynampi::get_debug_log()
              << "[MPI_COMM] ~MPICommunicator(): MPI initialized=" << mpi_initialized << " on rank "
              << rank << ", about to call MPI_Comm_free" << std::endl;
          dynampi::get_debug_log().flush();
        }
        // Ignore errors in destructor to avoid throwing exceptions during cleanup
        int free_result = MPI_Comm_free(&m_comm);
        (void)free_result;       // Suppress unused variable warning
        m_comm = MPI_COMM_NULL;  // Mark as freed
        if (mpi_initialized) {
          dynampi::get_debug_log()
              << "[MPI_COMM] ~MPICommunicator(): MPI_Comm_free completed on rank " << rank
              << std::endl;
        }
      }
    }
    if (mpi_initialized) {
      dynampi::get_debug_log() << "[MPI_COMM] ~MPICommunicator(): EXIT rank=" << rank << std::endl;
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

  inline MPI_Status probe(int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) {
    MPI_Status status;
    DYNAMPI_MPI_CHECK(MPI_Probe, (source, tag, m_comm, &status));
    return status;
  }

  template <typename T>
  inline void recv(T& data, int source, int tag = 0) {
    using mpi_type = MPI_Type<T>;
    DYNAMPI_MPI_CHECK(MPI_Recv, (mpi_type::ptr(data), mpi_type::count(data), mpi_type::value,
                                 source, tag, m_comm, MPI_STATUS_IGNORE));
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

  [[nodiscard]] MPI_Comm get() const { return m_comm; }
};

}  // namespace dynampi
