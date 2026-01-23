/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include "dynampi/utilities/assert.hpp"
#include "mpi_error.hpp"

namespace dynampi {

// Forward declaration
template <typename... Options>
class MPICommunicator;

class MPIGroup {
 private:
  MPI_Group m_group;

 public:
  // Create from a communicator (extracts the group)
  template <typename... Options>
  explicit MPIGroup(const MPICommunicator<Options...>& comm) {
    DYNAMPI_MPI_CHECK(MPI_Comm_group, (comm.get(), &m_group));
  }

  // Non-copyable
  MPIGroup(const MPIGroup& other) = delete;
  MPIGroup& operator=(const MPIGroup& other) = delete;

  // Movable
  MPIGroup(MPIGroup&& other) noexcept : m_group(other.m_group) { other.m_group = MPI_GROUP_NULL; }
  MPIGroup& operator=(MPIGroup&& other) noexcept {
    if (this != &other) {
      if (m_group != MPI_GROUP_NULL) {
        MPI_Group_free(&m_group);
      }
      m_group = other.m_group;
      other.m_group = MPI_GROUP_NULL;
    }
    return *this;
  }

  ~MPIGroup() {
    if (m_group != MPI_GROUP_NULL) {
      MPI_Group_free(&m_group);
    }
  }

  // Translate ranks from this group to another group
  void translate_ranks(const MPIGroup& to_group, int n, const int ranks[],
                       int translated_ranks[]) const {
    DYNAMPI_MPI_CHECK(MPI_Group_translate_ranks,
                      (m_group, n, ranks, to_group.m_group, translated_ranks));
  }

  // Convenience method for single rank translation
  int translate_rank(int rank, const MPIGroup& to_group) const {
    int translated_rank;
    translate_ranks(to_group, 1, &rank, &translated_rank);
    return translated_rank;
  }

  operator MPI_Group() const { return m_group; }

  [[nodiscard]] MPI_Group get() const { return m_group; }
};

}  // namespace dynampi
