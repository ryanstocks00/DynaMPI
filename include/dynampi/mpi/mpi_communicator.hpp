/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#include "mpi_error.hpp"

namespace dynampi {

class MPICommunicator {
 public:
  enum Ownership {
    NotOwned,  // The communicator is not owned by this class and should not be freed.
    Owned,     // The communicator is owned by this class and will be freed in the destructor.
  };

 private:
  MPI_Comm _comm;
  Ownership _ownership;

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

  [[nodiscard]] MPI_Comm get() const { return _comm; }
};

}  // namespace dynampi
