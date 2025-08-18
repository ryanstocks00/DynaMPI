/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#if __has_include(<source_location>)
#include <source_location>
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
#else
#define NO_SOURCE_LOCATION
#endif
#include <stdexcept>

#define DYNAMPI_MPI_CHECK(func, args)                               \
  do {                                                              \
    int err = func args;                                            \
    if (err != MPI_SUCCESS) {                                       \
      mpi_check(err, #func #args, std::source_location::current()); \
    }                                                               \
  } while (false)

namespace dynampi {
inline void mpi_check(int err, std::string_view command,
#ifndef NO_SOURCE_LOCATION
                      std::source_location loc = std::source_location::current()
#endif
) {
  if (err != MPI_SUCCESS) {
    char error_string[MPI_MAX_ERROR_STRING];
    int length_of_error_string;
    MPI_Error_string(err, error_string, &length_of_error_string);
    throw std::runtime_error(std::string("MPI error in ") + std::string(command) + ": " +
                             std::string(error_string, length_of_error_string) + " at " +
#ifndef NO_SOURCE_LOCATION
                             loc.file_name() + ":" + std::to_string(loc.line())
#endif
    );
  }
}
}  // namespace dynampi
