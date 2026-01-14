/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include <string>

#if __has_include(<source_location>)
#include <source_location>
#if defined(__cpp_lib_source_location)
#define DYNAMPI_HAS_SOURCE_LOCATION
#endif
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
#define DYNAMPI_HAS_SOURCE_LOCATION
#endif
#include <stdexcept>

#define DYNAMPI_MPI_CHECK(func, args)      \
  do {                                     \
    int err = func args;                   \
    if (err != MPI_SUCCESS) {              \
      dynampi::mpi_fail(err, #func #args); \
    }                                      \
  } while (false)

namespace dynampi {
inline void mpi_fail(int err, std::string_view command
#ifdef DYNAMPI_HAS_SOURCE_LOCATION
                     ,
                     std::source_location loc = std::source_location::current()
#endif
) {
  char error_string[MPI_MAX_ERROR_STRING];
  int length_of_error_string;
  MPI_Error_string(err, error_string, &length_of_error_string);
  throw std::runtime_error(std::string("MPI error in ") + std::string(command) + ": " +
                           std::string(error_string, length_of_error_string)
#ifdef DYNAMPI_HAS_SOURCE_LOCATION
                           + " at " + loc.file_name() + ":" + std::to_string(loc.line())
#endif
  );
}
}  // namespace dynampi
