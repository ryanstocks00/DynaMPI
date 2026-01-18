/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace dynampi {

// Returns a reference to the debug log file stream for the current rank
// Each rank gets its own file: debug_rank_N.log where N is the rank
// Files are created in the current working directory where the program runs
inline std::ostream& get_debug_log() {
  static std::map<int, std::unique_ptr<std::ofstream>> rank_files;
  static std::mutex files_mutex;
  static std::map<int, bool> use_cerr_fallback;
  static int cached_rank = -1;  // Cache rank to avoid calling MPI after finalization
  static bool rank_cached = false;

  int rank = 0;

  // Use cached rank if available - this avoids calling MPI_Comm_rank after finalization
  if (rank_cached && cached_rank >= 0) {
    rank = cached_rank;
  } else {
    // Try to get and cache rank only once, early in execution
    // After this, we never call MPI_Comm_rank again to avoid issues after finalization
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
      // MPI not initialized, use stderr
      return std::cerr;
    }

    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    if (mpi_finalized) {
      // MPI is finalized, use stderr
      return std::cerr;
    }

    // Only try to get rank if MPI is initialized and not finalized
    // Use error handler to avoid fatal errors
    MPI_Errhandler old_handler;
    MPI_Comm_get_errhandler(MPI_COMM_WORLD, &old_handler);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    rank = 0;  // Default
    int result = MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_set_errhandler(MPI_COMM_WORLD, old_handler);

    if (result == MPI_SUCCESS) {
      cached_rank = rank;
      rank_cached = true;
    } else {
      // Failed to get rank, use stderr and don't cache
      return std::cerr;
    }
  }

  std::lock_guard<std::mutex> lock(files_mutex);

  // Check if we should use cerr fallback for this rank
  if (use_cerr_fallback.find(rank) != use_cerr_fallback.end() && use_cerr_fallback[rank]) {
    return std::cerr;
  }

  auto it = rank_files.find(rank);
  if (it == rank_files.end()) {
    std::string filename = "debug_rank_" + std::to_string(rank) + ".log";
    auto file = std::make_unique<std::ofstream>(filename, std::ios::out | std::ios::trunc);

    // Check if file opened successfully
    if (!file->is_open() || !file->good()) {
      // Fallback to stderr if file can't be opened
      // Use a cross-platform way to get error message
      std::string error_msg;
#ifdef _WIN32
      char buffer[256];
      if (strerror_s(buffer, sizeof(buffer), errno) == 0) {
        error_msg = buffer;
      } else {
        error_msg = "Unknown error";
      }
#else
      error_msg = std::strerror(errno);
#endif
      std::cerr << "[RANK " << rank << "] WARNING: Failed to open debug log file '" << filename
                << "': " << error_msg << " (falling back to stderr)" << std::endl;
      use_cerr_fallback[rank] = true;
      return std::cerr;
    }

    // Set to unitbuf to ensure immediate flushing (useful for debugging stuck processes)
    file->setf(std::ios_base::unitbuf);
    rank_files[rank] = std::move(file);
    it = rank_files.find(rank);

    // Write initial message to confirm file creation
    *it->second << "[RANK " << rank << "] Debug log file opened: " << filename << std::endl;
    it->second->flush();

    // Also print to stderr so user knows where the file is
    std::cerr << "[RANK " << rank << "] Debug log file created: " << filename
              << " (in current working directory)" << std::endl;
  }

  return *(it->second);
}

}  // namespace dynampi
