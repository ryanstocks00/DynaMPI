// SPDX-FileCopyrightText: 2026 Ryan Stocks
// SPDX-License-Identifier: Apache-2.0

// Measures the raw ceiling of the one-sided RMA atomics LockFreeMPIWorkDistributor
// and AsyncPutLockFreeMPIWorkDistributor build on top of: MPI_Fetch_and_op
// (and MPI_Compare_and_swap for comparison) against a single int64 counter on
// rank 0's window, hammered concurrently by every other rank. No task claiming,
// no result staging, no gather rounds -- just the bare primitive, so this
// isolates "how fast can the fabric/MPI implementation do this operation" from
// anything about how the distributor uses it.
//
// Three phases:
//   faa_flushed   -- MPI_Fetch_and_op(SUM,1) + MPI_Win_flush every call. Always
//                     "succeeds" (no compare, nothing to lose a race on): the
//                     purest measure of one atomic round trip's rate.
//   cas_flushed   -- MPI_Compare_and_swap + MPI_Win_flush every call, with a
//                     locally-cached expected value corrected for free from a
//                     failed CAS's return. Kept for comparison against FAA;
//                     the distributors themselves use Fetch_and_op (SUM) for
//                     claiming, not CAS.
//   faa_pipelined -- `pipeline_depth` Fetch_and_op calls posted back-to-back
//                     before a single MPI_Win_flush, instead of flushing
//                     every call. Tests whether the flush-per-op pattern
//                     (used throughout the lock-free distributors)
//                     is itself the bottleneck, by seeing whether overlapping
//                     multiple outstanding atomics raises the ceiling.
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <iostream>
#include <vector>

namespace {

struct PhaseResult {
  long long attempted = 0;
  long long succeeded = 0;
};

double elapsed_since(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

PhaseResult run_faa_flushed(MPI_Win win, int root, double duration_s) {
  auto t0 = std::chrono::steady_clock::now();
  long long n = 0;
  while (elapsed_since(t0) < duration_s) {
    int64_t one = 1, out = 0;
    MPI_Fetch_and_op(&one, &out, MPI_INT64_T, root, 0, MPI_SUM, win);
    MPI_Win_flush(root, win);
    n++;
  }
  return {n, n};
}

PhaseResult run_cas_flushed(MPI_Win win, int root, double duration_s) {
  auto t0 = std::chrono::steady_clock::now();
  long long attempted = 0, succeeded = 0;
  int64_t cached = 0;
  while (elapsed_since(t0) < duration_s) {
    int64_t desired = cached + 1, compare = cached, out = 0;
    MPI_Compare_and_swap(&desired, &compare, &out, MPI_INT64_T, root, 0, win);
    MPI_Win_flush(root, win);
    attempted++;
    if (out == cached) {
      succeeded++;
      cached = desired;
    } else {
      cached = out;  // learned the real value for free; retry next iteration
    }
  }
  return {attempted, succeeded};
}

PhaseResult run_faa_pipelined(MPI_Win win, int root, double duration_s, int pipeline_depth) {
  auto t0 = std::chrono::steady_clock::now();
  long long n = 0;
  std::vector<int64_t> outs(static_cast<size_t>(pipeline_depth));
  int64_t one = 1;
  while (elapsed_since(t0) < duration_s) {
    for (int i = 0; i < pipeline_depth; ++i) {
      MPI_Fetch_and_op(&one, &outs[static_cast<size_t>(i)], MPI_INT64_T, root, 0, MPI_SUM, win);
    }
    MPI_Win_flush(root, win);  // one flush for the whole batch
    n += pipeline_depth;
  }
  return {n, n};
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  cxxopts::Options options("rma_atomic_microbench",
                           "Raw RMA atomic (Fetch_and_op / Compare_and_swap) throughput ceiling");
  options.add_options()("d,duration_s", "duration per phase, in seconds",
                        cxxopts::value<double>()->default_value("3"))(
      "p,pipeline_depths", "comma-separated pipeline depths to test for faa_pipelined",
      cxxopts::value<std::string>()->default_value("1,8,32,128"))("h,help", "print usage");
  auto parsed = options.parse(argc, argv);
  if (parsed.count("help")) {
    if (rank == 0) std::cout << options.help() << std::endl;
    MPI_Finalize();
    return 0;
  }
  const double duration_s = parsed["duration_s"].as<double>();
  std::vector<int> pipeline_depths;
  {
    std::string s = parsed["pipeline_depths"].as<std::string>();
    size_t pos = 0;
    while (pos < s.size()) {
      size_t comma = s.find(',', pos);
      if (comma == std::string::npos) comma = s.size();
      pipeline_depths.push_back(std::stoi(s.substr(pos, comma - pos)));
      pos = comma + 1;
    }
  }

  const int root = 0;
  int64_t counter = 0;
  MPI_Win win;
  void* base = (rank == root) ? static_cast<void*>(&counter) : nullptr;
  MPI_Aint winsize = (rank == root) ? static_cast<MPI_Aint>(sizeof(int64_t)) : 0;
  MPI_Win_create(base, winsize, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, win);

  const bool participates = (size > 1) ? (rank != root) : true;
  const int participants = (size > 1) ? (size - 1) : 1;

  auto run_phase = [&](const std::string& name, auto fn) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == root) {
      int64_t zero = 0, out = 0;
      MPI_Fetch_and_op(&zero, &out, MPI_INT64_T, root, 0, MPI_REPLACE, win);
      MPI_Win_flush(root, win);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    PhaseResult local;
    if (participates) local = fn();

    PhaseResult total;
    MPI_Reduce(&local.attempted, &total.attempted, 1, MPI_LONG_LONG, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Reduce(&local.succeeded, &total.succeeded, 1, MPI_LONG_LONG, MPI_SUM, root, MPI_COMM_WORLD);

    if (rank == root) {
      const double succ_rate = static_cast<double>(total.succeeded) / duration_s;
      const double att_rate = static_cast<double>(total.attempted) / duration_s;
      std::cout << "RESULT phase=" << name << " world_size=" << size
                << " participants=" << participants << " succeeded=" << total.succeeded
                << " attempted=" << total.attempted << " duration_s=" << duration_s
                << " succeeded_per_s=" << succ_rate << " attempted_per_s=" << att_rate
                << " succeeded_per_s_per_rank=" << (succ_rate / participants) << std::endl;
    }
  };

  run_phase("faa_flushed", [&]() { return run_faa_flushed(win, root, duration_s); });
  run_phase("cas_flushed", [&]() { return run_cas_flushed(win, root, duration_s); });
  for (int depth : pipeline_depths) {
    run_phase("faa_pipelined_depth" + std::to_string(depth),
              [&]() { return run_faa_pipelined(win, root, duration_s, depth); });
  }

  MPI_Win_unlock_all(win);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Win_free(&win);
  MPI_Finalize();
  return 0;
}
