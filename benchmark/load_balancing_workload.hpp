/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

// Workload and CSV schema shared by load_balancing.cpp and
// load_balancing_adlb.cpp. Both plot into the same figures against a floor
// simulated from this duration distribution (plot_load_balancing.py's
// duration_sampler), so it has to be one implementation rather than two that
// agree by inspection.

#ifndef DYNAMPI_BENCHMARK_LOAD_BALANCING_WORKLOAD_HPP
#define DYNAMPI_BENCHMARK_LOAD_BALANCING_WORKLOAD_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>

namespace load_balancing {

using Task = uint32_t;
using Result = uint32_t;

enum class DurationMode {
  Fixed,
  Uniform,
  LogNormal,
};

inline std::string to_string(DurationMode mode) {
  switch (mode) {
    case DurationMode::Fixed:
      return "fixed";
    case DurationMode::Uniform:
      return "uniform";
    case DurationMode::LogNormal:
      return "lognormal";
  }
  return "unknown";
}

inline DurationMode parse_duration_mode(const std::string& value) {
  if (value == "fixed") return DurationMode::Fixed;
  if (value == "uniform") return DurationMode::Uniform;
  if (value == "lognormal") return DurationMode::LogNormal;
  throw std::runtime_error("Unknown duration_mode: " + value);
}

inline void spin_wait(std::chrono::microseconds duration) {
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start < duration) {
  }
}

// Fixed by default so a run isolates scheduling/distribution overhead from
// task-duration variance. Two ways to opt back into variance, both mean
// expected_us: uniform on [0, 2*expected_us] -- the same symmetric spread
// weak_scaling_distribution_rate's WorkerFunctor calls its "random" mode --
// and lognormal, deliberately shaped like real task-duration distributions
// instead (heavily right-skewed: most tasks well under the mean, a long tail
// of much longer ones, rather than uniform's hard cutoff at 2x).
struct WorkerFunctor {
  uint64_t expected_us;
  DurationMode duration_mode;
  std::mt19937_64 rng;
  std::uniform_int_distribution<uint64_t> uniform;
  std::lognormal_distribution<double> lognormal;

  WorkerFunctor(int rank, uint64_t expected_us_in, DurationMode mode, double task_duration_cv)
      : expected_us(expected_us_in),
        duration_mode(mode),
        rng([rank]() {
          std::random_device rd;
          std::mt19937_64 seed_gen(rd());
          return seed_gen() + static_cast<uint64_t>(rank);
        }()),
        uniform(0, 2 * expected_us_in),
        lognormal(lognormal_mu(static_cast<double>(expected_us_in), task_duration_cv),
                  lognormal_sigma(task_duration_cv)) {}

  // Derives the underlying normal distribution's (mu, sigma) so the
  // lognormal's own mean and coefficient of variation match expected_us and
  // task_duration_cv exactly, rather than expecting the caller to reason
  // about the underlying-normal parameterization std::lognormal_distribution
  // itself takes.
  static double lognormal_sigma(double cv) { return std::sqrt(std::log1p(cv * cv)); }
  static double lognormal_mu(double mean, double cv) {
    const double sigma = lognormal_sigma(cv);
    return std::log(mean) - sigma * sigma / 2.0;
  }

  Result operator()(Task task) {
    uint64_t duration_us = expected_us;
    if (duration_mode == DurationMode::Uniform) {
      duration_us = uniform(rng);
    } else if (duration_mode == DurationMode::LogNormal) {
      // >= 1: a 0us spin_wait is a no-op distinguishable from "ran, just
      // fast" only by timing noise, not a meaningful sample of the
      // distribution's near-zero left tail.
      duration_us = std::max<uint64_t>(1, static_cast<uint64_t>(std::llround(lognormal(rng))));
    }
    spin_wait(std::chrono::microseconds(duration_us));
    return static_cast<Result>(task);
  }
};

// One CSV line: the configuration that produced a measurement, and the
// measurement itself.
struct ResultRow {
  std::string system;
  std::string distributor;
  uint64_t nodes = 0;
  uint64_t world_size = 0;
  uint64_t workers = 0;
  int max_upper_fanout = -1;
  int pipeline_depth = -1;
  int max_pending_rounds = -1;
  int adlb_servers = -1;  // ADLB only; -1 from the DynaMPI drivers
  uint64_t expected_us = 0;
  DurationMode duration_mode = DurationMode::Fixed;
  double task_duration_cv = 0.0;
  uint64_t tasks_per_worker = 0;
  uint64_t repeat = 0;
  uint64_t total_tasks = 0;
  double elapsed_s = 0.0;
  // Phases outside the timed drain, reported separately rather than folded
  // into elapsed_s.
  double construct_s = 0.0;
  double warmup_s = 0.0;
  double finalize_s = 0.0;
  double destruct_s = 0.0;
};

inline void write_csv_header(std::ostream& os) {
  os << "system,distributor,nodes,world_size,workers,max_upper_fanout,pipeline_depth,"
        "max_pending_rounds,adlb_servers,expected_us,duration_mode,task_duration_cv,"
        "tasks_per_worker,repeat,total_tasks,elapsed_s,construct_s,warmup_s,finalize_s,"
        "destruct_s\n";
}

inline void write_csv_row(std::ostream& os, const ResultRow& row) {
  os << row.system << "," << row.distributor << "," << row.nodes << "," << row.world_size << ","
     << row.workers << "," << row.max_upper_fanout << "," << row.pipeline_depth << ","
     << row.max_pending_rounds << "," << row.adlb_servers << "," << row.expected_us << ","
     << to_string(row.duration_mode) << "," << row.task_duration_cv << "," << row.tasks_per_worker
     << "," << row.repeat << "," << row.total_tasks << "," << row.elapsed_s << ","
     << row.construct_s << "," << row.warmup_s << "," << row.finalize_s << "," << row.destruct_s
     << "\n";
}

inline void append_result(const std::string& output_path, const ResultRow& row) {
  std::cout << "RESULT distributor=" << row.distributor << " nodes=" << row.nodes
            << " world_size=" << row.world_size << " workers=" << row.workers
            << " max_upper_fanout=" << row.max_upper_fanout
            << " pipeline_depth=" << row.pipeline_depth
            << " max_pending_rounds=" << row.max_pending_rounds
            << " adlb_servers=" << row.adlb_servers << " expected_us=" << row.expected_us
            << " duration_mode=" << to_string(row.duration_mode)
            << " task_duration_cv=" << row.task_duration_cv
            << " tasks_per_worker=" << row.tasks_per_worker << " repeat=" << row.repeat
            << " total_tasks=" << row.total_tasks << " elapsed_s=" << row.elapsed_s
            << " construct_s=" << row.construct_s << " warmup_s=" << row.warmup_s
            << " finalize_s=" << row.finalize_s << " destruct_s=" << row.destruct_s << std::endl;
  if (output_path.empty()) return;
  std::ifstream check(output_path);
  const bool needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
  check.close();
  std::ofstream out(output_path, std::ios::app);
  if (needs_header) {
    write_csv_header(out);
  }
  write_csv_row(out, row);
}

}  // namespace load_balancing

#endif  // DYNAMPI_BENCHMARK_LOAD_BALANCING_WORKLOAD_HPP
