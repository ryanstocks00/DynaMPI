/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <dynampi/impl/hierarchical_distributor.hpp>
#include <dynampi/impl/naive_distributor.hpp>
#include <dynampi/mpi/mpi_communicator.hpp>
#include <dynampi/utilities/timer.hpp>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>

using Task = uint32_t;

enum class DistributorKind { Naive, Hierarchical };
enum class DurationMode { Fixed, Poisson };

struct BenchmarkOptions {
  uint64_t expected_us = 1;
  double duration_s = 10.0;
  DistributorKind distributor = DistributorKind::Hierarchical;
  DurationMode duration_mode = DurationMode::Fixed;
  uint64_t nodes = 0;
  std::string system;
  std::string output_path;
};

struct BenchmarkResult {
  uint64_t total_tasks = 0;
  uint64_t workers = 0;
  uint64_t world_size = 0;
  double elapsed_s = 0.0;
};

static DistributorKind parse_distributor(const std::string& value) {
  if (value == "naive") return DistributorKind::Naive;
  if (value == "hierarchical") return DistributorKind::Hierarchical;
  throw std::runtime_error("Unknown distributor: " + value);
}

static DurationMode parse_duration_mode(const std::string& value) {
  if (value == "fixed") return DurationMode::Fixed;
  if (value == "poisson" || value == "random") return DurationMode::Poisson;
  throw std::runtime_error("Unknown duration mode: " + value);
}

static std::string to_string(DistributorKind kind) {
  return kind == DistributorKind::Naive ? "naive" : "hierarchical";
}

static std::string to_string(DurationMode mode) {
  return mode == DurationMode::Fixed ? "fixed" : "random";
}

static void spin_wait(std::chrono::microseconds duration) {
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start < duration) {
  }
}

static void write_csv_header(std::ostream& os) {
  os << "system,distributor,mode,expected_us,"
        "duration_s,nodes,world_size,workers,total_tasks,elapsed_s,"
        "throughput_tasks_per_s\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts,
                          const BenchmarkResult& result) {
  const double throughput =
      result.elapsed_s > 0.0 ? static_cast<double>(result.total_tasks) / result.elapsed_s : 0.0;
  os << opts.system << "," << to_string(opts.distributor) << "," << to_string(opts.duration_mode)
     << "," << opts.expected_us << "," << opts.duration_s << "," << opts.nodes << ","
     << result.world_size << "," << result.workers << "," << result.total_tasks << ","
     << result.elapsed_s << "," << throughput << "\n";
}

class CommWrapper {
 public:
  explicit CommWrapper(MPI_Comm comm) : comm_(comm) {}
  void barrier() const { MPI_Barrier(comm_); }
  MPI_Comm get() const { return comm_; }

 private:
  MPI_Comm comm_;
};

template <typename Distributor>
static BenchmarkResult run_benchmark(const BenchmarkOptions& opts, MPI_Comm comm) {
  CommWrapper comm_wrapper(comm);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const uint64_t num_workers = (size == 1) ? 1 : static_cast<uint64_t>(size - 1);

  struct WorkerFunctor {
    std::mt19937_64 rng;
    std::uniform_int_distribution<uint64_t> uniform;
    uint64_t expected_us;
    DurationMode duration_mode;

    WorkerFunctor(int rank, uint64_t expected_us, DurationMode mode)
        : rng([rank]() {
            std::random_device rd;
            std::mt19937_64 seed_gen(rd());
            return seed_gen() + static_cast<uint64_t>(rank);
          }()),
          uniform(0, 2 * expected_us),
          expected_us(expected_us),
          duration_mode(mode) {}

    uint32_t operator()(Task task) {
      uint32_t value = task;
      uint64_t duration_us = expected_us;
      if (duration_mode == DurationMode::Poisson) {
        duration_us = uniform(rng);
      }
      spin_wait(std::chrono::microseconds(duration_us));
      const uint64_t squared = static_cast<uint64_t>(value) * static_cast<uint64_t>(value);
      return static_cast<uint32_t>(squared);
    }
  };

  WorkerFunctor worker_function(rank, opts.expected_us, opts.duration_mode);

  comm_wrapper.barrier();
  dynampi::Timer timer(dynampi::Timer::AutoStart::No);
  uint64_t total_tasks = 0;

  Distributor distributor(worker_function, {.comm = comm, .manager_rank = 0});

  if (distributor.is_root_manager()) {
    timer.start();

    const uint64_t target_queue_size = num_workers * 4;
    while (timer.elapsed().count() < opts.duration_s) {
      const uint64_t remaining = distributor.remaining_tasks_count();
      uint64_t to_insert = 0;
      if (remaining < target_queue_size) {
        to_insert = target_queue_size - remaining;
      }
      if (timer.elapsed().count() > opts.duration_s / 2.0 && total_tasks > 0) {
        double current_rate = static_cast<double>(total_tasks) / timer.elapsed().count();
        double estimated_total_tasks = current_rate * opts.duration_s;
        if (estimated_total_tasks > static_cast<double>(total_tasks) && current_rate > 0.0) {
          double remaining_time = opts.duration_s - timer.elapsed().count();
          uint64_t can_complete_tasks_remaining =
              static_cast<uint64_t>(current_rate * remaining_time);
          if (can_complete_tasks_remaining > remaining) {
            uint64_t max_to_insert = can_complete_tasks_remaining - remaining;
            to_insert = std::min(to_insert, max_to_insert);
          } else {
            // Already have more tasks queued than can be completed, don't insert more
            to_insert = 0;
          }
        }
      }
      // Clamp to_insert to be non-negative and <= target_queue_size
      to_insert = std::min(to_insert, target_queue_size);

      if (to_insert > 0) {
        std::vector<Task> tasks;
        tasks.reserve(to_insert);
        for (uint64_t i = 0; i < to_insert; ++i) {
          tasks.push_back(static_cast<Task>(total_tasks + i));
        }
        distributor.insert_tasks(tasks);
      }
      auto results = distributor.run_tasks({.max_tasks = num_workers * 2, .max_seconds = 0.1});
      total_tasks += results.size();
    }
    {
      auto results = distributor.finish_remaining_tasks();
      total_tasks += results.size();
    }
    timer.stop();
    distributor.finalize();
  }

  return BenchmarkResult{total_tasks, num_workers, static_cast<uint64_t>(size),
                         timer.elapsed().count()};
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cxxopts::Options options("strong_scaling_distribution_rate",
                           "Benchmark strong scaling task distribution throughput");
  options.add_options()("t,expected_us", "Expected task duration in microseconds",
                        cxxopts::value<uint64_t>()->default_value("1"))(
      "d,duration_s", "Target duration in seconds", cxxopts::value<double>()->default_value("10"))(
      "D,distribution", "Distribution strategy: naive or hierarchical",
      cxxopts::value<std::string>()->default_value("hierarchical"))(
      "m,mode", "Duration mode: fixed or random (uniform 0-2x expected)",
      cxxopts::value<std::string>()->default_value("fixed"))(
      "n,nodes", "Number of nodes for labeling output (defaults to world size)",
      cxxopts::value<uint64_t>()->default_value("0"))(
      "S,system", "System label for plotting (frontier, aurora, ...)",
      cxxopts::value<std::string>()->default_value(""))(
      "o,output", "Append results to CSV file", cxxopts::value<std::string>()->default_value(""))(
      "h,help", "Print usage");

  cxxopts::ParseResult args;
  try {
    args = options.parse(argc, argv);
  } catch (const std::exception& e) {
    if (world_rank == 0) {
      std::cerr << "Error parsing options: " << e.what() << "\n" << options.help() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  if (args.count("help")) {
    if (world_rank == 0) {
      std::cout << options.help() << std::endl;
    }
    MPI_Finalize();
    return 0;
  }

  BenchmarkOptions opts;
  opts.expected_us = args["expected_us"].as<uint64_t>();
  opts.duration_s = args["duration_s"].as<double>();
  opts.distributor = parse_distributor(args["distribution"].as<std::string>());
  opts.duration_mode = parse_duration_mode(args["mode"].as<std::string>());
  opts.nodes = args["nodes"].as<uint64_t>();
  opts.system = args["system"].as<std::string>();
  opts.output_path = args["output"].as<std::string>();

  if (opts.expected_us == 0) {
    if (world_rank == 0) {
      std::cerr << "Expected task duration must be >= 1 microsecond." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (opts.nodes == 0) {
      opts.nodes = static_cast<uint64_t>(size);
    }

    BenchmarkResult result;
    if (opts.distributor == DistributorKind::Naive) {
      result = run_benchmark<dynampi::NaiveMPIWorkDistributor<Task, uint32_t>>(opts, comm);
    } else {
      result = run_benchmark<dynampi::HierarchicalMPIWorkDistributor<Task, uint32_t>>(opts, comm);
    }

    if (rank == 0) {
      const double throughput =
          result.elapsed_s > 0.0 ? static_cast<double>(result.total_tasks) / result.elapsed_s : 0.0;
      std::cout << "RESULT"
                << " distributor=" << to_string(opts.distributor)
                << " mode=" << to_string(opts.duration_mode) << " expected_us=" << opts.expected_us
                << " nodes=" << opts.nodes << " world_size=" << result.world_size
                << " total_tasks=" << result.total_tasks << " elapsed_s=" << result.elapsed_s
                << " throughput_tasks_per_s=" << throughput << std::endl;
      if (!opts.output_path.empty()) {
        std::ifstream check(opts.output_path);
        const bool needs_header =
            !check.good() || check.peek() == std::ifstream::traits_type::eof();
        check.close();
        std::ofstream out(opts.output_path, std::ios::app);
        if (needs_header) {
          write_csv_header(out);
        }
        write_csv_row(out, opts, result);
      }
    }
  }
  MPI_Finalize();
  return 0;
}
