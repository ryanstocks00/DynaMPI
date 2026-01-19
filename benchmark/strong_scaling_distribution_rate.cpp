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
#include <limits>
#include <optional>
#include <random>
#include <string>

using Task = uint32_t;

enum class DistributorKind { Naive, Hierarchical };
enum class DurationMode { Fixed, Poisson };

struct BenchmarkOptions {
  uint64_t expected_ns = 1000;
  double duration_s = 10.0;
  uint64_t bundle_target_ms = 10;
  uint64_t round_target_ms = 200;
  bool remove_root_from_distribution = false;
  DistributorKind distributor = DistributorKind::Hierarchical;
  DurationMode duration_mode = DurationMode::Fixed;
  uint64_t seed = 0;
  uint64_t nodes = 0;
  std::string system;
  std::string output_path;
};

struct BenchmarkResult {
  uint64_t total_tasks = 0;
  uint64_t total_subtasks = 0;
  uint64_t repetitions = 1;
  uint64_t workers = 0;
  uint64_t world_size = 0;
  double elapsed_s = 0.0;
};

static std::optional<dynampi::MPICommunicator<>> make_dynamic_communicator(
    bool remove_root_from_distribution) {
  if (remove_root_from_distribution) {
    dynampi::MPICommunicator<> world_communicator(MPI_COMM_WORLD);
    dynampi::MPICommunicator<> node_communicator = world_communicator.split_by_node();
    int is_root_node = world_communicator.rank() == 0;
    node_communicator.broadcast(is_root_node, 0);
    std::optional<dynampi::MPICommunicator<>> skip_manager_node_communicator =
        node_communicator.split((world_communicator.rank() != 0 && is_root_node) ? MPI_UNDEFINED
                                                                                 : 1);
    if (skip_manager_node_communicator) {
      return std::optional<dynampi::MPICommunicator<>>(
          std::move(skip_manager_node_communicator.value()));
    }
    return std::nullopt;
  }
  return std::optional<dynampi::MPICommunicator<>>(dynampi::MPICommunicator<>(MPI_COMM_WORLD));
}

static DistributorKind parse_distributor(const std::string& value) {
  if (value == "naive") return DistributorKind::Naive;
  if (value == "hierarchical") return DistributorKind::Hierarchical;
  throw std::runtime_error("Unknown distributor: " + value);
}

static DurationMode parse_duration_mode(const std::string& value) {
  if (value == "fixed") return DurationMode::Fixed;
  if (value == "poisson") return DurationMode::Poisson;
  throw std::runtime_error("Unknown duration mode: " + value);
}

static std::string to_string(DistributorKind kind) {
  return kind == DistributorKind::Naive ? "naive" : "hierarchical";
}

static std::string to_string(DurationMode mode) {
  return mode == DurationMode::Fixed ? "fixed" : "poisson";
}

static void spin_wait(std::chrono::microseconds duration) {
  auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < duration) {
  }
}

static uint64_t compute_repetitions(uint64_t expected_ns, uint64_t bundle_target_ms) {
  if (expected_ns == 0) return 1;
  const uint64_t target_ns = bundle_target_ms * 1'000'000;
  const uint64_t repetitions = (target_ns + expected_ns - 1) / expected_ns;
  return std::max<uint64_t>(1, repetitions);
}

static uint64_t compute_tasks_per_round(uint64_t expected_ns, uint64_t repetitions,
                                        uint64_t round_target_ms) {
  const uint64_t task_duration_ns = expected_ns * repetitions;
  if (task_duration_ns == 0) return 1;
  const uint64_t target_ns = round_target_ms * 1'000'000;
  const uint64_t tasks = (target_ns + task_duration_ns - 1) / task_duration_ns;
  return std::max<uint64_t>(1, tasks);
}

static void write_csv_header(std::ostream& os) {
  os << "system,distributor,mode,expected_ns,bundle_repetitions,bundle_target_ms,round_target_ms,"
        "duration_s,nodes,world_size,workers,total_tasks,total_subtasks,elapsed_s,"
        "throughput_tasks_per_s\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts,
                          const BenchmarkResult& result) {
  const double throughput =
      result.elapsed_s > 0.0 ? static_cast<double>(result.total_subtasks) / result.elapsed_s : 0.0;
  os << opts.system << "," << to_string(opts.distributor) << "," << to_string(opts.duration_mode)
     << "," << opts.expected_ns << "," << result.repetitions << "," << opts.bundle_target_ms << ","
     << opts.round_target_ms << "," << opts.duration_s << "," << opts.nodes << ","
     << result.world_size << "," << result.workers << "," << result.total_tasks << ","
     << result.total_subtasks << "," << result.elapsed_s << "," << throughput << "\n";
}

template <typename Distributor>
static BenchmarkResult run_benchmark(const BenchmarkOptions& opts, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const uint64_t repetitions = compute_repetitions(opts.expected_ns, opts.bundle_target_ms);
  if (repetitions > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("Bundle repetitions exceed uint32_t maximum");
  }
  const uint64_t workers = (size == 1) ? 1 : static_cast<uint64_t>(size - 1);
  const uint64_t tasks_per_round =
      compute_tasks_per_round(opts.expected_ns, repetitions, opts.round_target_ms);

  const uint64_t cap_ns = opts.expected_ns * 10;
  const uint64_t base_seed =
      opts.seed == 0 ? static_cast<uint64_t>(
                           std::chrono::high_resolution_clock::now().time_since_epoch().count())
                     : opts.seed;
  auto worker_function = [=]() {
    std::mt19937_64 rng(base_seed + static_cast<uint64_t>(rank));
    std::poisson_distribution<uint64_t> poisson(static_cast<double>(opts.expected_ns));
    return [=](Task task) mutable -> uint32_t {
      for (uint32_t i = 0; i < task; ++i) {
        uint64_t duration_ns = opts.expected_ns;
        if (opts.duration_mode == DurationMode::Poisson) {
          uint64_t sample = poisson(rng);
          if (sample == 0) sample = 1;
          if (sample > cap_ns) sample = cap_ns;
          duration_ns = sample;
        }
        spin_wait(std::chrono::nanoseconds(duration_ns));
      }
      return task;
    };
  }();

  MPI_Barrier(comm);
  dynampi::Timer timer(dynampi::Timer::AutoStart::No);
  uint64_t total_tasks = 0;
  uint64_t total_subtasks = 0;

  if (rank == 0) {
    timer.start();
  }

  while (true) {
    int continue_round = 0;
    if (rank == 0) {
      continue_round = timer.elapsed().count() < opts.duration_s ? 1 : 0;
    }
    MPI_Bcast(&continue_round, 1, MPI_INT, 0, comm);
    if (continue_round == 0) {
      break;
    }

    Distributor distributor(worker_function,
                            {.comm = comm, .manager_rank = 0, .auto_run_workers = false});

    if (distributor.is_root_manager()) {
      const uint64_t round_tasks = tasks_per_round * workers;
      for (uint64_t i = 0; i < round_tasks; ++i) {
        distributor.insert_task(static_cast<Task>(repetitions));
      }
      auto results = distributor.finish_remaining_tasks();
      (void)results;
      distributor.finalize();
      total_tasks += round_tasks;
      total_subtasks += round_tasks * repetitions;
    } else {
      distributor.run_worker();
    }
  }

  if (rank == 0) {
    timer.stop();
  }

  return BenchmarkResult{total_tasks,
                         total_subtasks,
                         repetitions,
                         workers,
                         static_cast<uint64_t>(size),
                         timer.elapsed().count()};
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cxxopts::Options options("strong_scaling_distribution_rate",
                           "Benchmark strong scaling task distribution throughput");
  options.add_options()("expected_us", "Expected task duration in microseconds (deprecated)",
                        cxxopts::value<uint64_t>()->default_value("0"))(
      "t,expected_ns", "Expected task duration in nanoseconds",
      cxxopts::value<uint64_t>()->default_value("1000"))(
      "d,duration_s", "Target duration in seconds", cxxopts::value<double>()->default_value("10"))(
      "b,bundle_target_ms", "Target bundle duration in milliseconds",
      cxxopts::value<uint64_t>()->default_value("10"))("r,rm_root",
                                                       "Remove root node from task distribution")(
      "round_target_ms", "Target per-round duration in milliseconds",
      cxxopts::value<uint64_t>()->default_value("200"))(
      "D,distribution", "Distribution strategy: naive or hierarchical",
      cxxopts::value<std::string>()->default_value("hierarchical"))(
      "m,mode", "Duration mode: fixed or poisson",
      cxxopts::value<std::string>()->default_value("fixed"))(
      "s,seed", "Random seed (0 uses time-based seed)",
      cxxopts::value<uint64_t>()->default_value("0"))(
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
  opts.expected_ns = args["expected_ns"].as<uint64_t>();
  const uint64_t expected_us = args["expected_us"].as<uint64_t>();
  if (expected_us > 0) {
    opts.expected_ns = expected_us * 1000;
  }
  opts.duration_s = args["duration_s"].as<double>();
  opts.bundle_target_ms = args["bundle_target_ms"].as<uint64_t>();
  opts.round_target_ms = args["round_target_ms"].as<uint64_t>();
  opts.remove_root_from_distribution = args.count("rm_root") > 0;
  opts.distributor = parse_distributor(args["distribution"].as<std::string>());
  opts.duration_mode = parse_duration_mode(args["mode"].as<std::string>());
  opts.seed = args["seed"].as<uint64_t>();
  opts.nodes = args["nodes"].as<uint64_t>();
  opts.system = args["system"].as<std::string>();
  opts.output_path = args["output"].as<std::string>();

  if (opts.expected_ns == 0) {
    if (world_rank == 0) {
      std::cerr << "Expected task duration must be >= 1 nanosecond." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  auto dynamic_comm = make_dynamic_communicator(opts.remove_root_from_distribution);
  if (!dynamic_comm.has_value()) {
    MPI_Finalize();
    return 0;
  }

  MPI_Comm comm = dynamic_comm.value().get();
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
    const double throughput = result.elapsed_s > 0.0
                                  ? static_cast<double>(result.total_subtasks) / result.elapsed_s
                                  : 0.0;
    std::cout << "RESULT"
              << " distributor=" << to_string(opts.distributor)
              << " mode=" << to_string(opts.duration_mode) << " expected_ns=" << opts.expected_ns
              << " repetitions=" << result.repetitions << " nodes=" << opts.nodes
              << " world_size=" << result.world_size << " total_subtasks=" << result.total_subtasks
              << " elapsed_s=" << result.elapsed_s << " throughput_tasks_per_s=" << throughput
              << std::endl;
    if (!opts.output_path.empty()) {
      std::ifstream check(opts.output_path);
      const bool needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
      check.close();
      std::ofstream out(opts.output_path, std::ios::app);
      if (needs_header) {
        write_csv_header(out);
      }
      write_csv_row(out, opts, result);
    }
  }

  MPI_Finalize();
  return 0;
}
