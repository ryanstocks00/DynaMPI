/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>

#include <cstdint>
#include <cxxopts.hpp>
#include <dynampi/impl/naive_distributor.hpp>
#include <dynampi/mpi/mpi_communicator.hpp>
#include <dynampi/utilities/timer.hpp>
#include <fstream>
#include <iostream>
#include <string>

using Task = uint32_t;
using Result = uint32_t;

struct BenchmarkOptions {
  uint64_t nodes = 0;
  std::string system;
  std::string output_path;
};

struct BenchmarkResult {
  uint64_t workers = 0;
  uint64_t world_size = 0;
  double time_per_shutdown_us = 0.0;
  uint64_t iterations = 0;
};

static void write_csv_header(std::ostream& os) {
  os << "system,nodes,world_size,workers,time_per_shutdown_us,iterations\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts,
                          const BenchmarkResult& result) {
  os << opts.system << "," << opts.nodes << "," << result.world_size << "," << result.workers << ","
     << result.time_per_shutdown_us << "," << result.iterations << "\n";
}

static BenchmarkResult run_benchmark([[maybe_unused]] const BenchmarkOptions& opts, MPI_Comm comm) {
  dynampi::MPICommunicator<> comm_wrapper(comm, dynampi::MPICommunicator<>::Ownership::Reference);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const uint64_t num_workers = (size == 1) ? 1 : static_cast<uint64_t>(size - 1);

  // Simple worker function that does nothing
  auto worker_function = [](Task task) -> Result { return static_cast<Result>(task); };

  MPI_Barrier(comm_wrapper);

  // Overall timer for 10-second duration
  dynampi::Timer overall_timer(dynampi::Timer::AutoStart::Yes);
  const double target_duration_s = 10.0;

  // Per-iteration timer
  dynampi::Timer iteration_timer(dynampi::Timer::AutoStart::No);

  double total_shutdown_time = 0.0;
  uint64_t iterations = 0;

  while (true) {
    bool should_continue = overall_timer.elapsed().count() < target_duration_s;
    comm_wrapper.broadcast(should_continue);
    if (!should_continue) {
      break;
    }
    // Ensure all workers are ready
    MPI_Barrier(comm_wrapper);

    {
      dynampi::NaiveMPIWorkDistributor<Task, Result> distributor(
          worker_function, {.comm = comm, .manager_rank = 0, .auto_run_workers = true});

      if (distributor.is_root_manager()) {
        iteration_timer.reset(dynampi::Timer::AutoStart::Yes);
        auto _ = distributor.finish_remaining_tasks();
        (void)_;
        iteration_timer.stop();
        total_shutdown_time += iteration_timer.elapsed().count();
        iterations++;
      }
    }

    // Barrier to ensure all processes complete shutdown before next iteration
    MPI_Barrier(comm_wrapper);
  }

  // Calculate average shutdown time in microseconds
  const double avg_shutdown_time_us =
      (iterations > 0) ? (total_shutdown_time / static_cast<double>(iterations)) * 1'000'000.0
                       : 0.0;

  return BenchmarkResult{num_workers, static_cast<uint64_t>(size), avg_shutdown_time_us,
                         iterations};
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cxxopts::Options options("naive_shutdown_time",
                           "Benchmark naive distributor shutdown time with no tasks");
  options.add_options()("n,nodes", "Number of nodes for labeling output (defaults to world size)",
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
  opts.nodes = args["nodes"].as<uint64_t>();
  opts.system = args["system"].as<std::string>();
  opts.output_path = args["output"].as<std::string>();

  {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (opts.nodes == 0) {
      opts.nodes = static_cast<uint64_t>(size);
    }

    BenchmarkResult result = run_benchmark(opts, comm);

    if (rank == 0) {
      std::cout << "RESULT"
                << " nodes=" << opts.nodes << " world_size=" << result.world_size
                << " workers=" << result.workers
                << " time_per_shutdown_us=" << result.time_per_shutdown_us
                << " iterations=" << result.iterations << std::endl;
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
