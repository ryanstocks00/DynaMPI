/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>

#include <cstdint>
#include <cxxopts.hpp>
#include <dynampi/impl/naive_distributor.hpp>
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
  double shutdown_time_s = 0.0;
};

static void write_csv_header(std::ostream& os) {
  os << "system,nodes,world_size,workers,shutdown_time_s\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts,
                          const BenchmarkResult& result) {
  os << opts.system << "," << opts.nodes << "," << result.world_size << "," << result.workers << ","
     << result.shutdown_time_s << "\n";
}

class CommWrapper {
 public:
  explicit CommWrapper(MPI_Comm comm) : comm_(comm) {}
  void barrier() const { MPI_Barrier(comm_); }
  MPI_Comm get() const { return comm_; }

 private:
  MPI_Comm comm_;
};

static BenchmarkResult run_benchmark([[maybe_unused]] const BenchmarkOptions& opts, MPI_Comm comm) {
  CommWrapper comm_wrapper(comm);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const uint64_t num_workers = (size == 1) ? 1 : static_cast<uint64_t>(size - 1);

  // Simple worker function that does nothing
  auto worker_function = [](Task task) -> Result { return static_cast<Result>(task); };

  comm_wrapper.barrier();

  dynampi::Timer timer(dynampi::Timer::AutoStart::No);
  double shutdown_time = 0.0;

  {
    dynampi::NaiveMPIWorkDistributor<Task, Result> distributor(worker_function,
                                                               {.comm = comm, .manager_rank = 0});

    // Ensure all workers are ready
    comm_wrapper.barrier();

    if (distributor.is_root_manager()) {
      timer.start();
      distributor.finalize();
      timer.stop();
      shutdown_time = timer.elapsed().count();
    } else {
      // Workers will exit when they receive DONE message
      // The distributor destructor will handle cleanup
    }
  }

  // Barrier to ensure all processes complete shutdown
  comm_wrapper.barrier();

  return BenchmarkResult{num_workers, static_cast<uint64_t>(size), shutdown_time};
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
                << " workers=" << result.workers << " shutdown_time_s=" << result.shutdown_time_s
                << std::endl;
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
