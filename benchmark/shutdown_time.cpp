/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>

#include <cstdint>
#include <cxxopts.hpp>
#include <dynampi/impl/hierarchical_distributor.hpp>
#include <dynampi/impl/hierarchical_lockfree_rma_distributor.hpp>
#include <dynampi/impl/lockfree_rma_distributor.hpp>
#include <dynampi/impl/naive_distributor.hpp>
#include <dynampi/mpi/mpi_communicator.hpp>
#include <dynampi/utilities/timer.hpp>
#include <fstream>
#include <iostream>
#include <string>

using Task = uint32_t;
using Result = uint32_t;

enum class DistributorKind {
  Naive,
  Hierarchical,
  LockFreeRMA,
  HierarchicalLockFreeRMA,
};

struct BenchmarkOptions {
  DistributorKind distributor = DistributorKind::Naive;
  uint64_t nodes = 0;
  int max_upper_fanout = -1;
  std::string system;
  std::string output_path;
};

struct BenchmarkResult {
  uint64_t workers = 0;
  uint64_t world_size = 0;
  double time_per_shutdown_us = 0.0;
  uint64_t iterations = 0;
};

static DistributorKind parse_distributor(const std::string& value) {
  if (value == "naive") return DistributorKind::Naive;
  if (value == "hierarchical") return DistributorKind::Hierarchical;
  if (value == "lockfree_rma") return DistributorKind::LockFreeRMA;
  if (value == "hierarchical_lockfree_rma") return DistributorKind::HierarchicalLockFreeRMA;
  throw std::runtime_error("Unknown distributor: " + value);
}

static std::string to_string(DistributorKind kind) {
  switch (kind) {
    case DistributorKind::Naive:
      return "naive";
    case DistributorKind::Hierarchical:
      return "hierarchical";
    case DistributorKind::LockFreeRMA:
      return "lockfree_rma";
    case DistributorKind::HierarchicalLockFreeRMA:
      return "hierarchical_lockfree_rma";
  }
  return "unknown";
}

static void write_csv_header(std::ostream& os) {
  os << "system,distributor,nodes,max_upper_fanout,world_size,workers,time_per_shutdown_us,"
        "iterations\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts,
                          const BenchmarkResult& result) {
  os << opts.system << "," << to_string(opts.distributor) << "," << opts.nodes << ","
     << opts.max_upper_fanout << "," << result.world_size << "," << result.workers << ","
     << result.time_per_shutdown_us << "," << result.iterations << "\n";
}

template <typename Distributor>
static BenchmarkResult run_benchmark([[maybe_unused]] const BenchmarkOptions& opts, MPI_Comm comm) {
  dynampi::MPICommunicator<> comm_wrapper(comm, dynampi::MPICommunicator<>::Ownership::Reference);
  int size = 0;
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

    // Must match the is_root_manager() start path (not rank==0 alone).
    bool timed_this_iteration = false;
    {
      // No max_tasks override needed here (unlike strong_scaling_distribution_rate.cpp):
      // this benchmark never calls insert_task/insert_tasks, so lockfree's
      // task-table capacity is never exercised regardless of its size --
      // each iteration constructs a fresh, empty distributor. The library
      // default is fine.
      typename Distributor::Config config{
          .comm = comm, .manager_rank = 0, .auto_run_workers = true};
      if constexpr (requires { config.max_upper_fanout; }) {
        config.max_upper_fanout = opts.max_upper_fanout;
      }
      Distributor distributor(worker_function, config);

      if (distributor.is_root_manager()) {
        iteration_timer.reset(dynampi::Timer::AutoStart::Yes);
        timed_this_iteration = true;
        auto _ = distributor.finish_remaining_tasks();
        (void)_;
        // Explicitly finalize here, inside the timed region, rather than
        // leaving it to distributor's destructor at the closing brace
        // below. finalize() is what actually does the real shutdown
        // signaling (e.g. HierarchicalWorkDistributor's
        // send_done_to_children_when_free(), which tells the whole
        // coordinator tree to stop) -- finish_remaining_tasks() alone has
        // nothing to do here since this benchmark never publishes any
        // tasks, so it returns near-instantly regardless of distributor
        // kind. Before this fix, finalize() only ran via "if
        // (!m_finalized) finalize();" in the destructor, which fires
        // *after* the timer already stopped -- so every hierarchical-class
        // measurement was timing a no-op, not real teardown (confirmed:
        // consistently ~60-70ns, flat across every node count from 1 to
        // 128, orders of magnitude faster than a single network round
        // trip, while naive -- whose real signaling apparently happens
        // synchronously inside finish_remaining_tasks() rather than being
        // deferred -- measured real, node-count-scaling numbers).
        distributor.finalize();
      }
      // Left in scope deliberately: distributor's destructor (window
      // free, communicator cleanup, etc. -- real teardown cost, not just
      // the done-signaling finalize() already covers above) still needs
      // to run before we stop the clock, so the timer stop is placed
      // after this block closes rather than immediately after
      // finalize().
    }
    if (timed_this_iteration) {
      iteration_timer.stop();
      total_shutdown_time += iteration_timer.elapsed().count();
      iterations++;
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

  cxxopts::Options options("shutdown_time",
                           "Benchmark distributor construct/shutdown time with no tasks");
  options.add_options()("D,distribution",
                        "Distribution strategy: naive, hierarchical, lockfree_rma, "
                        "or hierarchical_lockfree_rma",
                        cxxopts::value<std::string>()->default_value("naive"))(
      "n,nodes", "Number of nodes for labeling output (defaults to world size)",
      cxxopts::value<uint64_t>()->default_value("0"))(
      "max_upper_fanout",
      "hierarchical and hierarchical_lockfree_rma only: max direct children per "
      "coordinator above the node-local level. Negative (default) = auto; 0 = single "
      "unbounded coordinator level (1-layer); >0 activates k-ary grouping into multiple "
      "upper levels once coordinator count exceeds this fanout.",
      cxxopts::value<int>()->default_value("-1"))(
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
  try {
    opts.distributor = parse_distributor(args["distribution"].as<std::string>());
    opts.nodes = args["nodes"].as<uint64_t>();
    opts.max_upper_fanout = args["max_upper_fanout"].as<int>();
    opts.system = args["system"].as<std::string>();
    opts.output_path = args["output"].as<std::string>();
  } catch (const std::exception& e) {
    if (world_rank == 0) {
      std::cerr << "Error: " << e.what() << "\n" << options.help() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  try {
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (opts.nodes == 0) {
      opts.nodes = static_cast<uint64_t>(size);
    }

    BenchmarkResult result;
    switch (opts.distributor) {
      case DistributorKind::Naive:
        result = run_benchmark<dynampi::NaiveWorkDistributor<Task, Result>>(opts, comm);
        break;
      case DistributorKind::Hierarchical:
        result = run_benchmark<dynampi::HierarchicalWorkDistributor<Task, Result>>(opts, comm);
        break;
      case DistributorKind::LockFreeRMA:
        result = run_benchmark<dynampi::LockFreeRMAWorkDistributor<Task, Result>>(opts, comm);
        break;
      case DistributorKind::HierarchicalLockFreeRMA:
        result = run_benchmark<dynampi::HierarchicalLockFreeRMAWorkDistributor<Task, Result>>(opts,
                                                                                              comm);
        break;
    }

    if (rank == 0) {
      std::cout << "RESULT"
                << " distributor=" << to_string(opts.distributor) << " nodes=" << opts.nodes
                << " max_upper_fanout=" << opts.max_upper_fanout
                << " world_size=" << result.world_size << " workers=" << result.workers
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
  } catch (const std::exception& e) {
    if (world_rank == 0) {
      std::cerr << "Benchmark failed: " << e.what() << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  } catch (...) {
    if (world_rank == 0) {
      std::cerr << "Benchmark failed: unknown error" << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Finalize();
  return 0;
}
