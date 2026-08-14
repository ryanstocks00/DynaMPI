/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>

#include <array>
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
#include <vector>

using Task = uint32_t;
using Result = uint32_t;

enum class DistributorKind {
  Naive,
  Hierarchical,
  LockFreeRMA,
  HierarchicalLockFreeRMA,
};

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

struct BenchmarkOptions {
  uint64_t expected_us = 1000;
  uint64_t min_tasks_per_worker = 1;
  uint64_t max_tasks_per_worker = 10;
  uint64_t repeats = 3;
  int max_upper_fanout = -1;
  uint64_t nodes = 0;
  std::string system;
  std::string output_path;
};

struct ResultRow {
  DistributorKind distributor;
  uint64_t tasks_per_worker = 0;
  uint64_t repeat = 0;
  uint64_t world_size = 0;
  uint64_t workers = 0;
  uint64_t total_tasks = 0;
  double elapsed_s = 0.0;
};

static void spin_wait(std::chrono::microseconds duration) {
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start < duration) {
  }
}

// Fixed-duration-only worker: this benchmark isolates scheduling/distribution
// overhead (load balancing) from task-duration variance, so unlike
// strong_scaling_distribution_rate's WorkerFunctor there is no random/Poisson
// mode here.
struct WorkerFunctor {
  uint64_t expected_us;

  Result operator()(Task task) {
    spin_wait(std::chrono::microseconds(expected_us));
    return static_cast<Result>(task);
  }
};

static void write_csv_header(std::ostream& os) {
  os << "system,distributor,nodes,world_size,workers,max_upper_fanout,expected_us,"
        "tasks_per_worker,repeat,total_tasks,elapsed_s\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts, const ResultRow& row) {
  os << opts.system << "," << to_string(row.distributor) << "," << opts.nodes << ","
     << row.world_size << "," << row.workers << "," << opts.max_upper_fanout << ","
     << opts.expected_us << "," << row.tasks_per_worker << "," << row.repeat << ","
     << row.total_tasks << "," << row.elapsed_s << "\n";
}

static void append_result(const BenchmarkOptions& opts, const ResultRow& row) {
  std::cout << "RESULT distributor=" << to_string(row.distributor) << " nodes=" << opts.nodes
            << " world_size=" << row.world_size << " workers=" << row.workers
            << " max_upper_fanout=" << opts.max_upper_fanout << " expected_us=" << opts.expected_us
            << " tasks_per_worker=" << row.tasks_per_worker << " repeat=" << row.repeat
            << " total_tasks=" << row.total_tasks << " elapsed_s=" << row.elapsed_s << std::endl;
  if (opts.output_path.empty()) return;
  std::ifstream check(opts.output_path);
  const bool needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
  check.close();
  std::ofstream out(opts.output_path, std::ios::app);
  if (needs_header) {
    write_csv_header(out);
  }
  write_csv_row(out, opts, row);
}

// Runs one (distributor kind, tasks_per_worker, repeat) combo: constructs a
// fresh distributor, publishes exactly tasks_per_worker * num_workers tasks,
// and times the manager's insert_tasks()+run_tasks() call. run_tasks() with a
// default RunConfig blocks until every published task is drained (its
// "total exhaustion" exit condition), for every distributor kind here -- so
// by construction there is nothing left outstanding when it returns, and the
// distributor's destructor (which runs when this function returns, releasing
// the workers via finalize()) is a cheap done-signal rather than an unbounded
// drain. All ranks call this the same number of times with the same static
// loop bounds (every rank parsed the same argv), so no cross-rank
// continue/stop coordination is needed the way shutdown_time.cpp needs one
// for its wall-clock-bounded loop.
template <typename Distributor>
static void run_batch(DistributorKind kind, uint64_t tasks_per_worker, uint64_t repeat,
                      const BenchmarkOptions& opts, MPI_Comm comm,
                      std::vector<ResultRow>& results) {
  int size = 0;
  MPI_Comm_size(comm, &size);
  const uint64_t num_workers = (size == 1) ? 1 : static_cast<uint64_t>(size - 1);
  // Every rank needs the same capacity (it's part of collective Config setup
  // below), not just the manager, so compute it up front rather than inside
  // the is_root_manager() branch.
  const uint64_t total_tasks = tasks_per_worker * num_workers;
  // One warmup task per worker, run and fully drained before the timer
  // starts. This absorbs each fresh distributor's one-time first-touch cost
  // (e.g. the RMA classes' first one-sided op per target lazily triggers
  // fabric-level connection/registration work, all landing on a single
  // window for the flat class) outside the timed region, so the timed batch
  // measures steady-state scheduling overhead instead of construction-
  // adjacent noise. Confirmed necessary: without it, lockfree_rma's k=1
  // measurement was 3-10x its own trend line and fell with successive
  // repeats within the same run (each repeat rebuilds the distributor from
  // scratch, so that cost recurs every time, not just once per process).
  const uint64_t warmup_tasks = num_workers;

  WorkerFunctor worker_function{opts.expected_us};

  typename Distributor::Config config{.comm = comm, .manager_rank = 0};
  if constexpr (requires { config.max_upper_fanout; }) {
    config.max_upper_fanout = opts.max_upper_fanout;
  }
  // The lock-free RMA classes preallocate their task/result window to this
  // capacity (library default is a modest 8192, not the 500M constant
  // strong_scaling_distribution_rate.cpp always overrides it with) -- size it
  // to cover both the warmup and the real batch, plus headroom for the "-1"
  // reserved slot other drivers in this codebase budget for.
  const int max_tasks_capacity = static_cast<int>(warmup_tasks + total_tasks) + 8;
  if constexpr (requires { config.max_tasks; }) config.max_tasks = max_tasks_capacity;
  if constexpr (requires { config.max_local_tasks; }) config.max_local_tasks = max_tasks_capacity;
  Distributor distributor(worker_function, config);

  if (distributor.is_root_manager()) {
    std::vector<Task> warmup(warmup_tasks);
    for (uint64_t i = 0; i < warmup_tasks; ++i) warmup[i] = static_cast<Task>(i);
    distributor.insert_tasks(warmup);
    auto warmup_results = distributor.run_tasks();
    (void)warmup_results;

    std::vector<Task> tasks(total_tasks);
    for (uint64_t i = 0; i < total_tasks; ++i) tasks[i] = static_cast<Task>(warmup_tasks + i);

    dynampi::Timer timer;
    distributor.insert_tasks(tasks);
    auto task_results = distributor.run_tasks();
    (void)task_results;
    const double elapsed_s = timer.stop().count();

    results.push_back(ResultRow{kind, tasks_per_worker, repeat, static_cast<uint64_t>(size),
                                num_workers, total_tasks, elapsed_s});
  }
  // distributor's destructor runs finalize() here, releasing worker ranks
  // (already fully drained above, so this is just the done-signal, not a
  // drain -- safe to repeat across every combo in this loop).
}

static void run_all(DistributorKind kind, const BenchmarkOptions& opts, MPI_Comm comm,
                    std::vector<ResultRow>& results) {
  for (uint64_t k = opts.min_tasks_per_worker; k <= opts.max_tasks_per_worker; ++k) {
    for (uint64_t repeat = 0; repeat < opts.repeats; ++repeat) {
      MPI_Barrier(comm);
      switch (kind) {
        case DistributorKind::Naive:
          run_batch<dynampi::NaiveWorkDistributor<Task, Result>>(kind, k, repeat, opts, comm,
                                                                 results);
          break;
        case DistributorKind::Hierarchical:
          run_batch<dynampi::HierarchicalWorkDistributor<Task, Result>>(kind, k, repeat, opts, comm,
                                                                        results);
          break;
        case DistributorKind::LockFreeRMA:
          run_batch<dynampi::LockFreeRMAWorkDistributor<Task, Result>>(kind, k, repeat, opts, comm,
                                                                       results);
          break;
        case DistributorKind::HierarchicalLockFreeRMA:
          run_batch<dynampi::HierarchicalLockFreeRMAWorkDistributor<Task, Result>>(
              kind, k, repeat, opts, comm, results);
          break;
      }
    }
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cxxopts::Options options(
      "load_balancing_makespan",
      "Times each distributor's wall-clock drain of a fixed-size task batch "
      "(tasks_per_worker * workers tasks) across a range of tasks_per_worker "
      "values, to compare load-balancing/scheduling overhead between distributors.");
  options.add_options()("t,expected_us", "Fixed task duration in microseconds",
                        cxxopts::value<uint64_t>()->default_value("1000"))(
      "min_tasks_per_worker", "Smallest tasks-per-worker batch size to test",
      cxxopts::value<uint64_t>()->default_value("1"))(
      "max_tasks_per_worker", "Largest tasks-per-worker batch size to test",
      cxxopts::value<uint64_t>()->default_value("10"))(
      "r,repeats", "Number of repeats averaged per (distributor, tasks_per_worker) combo",
      cxxopts::value<uint64_t>()->default_value("3"))(
      "max_upper_fanout",
      "hierarchical and hierarchical_lockfree_rma only: max direct children per "
      "coordinator above the node-local level. Negative (default) = auto.",
      cxxopts::value<int>()->default_value("-1"))(
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
  try {
    opts.expected_us = args["expected_us"].as<uint64_t>();
    opts.min_tasks_per_worker = args["min_tasks_per_worker"].as<uint64_t>();
    opts.max_tasks_per_worker = args["max_tasks_per_worker"].as<uint64_t>();
    opts.repeats = args["repeats"].as<uint64_t>();
    opts.max_upper_fanout = args["max_upper_fanout"].as<int>();
    opts.nodes = args["nodes"].as<uint64_t>();
    opts.system = args["system"].as<std::string>();
    opts.output_path = args["output"].as<std::string>();
  } catch (const std::exception& e) {
    if (world_rank == 0) {
      std::cerr << "Error: " << e.what() << "\n" << options.help() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  if (opts.expected_us == 0) {
    if (world_rank == 0) {
      std::cerr << "--expected_us must be >= 1 microsecond." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }
  if (opts.min_tasks_per_worker == 0 || opts.min_tasks_per_worker > opts.max_tasks_per_worker) {
    if (world_rank == 0) {
      std::cerr << "--min_tasks_per_worker must be >= 1 and <= --max_tasks_per_worker."
                << std::endl;
    }
    MPI_Finalize();
    return 1;
  }
  if (opts.repeats == 0) {
    if (world_rank == 0) {
      std::cerr << "--repeats must be >= 1." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  try {
    MPI_Comm comm = MPI_COMM_WORLD;
    int size = 0;
    MPI_Comm_size(comm, &size);
    if (opts.nodes == 0) {
      opts.nodes = static_cast<uint64_t>(size);
    }

    static constexpr std::array<DistributorKind, 4> kKinds = {
        DistributorKind::Naive, DistributorKind::Hierarchical, DistributorKind::LockFreeRMA,
        DistributorKind::HierarchicalLockFreeRMA};

    std::vector<ResultRow> results;
    for (auto kind : kKinds) {
      const size_t before = results.size();
      run_all(kind, opts, comm, results);
      if (world_rank == 0) {
        for (size_t i = before; i < results.size(); ++i) {
          append_result(opts, results[i]);
        }
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
