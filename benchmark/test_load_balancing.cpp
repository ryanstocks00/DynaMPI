/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>

#include <array>
#include <cmath>
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
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Task = uint32_t;
using Result = uint32_t;

enum class DurationMode {
  Fixed,
  Uniform,
  LogNormal,
};

enum class DistributorKind {
  Naive,
  Hierarchical,
  LockFreeRMA,
  HierarchicalLockFreeRMA,
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

struct BenchmarkOptions {
  uint64_t expected_us = 1000;
  DurationMode duration_mode = DurationMode::Fixed;
  // Coefficient of variation (stddev/mean) for --duration_mode=lognormal.
  // 1.0 is a standard "high variability" choice for this kind of sweep --
  // stddev equal to the mean, heavily right-skewed (most tasks well under
  // expected_us, a long tail of much longer ones) -- rather than anything
  // measured from a specific real workload.
  double task_duration_cv = 1.0;
  uint64_t min_tasks_per_worker = 1;
  uint64_t max_tasks_per_worker = 10;
  uint64_t repeats = 3;
  int max_upper_fanout = -1;
  // <0 (default): leave the library's own default in place. Only forwarded
  // to distributors whose Config actually has the matching field (see the
  // if constexpr guards in run_batch), so an override for one class's knob
  // never gets silently applied where it doesn't apply.
  int pipeline_depth = -1;
  int max_pending_rounds = -1;
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
  // Manager-side wall time for each phase outside the timed batch drain
  // above, so the fixed per-combo cost (construction, warmup, the done
  // broadcast, and destructor teardown) can be reported separately instead
  // of folded into elapsed_s or left to the sub-microsecond floor a k=0 row
  // shows on its own.
  double construct_s = 0.0;
  double warmup_s = 0.0;
  double finalize_s = 0.0;
  double destruct_s = 0.0;
};

static void spin_wait(std::chrono::microseconds duration) {
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start < duration) {
  }
}

// Fixed by default so a run isolates scheduling/distribution overhead from
// task-duration variance. Two ways to opt back into variance, both mean
// expected_us: uniform on [0, 2*expected_us] -- the same symmetric spread
// strong_scaling_distribution_rate's WorkerFunctor calls its "random" mode --
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
        lognormal(lognormal_mu(expected_us_in, task_duration_cv),
                  lognormal_sigma(task_duration_cv)) {}

  // Derives the underlying normal distribution's (mu, sigma) so the
  // lognormal's own mean and coefficient of variation match expected_us and
  // task_duration_cv exactly, rather than expecting the caller to reason
  // about the underlying-normal parameterization std::lognormal_distribution
  // itself takes.
  static double lognormal_sigma(double cv) { return std::sqrt(std::log(1.0 + cv * cv)); }
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

static std::string to_string(DurationMode mode) {
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

static void write_csv_header(std::ostream& os) {
  os << "system,distributor,nodes,world_size,workers,max_upper_fanout,pipeline_depth,"
        "max_pending_rounds,expected_us,duration_mode,task_duration_cv,tasks_per_worker,repeat,"
        "total_tasks,elapsed_s,construct_s,warmup_s,finalize_s,destruct_s\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts, const ResultRow& row) {
  os << opts.system << "," << to_string(row.distributor) << "," << opts.nodes << ","
     << row.world_size << "," << row.workers << "," << opts.max_upper_fanout << ","
     << opts.pipeline_depth << "," << opts.max_pending_rounds << "," << opts.expected_us << ","
     << to_string(opts.duration_mode) << "," << opts.task_duration_cv << ","
     << row.tasks_per_worker << "," << row.repeat << "," << row.total_tasks << ","
     << row.elapsed_s << "," << row.construct_s << "," << row.warmup_s << "," << row.finalize_s
     << "," << row.destruct_s << "\n";
}

static void append_result(const BenchmarkOptions& opts, const ResultRow& row) {
  std::cout << "RESULT distributor=" << to_string(row.distributor) << " nodes=" << opts.nodes
            << " world_size=" << row.world_size << " workers=" << row.workers
            << " max_upper_fanout=" << opts.max_upper_fanout
            << " pipeline_depth=" << opts.pipeline_depth
            << " max_pending_rounds=" << opts.max_pending_rounds
            << " expected_us=" << opts.expected_us
            << " duration_mode=" << to_string(opts.duration_mode)
            << " task_duration_cv=" << opts.task_duration_cv
            << " tasks_per_worker=" << row.tasks_per_worker << " repeat=" << row.repeat
            << " total_tasks=" << row.total_tasks << " elapsed_s=" << row.elapsed_s
            << " construct_s=" << row.construct_s << " warmup_s=" << row.warmup_s
            << " finalize_s=" << row.finalize_s << " destruct_s=" << row.destruct_s << std::endl;
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
// fresh distributor, publishes exactly tasks_per_worker * hierarchical's
// worker count tasks (the same total batch size across all four classes at a
// given k, even though the other three have more real workers -- see
// hierarchical_worker_count below), and times the manager's
// insert_tasks()+run_tasks() call. run_tasks() with a
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
  int rank = 0;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  const uint64_t num_workers = (size == 1) ? 1 : static_cast<uint64_t>(size - 1);

  // world_size - 1 counts every non-manager rank, but hierarchical's node
  // managers route/coordinate rather than execute worker_function
  // themselves (unless their node is single-core, i.e. nodes == world_size)
  // -- so for that class alone, world_size - 1 overcounts task-executing
  // ranks by one node manager per node. Computed analytically rather than
  // queried from the constructed distributor: HierarchicalWorkDistributor's
  // constructor blocks every non-manager rank inside its own worker loop
  // (auto_run_workers) until finalize() runs, so a post-construction
  // collective (e.g. reducing over is_leaf_worker()) only the manager rank
  // ever reaches deadlocks immediately -- confirmed the hard way, hung for
  // the full job walltime on the very first hierarchical combo. Exactly one
  // non-leaf rank exists per node (that node's manager, or the root manager
  // on the root's node -- see hierarchical_distributor.hpp's
  // is_leaf_worker()), so real leaf-worker count is world_size - nodes.
  const bool nodes_known = opts.nodes > 0 && static_cast<uint64_t>(size) > opts.nodes;
  const uint64_t real_worker_count =
      (kind == DistributorKind::Hierarchical && nodes_known)
          ? static_cast<uint64_t>(size) - opts.nodes
          : num_workers;
  // hierarchical's worker count (world_size - nodes, always <= the other
  // three classes' world_size - 1) is the shared baseline for how many
  // tasks a combo actually publishes, so tasks_per_worker means the same
  // total batch size across all four distributors at a given k -- otherwise
  // the flat/RMA classes' extra node-manager-turned-worker ranks would make
  // their batches larger than hierarchical's for the "same" k, comparing
  // different amounts of total work rather than just scheduling overhead.
  // real_worker_count above is still used for warmup sizing and the
  // reported "workers" column, since that's about warming/reporting each
  // class's own real ranks, not the shared batch size.
  const uint64_t hierarchical_worker_count =
      nodes_known ? static_cast<uint64_t>(size) - opts.nodes : num_workers;
  // Upper bound only, used purely to size the RMA classes' preallocated
  // window before construction. Every non-manager rank is a safe
  // over-estimate of how many ranks will ever actually claim a task.
  const uint64_t max_possible_tasks = opts.max_tasks_per_worker * num_workers;

  WorkerFunctor worker_function(rank, opts.expected_us, opts.duration_mode, opts.task_duration_cv);

  typename Distributor::Config config{.comm = comm, .manager_rank = 0};
  if constexpr (requires { config.max_upper_fanout; }) {
    config.max_upper_fanout = opts.max_upper_fanout;
  }
  if constexpr (requires { config.pipeline_depth; }) {
    if (opts.pipeline_depth >= 0) config.pipeline_depth = opts.pipeline_depth;
  }
  if constexpr (requires { config.max_pending_rounds; }) {
    if (opts.max_pending_rounds >= 0) config.max_pending_rounds = opts.max_pending_rounds;
  }
  // The lock-free RMA classes preallocate their task/result window to this
  // capacity (library default is a modest 8192, not the 500M constant
  // strong_scaling_distribution_rate.cpp always overrides it with) -- size it
  // to cover both the warmup and the largest real batch this run will ever
  // publish, plus headroom for the "-1" reserved slot other drivers in this
  // codebase budget for.
  const int max_tasks_capacity = static_cast<int>(2 * max_possible_tasks) + 8;
  if constexpr (requires { config.max_tasks; }) config.max_tasks = max_tasks_capacity;
  if constexpr (requires { config.max_local_tasks; }) config.max_local_tasks = max_tasks_capacity;

  // Heap-allocated and explicitly reset() (rather than a plain stack object)
  // so construction and destruction can each be timed as their own phase --
  // a plain local can't have its destructor's own duration measured, since
  // whatever Timer would stop it is itself out of scope by the time the
  // destructor runs at the closing brace.
  //
  // On non-manager ranks, construct_timer only measures until this rank
  // returns from the constructor -- which (auto_run_workers) blocks for this
  // combo's *entire* lifetime, not just its own setup, so the value is
  // meaningless there. Same for destruct_timer's local reading on managers
  // vs workers below. Harmless: only the manager's readings are ever pushed
  // to results.
  dynampi::Timer construct_timer;
  auto distributor = std::make_unique<Distributor>(worker_function, config);
  const double construct_s = construct_timer.stop().count();

  const bool is_manager = distributor->is_root_manager();
  double warmup_s = 0.0;
  double elapsed_s = 0.0;
  double finalize_s = 0.0;
  uint64_t total_tasks = 0;

  if (is_manager) {
    // One warmup task per real worker, run and fully drained before the
    // batch timer starts. This absorbs each fresh distributor's one-time
    // first-touch cost (e.g. the RMA classes' first one-sided op per target
    // lazily triggers fabric-level connection/registration work, all
    // landing on a single window for the flat class) outside the timed
    // region, so the timed batch measures steady-state scheduling overhead
    // instead of construction-adjacent noise. Confirmed necessary: without
    // it, lockfree_rma's k=1 measurement was 3-10x its own trend line and
    // fell with successive repeats within the same run (each repeat rebuilds
    // the distributor from scratch, so that cost recurs every time, not just
    // once per process).
    const uint64_t warmup_tasks = real_worker_count;
    dynampi::Timer warmup_timer;
    if (warmup_tasks > 0) {
      std::vector<Task> warmup(warmup_tasks);
      for (uint64_t i = 0; i < warmup_tasks; ++i) warmup[i] = static_cast<Task>(i);
      distributor->insert_tasks(warmup);
      auto warmup_results = distributor->run_tasks();
      (void)warmup_results;
    }
    warmup_s = warmup_timer.stop().count();

    total_tasks = tasks_per_worker * hierarchical_worker_count;
    std::vector<Task> tasks(total_tasks);
    for (uint64_t i = 0; i < total_tasks; ++i) tasks[i] = static_cast<Task>(warmup_tasks + i);

    dynampi::Timer batch_timer;
    if (total_tasks > 0) {
      distributor->insert_tasks(tasks);
      auto task_results = distributor->run_tasks();
      (void)task_results;
    }
    elapsed_s = batch_timer.stop().count();

    // Explicit rather than left to the destructor below, so its cost (the
    // done broadcast that releases every worker from its blocked
    // constructor call, see setup_leader_hierarchy()/run_worker() for
    // hierarchical, broadcast_done() for the flat classes) is its own
    // measurement instead of folded into destruct_s.
    dynampi::Timer finalize_timer;
    distributor->finalize();
    finalize_s = finalize_timer.stop().count();
  }

  // Runs on every rank at roughly the same point: the manager reaches it
  // immediately after finalize() above; workers reach it as soon as that
  // finalize() call's broadcast releases them from the constructor. Window
  // free, communicator teardown, etc. -- real cleanup cost, distinct from
  // finalize()'s done-signal.
  dynampi::Timer destruct_timer;
  distributor.reset();
  const double destruct_s = destruct_timer.stop().count();

  if (is_manager) {
    results.push_back(ResultRow{kind, tasks_per_worker, repeat, static_cast<uint64_t>(size),
                                real_worker_count, total_tasks, elapsed_s, construct_s, warmup_s,
                                finalize_s, destruct_s});
  }
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
      "test_load_balancing",
      "Times each distributor's wall-clock drain of a fixed-size task batch "
      "(tasks_per_worker * workers tasks) across a range of tasks_per_worker "
      "values, to compare load-balancing/scheduling overhead between distributors.");
  options.add_options()("t,expected_us", "Fixed task duration in microseconds",
                        cxxopts::value<uint64_t>()->default_value("1000"))(
      "duration_mode",
      "Task duration distribution, all with mean expected_us: fixed (every "
      "task takes exactly expected_us), uniform (symmetric on [0, "
      "2*expected_us], matches strong_scaling_distribution_rate's \"random\" "
      "mode), or lognormal (shaped like real task-duration variance -- "
      "mostly short, a long right tail, no hard cutoff).",
      cxxopts::value<std::string>()->default_value("fixed"))(
      "task_duration_cv",
      "duration_mode=lognormal only: coefficient of variation (stddev/mean).",
      cxxopts::value<double>()->default_value("1.0"))(
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
      "pipeline_depth",
      "hierarchical only: batches kept in flight at once (1 disables "
      "prefetching, 2 is double-buffering). Negative (default) leaves the "
      "library default (2) in place.",
      cxxopts::value<int>()->default_value("-1"))(
      "max_pending_rounds",
      "hierarchical_lockfree_rma only: rounds (at the parent's own claim "
      "granularity) a relay hop may claim ahead of its parent before backing "
      "off. Negative (default) leaves the library default (8) in place.",
      cxxopts::value<int>()->default_value("-1"))(
      "D,distribution",
      "Comma-separated distributor(s) to run: naive, hierarchical, lockfree_rma, "
      "hierarchical_lockfree_rma, or all (default).",
      cxxopts::value<std::string>()->default_value("all"))(
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
    const std::string duration_mode_arg = args["duration_mode"].as<std::string>();
    if (duration_mode_arg == "fixed") {
      opts.duration_mode = DurationMode::Fixed;
    } else if (duration_mode_arg == "uniform") {
      opts.duration_mode = DurationMode::Uniform;
    } else if (duration_mode_arg == "lognormal") {
      opts.duration_mode = DurationMode::LogNormal;
    } else {
      throw std::runtime_error("Unknown duration_mode: " + duration_mode_arg);
    }
    opts.task_duration_cv = args["task_duration_cv"].as<double>();
    opts.min_tasks_per_worker = args["min_tasks_per_worker"].as<uint64_t>();
    opts.max_tasks_per_worker = args["max_tasks_per_worker"].as<uint64_t>();
    opts.repeats = args["repeats"].as<uint64_t>();
    opts.max_upper_fanout = args["max_upper_fanout"].as<int>();
    opts.pipeline_depth = args["pipeline_depth"].as<int>();
    opts.max_pending_rounds = args["max_pending_rounds"].as<int>();
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
  if (opts.min_tasks_per_worker > opts.max_tasks_per_worker) {
    if (world_rank == 0) {
      std::cerr << "--min_tasks_per_worker must be <= --max_tasks_per_worker." << std::endl;
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
    const std::string distribution_arg = args["distribution"].as<std::string>();
    MPI_Comm comm = MPI_COMM_WORLD;
    int size = 0;
    MPI_Comm_size(comm, &size);
    if (opts.nodes == 0) {
      opts.nodes = static_cast<uint64_t>(size);
    }

    std::vector<DistributorKind> kinds;
    if (distribution_arg == "all") {
      kinds = {DistributorKind::Naive, DistributorKind::Hierarchical,
               DistributorKind::LockFreeRMA, DistributorKind::HierarchicalLockFreeRMA};
    } else {
      size_t start = 0;
      while (start <= distribution_arg.size()) {
        const size_t comma = distribution_arg.find(',', start);
        const size_t end = (comma == std::string::npos) ? distribution_arg.size() : comma;
        kinds.push_back(parse_distributor(distribution_arg.substr(start, end - start)));
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
    }

    std::vector<ResultRow> results;
    for (auto kind : kinds) {
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
