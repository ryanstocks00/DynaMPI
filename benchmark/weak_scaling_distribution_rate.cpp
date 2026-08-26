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
#include <dynampi/impl/hierarchical_lockfree_rma_distributor.hpp>
#include <dynampi/impl/lockfree_rma_distributor.hpp>
#include <dynampi/impl/naive_distributor.hpp>
#include <dynampi/mpi/mpi_communicator.hpp>
#include <dynampi/utilities/timer.hpp>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <string>

using Task = uint32_t;

// Default --max_tasks: RMA ring capacity and this driver's cumulative
// publication budget. Library rings reuse harvested slots; the driver still
// stops after this many published tasks so measurements match the archived
// sweeps. 500M (~19 GiB for this scalar workload) covers the 20 s paper
// window; lower it on memory-constrained hosts (the run warns if it ends early).
constexpr int kDefaultLockFreeMaxTasks = 500'000'000;

enum class DistributorKind {
  Naive,
  Hierarchical,
  LockFreeRMA,
  HierarchicalLockFreeRMA,
};
enum class DurationMode { Fixed, Uniform };

struct BenchmarkOptions {
  uint64_t expected_us = 1;
  double duration_s = 10.0;
  DistributorKind distributor = DistributorKind::Hierarchical;
  DurationMode duration_mode = DurationMode::Fixed;
  int max_upper_fanout = -1;
  int max_tasks = kDefaultLockFreeMaxTasks;
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
  if (value == "lockfree_rma") return DistributorKind::LockFreeRMA;
  if (value == "hierarchical_lockfree_rma") return DistributorKind::HierarchicalLockFreeRMA;
  throw std::runtime_error("Unknown distributor: " + value);
}

static DurationMode parse_duration_mode(const std::string& value) {
  if (value == "fixed") return DurationMode::Fixed;
  if (value == "uniform") return DurationMode::Uniform;
  throw std::runtime_error("Unknown duration mode: " + value);
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

static std::string to_string(DurationMode mode) {
  return mode == DurationMode::Fixed ? "fixed" : "uniform";
}

static void spin_wait(std::chrono::microseconds duration) {
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start < duration) {
  }
}

static void write_csv_header(std::ostream& os) {
  os << "system,distributor,mode,expected_us,"
        "duration_s,nodes,world_size,workers,max_upper_fanout,total_tasks,elapsed_s,"
        "throughput_tasks_per_s\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts,
                          const BenchmarkResult& result) {
  const double throughput =
      result.elapsed_s > 0.0 ? static_cast<double>(result.total_tasks) / result.elapsed_s : 0.0;
  os << opts.system << "," << to_string(opts.distributor) << "," << to_string(opts.duration_mode)
     << "," << opts.expected_us << "," << opts.duration_s << "," << opts.nodes << ","
     << result.world_size << "," << result.workers << "," << opts.max_upper_fanout << ","
     << result.total_tasks << "," << result.elapsed_s << "," << throughput << "\n";
}

// Prints the result and tears the whole job down immediately via MPI_Abort
// instead of returning normally and letting the distributor finalize/
// destruct cooperatively. finalize() (called explicitly or from the
// destructor if not already finalized) unconditionally drains every
// published-but-unclaimed task before returning; for the lock-free RMA classes
// that drain has no time bound at all, and at compute-bound or
// contention-collapsed configs it can take orders of magnitude longer than
// duration_s -- burning node-hours for a drain nobody needs once the
// measurement is already captured (confirmed: a single expected_us=1000000
// config left ~100K+ tasks unclaimed after a 20s window, whose finalize()
// drain alone would need ~1000s+ at that class's achievable rate). Must be
// called with the distributor still alive/in scope: returning normally and
// letting it go out of scope hits the exact same unbounded drain one frame
// up, via the destructor.
[[noreturn]] static void print_result_and_abort(MPI_Comm comm, const BenchmarkOptions& opts,
                                                const BenchmarkResult& result) {
  const double throughput =
      result.elapsed_s > 0.0 ? static_cast<double>(result.total_tasks) / result.elapsed_s : 0.0;
  std::cout << "RESULT"
            << " distributor=" << to_string(opts.distributor)
            << " mode=" << to_string(opts.duration_mode) << " expected_us=" << opts.expected_us
            << " nodes=" << opts.nodes << " world_size=" << result.world_size
            << " max_upper_fanout=" << opts.max_upper_fanout << " max_tasks=" << opts.max_tasks
            << " total_tasks=" << result.total_tasks << " elapsed_s=" << result.elapsed_s
            << " throughput_tasks_per_s=" << throughput << std::endl;
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
  std::cout.flush();
  MPI_Abort(comm, 0);
  std::_Exit(1);  // MPI_Abort should not return; fallback in case it does
}

struct WorkerFunctor {
  uint64_t expected_us;
  DurationMode duration_mode;
  std::mt19937_64 rng;
  std::uniform_int_distribution<uint64_t> uniform;

  WorkerFunctor(int rank, uint64_t expected_us_in, DurationMode mode)
      : expected_us(expected_us_in),
        duration_mode(mode),
        rng([rank]() {
          std::random_device rd;
          std::mt19937_64 seed_gen(rd());
          return seed_gen() + static_cast<uint64_t>(rank);
        }()),
        uniform(0, 2 * expected_us_in) {}

  uint32_t operator()(Task task) {
    uint32_t value = task;
    uint64_t duration_us = expected_us;
    if (duration_mode == DurationMode::Uniform) {
      duration_us = uniform(rng);
    }
    spin_wait(std::chrono::microseconds(duration_us));
    const uint64_t squared = static_cast<uint64_t>(value) * static_cast<uint64_t>(value);
    return static_cast<uint32_t>(squared);
  }
};

template <typename Distributor>
static BenchmarkResult run_benchmark(const BenchmarkOptions& opts, MPI_Comm comm) {
  dynampi::MPICommunicator<> comm_wrapper(comm, dynampi::MPICommunicator<>::Ownership::Reference);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const uint64_t num_workers = (size == 1) ? 1 : static_cast<uint64_t>(size - 1);

  WorkerFunctor worker_function(rank, opts.expected_us, opts.duration_mode);

  MPI_Barrier(comm_wrapper);
  dynampi::Timer timer(dynampi::Timer::AutoStart::No);
  uint64_t total_tasks = 0;

  typename Distributor::Config config{.comm = comm, .manager_rank = 0};
  if constexpr (requires { config.max_tasks; }) {
    config.max_tasks = opts.max_tasks;
  }
  if constexpr (requires { config.max_upper_fanout; }) {
    config.max_upper_fanout = opts.max_upper_fanout;
  }
  Distributor distributor(worker_function, config);

  if (distributor.is_root_manager()) {
    timer.start();

    // Keep a small, fixed-size queue topped up for the whole run and just
    // stop at duration_s -- no mid-run recalibration. An earlier version
    // estimated a "current_rate" from the cumulative total at the
    // duration_s/2 mark and throttled further inserts against it; that's a
    // single noisy snapshot (poisoned by topology-construction/warm-up time
    // dominating the early portion of a run, worse at larger scale) driving
    // every remaining insertion decision, and it produced wildly
    // irreproducible throughput at scale (e.g. hierarchical at
    // expected_us=1000: 5.1M tasks/s at 128 nodes, 600-700K at 256 nodes,
    // 2.3M at 512 nodes, all nominally the same configuration). That
    // throttle was guarding against a risk this driver doesn't actually
    // have: unlike the lock-free RMA classes' one-shot giant-batch drivers,
    // where overpublishing means an unbounded final drain, this queue is
    // continuously topped up in small increments (target_queue_size), so
    // the outstanding backlog at any moment -- including at
    // finish_remaining_tasks() -- is inherently bounded by
    // target_queue_size regardless of how fast or slow the run turns out
    // to be.
    const uint64_t target_queue_size = num_workers * 4;
    while (timer.elapsed().count() < opts.duration_s) {
      const uint64_t remaining = distributor.remaining_tasks_count();
      uint64_t to_insert = remaining < target_queue_size ? target_queue_size - remaining : 0;

      if (to_insert > 0) {
        std::vector<Task> tasks;
        tasks.reserve(to_insert);
        for (uint64_t i = 0; i < to_insert; ++i) {
          tasks.push_back(static_cast<Task>(total_tasks + i));
        }
        distributor.insert_tasks(tasks);
      }
      auto results =
          distributor.run_tasks({.target_num_tasks = num_workers * 2, .max_seconds = 0.1});
      total_tasks += results.size();
    }
    {
      auto results = distributor.finish_remaining_tasks();
      total_tasks += results.size();
    }
    timer.stop();
    print_result_and_abort(comm, opts,
                           BenchmarkResult{total_tasks, num_workers, static_cast<uint64_t>(size),
                                           timer.elapsed().count()});
  }

  return BenchmarkResult{total_tasks, num_workers, static_cast<uint64_t>(size),
                         timer.elapsed().count()};
}

// Shared by LockFreeRMAWorkDistributor and
// HierarchicalLockFreeRMAWorkDistributor -- they already implement
// the exact same insert_tasks()/run_tasks()/finalize() API, so the two
// drivers used to be byte-identical copy-pasted functions apart from Config
// setup. Shape: publish a bounded chunk of tasks, then just ask the
// distributor to run for the remaining time budget --
// run_tasks({.max_seconds = t}) already loops harvesting until either the
// time bound or task exhaustion, whichever comes first, so it correctly
// measures "how much got done in this much time" regardless of how the
// chunk was sized. If the chunk drains before duration_s elapses (the
// while condition is still true), publish another one and keep going.
//
// No calibration anywhere. An earlier version measured a rate from a short
// calibration window and used it to size one big batch for the entire rest
// of the run -- exactly the same "one snapshot decides everything"
// fragility that caused unreliable message-passing hierarchical
// measurements at scale (see run_benchmark()'s comment: the same
// expected_us=1000 config measured 5.1M tasks/s at 128 nodes, 600-700K at
// 256, 2.3M at 512, no trend, just noise from when the single snapshot was
// taken). Calibration was never actually needed for measurement accuracy
// -- run_tasks()'s own max_seconds bound already gets that right no matter
// how much was published. Its only real job was keeping the leftover small
// enough that finalize()'s mandatory full-drain (the protocol has no
// cancellation) doesn't blow up wall-clock time after the measurement is
// already locked in behind timer.stop(). Bounding every chunk to the same
// modest, scale-independent size does that job just as well without
// needing to estimate anything: worst case, at most one chunk is left
// outstanding when duration_s hits, not however large a rate-based guess
// happened to compute.
template <typename Distributor>
static BenchmarkResult run_benchmark_lockfree_rma_style(const BenchmarkOptions& opts,
                                                        MPI_Comm comm) {
  dynampi::MPICommunicator<> comm_wrapper(comm, dynampi::MPICommunicator<>::Ownership::Reference);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  const uint64_t num_workers = (size == 1) ? 1 : static_cast<uint64_t>(size - 1);
  WorkerFunctor worker_function(rank, opts.expected_us, opts.duration_mode);

  MPI_Barrier(comm_wrapper);
  dynampi::Timer timer(dynampi::Timer::AutoStart::No);
  uint64_t total_tasks = 0;

  typename Distributor::Config config{};
  config.comm = comm;
  config.manager_rank = 0;
  if constexpr (requires { config.max_tasks; }) config.max_tasks = opts.max_tasks;
  if constexpr (requires { config.max_local_tasks; }) config.max_local_tasks = opts.max_tasks;
  if constexpr (requires { config.max_upper_fanout; }) {
    config.max_upper_fanout = opts.max_upper_fanout;
  }
  Distributor distributor(worker_function, config);

  if (distributor.is_root_manager()) {
    timer.start();

    // Every chunk (first one and every top-up) uses this same bounded
    // size: floor of num_workers*1000 so it actually reaches the cap at
    // realistic scale, cap of 2,000,000 -- proven safe throughout this
    // codebase's history as a one-shot batch size (a single bulk Put of a
    // few tens of MB, cheap regardless of scale, and never implicated in
    // any of the multi-minute drain incidents that motivated removing
    // calibration -- those all came from a *second*, rate-estimated batch
    // on top of a batch this size, not from a batch this size on its own).
    //
    // The floor matters a lot for the flat (non-hierarchical) RMA class:
    // every worker draws from one shared atomic counter, so a too-small
    // chunk means all of them race through it almost immediately and then
    // simultaneously stall on a refill -- and each refill cycle (allocate
    // + bulk Put + resync) has fixed overhead that, repeated often enough,
    // dominates the run. Confirmed via a real regression: an earlier
    // version of this floor was num_workers*10, which is smaller than the
    // 2,000,000 cap for any node count below ~2000 -- so min(cap, floor)
    // always evaluated to the (too-small) floor, the cap never actually
    // engaged, and flat lockfree_rma at 128 nodes measured
    // 27,404 tasks/s versus a historical (old, calibration-based driver)
    // 241,596 tasks/s at the same config -- a ~9x self-inflicted
    // regression, not a real architectural ceiling. num_workers*1000
    // reaches the 2,000,000 cap by ~20 nodes, matching how large the old
    // driver's batches always ended up being in practice.
    const uint64_t chunk_size =
        std::min<uint64_t>(2'000'000, std::max<uint64_t>(num_workers * 1000, 1000));

    uint64_t published = 0;
    while (timer.elapsed().count() < opts.duration_s) {
      const uint64_t capacity_left = static_cast<uint64_t>(opts.max_tasks) - 1 - published;
      const uint64_t to_insert = std::min(chunk_size, capacity_left);
      if (to_insert == 0) {
        // Cumulative publication budget exhausted (rings could still reuse slots).
        std::cerr << "WARNING: --max_tasks publication budget (" << opts.max_tasks
                  << ") exhausted after " << timer.elapsed().count() << "s of a " << opts.duration_s
                  << "s window; raise it for a full-length measurement." << std::endl;
        break;
      }
      {
        std::vector<Task> tasks;
        tasks.reserve(to_insert);
        for (uint64_t i = 0; i < to_insert; ++i) tasks.push_back(static_cast<Task>(published + i));
        distributor.insert_tasks(tasks);
      }
      published += to_insert;

      auto results =
          distributor.run_tasks({.max_seconds = opts.duration_s - timer.elapsed().count()});
      total_tasks += results.size();
    }
    timer.stop();

    print_result_and_abort(comm, opts,
                           BenchmarkResult{total_tasks, num_workers, static_cast<uint64_t>(size),
                                           timer.elapsed().count()});
  }

  return BenchmarkResult{total_tasks, num_workers, static_cast<uint64_t>(size),
                         timer.elapsed().count()};
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cxxopts::Options options("weak_scaling_distribution_rate",
                           "Benchmark weak scaling task distribution throughput");
  options.add_options()("t,expected_us", "Expected task duration in microseconds",
                        cxxopts::value<uint64_t>()->default_value("1"))(
      "d,duration_s", "Target duration in seconds", cxxopts::value<double>()->default_value("10"))(
      "D,distribution",
      "Distribution strategy: naive, hierarchical, lockfree_rma, "
      "or hierarchical_lockfree_rma",
      cxxopts::value<std::string>()->default_value("hierarchical"))(
      "m,mode", "Duration mode: fixed or uniform (uniform on [0, 2x expected])",
      cxxopts::value<std::string>()->default_value("fixed"))(
      "max_upper_fanout",
      "hierarchical_lockfree_rma only: max direct children per manager "
      "above the node-local level. Negative (default) = auto, picking a fanout "
      "from manager count (see HierarchicalLockFreeRMAWorkDistributor's "
      "setup_upper_chain() for the formula); 0 = single unbounded manager level "
      "(matches pre-N-level behavior); >0 activates iterative k-ary grouping into "
      "multiple upper levels once the manager count exceeds this fanout.",
      cxxopts::value<int>()->default_value("-1"))(
      "max_tasks",
      "lockfree_rma and hierarchical_lockfree_rma only: reusable ring capacity "
      "of each preallocated RMA window and this benchmark's total publication "
      "budget, in tasks. The window is allocated up front on the owning rank "
      "(~40 bytes per slot here); too low and the run ends before duration_s.",
      cxxopts::value<int>()->default_value(std::to_string(kDefaultLockFreeMaxTasks)))(
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
    opts.duration_s = args["duration_s"].as<double>();
    opts.distributor = parse_distributor(args["distribution"].as<std::string>());
    opts.duration_mode = parse_duration_mode(args["mode"].as<std::string>());
    opts.max_upper_fanout = args["max_upper_fanout"].as<int>();
    opts.max_tasks = args["max_tasks"].as<int>();
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
      std::cerr << "Expected task duration must be >= 1 microsecond." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  if (opts.max_tasks <= 0) {
    if (world_rank == 0) {
      std::cerr << "--max_tasks must be positive." << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  {
    MPI_Comm comm = MPI_COMM_WORLD;
    int size = 0;
    MPI_Comm_size(comm, &size);
    if (opts.nodes == 0) {
      opts.nodes = static_cast<uint64_t>(size);
    }

    // Each driver prints the RESULT line and calls MPI_Abort itself, on the
    // manager rank, right after its own timer stops (see
    // print_result_and_abort's comment) -- so for every distributor kind,
    // either the manager rank aborts the whole job from inside the driver
    // call below, or (for non-manager ranks) control never returns from
    // constructing Distributor in the first place, since worker ranks block
    // there until they're killed by that same abort. Nothing below this
    // switch statement executes on a normal run; it only remains reachable
    // for build-configuration correctness.
    switch (opts.distributor) {
      case DistributorKind::Naive:
        run_benchmark<dynampi::NaiveWorkDistributor<Task, uint32_t>>(opts, comm);
        break;
      case DistributorKind::Hierarchical:
        run_benchmark<dynampi::HierarchicalWorkDistributor<Task, uint32_t>>(opts, comm);
        break;
      case DistributorKind::LockFreeRMA:
        run_benchmark_lockfree_rma_style<dynampi::LockFreeRMAWorkDistributor<Task, uint32_t>>(opts,
                                                                                              comm);
        break;
      case DistributorKind::HierarchicalLockFreeRMA:
        run_benchmark_lockfree_rma_style<
            dynampi::HierarchicalLockFreeRMAWorkDistributor<Task, uint32_t>>(opts, comm);
        break;
    }
  }
  MPI_Finalize();
  return 0;
}
