/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <dynampi/impl/async_put_lockfree_distributor.hpp>
#include <dynampi/impl/hierarchical_async_put_lockfree_distributor.hpp>
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

// Lockfree pre-allocates a fixed-capacity task/result table sized for the
// entire lifetime of the distributor (not a ring buffer). Sized generously
// above realistic sustained throughput at 2048 nodes; publish_task() only
// guards overflow with an assert(), which is compiled out under NDEBUG
// (our Release build), so undersizing this corrupts memory instead of
// erroring cleanly.
constexpr int kLockFreeMaxTasks = 100'000'000;

enum class DistributorKind { Naive, Hierarchical, AsyncPutLockFree, HierarchicalAsyncPutLockFree };
enum class DurationMode { Fixed, Poisson };

struct BenchmarkOptions {
  uint64_t expected_us = 1;
  double duration_s = 10.0;
  DistributorKind distributor = DistributorKind::Hierarchical;
  DurationMode duration_mode = DurationMode::Fixed;
  int max_upper_fanout = 0;
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
  if (value == "async_put_lockfree") return DistributorKind::AsyncPutLockFree;
  if (value == "hierarchical_async_put_lockfree")
    return DistributorKind::HierarchicalAsyncPutLockFree;
  throw std::runtime_error("Unknown distributor: " + value);
}

static DurationMode parse_duration_mode(const std::string& value) {
  if (value == "fixed") return DurationMode::Fixed;
  if (value == "poisson" || value == "random") return DurationMode::Poisson;
  throw std::runtime_error("Unknown duration mode: " + value);
}

static std::string to_string(DistributorKind kind) {
  switch (kind) {
    case DistributorKind::Naive:
      return "naive";
    case DistributorKind::Hierarchical:
      return "hierarchical";
    case DistributorKind::AsyncPutLockFree:
      return "async_put_lockfree";
    case DistributorKind::HierarchicalAsyncPutLockFree:
      return "hierarchical_async_put_lockfree";
  }
  return "unknown";
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
        "duration_s,nodes,world_size,workers,max_upper_fanout,total_tasks,elapsed_s,"
        "throughput_tasks_per_s\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts,
                          const BenchmarkResult& result) {
  const double throughput =
      result.elapsed_s > 0.0 ? static_cast<double>(result.total_tasks) / result.elapsed_s : 0.0;
  os << opts.system << "," << to_string(opts.distributor) << "," << to_string(opts.duration_mode)
     << "," << opts.expected_us << "," << opts.duration_s
     << "," << opts.nodes << "," << result.world_size << "," << result.workers << ","
     << opts.max_upper_fanout << "," << result.total_tasks << "," << result.elapsed_s << ","
     << throughput << "\n";
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
    if (duration_mode == DurationMode::Poisson) {
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
    config.max_tasks = kLockFreeMaxTasks;
  }
  Distributor distributor(worker_function, config);

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
      auto results =
          distributor.run_tasks({.target_num_tasks = num_workers * 2, .max_seconds = 0.1});
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

// A cheap, non-blocking MPI call whose only purpose is to give the library a
// chance to drive its progress engine. See the comment at its call sites in
// run_benchmark_async_put_lockfree() for why a manager rank that never
// otherwise touches MPI needs this under FI_CXI_RX_MATCH_MODE=software.
static void pump_mpi_progress(MPI_Comm comm) {
  int flag = 0;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
}

// AsyncPutLockFreeMPIWorkDistributor: calibrate, then spin completely
// uninterrupted publishing/claiming/writing results, then harvest exactly
// once at the end -- isolating the claim+execute rate from collection
// frequency. Calibration must measure throughput the same way the main
// phase runs it (one uninterrupted spin + one snapshot, not repeated
// polling), and the spin loop needs pump_mpi_progress() under
// FI_CXI_RX_MATCH_MODE=software. This distributor needs its own driver
// (rather than fitting run_benchmark<Distributor>()) because the generic
// driver's incremental small-batch insert_tasks()/run_tasks(max_seconds=0.1) cycle
// pays this class's per-call overhead far more often than necessary,
// capping measured throughput around 10-15K tasks/s regardless of how fast
// the underlying claim+write protocol actually is -- confirmed via a
// now-removed dedicated isolated benchmark that this class's true
// uninterrupted throughput reaches ~650K-965K tasks/s (matching
// rma_atomic_microbench's raw one-sided-atomic ceiling), a ~70-100x
// difference explained entirely by driver pacing, not the protocol itself.
static BenchmarkResult run_benchmark_async_put_lockfree(const BenchmarkOptions& opts,
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

  using Distributor = dynampi::AsyncPutLockFreeMPIWorkDistributor<Task, uint32_t>;
  Distributor::Config config{.comm = comm, .manager_rank = 0, .max_tasks = kLockFreeMaxTasks};
  Distributor distributor(worker_function, config);

  if (distributor.is_root_manager()) {
    timer.start();

    const double expected_s = static_cast<double>(opts.expected_us) * 1e-6;
    const double calibration_window_s = std::min(1.0, opts.duration_s * 0.2);
    const double ideal_rate = static_cast<double>(num_workers) / expected_s;
    // Cap raised well above the 20,000 originally used for the now-removed
    // CAS-based LockFreeMPIWorkDistributor benchmark (copy-pasted from there
    // initially, then found to be a real bug here): that cap was sized for
    // its ~10-24K tasks/s ceiling, where 20,000 tasks takes close to the
    // full 1s calibration window to drain. AsyncPutLockFreeMPIWorkDistributor's
    // batched claim/write protocol reaches 600K-965K tasks/s uninterrupted --
    // a 20,000-task batch drains in well under 50ms, leaving the worker idle for
    // the rest of the calibration window. Since observed_rate below divides
    // by the *full* window regardless of how much of it was spent working,
    // an early-draining batch silently collapses observed_rate toward
    // batch/window instead of the true achievable rate -- confirmed via
    // measurement: this alone was capping the real benchmark path at
    // ~25K tasks/s (vs ~57K in the isolated, uncalibrated benchmark, same
    // settings) despite every other part of the class performing correctly.
    // A 2,000,000-task cap costs one bulk Put of a few tens of MB either
    // way -- cheap regardless of whether it fully drains within the window.
    const uint64_t calibration_batch = std::min<uint64_t>(
        std::min<uint64_t>(2'000'000, kLockFreeMaxTasks - 1),
        std::max<uint64_t>(num_workers * 10,
                           static_cast<uint64_t>(ideal_rate * calibration_window_s)));
    {
      std::vector<Task> tasks;
      tasks.reserve(calibration_batch);
      for (uint64_t i = 0; i < calibration_batch; ++i) tasks.push_back(static_cast<Task>(i));
      distributor.insert_tasks(tasks);
    }
    const double calibration_start_s = timer.elapsed().count();
    while (timer.elapsed().count() - calibration_start_s < calibration_window_s) {
      pump_mpi_progress(comm);
    }
    auto calibration_results = distributor.gather_once();
    const uint64_t calibration_collected = calibration_results.size();
    total_tasks += calibration_collected;
    const double calibration_elapsed_s = timer.elapsed().count() - calibration_start_s;
    const double observed_rate =
        calibration_collected > 0 && calibration_elapsed_s > 0.0
            ? static_cast<double>(calibration_collected) / calibration_elapsed_s
            : static_cast<double>(num_workers) / expected_s;

    const double remaining_budget_s = std::max(0.0, opts.duration_s - timer.elapsed().count());
    const double estimate = 1.3 * observed_rate * remaining_budget_s;
    // Clamped against remaining capacity, not the raw table size: the
    // calibration batch above already consumed calibration_batch slots of
    // it, and this batch is published on top -- clamping against the full
    // kLockFreeMaxTasks here (as this used to do) let the two batches
    // together publish up to kLockFreeMaxTasks - 1 + calibration_batch
    // tasks, silently exceeding the table's real capacity by however large
    // calibration_batch was (confirmed: an actual run published 101,999,999
    // tasks against a 100,000,000 cap, exactly matching a ~2,000,000
    // calibration_batch overshoot for this distributor). publish_tasks()'s
    // own overflow guard is an assert(), compiled out under NDEBUG/Release
    // (see kLockFreeMaxTasks's own comment), so this silently corrupted
    // memory instead of erroring -- manifesting many calls later as an
    // unrelated-looking segfault deep inside a harvest's bulk RMA read.
    const double capacity_remaining =
        static_cast<double>(kLockFreeMaxTasks) - 1.0 - static_cast<double>(calibration_batch);
    const double clamped_estimate = std::min(estimate, std::max(0.0, capacity_remaining));
    const uint64_t batch_size =
        clamped_estimate >= 1.0 ? static_cast<uint64_t>(clamped_estimate) : 1;

    {
      std::vector<Task> tasks;
      tasks.reserve(batch_size);
      for (uint64_t i = 0; i < batch_size; ++i) {
        tasks.push_back(static_cast<Task>(calibration_batch + i));
      }
      distributor.insert_tasks(tasks);
    }
    while (timer.elapsed().count() < opts.duration_s) {
      pump_mpi_progress(comm);
    }
    auto results = distributor.gather_once();
    total_tasks += results.size();
    timer.stop();

    distributor.finalize();
  }

  return BenchmarkResult{total_tasks, num_workers, static_cast<uint64_t>(size),
                         timer.elapsed().count()};
}

// HierarchicalAsyncPutLockFreeMPIWorkDistributor: same calibrate-then-spin-
// uninterrupted-then-harvest-once shape as run_benchmark_async_put_lockfree()
// above, against the leader level instead of a single flat window. Built to
// test whether spreading the async-put protocol's RMA load across per-node
// coordinator windows (instead of concentrating it all on one manager
// window) breaks through the ~2.1-2.26M tasks/s plateau the flat class hits
// from 32 nodes onward (see the strong-scaling sweep this benchmark
// produced for async_put_lockfree at multi-node scale).
static BenchmarkResult run_benchmark_hierarchical_async_put_lockfree(const BenchmarkOptions& opts,
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

  using Distributor = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<Task, uint32_t>;
  Distributor::Config config;
  config.comm = comm;
  config.manager_rank = 0;
  config.max_tasks = kLockFreeMaxTasks;
  config.max_local_tasks = kLockFreeMaxTasks;
  config.max_upper_fanout = opts.max_upper_fanout;
  Distributor distributor(worker_function, config);

  if (distributor.is_root_manager()) {
    timer.start();

    const double expected_s = static_cast<double>(opts.expected_us) * 1e-6;
    const double calibration_window_s = std::min(1.0, opts.duration_s * 0.2);
    const double ideal_rate = static_cast<double>(num_workers) / expected_s;
    const uint64_t calibration_batch = std::min<uint64_t>(
        std::min<uint64_t>(2'000'000, kLockFreeMaxTasks - 1),
        std::max<uint64_t>(num_workers * 10,
                           static_cast<uint64_t>(ideal_rate * calibration_window_s)));
    {
      std::vector<Task> tasks;
      tasks.reserve(calibration_batch);
      for (uint64_t i = 0; i < calibration_batch; ++i) tasks.push_back(static_cast<Task>(i));
      distributor.insert_tasks(tasks);
    }
    const double calibration_start_s = timer.elapsed().count();
    while (timer.elapsed().count() - calibration_start_s < calibration_window_s) {
      pump_mpi_progress(comm);
    }
    auto calibration_results = distributor.gather_once();
    const uint64_t calibration_collected = calibration_results.size();
    total_tasks += calibration_collected;
    const double calibration_elapsed_s = timer.elapsed().count() - calibration_start_s;
    const double observed_rate =
        calibration_collected > 0 && calibration_elapsed_s > 0.0
            ? static_cast<double>(calibration_collected) / calibration_elapsed_s
            : static_cast<double>(num_workers) / expected_s;

    const double remaining_budget_s = std::max(0.0, opts.duration_s - timer.elapsed().count());
    const double estimate = 1.3 * observed_rate * remaining_budget_s;
    // Clamped against remaining capacity, not the raw table size: the
    // calibration batch above already consumed calibration_batch slots of
    // it, and this batch is published on top -- clamping against the full
    // kLockFreeMaxTasks here (as this used to do) let the two batches
    // together publish up to kLockFreeMaxTasks - 1 + calibration_batch
    // tasks, silently exceeding the table's real capacity by however large
    // calibration_batch was (confirmed: an actual run published 101,999,999
    // tasks against a 100,000,000 cap, exactly matching a ~2,000,000
    // calibration_batch overshoot for this distributor). publish_tasks()'s
    // own overflow guard is an assert(), compiled out under NDEBUG/Release
    // (see kLockFreeMaxTasks's own comment), so this silently corrupted
    // memory instead of erroring -- manifesting many calls later as an
    // unrelated-looking segfault deep inside a harvest's bulk RMA read.
    const double capacity_remaining =
        static_cast<double>(kLockFreeMaxTasks) - 1.0 - static_cast<double>(calibration_batch);
    const double clamped_estimate = std::min(estimate, std::max(0.0, capacity_remaining));
    const uint64_t batch_size =
        clamped_estimate >= 1.0 ? static_cast<uint64_t>(clamped_estimate) : 1;

    {
      std::vector<Task> tasks;
      tasks.reserve(batch_size);
      for (uint64_t i = 0; i < batch_size; ++i) {
        tasks.push_back(static_cast<Task>(calibration_batch + i));
      }
      distributor.insert_tasks(tasks);
    }
    while (timer.elapsed().count() < opts.duration_s) {
      pump_mpi_progress(comm);
    }
    auto results = distributor.gather_once();
    total_tasks += results.size();
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
      "D,distribution",
      "Distribution strategy: naive, hierarchical, async_put_lockfree, "
      "or hierarchical_async_put_lockfree",
      cxxopts::value<std::string>()->default_value("hierarchical"))(
      "m,mode", "Duration mode: fixed or random (uniform 0-2x expected)",
      cxxopts::value<std::string>()->default_value("fixed"))(
      "max_upper_fanout",
      "hierarchical_async_put_lockfree only: max direct children per coordinator "
      "above the node-local level. 0 = single unbounded coordinator level (default, "
      "matches pre-N-level behavior); >0 activates iterative k-ary grouping into "
      "multiple upper levels once the coordinator count exceeds this fanout.",
      cxxopts::value<int>()->default_value("0"))(
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
    switch (opts.distributor) {
      case DistributorKind::Naive:
        result = run_benchmark<dynampi::NaiveMPIWorkDistributor<Task, uint32_t>>(opts, comm);
        break;
      case DistributorKind::Hierarchical:
        result = run_benchmark<dynampi::HierarchicalMPIWorkDistributor<Task, uint32_t>>(opts, comm);
        break;
      case DistributorKind::AsyncPutLockFree:
        result = run_benchmark_async_put_lockfree(opts, comm);
        break;
      case DistributorKind::HierarchicalAsyncPutLockFree:
        result = run_benchmark_hierarchical_async_put_lockfree(opts, comm);
        break;
    }

    if (rank == 0) {
      const double throughput =
          result.elapsed_s > 0.0 ? static_cast<double>(result.total_tasks) / result.elapsed_s : 0.0;
      std::cout << "RESULT"
                << " distributor=" << to_string(opts.distributor)
                << " mode=" << to_string(opts.duration_mode) << " expected_us=" << opts.expected_us
                << " nodes=" << opts.nodes << " world_size=" << result.world_size
                << " max_upper_fanout=" << opts.max_upper_fanout
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
