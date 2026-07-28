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
#include <dynampi/impl/lockfree_distributor.hpp>
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

enum class DistributorKind {
  Naive,
  Hierarchical,
  LockFree,
  AsyncPutLockFree,
  HierarchicalAsyncPutLockFree
};
enum class DurationMode { Fixed, Poisson };

// incremental: the standard driving loop below -- keep a small queue topped
// up and drain it via frequent run_tasks() calls. For LockFreeMPIWorkDistributor
// each drain is a full MPI_Barrier + Gather/Gatherv across the whole
// communicator (see lockfree_distributor.hpp), so at high rank counts this
// mode measures throughput dominated by that synchronization cost, not by
// how fast workers can claim+execute tasks.
// final: (lockfree only) insert one large up-front batch, let workers churn
// completely uninterrupted for the full duration, then take exactly one
// gather_once() snapshot at the end. Isolates the claim+execute rate from
// the collection-frequency cost.
enum class GatherMode { Incremental, Final };

struct BenchmarkOptions {
  uint64_t expected_us = 1;
  double duration_s = 10.0;
  DistributorKind distributor = DistributorKind::Hierarchical;
  DurationMode duration_mode = DurationMode::Fixed;
  GatherMode gather_mode = GatherMode::Incremental;
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
  if (value == "lockfree") return DistributorKind::LockFree;
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

static GatherMode parse_gather_mode(const std::string& value) {
  if (value == "incremental") return GatherMode::Incremental;
  if (value == "final") return GatherMode::Final;
  throw std::runtime_error("Unknown gather mode: " + value);
}

static std::string to_string(DistributorKind kind) {
  switch (kind) {
    case DistributorKind::Naive:
      return "naive";
    case DistributorKind::Hierarchical:
      return "hierarchical";
    case DistributorKind::LockFree:
      return "lockfree";
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

static std::string to_string(GatherMode mode) {
  return mode == GatherMode::Final ? "final" : "incremental";
}

static void spin_wait(std::chrono::microseconds duration) {
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start < duration) {
  }
}

static void write_csv_header(std::ostream& os) {
  os << "system,distributor,mode,gather_mode,expected_us,"
        "duration_s,nodes,world_size,workers,max_upper_fanout,total_tasks,elapsed_s,"
        "throughput_tasks_per_s\n";
}

static void write_csv_row(std::ostream& os, const BenchmarkOptions& opts,
                          const BenchmarkResult& result) {
  const double throughput =
      result.elapsed_s > 0.0 ? static_cast<double>(result.total_tasks) / result.elapsed_s : 0.0;
  os << opts.system << "," << to_string(opts.distributor) << "," << to_string(opts.duration_mode)
     << "," << to_string(opts.gather_mode) << "," << opts.expected_us << "," << opts.duration_s
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
// run_benchmark_lockfree_final_gather() for why a manager rank that never
// otherwise touches MPI needs this under FI_CXI_RX_MATCH_MODE=software.
static void pump_mpi_progress(MPI_Comm comm) {
  int flag = 0;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
}

// GatherMode::Final, lockfree only: insert one large up-front batch, let
// workers claim+execute completely uninterrupted for the full duration (no
// gather calls at all during that window -- workers don't block on the
// manager for anything once tasks are published), then take exactly one
// gather_once() snapshot at the end. See the GatherMode comment above for
// why this differs from run_benchmark<LockFreeMPIWorkDistributor>: that path
// pays a full MPI_Barrier + Gather/Gatherv (across the whole communicator)
// on every drain, which at high rank counts can dominate over actual task
// throughput for fine-grained tasks.
static BenchmarkResult run_benchmark_lockfree_final_gather(const BenchmarkOptions& opts,
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

  using Distributor = dynampi::LockFreeMPIWorkDistributor<Task, uint32_t>;
  Distributor::Config config{.comm = comm, .manager_rank = 0, .max_tasks = kLockFreeMaxTasks};
  Distributor distributor(worker_function, config);

  if (distributor.is_root_manager()) {
    timer.start();

    // Calibration phase: measure real achievable throughput instead of
    // guessing from expected_us. expected_us assumes zero per-task
    // claim/RMA overhead, which is a poor predictor of the actual rate --
    // observed throughput can be an order of magnitude below the naive
    // num_workers/expected_us estimate. Sizing the main batch directly from
    // that estimate risks either running dry early (undersized) or, as
    // happened during testing, over-provisioning so heavily that draining
    // the leftover in finalize() afterward takes far longer than
    // duration_s itself.
    //
    // Crucially, calibration must measure throughput the *same way* the
    // main phase runs it -- insert, spin completely uninterrupted, take one
    // snapshot -- not via repeated polling. An earlier version of this
    // calibrated by looping gather_once() until a small batch drained: each
    // call pays the same full-communicator barrier the incremental gather
    // mode pays, so for fine-grained tasks the polling overhead itself
    // dominated the measurement, producing a rate far below what
    // uninterrupted execution actually achieves and badly under-sizing the
    // main batch (workers ran dry almost immediately and then sat idle for
    // the rest of duration_s).
    const double expected_s = static_cast<double>(opts.expected_us) * 1e-6;
    // Deliberately modest, not "large enough to definitely survive the
    // window": real lockfree throughput has been observed anywhere from
    // roughly 1/10th to 1/50th of the naive num_workers/expected_us ideal,
    // so sizing the calibration batch *from* that ideal (or padding it
    // generously "to be safe") means most of it goes unconsumed within the
    // short calibration window -- and that unconsumed remainder becomes
    // backlog finalize() must drain later, on top of the main batch's own
    // backlog. If this batch happens to fully drain before the window
    // ends, the measured rate is a (safe) underestimate -- workers just sit
    // idle for the remainder, no different in kind from a slightly
    // undersized main batch, which is the safe failure mode here.
    const double calibration_window_s = std::min(1.0, opts.duration_s * 0.2);
    // The naive ideal-case estimate (num_workers/expected_s) is already an
    // upper bound on real achievable throughput -- overhead only ever
    // subtracts from it. Scaling the calibration batch by that ideal
    // therefore auto-shrinks it for coarse-grained tasks (fewer will ever
    // complete in a short window regardless of overhead), while the 20,000
    // cap keeps it from ballooning for fine-grained tasks, where the ideal
    // is wildly larger than what's actually achievable.
    const double ideal_rate = static_cast<double>(num_workers) / expected_s;
    const uint64_t calibration_batch = std::min<uint64_t>(
        std::min<uint64_t>(20'000, kLockFreeMaxTasks - 1),
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
    // Fall back to the naive ideal-case estimate only if calibration
    // couldn't measure anything (e.g. finished instantaneously).
    const double observed_rate =
        calibration_collected > 0 && calibration_elapsed_s > 0.0
            ? static_cast<double>(calibration_collected) / calibration_elapsed_s
            : static_cast<double>(num_workers) / expected_s;

    // Modest safety margin now that the rate is measured, not guessed: just
    // enough to not run dry before duration_s, without a large leftover to
    // drain afterward. Based on total elapsed so far (not just the
    // calibration spin window), so it also accounts for the calibration
    // batch's insert time and the calibration gather_once() call itself.
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
    // Compare the post-clamp value against 1 (not the pre-truncation
    // estimate against 0): a small positive double like 0.4 passes ">0" but
    // truncates to a batch_size of 0, which would insert nothing.
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
    // Workers claim and execute freely here: no gather calls, so nothing on
    // the manager side normally interrupts them until this spin exits. But
    // "normally" is doing real work: with FI_CXI_RX_MATCH_MODE=software (see
    // the launch scripts), receive-side matching for RMA targeting this
    // rank's window is not fully NIC-offloaded -- it depends on this rank
    // making MPI library calls periodically to pump progress. A pure C++
    // busy-wait here that never touches MPI starves that progress engine
    // almost entirely: workers' Fetch_and_op/CAS/Get calls queue up against
    // this rank's window and complete only sporadically, which is why this
    // path originally measured *lower* throughput than the incremental
    // gather mode despite doing strictly less synchronization -- confirmed
    // by adding pump_mpi_progress() below, which alone took expected_us=1000
    // from ~15 tasks/s to ~6900/s. The incremental path never hit this
    // because every run_tasks() call already makes real MPI collective
    // calls (Gather/Gatherv), pumping progress as a side effect.
    while (timer.elapsed().count() < opts.duration_s) {
      pump_mpi_progress(comm);
    }
    auto results = distributor.gather_once();
    total_tasks += results.size();
    // Stop the clock right at the measurement snapshot: total_tasks/elapsed_s
    // above is exactly "calibrate + insert + uninterrupted work + one
    // gather" and must not include what follows.
    timer.stop();

    // The over-provisioned batch deliberately leaves tasks unclaimed/
    // uncomputed at snapshot time (that's what avoids running dry early).
    // finalize() itself blocks until every published task is collected
    // (same loop finish_remaining_tasks() would run), so it alone drains
    // the remainder -- this cleanup cost happens after the timed window and
    // isn't counted above.
    distributor.finalize();
  }

  return BenchmarkResult{total_tasks, num_workers, static_cast<uint64_t>(size),
                         timer.elapsed().count()};
}

// AsyncPutLockFreeMPIWorkDistributor: same calibrate-then-spin-uninterrupted-
// then-harvest-once shape as run_benchmark_lockfree_final_gather() above --
// see that function's comments for why calibration must measure throughput
// the same way the main phase runs it (one uninterrupted spin + one
// snapshot, not repeated polling) and why the spin loop needs
// pump_mpi_progress() under FI_CXI_RX_MATCH_MODE=software. This distributor
// needs its own driver (rather than fitting run_benchmark<Distributor>())
// for the same reason gather_mode=final does: the generic driver's
// incremental small-batch insert_tasks()/run_tasks(max_seconds=0.1) cycle
// pays this class's per-call overhead far more often than necessary,
// capping measured throughput around 10-15K tasks/s regardless of how fast
// the underlying claim+write protocol actually is -- confirmed via a
// dedicated isolated benchmark (async_put_lockfree_isolated_bench.cpp) that
// this class's true uninterrupted throughput reaches ~650K-965K tasks/s
// (matching rma_atomic_microbench's raw one-sided-atomic ceiling), a ~70-100x
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
    // Cap raised well above run_benchmark_lockfree_final_gather()'s 20,000
    // (copy-pasted from there initially, then found to be a real bug here):
    // that cap was sized for the CAS-based LockFreeMPIWorkDistributor's
    // ~10-24K tasks/s ceiling, where 20,000 tasks takes close to the full
    // 1s calibration window to drain. AsyncPutLockFreeMPIWorkDistributor's
    // batched claim/write protocol reaches 600K-965K tasks/s uninterrupted
    // (see async_put_lockfree_isolated_bench.cpp) -- a 20,000-task batch
    // drains in well under 50ms, leaving the worker completely idle for
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
      "Distribution strategy: naive, hierarchical, lockfree, async_put_lockfree, "
      "or hierarchical_async_put_lockfree",
      cxxopts::value<std::string>()->default_value("hierarchical"))(
      "m,mode", "Duration mode: fixed or random (uniform 0-2x expected)",
      cxxopts::value<std::string>()->default_value("fixed"))(
      "G,gather_mode",
      "Gather mode (lockfree only): incremental (frequent small drains) or "
      "final (one uninterrupted batch, single gather at the end)",
      cxxopts::value<std::string>()->default_value("incremental"))(
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
    opts.gather_mode = parse_gather_mode(args["gather_mode"].as<std::string>());
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

  if (opts.gather_mode == GatherMode::Final && opts.distributor != DistributorKind::LockFree) {
    if (world_rank == 0) {
      std::cerr << "--gather_mode final is only supported with --distribution lockfree."
                << std::endl;
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
      case DistributorKind::LockFree:
        result =
            opts.gather_mode == GatherMode::Final
                ? run_benchmark_lockfree_final_gather(opts, comm)
                : run_benchmark<dynampi::LockFreeMPIWorkDistributor<Task, uint32_t>>(opts, comm);
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
