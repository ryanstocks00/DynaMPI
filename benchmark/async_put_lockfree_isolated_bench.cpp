// SPDX-FileCopyrightText: 2026 Ryan Stocks
// SPDX-License-Identifier: Apache-2.0

// Isolates AsyncPutLockFreeMPIWorkDistributor's true claim+compute+write
// throughput from strong_scaling_distribution_rate's generic incremental
// insert/run_tasks driver loop, which replaces insertion in small batches
// (num_workers*4) and calls run_tasks() with a 0.1s budget -- overhead
// that's shared fairly across all distributors there, but worth ruling out
// as this class's specific bottleneck. Pattern: publish one huge batch up
// front, let workers claim+compute+write completely uninterrupted (zero
// manager RMA activity) for the timed window, then harvest exactly once at
// the end. This measures the same thing rma_atomic_microbench measures for
// the bare atomic, but for this distributor's whole protocol.
#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <dynampi/impl/async_put_lockfree_distributor.hpp>
#include <dynampi/mpi/mpi_communicator.hpp>
#include <dynampi/utilities/timer.hpp>
#include <iostream>
#include <limits>

using Task = uint32_t;

static void spin_wait(std::chrono::microseconds duration) {
  auto start = std::chrono::high_resolution_clock::now();
  while (std::chrono::high_resolution_clock::now() - start < duration) {
  }
}

// See the call site in main()'s spin loop: under FI_CXI_RX_MATCH_MODE=software
// a rank that never touches MPI stops pumping the fabric progress engine,
// stalling other ranks' RMA ops targeting its window -- this is exactly the
// bug found and fixed in strong_scaling_distribution_rate.cpp's
// gather_mode=final path earlier this session.
static void pump_mpi_progress(MPI_Comm comm) {
  int flag = 0;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  cxxopts::Options options("async_put_lockfree_isolated_bench", "Isolated claim+write throughput");
  options.add_options()("t,expected_us", "task duration us",
                        cxxopts::value<uint64_t>()->default_value("1"))(
      "d,duration_s", "spin duration s", cxxopts::value<double>()->default_value("3"))(
      "b,claim_batch_size", "claim batch size", cxxopts::value<int>()->default_value("8"))(
      "n,num_tasks", "tasks to publish up front",
      cxxopts::value<uint64_t>()->default_value("2000000"));
  auto args = options.parse(argc, argv);
  const uint64_t expected_us = args["expected_us"].as<uint64_t>();
  const double duration_s = args["duration_s"].as<double>();
  const int claim_batch_size = args["claim_batch_size"].as<int>();
  const uint64_t num_tasks = args["num_tasks"].as<uint64_t>();

  const int num_workers = std::max(1, size - 1);
  auto worker_function = [expected_us](Task t) -> uint32_t {
    spin_wait(std::chrono::microseconds(expected_us));
    return t * t;
  };

  using Distributor = dynampi::AsyncPutLockFreeMPIWorkDistributor<Task, uint32_t>;
  {
    // Barrier BEFORE construction, not after: auto_run_workers defaults to
    // true, so a non-manager rank's constructor call blocks internally
    // running run_worker() until FINISHED_OFF is set -- a barrier placed
    // after construction would deadlock, since the manager (which alone can
    // eventually trigger FINISHED_OFF, via finalize()) would itself be
    // stuck at that same barrier waiting on workers that can't reach it
    // either. Confirmed as a real, reproducible hang during isolated
    // testing (not a bug in the distributor itself).
    MPI_Barrier(MPI_COMM_WORLD);
    dynampi::Timer timer;

    Distributor::Config config;
    config.comm = MPI_COMM_WORLD;
    config.manager_rank = 0;
    if (num_tasks > static_cast<uint64_t>(std::numeric_limits<int>::max()) - 1000u) {
      if (rank == 0) {
        std::cerr << "num_tasks too large for max_tasks (int capacity)" << std::endl;
      }
      MPI_Finalize();
      return 1;
    }
    config.max_tasks = static_cast<int>(num_tasks) + 1000;
    config.claim_batch_size = claim_batch_size;
    Distributor distributor(worker_function, config);

    if (distributor.is_root_manager()) {
      std::vector<Task> tasks(num_tasks);
      for (uint64_t i = 0; i < num_tasks; ++i) tasks[i] = static_cast<Task>(i);
      distributor.insert_tasks(tasks);

      const double spin_start = timer.elapsed().count();
      while (timer.elapsed().count() - spin_start < duration_s) {
        pump_mpi_progress(MPI_COMM_WORLD);
      }
      auto results = distributor.run_tasks();  // one harvest at the end
      timer.stop();
      const double throughput = static_cast<double>(results.size()) / timer.elapsed().count();
      std::cout << "RESULT claim_batch_size=" << claim_batch_size << " expected_us=" << expected_us
                << " num_workers=" << num_workers << " collected=" << results.size()
                << " elapsed_s=" << timer.elapsed().count()
                << " throughput_tasks_per_s=" << throughput << std::endl;
    }
    // distributor's destructor (which calls finalize(), issuing a real RMA
    // call) must run before MPI_Finalize() -- hence this nested block.
  }

  MPI_Finalize();
  return 0;
}
