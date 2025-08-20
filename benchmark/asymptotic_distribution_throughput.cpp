/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <mpi.h>

#include <cxxopts.hpp>
#include <dynampi/dynampi.hpp>
#include <dynampi/utilities/timer.hpp>
#include <iostream>
#include <numeric>
#include <ranges>
#include <sched.h>

#include "dynampi/mpi/mpi_communicator.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  cxxopts::Options options("asymptotic_distribution_throughput",
                           "Test dynamic MPI task distribution throughput");
  options.add_options()("n,n_tasks", "Number of tasks to distribute",
                        cxxopts::value<size_t>()->default_value("1000000"))("h,help",
                                                                            "Print usage");
  cxxopts::ParseResult args;
  try {
    args = options.parse(argc, argv);
  } catch (std::exception& e) {
    if (rank == 0) {
      std::cerr << "Error parsing options: " << e.what() << std::endl;
      std::cerr << options.help() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  if (args.count("help")) {
    if (rank == 0) {
      std::cout << options.help() << std::endl;
    }
    MPI_Finalize();
    return 0;
  }

  size_t num_tasks = args["n_tasks"].as<size_t>();

  if (rank == 0) {
    std::cout << "Beginning testing dynamic MPI task distribution throughput" << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlength;
  MPI_Get_processor_name(name, &resultlength);
  int hwthread = sched_getcpu();

  printf("MPI %03d - HWT %03d - Node %s\n", rank, hwthread, name);

  MPI_Barrier(MPI_COMM_WORLD);
  dynampi::Timer total_timer;

  using Task = size_t;
  using Result = std::vector<size_t>;
  auto worker_task = [](Task task) -> Result { return std::vector<size_t>(10, task); };

  dynampi::Timer dynamic_timer;
  {
    dynampi::MPIDynamicWorkDistributor<Task, Result,
                                       dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>
        work_distributer(worker_task);
    if (work_distributer.is_manager()) {
      // work_distributer.insert_tasks(std::views::iota(0ul, num_tasks));
      std::vector<Task> tasks(num_tasks);
      std::iota(tasks.begin(), tasks.end(), 0);
      work_distributer.insert_tasks(tasks);
      auto result = work_distributer.finish_remaining_tasks();
      work_distributer.finalize();
    }
    dynamic_timer.stop();

    if (work_distributer.is_manager()) {
      std::cout << "Dynamic task distribution completed successfully." << std::endl;
      const auto& stats = work_distributer.get_statistics();
      for (size_t i = 0; i < stats.worker_task_counts.size(); i++) {
        std::cout << "Rank " << i << ": "
                  << "Tasks: " << stats.worker_task_counts[i] << std::endl;
      }
      std::cout << "Total messages sent: " << stats.comm_statistics.send_count << std::endl;
      std::cout << "Total messages received: " << stats.comm_statistics.recv_count << std::endl;
      std::cout << "Total bytes sent: " << stats.comm_statistics.bytes_sent << std::endl;
      std::cout << "Total bytes received: " << stats.comm_statistics.bytes_received << std::endl;
      std::cout << "Average send size: " << stats.comm_statistics.average_send_size() << " bytes"
                << std::endl;
      std::cout << "Average receive size: " << stats.comm_statistics.average_receive_size()
                << " bytes" << std::endl;
      std::cout << "Average send bandwidth: "
                << stats.comm_statistics.bytes_sent / dynamic_timer.elapsed().count()
                << " bytes/second" << std::endl;
      std::cout << "Average receive bandwidth: "
                << stats.comm_statistics.bytes_received / dynamic_timer.elapsed().count()
                << " bytes/second" << std::endl;
    }
  }

  if (rank == 0) {
    std::cout << "Distributed " << num_tasks << " tasks to " << size << " MPI processes in "
              << dynamic_timer << ". Throughput: "
              << static_cast<double>(num_tasks) / dynamic_timer.elapsed().count() << " tasks/second"
              << std::endl;
  }
  total_timer.stop();
  if (rank == 0) {
    std::cout
        << "Dynamic MPI task distribution throughput test completed successfully. Total run time: "
        << total_timer << std::endl;
  }

  MPI_Finalize();
  return 0;
}
