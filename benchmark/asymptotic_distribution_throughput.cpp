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
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

#include "dynampi/mpi/mpi_communicator.hpp"

struct BenchmarkOptions {
  size_t num_tasks;
  size_t message_size;
  bool remove_root_from_distribution;
};

static std::vector<size_t> parse_size_t_list(const std::string& csv) {
  std::vector<size_t> values;
  std::stringstream ss(csv);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      values.push_back(static_cast<size_t>(std::stoull(token)));
    }
  }
  return values;
}

static void print_host_info_if_linux(int rank) {
#if defined(__linux__)
  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlength;
  MPI_Get_processor_name(name, &resultlength);
  int hardware_thread = sched_getcpu();
  printf("Rank %03d on Node %s, Hardware Thread %d\n", rank, name, hardware_thread);
#else
  (void)rank;  // Avoid unused variable warning if not on Linux
#endif
}

static std::optional<dynampi::MPICommunicator<>> make_dynamic_communicator(
    bool remove_root_from_distribution) {
  if (remove_root_from_distribution) {
    dynampi::MPICommunicator<> world_communicator(MPI_COMM_WORLD);
    dynampi::MPICommunicator<> node_communicator = world_communicator.split_by_node();
    int is_root_node = world_communicator.rank() == 0;
    node_communicator.broadcast(is_root_node, 0);
    std::optional<dynampi::MPICommunicator<>> skip_manager_node_communicator =
        node_communicator.split((world_communicator.rank() != 0 && is_root_node) ? MPI_UNDEFINED
                                                                                 : 1);
    if (skip_manager_node_communicator) {
      return std::optional<dynampi::MPICommunicator<>>(
          std::move(skip_manager_node_communicator.value()));
    }
    return std::nullopt;
  } else {
    return std::optional<dynampi::MPICommunicator<>>(dynampi::MPICommunicator<>(MPI_COMM_WORLD));
  }
}

static double run_single_benchmark(const BenchmarkOptions& opts) {
  MPI_Barrier(MPI_COMM_WORLD);

  using Task = size_t;
  //using Result = std::vector<std::byte>;
  using Result = size_t;

  //auto worker_task = [&opts](Task task) -> Result {
    //return std::vector<std::byte>(opts.message_size, std::byte(task));
  auto worker_task = [](Task task) -> Result {
    return task;
  };

  dynampi::Timer dynamic_timer;
  auto dynamic_communicator = make_dynamic_communicator(opts.remove_root_from_distribution);
  if (dynamic_communicator.has_value()) {
    int rank = dynamic_communicator.value().rank();
    int size = dynamic_communicator.value().size();
    dynampi::MPIDynamicWorkDistributor<Task, Result,
                                       dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>
        work_distributer(worker_task, {.comm = dynamic_communicator.value()});

    if (work_distributer.is_root_manager()) {
      std::vector<Task> tasks(opts.num_tasks);
      std::iota(tasks.begin(), tasks.end(), 0);
      work_distributer.insert_tasks(tasks);
      auto result = work_distributer.finish_remaining_tasks();
      (void)result;
      work_distributer.finalize();
    }

    dynamic_timer.stop();

    if (work_distributer.is_root_manager()) {
      std::cout << "Dynamic task distribution completed successfully." << std::endl;
      const auto& stats = work_distributer.get_statistics();
      for (size_t i = 0; i < stats.worker_task_counts->size(); i++) {
        std::cout << "Rank " << i << ": " << "Tasks: " << stats.worker_task_counts->at(i)
                  << std::endl;
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

    if (rank == 0) {
      std::cout << "Distributed " << opts.num_tasks << " tasks to " << size << " MPI processes in "
                << dynamic_timer << ". Throughput: "
                << static_cast<double>(opts.num_tasks) / dynamic_timer.elapsed().count()
                << " tasks/second" << std::endl;
    }
  }
  return dynamic_timer.elapsed().count();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  cxxopts::Options options("asymptotic_distribution_throughput",
                           "Test dynamic MPI task distribution throughput");
  options.add_options()("n,n_tasks", "Number of tasks to distribute",
                        cxxopts::value<size_t>()->default_value("1000000"))(
      "m,message_size", "Size of each message in bytes",
      cxxopts::value<size_t>()->default_value("1"))(
      "N,n_tasks_list", "Comma-separated list of task counts (overrides --n_tasks)",
      cxxopts::value<std::string>()->default_value(""))(
      "M,message_size_list", "Comma-separated list of message sizes (overrides --message_size)",
      cxxopts::value<std::string>()->default_value(""))(
      "r,rm_root", "Remove root node from task distribution")("h,help", "Print usage");

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

  bool remove_root = args.count("rm_root") > 0;
  size_t default_num_tasks = args["n_tasks"].as<size_t>();
  size_t default_message_size = args["message_size"].as<size_t>();
  std::string num_tasks_list_csv = args["n_tasks_list"].as<std::string>();
  std::string message_size_list_csv = args["message_size_list"].as<std::string>();

  std::vector<size_t> num_tasks_list = num_tasks_list_csv.empty()
                                           ? std::vector<size_t>{default_num_tasks}
                                           : parse_size_t_list(num_tasks_list_csv);
  std::vector<size_t> message_size_list = message_size_list_csv.empty()
                                              ? std::vector<size_t>{default_message_size}
                                              : parse_size_t_list(message_size_list_csv);

  if (rank == 0) {
    std::cout << "Beginning testing dynamic MPI task distribution throughput" << std::endl;
  }

  print_host_info_if_linux(rank);

  dynampi::Timer total_timer;

  for (size_t tasks : num_tasks_list) {
    for (size_t msg_size : message_size_list) {
      if (rank == 0) {
        std::cout << "\n=== Running benchmark: n_tasks=" << tasks << ", message_size=" << msg_size
                  << ", rm_root=" << (remove_root ? "true" : "false") << " ===" << std::endl;
      }

      BenchmarkOptions run_opts{tasks, msg_size, remove_root};
      run_single_benchmark(run_opts);
    }
  }

  total_timer.stop();
  if (rank == 0) {
    std::cout << "\nDynamic MPI task distribution throughput test completed successfully. Total "
                 "run time: "
              << total_timer << std::endl;
  }

  MPI_Finalize();
  return 0;
}
