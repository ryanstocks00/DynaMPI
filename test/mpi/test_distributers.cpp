/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <dynampi/dynampi.hpp>
#include <vector>

#include "dynampi/impl/hierarchical_distributor.hpp"
#include "dynampi/mpi/mpi_communicator.hpp"
#include "mpi_test_environment.hpp"

// --- Configuration Wrapper ---
template <template <typename, typename, typename...> class DistributorT, bool ImmediateRecv = false>
struct TestConfig {
  template <typename TaskT, typename ResultT, typename... Options>
  using type = DistributorT<TaskT, ResultT, Options...>;

  static constexpr bool use_immediate_recv = ImmediateRecv;
  static constexpr size_t max_result_size = 1024;
};

// --- Unified Test Fixture ---
template <typename T>
class DynamicDistribution : public ::testing::Test {
 protected:
  template <typename TaskT, typename ResultT, typename... Options>
  auto make_distributor(auto worker_task, bool auto_run = false) {
    using DistT = typename T::template type<TaskT, ResultT, Options...>;

    // Use decltype to get the correct Options type regardless of its internal name
    using ConfigT = typename DistT::Config;

    ConfigT opts{};
    opts.comm = MPI_COMM_WORLD;
    opts.auto_run_workers = auto_run;

    if constexpr (T::use_immediate_recv) {
      opts.use_immediate_recv = true;
      opts.max_result_size = T::max_result_size;
    }

    return DistT(worker_task, opts);
  }
};

using DistributerTypes =
    ::testing::Types<TestConfig<dynampi::NaiveMPIWorkDistributor, false>,
                     TestConfig<dynampi::NaiveMPIWorkDistributor, true>,
                     TestConfig<dynampi::HierarchicalMPIWorkDistributor, false>>;

TYPED_TEST_SUITE(DynamicDistribution, DistributerTypes);

// --- Tests are now much leaner ---

TYPED_TEST(DynamicDistribution, BasicFlow) {
  auto worker_task = [](uint32_t task) -> double { return sqrt(static_cast<double>(task)); };
  auto dist = this->template make_distributor<uint32_t, double>(worker_task);

  if (dist.is_root_manager()) {
    for (int i = 0; i < 10; ++i) dist.insert_task(i);
    auto results = dist.finish_remaining_tasks();
    EXPECT_EQ(results.size(), 10);
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_DOUBLE_EQ(results[i] * results[i], static_cast<double>(i));
    }
  } else {
    dist.run_worker();
  }
}

TYPED_TEST(DynamicDistribution, MultiStageTasks) {
  using Result = std::vector<int>;
  auto worker_task = [](int task) -> Result { return {task, task * task, task * task * task}; };
  auto dist = this->template make_distributor<int, Result>(worker_task);

  if (dist.is_root_manager()) {
    dist.insert_tasks({1, 2, 3, 4, 5});
    EXPECT_EQ(dist.finish_remaining_tasks().size(), 5);

    dist.insert_tasks({6, 7, 8});
    auto results = dist.finish_remaining_tasks();
    EXPECT_EQ(results.size(), 8);
    EXPECT_EQ(results.back(), (Result{8, 64, 512}));
  } else {
    dist.run_worker();
  }
}

TYPED_TEST(DynamicDistribution, PriorityQueue) {
  auto worker_task = [](int task) -> int { return task * task; };
  // Pass the priority option as a template argument to the factory
  auto dist =
      this->template make_distributor<int, int, dynampi::enable_prioritization>(worker_task);

  if (dist.is_root_manager()) {
    std::vector<std::pair<int, double>> tasks = {{1, 1.0}, {7, 7.0}, {3, 3.0}, {6, 6.0},
                                                 {2, 2.0}, {4, 5.0}, {5, 4.0}};
    for (auto& t : tasks) dist.insert_task(t.first, t.second);

    auto result = dist.finish_remaining_tasks();
    EXPECT_EQ(result, (std::vector<int>{49, 36, 16, 25, 9, 4, 1}));
  } else {
    dist.run_worker();
  }
}

TYPED_TEST(DynamicDistribution, Statistics) {
  auto worker_task = [](int task) -> int { return task * task; };
  using StatsOpt = dynampi::track_statistics<dynampi::StatisticsMode::Detailed>;
  auto dist = this->template make_distributor<int, int, StatsOpt>(worker_task);

  if (dist.is_root_manager()) {
    dist.insert_tasks({1, 2, 3, 4, 5});
    auto results = dist.finish_remaining_tasks();

    size_t expected_size = (MPIEnvironment::world_comm_size() == 1) ? 0 : 5;
    EXPECT_EQ(results, (std::vector<int>{1, 4, 9, 16, 25}));
    EXPECT_EQ(dist.get_statistics().comm_statistics.send_count, expected_size);
    EXPECT_EQ(dist.get_statistics().comm_statistics.bytes_sent, expected_size * sizeof(int));
    EXPECT_EQ(dist.get_statistics().comm_statistics.recv_count,
              expected_size + MPIEnvironment::world_comm_size() - 1);
    EXPECT_EQ(dist.get_statistics().comm_statistics.bytes_received, expected_size * sizeof(int));

    dist.finalize();
    EXPECT_EQ(dist.get_statistics().comm_statistics.send_count,
              expected_size + MPIEnvironment::world_comm_size() - 1);
    EXPECT_EQ(dist.get_statistics().comm_statistics.bytes_sent, expected_size * sizeof(int));

    double expected_num_bytes = 0;
    if (MPIEnvironment::world_comm_size() > 1) {
      expected_num_bytes = static_cast<double>(expected_size * sizeof(int)) /
                           (expected_size + MPIEnvironment::world_comm_size() - 1);
    }
    EXPECT_DOUBLE_EQ(dist.get_statistics().comm_statistics.average_receive_size(),
                     expected_num_bytes);
    EXPECT_DOUBLE_EQ(dist.get_statistics().comm_statistics.average_send_size(), expected_num_bytes);
  } else {
    dist.run_worker();
  }
}

TYPED_TEST(DynamicDistribution, AutoRunWorkers) {
  auto worker_task = [](int task) -> int { return task * task; };
  // Test with auto_run_workers = true - workers should start automatically
  auto dist = this->template make_distributor<int, int>(worker_task, true);

  if (dist.is_root_manager()) {
    // Workers should already be running, so we can just insert tasks
    dist.insert_tasks({1, 2, 3, 4, 5});
    auto results = dist.finish_remaining_tasks();
    EXPECT_EQ(results, (std::vector<int>{1, 4, 9, 16, 25}));
  }
  // Workers run automatically in constructor, no need to call run_worker()
}
