/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <dynampi/dynampi.hpp>
#include <type_traits>
#include <vector>

#include "dynampi/impl/hierarchical_distributor.hpp"
#include "dynampi/mpi/mpi_communicator.hpp"
#include "mpi_test_environment.hpp"

template <template <typename...> class Template, typename T>
struct is_specialization_of : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template, Template<Args...>> : std::true_type {};

template <template <typename, typename, typename...> class TT>
struct DistributerTypeWrapper {
  template <typename TaskT, typename ResultT, typename... Options>
  using type = TT<TaskT, ResultT, Options...>;

  static constexpr bool use_immediate_recv = false;
  static constexpr size_t max_result_size = 1024;

  template <typename TaskT, typename ResultT, typename... Options>
  static typename TT<TaskT, ResultT, Options...>::Config get_config() {
    return typename TT<TaskT, ResultT, Options...>::Config{};
  }
};

// Specialized wrapper for HierarchicalMPIWorkDistributor with coordinator_per_node config
template <bool CoordinatorPerNode>
struct HierarchicalDistributerTypeWrapper {
  template <typename TaskT, typename ResultT, typename... Options>
  using type = dynampi::HierarchicalMPIWorkDistributor<TaskT, ResultT, Options...>;

  static constexpr bool use_immediate_recv = false;
  static constexpr size_t max_result_size = 1024;

  template <typename TaskT, typename ResultT, typename... Options>
  static typename dynampi::HierarchicalMPIWorkDistributor<TaskT, ResultT, Options...>::Config
  get_config() {
    typename dynampi::HierarchicalMPIWorkDistributor<TaskT, ResultT, Options...>::Config config;
    config.coordinator_per_node = CoordinatorPerNode;
    return config;
  }
};

// Helper to get config from wrapper
template <typename Wrapper, typename TaskT, typename ResultT, typename... Options>
auto get_distributer_config() {
  return Wrapper::template get_config<TaskT, ResultT, Options...>();
}

template <typename Wrapper, typename... T>
using DistributerOf = typename Wrapper::template type<T...>;

// Test fixture
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

using DistributerTypes = ::testing::Types<DistributerTypeWrapper<dynampi::NaiveMPIWorkDistributor>,
                                          HierarchicalDistributerTypeWrapper<true>,
                                          HierarchicalDistributerTypeWrapper<false>>;

TYPED_TEST_SUITE(DynamicDistribution, DistributerTypes);

// --- Tests are now much leaner ---

TYPED_TEST(DynamicDistribution, BasicFlow) {
  using TaskT = int;
  using Distributer = DistributerOf<TypeParam, TaskT, double>;
  auto worker_task = [](TaskT task) -> double { return sqrt(static_cast<double>(task)); };

  auto config = get_distributer_config<TypeParam, TaskT, double>();
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = false;
  Distributer distributor(worker_task, config);

  EXPECT_EQ(distributor.is_root_manager(), MPIEnvironment::world_comm_rank() == 0);

  if (distributor.is_root_manager()) {
    for (int i = 0; i < 10; ++i) distributor.insert_task(i);
  }

  if (distributor.is_root_manager()) {
    auto results =
        distributor.run_tasks({.target_num_tasks = 5, .allow_more_than_target_tasks = false});
    EXPECT_EQ(results.size(), 5);
    EXPECT_LE(distributor.remaining_tasks_count(), 5);
    auto second_results = distributor.finish_remaining_tasks();
    EXPECT_EQ(second_results.size(), 5);
    EXPECT_EQ(distributor.remaining_tasks_count(), 0);
    results.insert(results.end(), second_results.begin(), second_results.end());
    if (!Distributer::ordered) {
      std::sort(results.begin(), results.end());
    }
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_DOUBLE_EQ(results[i] * results[i], static_cast<double>(i));
    }
  } else {
    distributor.run_worker();
  }
}

TYPED_TEST(DynamicDistribution, Naive2) {
  using DistributerWrapper = TypeParam;

  auto worker_task = [](size_t task) -> char { return "Hi"[task]; };

  auto result = dynampi::mpi_manager_worker_distribution<char, DistributerWrapper::template type>(
      2, worker_task);

  if (MPIEnvironment::world_comm_rank() == 0) {
    ASSERT_TRUE(result.has_value());
    if constexpr (!DistributerWrapper::template type<int, int>::ordered) {
      std::sort(result->begin(), result->end());
    }
    EXPECT_EQ(result.value(), std::vector<char>({'H', 'i'}));
  } else {
    EXPECT_FALSE(result.has_value());
  }
}

// Exercises manager_rank != 0 for both Naive and Hierarchical (e.g. idx_for_worker branches).
TYPED_TEST(DynamicDistribution, ManagerRankNonZero) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "Need at least 2 ranks for non-zero manager rank";
  }
  const int manager_rank = 1;
  using TaskT = int;
  using ResultT = double;
  using Distributer = DistributerOf<TypeParam, TaskT, ResultT>;
  auto worker_task = [](TaskT task) -> ResultT { return sqrt(static_cast<double>(task)); };
  auto config = get_distributer_config<TypeParam, TaskT, ResultT>();
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = false;
  config.manager_rank = manager_rank;
  Distributer distributor(worker_task, config);

  EXPECT_EQ(distributor.is_root_manager(), MPIEnvironment::world_comm_rank() == manager_rank);

  if (distributor.is_root_manager()) {
    for (int i = 0; i < 10; ++i) distributor.insert_task(i);
    auto results = distributor.finish_remaining_tasks();
    if (!Distributer::ordered) {
      std::sort(results.begin(), results.end());
    }
    EXPECT_EQ(results.size(), 10u);
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_DOUBLE_EQ(results[i] * results[i], static_cast<double>(i));
    }
  } else {
    distributor.run_worker();
  }
}

TYPED_TEST(DynamicDistribution, Example1) {
  using DistributerWrapper = TypeParam;

  for (int manager_rank : {0, MPIEnvironment::world_comm_size() - 1}) {
    auto worker_task = [](size_t task) -> size_t { return task * task; };
    auto result =
        dynampi::mpi_manager_worker_distribution<size_t, DistributerWrapper::template type>(
            4, worker_task, MPI_COMM_WORLD, manager_rank);
    if (result.has_value()) {
      if constexpr (!DistributerWrapper::template type<int, int>::ordered) {
        std::sort(result->begin(), result->end());
      }
      EXPECT_EQ(MPIEnvironment::world_comm_rank(), manager_rank);
      EXPECT_EQ(result, std::vector<size_t>({0, 1, 4, 9}));
    }
  }
}

TYPED_TEST(DynamicDistribution, Example2) {
  using Task = int;
  using Result = std::vector<int>;
  using Distributer = DistributerOf<TypeParam, Task, Result>;
  if constexpr (is_specialization_of<dynampi::HierarchicalMPIWorkDistributor, Distributer>::value) {
    GTEST_SKIP() << "This test is not applicable for HierarchicalMPIWorkDistributor.";
  } else {
    auto worker_task = [](Task task) -> Result {
      return Result{task, task * task, task * task * task};
    };
    {
      auto config = get_distributer_config<TypeParam, Task, Result>();
      Distributer work_distributer(worker_task, config);
      if (work_distributer.is_root_manager()) {
        work_distributer.insert_tasks({1, 2, 3, 4, 5});
        auto results = work_distributer.finish_remaining_tasks();
        EXPECT_EQ(results, (std::vector<std::vector<int>>{
                               {1, 1, 1}, {2, 4, 8}, {3, 9, 27}, {4, 16, 64}, {5, 25, 125}}));
        work_distributer.insert_tasks({6, 7, 8});
        results = work_distributer.finish_remaining_tasks();
        EXPECT_EQ(results,
                  (std::vector<std::vector<int>>{{6, 36, 216}, {7, 49, 343}, {8, 64, 512}}));
      }
    }
  }
}

TYPED_TEST(DynamicDistribution, RunTasksMaxTasks) {
  using Task = int;
  using Result = int;
  using Distributer = DistributerOf<TypeParam, Task, Result>;

  auto worker_task = [](Task task) -> Result { return task * 2; };

  auto config = get_distributer_config<TypeParam, Task, Result>();
  Distributer work_distributer(worker_task, config);
  if (work_distributer.is_root_manager()) {
    work_distributer.insert_tasks({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    typename Distributer::RunConfig run_config;
    run_config.target_num_tasks = 3;
    run_config.allow_more_than_target_tasks = false;
    auto results = work_distributer.run_tasks(run_config);
    EXPECT_EQ(results.size(), 3u);

    run_config.target_num_tasks = 4;
    auto more_results = work_distributer.run_tasks(run_config);
    EXPECT_EQ(more_results.size(), 4u);

    auto remaining_results = work_distributer.run_tasks();
    EXPECT_EQ(remaining_results.size(), 3u);

    std::vector<int> all_results;
    all_results.insert(all_results.end(), results.begin(), results.end());
    all_results.insert(all_results.end(), more_results.begin(), more_results.end());
    all_results.insert(all_results.end(), remaining_results.begin(), remaining_results.end());
    std::sort(all_results.begin(), all_results.end());
    EXPECT_EQ(all_results, (std::vector<int>{2, 4, 6, 8, 10, 12, 14, 16, 18, 20}));
  }
}

TYPED_TEST(DynamicDistribution, RunTasksMinTasksWithTimeLimit) {
  using Task = int;
  using Result = int;
  using Distributer = DistributerOf<TypeParam, Task, Result>;

  auto worker_task = [](Task task) -> Result { return task * 3; };

  auto config = get_distributer_config<TypeParam, Task, Result>();
  Distributer work_distributer(worker_task, config);
  if (work_distributer.is_root_manager()) {
    work_distributer.insert_tasks({1, 2, 3, 4, 5});

    typename Distributer::RunConfig run_config;
    run_config.target_num_tasks = 2;
    run_config.max_seconds = 0.0;
    auto results = work_distributer.run_tasks(run_config);
    EXPECT_EQ(results.size(), 0u);

    auto remaining_results = work_distributer.run_tasks();
    EXPECT_EQ(results.size() + remaining_results.size(), 5u);
  }
}

TYPED_TEST(DynamicDistribution, PriorityQueue) {
  using Task = int;
  using Result = int;
  using Distributer = DistributerOf<TypeParam, Task, Result, dynampi::enable_prioritization>;
  if (!Distributer::ordered) {
    GTEST_SKIP()
        << "This test requires ordered results, which is not supported by this distributer.";
  }
  auto worker_task = [](Task task) -> Result { return task * task; };
  {
    auto config = get_distributer_config<TypeParam, Task, Result, dynampi::enable_prioritization>();
    Distributer work_distributer(worker_task, config);
    if (work_distributer.is_root_manager()) {
      work_distributer.insert_task(1, 1.0);
      work_distributer.insert_task(7, 7.0);
      work_distributer.insert_task(3, 3.0);
      work_distributer.insert_task(6, 6.0);
      work_distributer.insert_task(2, 2.0);
      work_distributer.insert_task(4, 5.0);
      work_distributer.insert_task(5, 4.0);
      auto result = work_distributer.finish_remaining_tasks();
      EXPECT_EQ(result, (std::vector<int>{49, 36, 16, 25, 9, 4, 1}));
    }
  }
}

TYPED_TEST(DynamicDistribution, Statistics) {
  using Task = int;
  using Result = int;
  using Distributer = DistributerOf<TypeParam, Task, Result,
                                    dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>;
  auto worker_task = [](Task task) -> Result { return task * task; };
  {
    auto config =
        get_distributer_config<TypeParam, Task, Result,
                               dynampi::track_statistics<dynampi::StatisticsMode::Detailed>>();
    Distributer work_distributer(worker_task, config);
    if (work_distributer.is_root_manager()) {
      work_distributer.insert_tasks({1, 2, 3, 4, 5});
      auto results = work_distributer.finish_remaining_tasks();
      size_t expected_size = 5;
      if (MPIEnvironment::world_comm_size() == 1) {
        expected_size = 0;
      }
      if constexpr (!Distributer::ordered) {
        std::sort(results.begin(), results.end());
      }
      EXPECT_EQ(results, (std::vector<int>{1, 4, 9, 16, 25}));
      EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_sent,
                expected_size * sizeof(int));
      if constexpr (is_specialization_of<dynampi::NaiveMPIWorkDistributor, Distributer>::value) {
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.send_count, expected_size);
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.recv_count,
                  expected_size + MPIEnvironment::world_comm_size() - 1);
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_received,
                  expected_size * sizeof(int));
      }
      work_distributer.finalize();
      EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_sent,
                expected_size * sizeof(int));
      if constexpr (is_specialization_of<dynampi::NaiveMPIWorkDistributor, Distributer>::value) {
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.send_count,
                  expected_size + MPIEnvironment::world_comm_size() - 1);
        double expected_num_bytes = 0;
        if (MPIEnvironment::world_comm_size() > 1) {
          expected_num_bytes = static_cast<double>(expected_size * sizeof(int)) /
                               (expected_size + MPIEnvironment::world_comm_size() - 1);
        }
        EXPECT_DOUBLE_EQ(work_distributer.get_statistics().comm_statistics.average_receive_size(),
                         expected_num_bytes);
        EXPECT_DOUBLE_EQ(work_distributer.get_statistics().comm_statistics.average_send_size(),
                         expected_num_bytes);
      }
    }
  }
}

TYPED_TEST(DynamicDistribution, AutoRunWorkers) {
  using Distributer = DistributerOf<TypeParam, int, int>;
  auto worker_task = [](int task) -> int { return task * task; };
  // Test with auto_run_workers = true - workers should start automatically
  auto dist = this->template make_distributor<int, int>(worker_task, true);

  if (dist.is_root_manager()) {
    // Workers should already be running, so we can just insert tasks
    dist.insert_tasks({1, 2, 3, 4, 5});
    auto results = dist.finish_remaining_tasks();
    if constexpr (!Distributer::ordered) {
      std::sort(results.begin(), results.end());
    }
    EXPECT_EQ(results, (std::vector<int>{1, 4, 9, 16, 25}));
  }
  // Workers run automatically in constructor, no need to call run_worker()
}
