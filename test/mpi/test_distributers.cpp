/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <mpi.h>

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
};

template <typename Wrapper, typename... T>
using DistributerOf = typename Wrapper::template type<T...>;

// Test fixture
template <typename T>
class DynamicDistribution : public ::testing::Test {};

using DistributerTypes =
    ::testing::Types<DistributerTypeWrapper<dynampi::NaiveMPIWorkDistributor>,
                     DistributerTypeWrapper<dynampi::HierarchicalMPIWorkDistributor>>;

TYPED_TEST_SUITE(DynamicDistribution, DistributerTypes);

TYPED_TEST(DynamicDistribution, Naive) {
  using TaskT = uint32_t;
  using Distributer = DistributerOf<TypeParam, TaskT, double>;

  auto worker_task = [](TaskT task) -> double { return sqrt(static_cast<double>(task)); };

  std::vector<TaskT> tasks(10);
  for (size_t i = 0; i < tasks.size(); ++i) tasks[i] = static_cast<TaskT>(i);

  Distributer distributor(worker_task, {.comm = MPI_COMM_WORLD, .auto_run_workers = false});

  EXPECT_EQ(distributor.is_root_manager(), MPIEnvironment::world_comm_rank() == 0);

  if (distributor.is_root_manager()) {
    for (int i = 0; i < 10; ++i) distributor.insert_task(i);
  }

  if (distributor.is_root_manager()) {
    auto results = distributor.finish_remaining_tasks();
    EXPECT_EQ(results.size(), 10);
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
    EXPECT_TRUE(result.has_value());
    if constexpr (!DistributerWrapper::template type<int, int>::ordered) {
      std::sort(result->begin(), result->end());
    }
    EXPECT_EQ(result.value(), std::vector<char>({'H', 'i'}));
  } else {
    EXPECT_FALSE(result.has_value());
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
      assert(result == std::vector<size_t>({0, 1, 4, 9}));
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
      Distributer work_distributer(worker_task);
      if (work_distributer.is_root_manager()) {
        work_distributer.insert_tasks({1, 2, 3, 4, 5});
        auto results = work_distributer.finish_remaining_tasks();
        EXPECT_EQ(results, (std::vector<std::vector<int>>{
                               {1, 1, 1}, {2, 4, 8}, {3, 9, 27}, {4, 16, 64}, {5, 25, 125}}));
        work_distributer.insert_tasks({6, 7, 8});
        results = work_distributer.finish_remaining_tasks();
        EXPECT_EQ(results, (std::vector<std::vector<int>>{{1, 1, 1},
                                                          {2, 4, 8},
                                                          {3, 9, 27},
                                                          {4, 16, 64},
                                                          {5, 25, 125},
                                                          {6, 36, 216},
                                                          {7, 49, 343},
                                                          {8, 64, 512}}));
      }
    }
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
    Distributer work_distributer(worker_task);
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
    Distributer work_distributer(worker_task);
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
