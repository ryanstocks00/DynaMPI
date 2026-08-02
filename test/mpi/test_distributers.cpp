/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <dynampi/dynampi.hpp>
#include <thread>
#include <type_traits>
#include <vector>

#include "dynampi/impl/async_put_lockfree_distributor.hpp"
#include "dynampi/impl/hierarchical_async_put_lockfree_distributor.hpp"
#include "dynampi/impl/hierarchical_distributor.hpp"
#include "dynampi/impl/hierarchical_nonblocking_distributor.hpp"
#include "dynampi/impl/lockfree_distributor.hpp"
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

// Specialized wrapper for HierarchicalNonBlockingMPIWorkDistributor with coordinator_per_node
// config
template <bool CoordinatorPerNode>
struct HierarchicalNonBlockingDistributerTypeWrapper {
  template <typename TaskT, typename ResultT, typename... Options>
  using type = dynampi::HierarchicalNonBlockingMPIWorkDistributor<TaskT, ResultT, Options...>;

  static constexpr bool use_immediate_recv = false;
  static constexpr size_t max_result_size = 1024;

  template <typename TaskT, typename ResultT, typename... Options>
  static typename dynampi::HierarchicalNonBlockingMPIWorkDistributor<TaskT, ResultT,
                                                                     Options...>::Config
  get_config() {
    typename dynampi::HierarchicalNonBlockingMPIWorkDistributor<TaskT, ResultT, Options...>::Config
        config;
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

using DistributerTypes =
    ::testing::Types<DistributerTypeWrapper<dynampi::NaiveMPIWorkDistributor>,
                     HierarchicalDistributerTypeWrapper<true>,
                     HierarchicalDistributerTypeWrapper<false>,
                     HierarchicalNonBlockingDistributerTypeWrapper<true>,
                     HierarchicalNonBlockingDistributerTypeWrapper<false>,
                     DistributerTypeWrapper<dynampi::AsyncPutLockFreeMPIWorkDistributor>>;

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
  // Variable-size ResultT is unimplemented for both hierarchical variants.
  if constexpr (is_specialization_of<dynampi::HierarchicalMPIWorkDistributor, Distributer>::value ||
                is_specialization_of<dynampi::HierarchicalNonBlockingMPIWorkDistributor,
                                     Distributer>::value) {
    GTEST_SKIP() << "This test is not applicable for this distributor.";
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
  if (!Distributer::ordered ||
      is_specialization_of<dynampi::AsyncPutLockFreeMPIWorkDistributor, Distributer>::value) {
    GTEST_SKIP() << "This test requires ordered results with priority, which is not supported by "
                    "this distributer.";
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
      // Message-passing distributors send bare TaskT values. Lock-free RMA
      // paths also count window headers and atomics via MPICommunicator.
      constexpr bool is_rma =
          is_specialization_of<dynampi::AsyncPutLockFreeMPIWorkDistributor, Distributer>::value;
      if constexpr (is_rma) {
        if (expected_size > 0) {
          EXPECT_GE(work_distributer.get_statistics().comm_statistics.bytes_sent,
                    expected_size * sizeof(int));
        } else {
          EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_sent, 0u);
        }
      } else {
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_sent,
                  expected_size * sizeof(int));
      }
      if constexpr (is_specialization_of<dynampi::NaiveMPIWorkDistributor, Distributer>::value) {
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.send_count, expected_size);
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.recv_count,
                  expected_size + MPIEnvironment::world_comm_size() - 1);
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_received,
                  expected_size * sizeof(int));
      }
      work_distributer.finalize();
      if constexpr (is_rma) {
        if (expected_size > 0) {
          EXPECT_GE(work_distributer.get_statistics().comm_statistics.bytes_sent,
                    expected_size * sizeof(int));
        } else {
          EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_sent, 0u);
        }
      } else {
        EXPECT_EQ(work_distributer.get_statistics().comm_statistics.bytes_sent,
                  expected_size * sizeof(int));
      }
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
      // Detailed exposes send_time/recv_time. Do not require them to be
      // strictly positive: MPI_Wtime can round short shared-memory sends to
      // zero (seen on macOS CI). AggregatedStatistics asserts the opposite
      // contract -- times stay exactly zero when timing is disabled.
      const auto& comm_stats = work_distributer.get_statistics().comm_statistics;
      EXPECT_GE(comm_stats.send_time, 0.0);
      EXPECT_GE(comm_stats.recv_time, 0.0);
    }
  }
}

// Aggregated tracks the same counters as Detailed but skips the per-call
// timing, so send_time/recv_time must stay zero while the counts still add up.
TYPED_TEST(DynamicDistribution, AggregatedStatistics) {
  using Task = int;
  using Result = int;
  using Aggregated = dynampi::track_statistics<dynampi::StatisticsMode::Aggregated>;
  using Distributer = DistributerOf<TypeParam, Task, Result, Aggregated>;
  auto worker_task = [](Task task) -> Result { return task * task; };
  {
    auto config = get_distributer_config<TypeParam, Task, Result, Aggregated>();
    Distributer work_distributer(worker_task, config);
    if (work_distributer.is_root_manager()) {
      work_distributer.insert_tasks({1, 2, 3, 4, 5});
      auto results = work_distributer.finish_remaining_tasks();
      if constexpr (!Distributer::ordered) {
        std::sort(results.begin(), results.end());
      }
      EXPECT_EQ(results, (std::vector<int>{1, 4, 9, 16, 25}));

      const auto& comm_stats = work_distributer.get_statistics().comm_statistics;
      if (MPIEnvironment::world_comm_size() > 1) {
        EXPECT_GT(comm_stats.send_count + comm_stats.atomic_count, 0);
      }
      EXPECT_DOUBLE_EQ(comm_stats.send_time, 0.0);
      EXPECT_DOUBLE_EQ(comm_stats.recv_time, 0.0);
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

// gather_once() is the non-looping snapshot API used by the strong-scaling
// bench to avoid per-retry busy-spin harvest. Only AsyncPutLockFree exposes
// it.
//
// Under SMPI, busy-polling gather_once() (or OS-sleeping in workers) can
// starve other ranks: AsyncPut's harvest is pure RMA and may not yield the
// simulator, so workers never run and remaining_tasks_count never drains.
// Exercise one snapshot, then drain with finish_remaining_tasks() which
// already has a proper no-progress idle path.
TYPED_TEST(DynamicDistribution, GatherOnce) {
  using Distributer = DistributerOf<TypeParam, int, int>;
  constexpr bool has_gather_once =
      is_specialization_of<dynampi::AsyncPutLockFreeMPIWorkDistributor, Distributer>::value;
  if constexpr (!has_gather_once) {
    GTEST_SKIP() << "gather_once is only on lock-free RMA distributors";
  } else {
    auto worker_task = [](int task) -> int { return task * task; };
    auto dist = this->template make_distributor<int, int>(worker_task, true);

    if (dist.is_root_manager()) {
      dist.insert_tasks({1, 2, 3, 4, 5, 6, 7, 8});

      auto snapshot = dist.gather_once();
      EXPECT_LE(snapshot.size(), 8u);

      auto drained = dist.finish_remaining_tasks();
      std::vector<int> all = std::move(snapshot);
      all.insert(all.end(), drained.begin(), drained.end());
      std::sort(all.begin(), all.end());
      EXPECT_EQ(all, (std::vector<int>{1, 4, 9, 16, 25, 36, 49, 64}));
      EXPECT_EQ(dist.remaining_tasks_count(), 0u);

      dist.insert_tasks({9, 10});
      auto rest = dist.finish_remaining_tasks();
      std::sort(rest.begin(), rest.end());
      EXPECT_EQ(rest, (std::vector<int>{81, 100}));
    }
  }
}

// --- MinimalLockFreeMPIWorkDistributor (index parallel-for) ---
// This distributor has a distinct, collective API (run(n)), so it is tested
// directly rather than through the generic DynamicDistribution suite.

TEST(MinimalLockFree, ScalarResults) {
  dynampi::MinimalLockFreeMPIWorkDistributor<double> dist(
      [](size_t i) -> double { return std::sqrt(static_cast<double>(i)); });

  auto results = dist.run(20);

  if (MPIEnvironment::world_comm_rank() == 0) {
    ASSERT_EQ(results.size(), 20u);
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_DOUBLE_EQ(results[i] * results[i], static_cast<double>(i));
    }
  } else {
    EXPECT_TRUE(results.empty());
  }
}

TEST(MinimalLockFree, ManagerRankNonZero) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "Need at least 2 ranks for non-zero manager rank";
  }
  const int manager_rank = 1;
  dynampi::MinimalLockFreeMPIWorkDistributor<size_t> dist(
      [](size_t i) -> size_t { return i * i; },
      {.comm = MPI_COMM_WORLD, .manager_rank = manager_rank});

  auto results = dist.run(8);

  if (MPIEnvironment::world_comm_rank() == manager_rank) {
    ASSERT_EQ(results.size(), 8u);
    for (size_t i = 0; i < results.size(); ++i) EXPECT_EQ(results[i], i * i);
  } else {
    EXPECT_TRUE(results.empty());
  }
}

TEST(MinimalLockFree, VariableSizeResults) {
  dynampi::MinimalLockFreeMPIWorkDistributor<std::vector<int>> dist(
      [](size_t i) -> std::vector<int> {
        int v = static_cast<int>(i);
        return {v, v * v, v * v * v};
      });

  auto results = dist.run(5);

  if (MPIEnvironment::world_comm_rank() == 0) {
    ASSERT_EQ(results.size(), 5u);
    for (size_t i = 0; i < results.size(); ++i) {
      int v = static_cast<int>(i);
      EXPECT_EQ(results[i], (std::vector<int>{v, v * v, v * v * v}));
    }
  }
}

TEST(MinimalLockFree, EmptyAndReusable) {
  dynampi::MinimalLockFreeMPIWorkDistributor<int> dist(
      [](size_t i) -> int { return static_cast<int>(i) + 1; });

  auto empty = dist.run(0);
  EXPECT_TRUE(empty.empty());

  // The distributor can be reused for multiple independent runs.
  auto results = dist.run(4);
  if (MPIEnvironment::world_comm_rank() == 0) {
    EXPECT_EQ(results, (std::vector<int>{1, 2, 3, 4}));
  }
}

TEST(LockFreeCapacity, RejectsTaskTableOverflow) {
  using Distributor = dynampi::AsyncPutLockFreeMPIWorkDistributor<int, int>;
  Distributor::Config config;
  config.comm = MPI_COMM_SELF;
  config.max_tasks = 1;
  Distributor dist([](int task) { return task; }, config);
  EXPECT_THROW(dist.insert_tasks(std::vector<int>{1, 2}), std::length_error);
}

// A fixed-size struct described by its scalar element type: one Vec3 is three
// MPI_DOUBLE elements, so everything that sizes a buffer per value -- the
// fixed-width RMA window slots, and the element count of a batched
// vector<Vec3> message -- has to account for all three. Exercised across every
// distributor, since each sizes those buffers differently.
namespace {
struct Vec3 {
  double x, y, z;
};

// 12 bytes described by an 8-byte datatype: not a whole number of elements, so
// no per-value element count can express it. Must still be rejected rather
// than silently truncated. (The fix for a type like this is to declare
// MPI_FLOAT with count() == 3, which Vec3-style handling then covers.)
struct Unaligned {
  float a, b, c;
};
}  // namespace

template <>
struct dynampi::MPI_Type<Vec3> {
  inline static const MPI_Datatype value = MPI_DOUBLE;
  inline static const bool resize_required = false;
  static int count(const Vec3&) noexcept { return 3; }
  static void resize(Vec3&, int) noexcept {}
  static void* ptr(Vec3& v) noexcept { return &v; }
  static const void* ptr(const Vec3& v) noexcept { return &v; }
};

template <>
struct dynampi::MPI_Type<Unaligned> {
  inline static const MPI_Datatype value = MPI_DOUBLE;  // deliberately mismatched
  inline static const bool resize_required = false;
  static int count(const Unaligned&) noexcept { return 2; }
  static void resize(Unaligned&, int) noexcept {}
  static void* ptr(Unaligned& v) noexcept { return &v; }
  static const void* ptr(const Unaligned& v) noexcept { return &v; }
};

TYPED_TEST(DynamicDistribution, MultiElementFixedSizePayload) {
  using Distributer = DistributerOf<TypeParam, Vec3, Vec3>;
  auto worker_task = [](Vec3 v) -> Vec3 { return Vec3{v.x * 2, v.y * 2, v.z * 2}; };

  auto config = get_distributer_config<TypeParam, Vec3, Vec3>();
  config.comm = MPI_COMM_WORLD;
  Distributer distributor(worker_task, config);

  if (distributor.is_root_manager()) {
    std::vector<Vec3> tasks;
    for (int i = 1; i <= 8; ++i) {
      const auto value = static_cast<double>(i);
      tasks.push_back(Vec3{value, value, value});
    }
    distributor.insert_tasks(tasks);

    auto results = distributor.finish_remaining_tasks();
    ASSERT_EQ(results.size(), tasks.size());

    // Sums every component, so a truncated element would show up.
    double sum = 0.0;
    for (const auto& r : results) sum += r.x + r.y + r.z;
    EXPECT_DOUBLE_EQ(sum, 2.0 * 3.0 * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8));
  }
}

TEST(FixedSizeMPIType, ElementsPerValue) {
  EXPECT_EQ(dynampi::mpi_elements_per_value<double>(), 1);
  EXPECT_EQ(dynampi::mpi_elements_per_value<int>(), 1);
  EXPECT_EQ(dynampi::mpi_elements_per_value<Vec3>(), 3);
  // Variable-length types size their buffers from count() instead.
  EXPECT_EQ(dynampi::mpi_elements_per_value<std::vector<int>>(), 1);

  // Batched messages count elements, not values, and resize() inverts that.
  const std::vector<Vec3> batch(4);
  EXPECT_EQ(dynampi::MPI_Type<std::vector<Vec3>>::count(batch), 12);
  std::vector<Vec3> received;
  dynampi::MPI_Type<std::vector<Vec3>>::resize(received, 12);
  EXPECT_EQ(received.size(), 4u);

  const std::vector<int> scalars(4);
  EXPECT_EQ(dynampi::MPI_Type<std::vector<int>>::count(scalars), 4);
}

TEST(FixedSizeMPIType, RejectsPayloadThatIsNotAWholeNumberOfElements) {
  auto identity = [](Unaligned v) { return v; };
  {
    using Distributor = dynampi::AsyncPutLockFreeMPIWorkDistributor<Unaligned, Unaligned>;
    typename Distributor::Config config;
    config.comm = MPI_COMM_SELF;
    EXPECT_THROW(Distributor(identity, config), std::invalid_argument);
  }
  {
    using Distributor = dynampi::HierarchicalMPIWorkDistributor<Unaligned, Unaligned>;
    typename Distributor::Config config;
    config.comm = MPI_COMM_SELF;
    EXPECT_THROW(Distributor(identity, config), std::invalid_argument);
  }
}

TEST(AsyncPutLevel, SingletonCommunicator) {
  using Level = dynampi::detail::AsyncPutLevel<int, int>;
  Level::Config config;
  config.comm = MPI_COMM_SELF;
  config.owner_rank = 0;
  config.max_tasks = 4;
  Level level(config);

  level.publish_tasks({2, 3});
  auto first = level.try_claim();
  ASSERT_EQ(first.start, 0);
  EXPECT_EQ(first.tasks, (std::vector<int>{2}));
  level.write_result_range(first.start, {4});
  auto second = level.try_claim();
  ASSERT_EQ(second.start, 1);
  EXPECT_EQ(second.tasks, (std::vector<int>{3}));
  level.write_result_range(second.start, {9});
  EXPECT_EQ(level.harvest_ready_results(), (std::vector<int>{4, 9}));
  level.mark_finished();
  EXPECT_TRUE(level.drained());

  EXPECT_THROW(level.publish_tasks({1, 2, 3, 4, 5}), std::length_error);
}

// --- HierarchicalAsyncPutLockFreeMPIWorkDistributor ---
// Node-aware tree topology (manager <-> per-node coordinators <-> local
// workers) with AsyncPutLockFreeMPIWorkDistributor's one-sided,
// collective-free protocol (fetch-and-add claiming, Put-based result return
// via a completion log) at each level. Results are unordered, so tests sort
// before comparing.

TEST(HierarchicalAsyncPutLockFree, BasicFlow) {
  using TaskT = int;
  using Distributer = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<TaskT, double>;
  auto worker_task = [](TaskT task) -> double { return sqrt(static_cast<double>(task)); };

  Distributer::Config config;
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = false;
  Distributer distributor(worker_task, config);

  EXPECT_EQ(distributor.is_root_manager(), MPIEnvironment::world_comm_rank() == 0);

  if (distributor.is_root_manager()) {
    for (int i = 0; i < 10; ++i) distributor.insert_task(i);
    auto results = distributor.finish_remaining_tasks();
    EXPECT_EQ(results.size(), 10u);
    std::sort(results.begin(), results.end());
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_DOUBLE_EQ(results[i] * results[i], static_cast<double>(i));
    }
  } else {
    distributor.run_worker();
  }
}

TEST(HierarchicalAsyncPutLockFree, ManagerRankNonZero) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "Need at least 2 ranks for non-zero manager rank";
  }
  const int manager_rank = 1;
  using TaskT = int;
  using ResultT = double;
  using Distributer = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<TaskT, ResultT>;
  auto worker_task = [](TaskT task) -> ResultT { return sqrt(static_cast<double>(task)); };

  Distributer::Config config;
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = false;
  config.manager_rank = manager_rank;
  Distributer distributor(worker_task, config);

  EXPECT_EQ(distributor.is_root_manager(), MPIEnvironment::world_comm_rank() == manager_rank);

  if (distributor.is_root_manager()) {
    for (int i = 0; i < 10; ++i) distributor.insert_task(i);
    auto results = distributor.finish_remaining_tasks();
    std::sort(results.begin(), results.end());
    EXPECT_EQ(results.size(), 10u);
    for (size_t i = 0; i < results.size(); ++i) {
      EXPECT_DOUBLE_EQ(results[i] * results[i], static_cast<double>(i));
    }
  } else {
    distributor.run_worker();
  }
}

TEST(HierarchicalAsyncPutLockFree, MultipleRoundsOfTasks) {
  using Task = int;
  using Result = int;
  using Distributer = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<Task, Result>;
  auto worker_task = [](Task task) -> Result { return task * task; };

  Distributer::Config config;
  Distributer work_distributer(worker_task, config);
  if (work_distributer.is_root_manager()) {
    work_distributer.insert_tasks({1, 2, 3, 4, 5});
    auto results = work_distributer.finish_remaining_tasks();
    std::sort(results.begin(), results.end());
    EXPECT_EQ(results, (std::vector<int>{1, 4, 9, 16, 25}));

    work_distributer.insert_tasks({6, 7, 8});
    results = work_distributer.finish_remaining_tasks();
    std::sort(results.begin(), results.end());
    EXPECT_EQ(results, (std::vector<int>{36, 49, 64}));
  }
}

TEST(HierarchicalAsyncPutLockFree, RunTasksMaxTasks) {
  using Task = int;
  using Result = int;
  using Distributer = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<Task, Result>;
  auto worker_task = [](Task task) -> Result { return task * 2; };

  Distributer::Config config;
  Distributer work_distributer(worker_task, config);
  if (work_distributer.is_root_manager()) {
    work_distributer.insert_tasks({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    Distributer::RunConfig run_config;
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

// Shared body for the hierarchical upper-chain topology tests below.
// Both HierarchicalMPIWorkDistributor and
// HierarchicalAsyncPutLockFreeMPIWorkDistributor expose the same
// max_upper_fanout / max_local_group_size Config knobs and the same
// insert_tasks / finish_remaining_tasks manager API, so one helper covers
// send/recv and RMA. Keep the TEST() names distinct: Open MPI 4's
// osc/pt2pt workaround in test/CMakeLists.txt filters the AsyncPut
// variants by exact suite.test name.
template <typename Distributer>
void expect_upper_hierarchy_squares(typename Distributer::Config config) {
  auto worker_task = [](int task) -> int { return task * task; };
  Distributer work_distributer(worker_task, config);
  if (!work_distributer.is_root_manager()) return;

  constexpr int kNumTasks = 500;
  std::vector<int> tasks(kNumTasks);
  for (int i = 0; i < kNumTasks; ++i) tasks[i] = i;
  work_distributer.insert_tasks(tasks);
  auto results = work_distributer.finish_remaining_tasks();
  std::sort(results.begin(), results.end());
  std::vector<int> expected(kNumTasks);
  for (int i = 0; i < kNumTasks; ++i) expected[i] = i * i;
  EXPECT_EQ(results, expected);
}

// Forces real multi-round grouping of the upper hierarchy (see
// max_upper_fanout's class comment). max_local_group_size=2 synthesizes
// multiple coordinators on a single shared-memory node so CI still
// exercises the grouping path; at tiny rank counts this degenerates to
// the flat level, which is still a useful no-op-path regression check.
TEST(HierarchicalAsyncPutLockFree, GroupedUpperHierarchy) {
  using Dist = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<int, int>;
  Dist::Config config;
  config.max_upper_fanout = 2;
  config.max_local_group_size = 2;
  expect_upper_hierarchy_squares<Dist>(config);
}

TEST(Hierarchical, GroupedUpperHierarchy) {
  using Dist = dynampi::HierarchicalMPIWorkDistributor<int, int>;
  Dist::Config config;
  config.max_upper_fanout = 2;
  config.max_local_group_size = 2;
  expect_upper_hierarchy_squares<Dist>(config);
}

// Exercises Config::max_upper_fanout auto mode (default -1).
// max_local_group_size=1 makes every non-manager rank a coordinator; at
// world size > 33 that crosses the flat-topology cutoff (~32) so the
// sqrt-based fanout pick and multi-round grouping both run. At smaller
// rank counts this stays on the flat auto path, which is still a useful
// smoke check. AsyncPut variant is excluded from Open MPI 4's default
// suite and run under osc/pt2pt (see test/CMakeLists.txt).
TEST(Hierarchical, AutoFanoutUpperHierarchy) {
  using Dist = dynampi::HierarchicalMPIWorkDistributor<int, int>;
  Dist::Config config;
  config.max_upper_fanout = -1;
  config.max_local_group_size = 1;
  expect_upper_hierarchy_squares<Dist>(config);
}

TEST(HierarchicalAsyncPutLockFree, AutoFanoutUpperHierarchy) {
  using Dist = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<int, int>;
  Dist::Config config;
  config.max_upper_fanout = -1;
  config.max_local_group_size = 1;
  expect_upper_hierarchy_squares<Dist>(config);
}

TEST(HierarchicalAsyncPutLockFree, SingletonGatherAndFinalize) {
  using Distributor = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<int, int>;

  {
    Distributor::Config config;
    config.comm = MPI_COMM_SELF;
    Distributor dist([](int task) { return task * task; }, config);
    dist.insert_tasks({1, 2, 3});
    EXPECT_EQ(dist.gather_once(), (std::vector<int>{1, 4, 9}));
  }

  {
    int tasks_run = 0;
    Distributor::Config config;
    config.comm = MPI_COMM_SELF;
    Distributor dist(
        [&tasks_run](int task) {
          ++tasks_run;
          return task;
        },
        config);
    dist.insert_tasks({1, 2, 3});
    dist.finalize();
    EXPECT_EQ(tasks_run, 3);
  }
}

TEST(HierarchicalAsyncPutLockFree, AutoRunWorkers) {
  using Distributer = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<int, int>;
  auto worker_task = [](int task) -> int { return task * task; };

  Distributer::Config config;
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = true;
  Distributer dist(worker_task, config);

  if (dist.is_root_manager()) {
    dist.insert_tasks({1, 2, 3, 4, 5});
    auto results = dist.finish_remaining_tasks();
    std::sort(results.begin(), results.end());
    EXPECT_EQ(results, (std::vector<int>{1, 4, 9, 16, 25}));
  }
}

TEST(HierarchicalAsyncPutLockFree, GatherOnce) {
  using Distributer = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<int, int>;
  auto worker_task = [](int task) -> int { return task * task; };

  Distributer::Config config;
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = true;
  Distributer dist(worker_task, config);

  if (dist.is_root_manager()) {
    dist.insert_tasks({1, 2, 3, 4, 5, 6, 7, 8});

    auto snapshot = dist.gather_once();
    EXPECT_LE(snapshot.size(), 8u);

    auto drained = dist.finish_remaining_tasks();
    std::vector<int> all = std::move(snapshot);
    all.insert(all.end(), drained.begin(), drained.end());
    std::sort(all.begin(), all.end());
    EXPECT_EQ(all, (std::vector<int>{1, 4, 9, 16, 25, 36, 49, 64}));
    EXPECT_EQ(dist.remaining_tasks_count(), 0u);
  }
}

TEST(LockFreeFinalization, FlatDrainsOutstandingWork) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "Need a worker rank to exercise remote finalization";
  }

  auto slow_worker = [](int task) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return task;
  };

  {
    using Distributor = dynampi::AsyncPutLockFreeMPIWorkDistributor<int, int>;
    Distributor dist(slow_worker);
    if (dist.is_root_manager()) dist.insert_tasks({1, 2, 3, 4, 5, 6, 7, 8});
    dist.finalize();
  }
}

TEST(LockFreeFinalization, HierarchicalDrainsOutstandingWork) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "Need a worker rank to exercise remote finalization";
  }

  auto slow_worker = [](int task) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return task;
  };

  {
    using Distributor = dynampi::HierarchicalAsyncPutLockFreeMPIWorkDistributor<int, int>;
    Distributor dist(slow_worker);
    if (dist.is_root_manager()) dist.insert_tasks({1, 2, 3, 4, 5, 6, 7, 8});
    dist.finalize();
  }
}
