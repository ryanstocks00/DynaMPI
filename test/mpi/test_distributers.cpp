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
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "dynampi/impl/hierarchical_distributor.hpp"
#include "dynampi/impl/hierarchical_lockfree_rma_distributor.hpp"
#include "dynampi/impl/lockfree_rma_distributor.hpp"
#include "dynampi/impl/variable_batch.hpp"
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

// Specialized wrapper for HierarchicalWorkDistributor with coordinator_per_node config
template <bool CoordinatorPerNode>
struct HierarchicalDistributerTypeWrapper {
  template <typename TaskT, typename ResultT, typename... Options>
  using type = dynampi::HierarchicalWorkDistributor<TaskT, ResultT, Options...>;

  static constexpr bool use_immediate_recv = false;
  static constexpr size_t max_result_size = 1024;

  template <typename TaskT, typename ResultT, typename... Options>
  static typename dynampi::HierarchicalWorkDistributor<TaskT, ResultT, Options...>::Config
  get_config() {
    typename dynampi::HierarchicalWorkDistributor<TaskT, ResultT, Options...>::Config config;
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
    ::testing::Types<DistributerTypeWrapper<dynampi::NaiveWorkDistributor>,
                     HierarchicalDistributerTypeWrapper<true>,
                     HierarchicalDistributerTypeWrapper<false>,
                     DistributerTypeWrapper<dynampi::LockFreeRMAWorkDistributor>>;

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
  if constexpr (is_specialization_of<dynampi::HierarchicalWorkDistributor, Distributer>::value) {
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

// Mirrors how EXESS ships serialized MBE fragment requests/results through
// DynaMPI: both TaskT and ResultT are variable-length std::vector<std::byte>
// buffers, well past the single-element case. HierarchicalMPIWorkDistributor
// and the hierarchical distributor's leaf-result path
// (receive_result_from) hits `if constexpr (result_mpi_type::resize_required)
// DYNAMPI_UNIMPLEMENTED(...)` for such ResultT types; under NDEBUG,
// DYNAMPI_ASSERT/DYNAMPI_UNIMPLEMENTED compile to nothing followed by
// __builtin_unreachable(), so a reachable "unimplemented" branch is undefined
// behavior rather than a clean abort -- in practice manifesting as heap
// corruption/segfaults far away from this function (e.g. in
// allocate_task_to_child()), not a clean failure here. See also Example2,
// which documents/skips this exact gap for ResultT = std::vector<int>.
TYPED_TEST(DynamicDistribution, VariableSizeTaskAndResult) {
  using Task = std::vector<std::byte>;
  using Result = std::vector<std::byte>;
  using Distributer = DistributerOf<TypeParam, Task, Result>;

  auto worker_task = [](Task task) -> Result {
    Result result(task.size() * 2);
    for (size_t i = 0; i < task.size(); ++i) {
      result[i] = task[i];
      result[task.size() + i] = static_cast<std::byte>(static_cast<unsigned char>(task[i]) + 1);
    }
    return result;
  };

  auto config = get_distributer_config<TypeParam, Task, Result>();
  Distributer distributor(worker_task, config);
  if (distributor.is_root_manager()) {
    std::vector<Task> tasks;
    for (int i = 0; i < 20; ++i) {
      Task task(static_cast<size_t>(3 + i));  // varying lengths 3..22 bytes
      for (size_t b = 0; b < task.size(); ++b) {
        task[b] = static_cast<std::byte>((i * 7 + b) % 251);
      }
      tasks.push_back(task);
    }
    for (auto& t : tasks) distributor.insert_task(t);
    std::vector<Result> results = distributor.finish_remaining_tasks();
    ASSERT_EQ(results.size(), tasks.size());

    std::vector<Result> expected;
    expected.reserve(tasks.size());
    for (auto& t : tasks) expected.push_back(worker_task(t));

    // A hand-rolled comparator, rather than vector<byte>'s default <=>,
    // sidesteps a GCC false positive (-Werror=stringop-overread on the
    // inlined memcmp) seen on some toolchains when three-way-comparing
    // variable-length byte buffers.
    auto byte_vector_less = [](const Result& a, const Result& b) {
      const size_t n = std::min(a.size(), b.size());
      for (size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) return a[i] < b[i];
      }
      return a.size() < b.size();
    };
    // The generated task/result bytes never happen to make one result a
    // prefix of another, so exercise the tie-break directly.
    EXPECT_TRUE(byte_vector_less(Result{std::byte{1}}, Result{std::byte{1}, std::byte{2}}));
    EXPECT_FALSE(byte_vector_less(Result{std::byte{1}, std::byte{2}}, Result{std::byte{1}}));
    std::sort(results.begin(), results.end(), byte_vector_less);
    std::sort(expected.begin(), expected.end(), byte_vector_less);
    EXPECT_EQ(results, expected);
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
      is_specialization_of<dynampi::LockFreeRMAWorkDistributor, Distributer>::value) {
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
          is_specialization_of<dynampi::LockFreeRMAWorkDistributor, Distributer>::value;
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
      if constexpr (is_specialization_of<dynampi::NaiveWorkDistributor, Distributer>::value) {
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
      if constexpr (is_specialization_of<dynampi::NaiveWorkDistributor, Distributer>::value) {
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
      // Detailed is the only mode that times the calls it counts.
      const auto& comm_stats = work_distributer.get_statistics().comm_statistics;
      EXPECT_GE(comm_stats.send_time, 0.0);
      EXPECT_GE(comm_stats.recv_time, 0.0);
      // macOS CI: MPI_Wtime can round short shared-memory sends to zero.
#if !defined(__APPLE__)
      if (MPIEnvironment::world_comm_size() > 1) {
        EXPECT_GT(comm_stats.send_time + comm_stats.recv_time, 0.0);
      }
#endif
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
// bench to avoid per-retry busy-spin harvest. Only LockFreeRMA exposes
// it.
//
// Under SMPI, busy-polling gather_once() (or OS-sleeping in workers) can
// starve other ranks: LockFreeRMA's harvest is pure RMA and may not yield the
// simulator, so workers never run and remaining_tasks_count never drains.
// Exercise one snapshot, then drain with finish_remaining_tasks() which
// already has a proper no-progress idle path.
TYPED_TEST(DynamicDistribution, GatherOnce) {
  using Distributer = DistributerOf<TypeParam, int, int>;
  constexpr bool has_gather_once =
      is_specialization_of<dynampi::LockFreeRMAWorkDistributor, Distributer>::value;
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

TEST(LockFreeCapacity, RejectsTaskTableOverflow) {
  using Distributor = dynampi::LockFreeRMAWorkDistributor<int, int>;
  Distributor::Config config;
  config.comm = MPI_COMM_SELF;
  config.max_tasks = 1;
  Distributor dist([](int task) { return task; }, config);
  EXPECT_THROW(dist.insert_tasks(std::vector<int>{1, 2}), std::length_error);
}

TEST(LockFreeCapacity, CapsRecordedTaskErrorsAt16) {
  // A single rank (num_workers() == 0) runs tasks through a locally-serial
  // fallback rather than the RMA error table this test targets, so a worker
  // rank is required.
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "Need a worker rank to exercise the RMA error table";
  }
  using Distributor = dynampi::LockFreeRMAWorkDistributor<int, int>;
  Distributor::Config config;
  config.rethrow_task_errors = false;
  constexpr int kFailures = 20;  // exceeds the 16-slot error table
  auto always_throws = [](int task) -> int {
    throw std::runtime_error("fail " + std::to_string(task));
  };
  Distributor dist(always_throws, config);

  if (dist.is_root_manager()) {
    std::vector<int> tasks(kFailures);
    for (int i = 0; i < kFailures; ++i) tasks[i] = i;
    dist.insert_tasks(tasks);
    auto results = dist.finish_remaining_tasks();
    EXPECT_EQ(results.size(), static_cast<size_t>(kFailures));

    auto errors = dist.take_task_errors();
    EXPECT_EQ(errors.size(), 16u);
  }
}

TEST(VariableBatch, RejectsBufferTooShortForItemCount) {
  using Item = std::vector<std::byte>;
  const std::vector<std::byte> truncated(4);  // shorter than the 8-byte item-count header
  EXPECT_THROW(dynampi::detail::unpack_variable_batch<Item>(truncated), std::runtime_error);
}

TEST(VariableBatch, RejectsLengthTableOverflowingBuffer) {
  using Item = std::vector<std::byte>;
  std::vector<std::byte> buf(16);  // room for the header plus one length-table slot
  const uint64_t n_items = 5;      // claims more items than the buffer can back
  std::copy_n(reinterpret_cast<const std::byte*>(&n_items), sizeof(n_items), buf.data());
  EXPECT_THROW(dynampi::detail::unpack_variable_batch<Item>(buf), std::runtime_error);
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

// MPI_Type_size(MPI_DATATYPE_NULL) fails, covering the "could not query"
// branch of check_fixed_size_mpi_type.
struct UnqueryableDatatype {
  int x;
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

template <>
struct dynampi::MPI_Type<UnqueryableDatatype> {
  inline static const MPI_Datatype value = MPI_DATATYPE_NULL;
  inline static const bool resize_required = false;
  static int count(const UnqueryableDatatype&) noexcept { return 1; }
  static void resize(UnqueryableDatatype&, int) noexcept {}
  static void* ptr(UnqueryableDatatype& v) noexcept { return &v; }
  static const void* ptr(const UnqueryableDatatype& v) noexcept { return &v; }
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
  // Rejection runs in check_fixed_size_mpi_type before any send/recv, so
  // exercise the MPI_Type surface explicitly for coverage.
  Unaligned value{1.f, 2.f, 3.f};
  EXPECT_EQ(dynampi::MPI_Type<Unaligned>::count(value), 2);
  dynampi::MPI_Type<Unaligned>::resize(value, 2);
  EXPECT_EQ(dynampi::MPI_Type<Unaligned>::ptr(value), static_cast<void*>(&value));
  EXPECT_EQ(dynampi::MPI_Type<Unaligned>::ptr(std::as_const(value)),
            static_cast<const void*>(&value));

  // Never actually invoked: both constructors below reject Unaligned before
  // running any task, so the body is structurally unreachable here.
  auto identity = [](Unaligned v) { return v; };  // LCOV_EXCL_LINE
  {
    using Distributor = dynampi::LockFreeRMAWorkDistributor<Unaligned, Unaligned>;
    typename Distributor::Config config;
    config.comm = MPI_COMM_SELF;
    EXPECT_THROW(Distributor(identity, config), std::invalid_argument);
  }
  {
    using Distributor = dynampi::HierarchicalWorkDistributor<Unaligned, Unaligned>;
    typename Distributor::Config config;
    config.comm = MPI_COMM_SELF;
    EXPECT_THROW(Distributor(identity, config), std::invalid_argument);
  }
}

TEST(FixedSizeMPIType, RejectsUnqueryableDatatype) {
  EXPECT_THROW((dynampi::check_fixed_size_mpi_type<UnqueryableDatatype>("task", "TestDistributor")),
               std::invalid_argument);
  try {
    dynampi::check_fixed_size_mpi_type<UnqueryableDatatype>("task", "TestDistributor");
    FAIL() << "expected invalid_argument";  // LCOV_EXCL_LINE -- only reached if the test above is
                                            // already failing
  } catch (const std::invalid_argument& e) {
    EXPECT_NE(std::string(e.what()).find("could not query the MPI datatype"), std::string::npos);
  }
}

TEST(LockFreeRMALevel, SingletonCommunicator) {
  using Level = dynampi::detail::LockFreeRMALevel<int, int>;
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

// --- HierarchicalLockFreeRMAWorkDistributor ---
// Node-aware tree topology (manager <-> per-node coordinators <-> local
// workers) with LockFreeRMAWorkDistributor's one-sided,
// collective-free protocol (fetch-and-add claiming, Put-based result return
// via a completion log) at each level. Results are unordered, so tests sort
// before comparing.

TEST(HierarchicalLockFreeRMA, BasicFlow) {
  using TaskT = int;
  using Distributer = dynampi::HierarchicalLockFreeRMAWorkDistributor<TaskT, double>;
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

TEST(HierarchicalLockFreeRMA, ManagerRankNonZero) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "Need at least 2 ranks for non-zero manager rank";
  }
  const int manager_rank = 1;
  using TaskT = int;
  using ResultT = double;
  using Distributer = dynampi::HierarchicalLockFreeRMAWorkDistributor<TaskT, ResultT>;
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

TEST(HierarchicalLockFreeRMA, MultipleRoundsOfTasks) {
  using Task = int;
  using Result = int;
  using Distributer = dynampi::HierarchicalLockFreeRMAWorkDistributor<Task, Result>;
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

TEST(HierarchicalLockFreeRMA, RunTasksMaxTasks) {
  using Task = int;
  using Result = int;
  using Distributer = dynampi::HierarchicalLockFreeRMAWorkDistributor<Task, Result>;
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
// Both HierarchicalWorkDistributor and
// HierarchicalLockFreeRMAWorkDistributor expose the same
// max_upper_fanout / max_local_group_size Config knobs and the same
// insert_tasks / finish_remaining_tasks manager API, so one helper covers
// send/recv and RMA. Keep the TEST() names distinct: Open MPI 4's
// osc/pt2pt workaround in test/CMakeLists.txt filters the LockFreeRMA
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
TEST(HierarchicalLockFreeRMA, GroupedUpperHierarchy) {
  using Dist = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;
  Dist::Config config;
  config.max_upper_fanout = 2;
  config.max_local_group_size = 2;
  expect_upper_hierarchy_squares<Dist>(config);
}

TEST(Hierarchical, GroupedUpperHierarchy) {
  using Dist = dynampi::HierarchicalWorkDistributor<int, int>;
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
// smoke check. LockFreeRMA variant is excluded from Open MPI 4's default
// suite and run under osc/pt2pt (see test/CMakeLists.txt).
TEST(Hierarchical, AutoFanoutUpperHierarchy) {
  using Dist = dynampi::HierarchicalWorkDistributor<int, int>;
  Dist::Config config;
  config.max_upper_fanout = -1;
  config.max_local_group_size = 1;
  expect_upper_hierarchy_squares<Dist>(config);
}

TEST(HierarchicalLockFreeRMA, AutoFanoutUpperHierarchy) {
  using Dist = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;
  Dist::Config config;
  config.max_upper_fanout = -1;
  config.max_local_group_size = 1;
  expect_upper_hierarchy_squares<Dist>(config);
}

TEST(HierarchicalLockFreeRMA, SingletonGatherAndFinalize) {
  using Distributor = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;

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

TEST(HierarchicalLockFreeRMA, AutoRunWorkers) {
  using Distributer = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;
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

TEST(HierarchicalLockFreeRMA, GatherOnce) {
  using Distributer = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;
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
    using Distributor = dynampi::LockFreeRMAWorkDistributor<int, int>;
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
    using Distributor = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;
    Distributor dist(slow_worker);
    if (dist.is_root_manager()) dist.insert_tasks({1, 2, 3, 4, 5, 6, 7, 8});
    dist.finalize();
  }
}

// --- Task error handling -----------------------------------------------
//
// A task that throws must not take the MPI job down with it. The manager
// either rethrows it as dynampi::TaskFailure (Config::rethrow_task_errors,
// the default) or hands it back from take_task_errors() so the caller can
// recover. Either way distribution completes, every dispatched task still
// yields exactly one result, and no rank is left waiting.

namespace {
constexpr int kFailingTask = 3;
constexpr int kTaskCount = 8;

int throwing_worker(int task) {
  if (task == kFailingTask) throw std::runtime_error("task blew up");
  return task * 2;
}
}  // namespace

TYPED_TEST(DynamicDistribution, TaskErrorIsRecoverable) {
  using Distributer = DistributerOf<TypeParam, int, int>;

  auto config = get_distributer_config<TypeParam, int, int>();
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = true;
  config.rethrow_task_errors = false;
  Distributer distributor(throwing_worker, config);

  if (!distributor.is_root_manager()) return;

  for (int i = 0; i < kTaskCount; ++i) distributor.insert_task(i);
  auto results = distributor.finish_remaining_tasks();

  // The failed task still occupies a slot, so nothing downstream has to
  // reason about a short result set.
  EXPECT_EQ(results.size(), static_cast<size_t>(kTaskCount));

  auto errors = distributor.take_task_errors();
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_NE(errors[0].message.find("task blew up"), std::string::npos);
  EXPECT_GE(errors[0].worker_rank, 0);
  EXPECT_LT(errors[0].worker_rank, MPIEnvironment::world_comm_size());
  EXPECT_FALSE(distributor.has_task_errors()) << "take_task_errors should drain";

  // Every task except the failing one produced its real result.
  std::sort(results.begin(), results.end());
  std::vector<int> expected;
  for (int i = 0; i < kTaskCount; ++i) expected.push_back(i == kFailingTask ? 0 : i * 2);
  std::sort(expected.begin(), expected.end());
  EXPECT_EQ(results, expected);
}

TYPED_TEST(DynamicDistribution, TaskErrorPropagatesToManager) {
  using Distributer = DistributerOf<TypeParam, int, int>;

  auto config = get_distributer_config<TypeParam, int, int>();
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = true;
  Distributer distributor(throwing_worker, config);

  if (!distributor.is_root_manager()) return;

  for (int i = 0; i < kTaskCount; ++i) distributor.insert_task(i);

  bool threw = false;
  try {
    auto ignored = distributor.finish_remaining_tasks();
    (void)ignored;
  } catch (const dynampi::TaskFailure& e) {
    threw = true;
    EXPECT_NE(std::string(e.what()).find("task blew up"), std::string::npos);
    EXPECT_EQ(e.error().message.find("task blew up"), 0u);
  }
  EXPECT_TRUE(threw) << "expected TaskFailure on the manager";

  // The throw consumed that error and left the results buffered, so a caller
  // that recovers can still collect them.
  EXPECT_FALSE(distributor.has_task_errors());
  auto results = distributor.finish_remaining_tasks();
  EXPECT_EQ(results.size(), static_cast<size_t>(kTaskCount));
}

TYPED_TEST(DynamicDistribution, SurvivesManyFailingTasks) {
  using Distributer = DistributerOf<TypeParam, int, int>;

  // Every task fails: the shutdown path has to stay balanced when no real
  // result is ever produced.
  auto always_throws = [](int task) -> int {
    throw std::runtime_error("always fails on " + std::to_string(task));
  };

  auto config = get_distributer_config<TypeParam, int, int>();
  config.comm = MPI_COMM_WORLD;
  config.auto_run_workers = true;
  config.rethrow_task_errors = false;
  Distributer distributor(always_throws, config);

  if (!distributor.is_root_manager()) return;

  for (int i = 0; i < kTaskCount; ++i) distributor.insert_task(i);
  auto results = distributor.finish_remaining_tasks();
  EXPECT_EQ(results.size(), static_cast<size_t>(kTaskCount));

  auto errors = distributor.take_task_errors();
  // The RMA distributors keep a bounded error table, so the count is capped
  // rather than exact; every distributor must report at least one.
  EXPECT_GE(errors.size(), 1u);
  EXPECT_LE(errors.size(), static_cast<size_t>(kTaskCount));
  for (const auto& error : errors) {
    EXPECT_NE(error.message.find("always fails"), std::string::npos);
  }
}

TEST(HierarchicalLockFreeRMA, TaskErrorIsRecoverable) {
  using Distributor = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;
  Distributor::Config config;
  config.rethrow_task_errors = false;
  Distributor distributor(throwing_worker, config);

  if (!distributor.is_root_manager()) return;

  std::vector<int> tasks;
  for (int i = 0; i < kTaskCount; ++i) tasks.push_back(i);
  distributor.insert_tasks(tasks);
  auto results = distributor.finish_remaining_tasks();
  EXPECT_EQ(results.size(), static_cast<size_t>(kTaskCount));

  auto errors = distributor.take_task_errors();
  ASSERT_EQ(errors.size(), 1u);
  EXPECT_NE(errors[0].message.find("task blew up"), std::string::npos);
}

TEST(HierarchicalLockFreeRMA, TaskErrorPropagatesToManager) {
  using Distributor = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;
  Distributor distributor(throwing_worker);

  if (!distributor.is_root_manager()) return;

  std::vector<int> tasks;
  for (int i = 0; i < kTaskCount; ++i) tasks.push_back(i);
  distributor.insert_tasks(tasks);

  EXPECT_THROW(
      {
        auto ignored = distributor.finish_remaining_tasks();
        (void)ignored;
      },
      dynampi::TaskFailure);
  auto rest = distributor.finish_remaining_tasks();
  EXPECT_EQ(rest.size(), static_cast<size_t>(kTaskCount));
}

TEST(HierarchicalLockFreeRMA, CapsRecordedTaskErrorsAt16) {
  if (MPIEnvironment::world_comm_size() < 2) {
    GTEST_SKIP() << "Need a worker rank to exercise the RMA error table";
  }
  using Distributor = dynampi::HierarchicalLockFreeRMAWorkDistributor<int, int>;
  Distributor::Config config;
  config.rethrow_task_errors = false;
  constexpr int kFailures = 20;  // exceeds the 16-slot error table
  auto always_throws = [](int task) -> int {
    throw std::runtime_error("fail " + std::to_string(task));
  };
  Distributor dist(always_throws, config);

  if (dist.is_root_manager()) {
    std::vector<int> tasks(kFailures);
    for (int i = 0; i < kFailures; ++i) tasks[i] = i;
    dist.insert_tasks(tasks);
    auto results = dist.finish_remaining_tasks();
    EXPECT_EQ(results.size(), static_cast<size_t>(kFailures));

    auto errors = dist.take_task_errors();
    EXPECT_EQ(errors.size(), 16u);
  }
}
