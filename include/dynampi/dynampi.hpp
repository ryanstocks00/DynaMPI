/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

namespace dynampi {

namespace version {

inline constexpr int major = DYNAMPI_VERSION_MAJOR;
inline constexpr int minor = DYNAMPI_VERSION_MINOR;
inline constexpr int patch = DYNAMPI_VERSION_PATCH;

// Macros for compile-time version string
#define DYNAMPI_STR_HELPER(x) #x
#define DYNAMPI_STR(x) DYNAMPI_STR_HELPER(x)
#define DYNAMPI_VERSION_STRING                                                                   \
  "v" DYNAMPI_STR(DYNAMPI_VERSION_MAJOR) "." DYNAMPI_STR(DYNAMPI_VERSION_MINOR) "." DYNAMPI_STR( \
      DYNAMPI_VERSION_PATCH)

inline constexpr std::string_view string = DYNAMPI_VERSION_STRING;

[[nodiscard]] constexpr bool is_at_least(int v_major, int v_minor, int v_patch) {
  return std::tie(major, minor, patch) >= std::tie(v_major, v_minor, v_patch);
}

[[nodiscard]] inline constexpr std::string_view compile_date() { return __DATE__ " " __TIME__; }

[[nodiscard]] inline constexpr std::string_view commit_hash() { return DYNAMPI_COMMIT_HASH; }

}  // namespace version

#define DYNAMPI_MPI_CHECK(func, args)                                              \
  do {                                                                             \
    int err = func args;                                                           \
    if (err != MPI_SUCCESS) {                                                      \
      char error_string[MPI_MAX_ERROR_STRING];                                     \
      int length_of_error_string;                                                  \
      MPI_Error_string(err, error_string, &length_of_error_string);                \
      throw std::runtime_error(std::string("MPI error in " #func ": ") +           \
                               std::string(error_string, length_of_error_string)); \
    }                                                                              \
  } while (false)

template <class T>
struct MPI_Type;
template <>
struct MPI_Type<char> {
  inline static MPI_Datatype value = MPI_CHAR;
};
template <>
struct MPI_Type<std::byte> {
  inline static MPI_Datatype value = MPI_BYTE;
};
#ifdef MPI_CXX_BOOL
template <>
struct MPI_Type<bool> {
  inline static MPI_Datatype value = MPI_CXX_BOOL;
};
#endif
template <>
struct MPI_Type<signed char> {
  inline static MPI_Datatype value = MPI_SIGNED_CHAR;
};
template <>
struct MPI_Type<unsigned char> {
  inline static MPI_Datatype value = MPI_UNSIGNED_CHAR;
};
template <>
struct MPI_Type<short> {
  inline static MPI_Datatype value = MPI_SHORT;
};
template <>
struct MPI_Type<unsigned short> {
  inline static MPI_Datatype value = MPI_UNSIGNED_SHORT;
};
template <>
struct MPI_Type<int> {
  inline static MPI_Datatype value = MPI_INT;
};
template <>
struct MPI_Type<unsigned int> {
  inline static MPI_Datatype value = MPI_UNSIGNED;
};
template <>
struct MPI_Type<long> {
  inline static MPI_Datatype value = MPI_LONG;
};
template <>
struct MPI_Type<unsigned long> {
  inline static MPI_Datatype value = MPI_UNSIGNED_LONG;
};
template <>
struct MPI_Type<long long> {
  inline static MPI_Datatype value = MPI_LONG_LONG_INT;
};
template <>
struct MPI_Type<unsigned long long> {
  inline static MPI_Datatype value = MPI_UNSIGNED_LONG_LONG;
};
template <>
struct MPI_Type<float> {
  inline static MPI_Datatype value = MPI_FLOAT;
};
template <>
struct MPI_Type<double> {
  inline static MPI_Datatype value = MPI_DOUBLE;
};
template <>
struct MPI_Type<long double> {
  inline static MPI_Datatype value = MPI_LONG_DOUBLE;
};

class MPICommunicator {
 public:
  enum Ownership {
    NotOwned,  // The communicator is not owned by this class and should not be freed.
    Owned,     // The communicator is owned by this class and will be freed in the destructor.
  };

 private:
  MPI_Comm _comm;
  Ownership _ownership;

 public:
  MPICommunicator(MPI_Comm comm, Ownership ownership = NotOwned)
      : _comm(comm), _ownership(ownership) {
    if (_ownership == Owned) {
      DYNAMPI_MPI_CHECK(MPI_Comm_dup, (comm, &_comm));
    }
  }

  ~MPICommunicator() {
    if (_ownership == Owned) {
      MPI_Comm_free(&_comm);
    }
  }

  int rank() const {
    int rank;
    DYNAMPI_MPI_CHECK(MPI_Comm_rank, (_comm, &rank));
    return rank;
  }

  int size() const {
    int size;
    DYNAMPI_MPI_CHECK(MPI_Comm_size, (_comm, &size));
    return size;
  }

  [[nodiscard]] MPI_Comm get() const { return _comm; }
};

template <typename TaskT, typename ResultT>
class NaiveMPIWorkDistributor {
  MPICommunicator _communicator;
  std::function<ResultT(TaskT)> _worker_function;
  int _manager_rank;
  std::queue<TaskT> _task_queue;
  std::vector<int64_t> _worker_task_indices;
  std::vector<ResultT> _results;

  size_t _tasks_sent = 0;
  size_t _results_received = 0;

  static constexpr int TASK_TAG = 0;
  static constexpr int DONE_TAG = 1;
  static constexpr int RESULT_TAG = 2;
  static constexpr int REQUEST_TAG = 3;

  MPI_Datatype _task_type = MPI_Type<TaskT>::value;
  MPI_Datatype _result_type = MPI_Type<ResultT>::value;

 public:
  NaiveMPIWorkDistributor(MPI_Comm comm, std::function<ResultT(TaskT)> worker_function,
                          int manager_rank = 0)
      : _communicator(comm, MPICommunicator::Owned),
        _worker_function(worker_function),
        _manager_rank(manager_rank) {
    if (is_manager()) _worker_task_indices.resize(_communicator.size() - 1, -1);
  }

  void run_worker() {
    assert(_communicator.rank() != _manager_rank && "Worker cannot run on the manager rank");
    MPI_Send(nullptr, 0, _task_type, _manager_rank, REQUEST_TAG, _communicator.get());
    while (true) {
      MPI_Status status;
      TaskT message;
      MPI_Recv(&message, 1, _task_type, _manager_rank, MPI_ANY_TAG, _communicator.get(), &status);
      if (status.MPI_TAG == DONE_TAG) {
        break;
      }
      _tasks_sent++;
      assert(status.MPI_TAG == TASK_TAG && "Unexpected tag received in worker");
      ResultT result = _worker_function(message);
      MPI_Send(&result, 1, _result_type, _manager_rank, RESULT_TAG, _communicator.get());
      _results_received++;
    }
  }

  bool is_manager() const { return _communicator.rank() == _manager_rank; }

  size_t remaining_tasks_count() const {
    assert(_communicator.rank() == _manager_rank && "Only the manager can check remaining tasks");
    return _task_queue.size();
  }

  void insert_task(TaskT task) {
    assert(_communicator.rank() == _manager_rank && "Only the manager can distribute tasks");
    _task_queue.push(task);
  }

  template <std::ranges::input_range Range>
  void insert_tasks(const Range& tasks) {
    assert(_communicator.rank() == _manager_rank && "Only the manager can distribute tasks");
    for (const auto& task : tasks) {
      _task_queue.push(task);
    }
  }

  void receive_result(MPI_Status& status) {
    assert(_communicator.rank() == _manager_rank &&
           "Only the manager can receive results and send tasks");
    DYNAMPI_MPI_CHECK(MPI_Probe, (MPI_ANY_SOURCE, MPI_ANY_TAG, _communicator.get(), &status));
    if (status.MPI_TAG == RESULT_TAG) {
      int64_t task_idx =
          _worker_task_indices[status.MPI_SOURCE - (status.MPI_SOURCE > _manager_rank)];
      _worker_task_indices[status.MPI_SOURCE - (status.MPI_SOURCE > _manager_rank)] = -1;
      assert(task_idx >= 0 && "Task index should be valid");
      if (static_cast<uint64_t>(task_idx) >= _results.size()) {
        _results.resize(task_idx + 1);
      }
      DYNAMPI_MPI_CHECK(MPI_Recv, (&_results[task_idx], 1, _result_type, status.MPI_SOURCE,
                                   RESULT_TAG, _communicator.get(), &status));
      _results_received++;
    } else {
      assert(status.MPI_TAG == REQUEST_TAG && "Unexpected tag received in worker");
      DYNAMPI_MPI_CHECK(MPI_Recv, (nullptr, 0, _result_type, status.MPI_SOURCE, REQUEST_TAG,
                                   _communicator.get(), &status));
    }
  }

  [[nodiscard]] std::vector<ResultT> distribute_tasks() {
    assert(_communicator.rank() == _manager_rank && "Only the manager can distribute tasks");
    while (!_task_queue.empty()) {
      MPI_Status status;
      receive_result(status);
      TaskT& task = _task_queue.front();
      int worker = status.MPI_SOURCE;
      _worker_task_indices[worker - (worker > _manager_rank)] = _tasks_sent;
      DYNAMPI_MPI_CHECK(MPI_Send, (&task, 1, _task_type, worker, TASK_TAG, _communicator.get()));
      _task_queue.pop();
      _tasks_sent++;
    }
    for (int i = 1; i < _communicator.size(); i++) {
      MPI_Status status;
      receive_result(status);
    }
    assert(_results_received == _tasks_sent && "Not all tasks were processed by workers");
    assert(_results.size() == _tasks_sent && "Results size should match tasks sent");
    return _results;
  }

  void finalize() {
    assert(_communicator.rank() == _manager_rank &&
           "Only the manager can finalize the work distribution");
    for (int i = 0; i < _communicator.size() - 1; i++) {
      DYNAMPI_MPI_CHECK(MPI_Send,
                        (nullptr, 0, _task_type, worker_for_idx(i), DONE_TAG, _communicator.get()));
    }
  }

  ~NaiveMPIWorkDistributor() {
    if (is_manager()) finalize();
    assert(_tasks_sent == _results_received && "Not all tasks were processed by workers");
  }

 private:
  int idx_for_worker(int worker_rank) const {
    assert(worker_rank != _manager_rank && "Manager rank should not be used as a worker rank");
    if (worker_rank < _manager_rank) {
      return worker_rank;
    } else {
      return worker_rank - 1;
    }
  }

  int worker_for_idx(int idx) const { return (idx < _manager_rank) ? idx : (idx + 1); }
};

template <typename TaskT, typename ResultT>
std::vector<ResultT> mpi_manager_worker_distribution(const std::span<TaskT>& tasks,
                                                     std::function<ResultT(TaskT)> worker_function,
                                                     MPI_Comm comm = MPI_COMM_WORLD,
                                                     int manager_rank = 0) {
  NaiveMPIWorkDistributor<TaskT, ResultT> distributor(comm, worker_function, manager_rank);
  if (distributor.is_manager()) {
    for (const auto& task : tasks) {
      distributor.insert_task(task);
    }
    return distributor.distribute_tasks();
  } else {
    assert(tasks.empty() && "Workers should not have tasks to process");
    distributor.run_worker();
    return {};
  }
}

}  // namespace dynampi
