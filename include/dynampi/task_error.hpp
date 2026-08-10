/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dynampi {

// Longest task-error message carried back to the manager. The RMA distributors
// report errors through fixed-width slots in a preallocated window, so the
// limit has to be a compile-time constant; the two-sided distributors truncate
// to the same length so a failing task reports identically whichever
// distributor ran it.
inline constexpr size_t kMaxTaskErrorMessage = 240;

// One task that threw, as reported by the rank that ran it.
struct TaskError {
  // Rank within the distributor's own (duplicated) communicator.
  int worker_rank = -1;
  // what() of the escaping exception, truncated to kMaxTaskErrorMessage.
  std::string message;
};

// Thrown on the manager by run_tasks()/finish_remaining_tasks() when a task
// threw and Config::rethrow_task_errors is set. Never thrown by finalize() or
// a destructor -- see BaseWorkDistributor's error accessors.
class TaskFailure : public std::runtime_error {
 public:
  explicit TaskFailure(TaskError error)
      : std::runtime_error("dynampi: task on rank " + std::to_string(error.worker_rank) +
                           " threw: " + error.message),
        m_error(std::move(error)) {}

  const TaskError& error() const noexcept { return m_error; }

 private:
  TaskError m_error;
};

namespace detail {

// Manager-side record of the tasks that threw, shared by every distributor.
//
// Each error surfaces exactly once: rethrow() hands back the oldest one and
// drops it, so a caller that catches and calls run_tasks() again sees the next
// one rather than the same one forever, and take() returns whatever has not
// been thrown yet.
class TaskErrorLog {
 public:
  void record(TaskError error) { m_errors.push_back(std::move(error)); }

  bool empty() const { return m_errors.empty(); }
  size_t size() const { return m_errors.size(); }

  std::vector<TaskError> take() {
    std::vector<TaskError> taken;
    taken.swap(m_errors);
    return taken;
  }

  // Call at a point where throwing is safe -- never from finalize() or a
  // destructor.
  void rethrow_first_if(bool enabled) {
    if (!enabled || m_errors.empty()) return;
    TaskError error = std::move(m_errors.front());
    m_errors.erase(m_errors.begin());
    throw TaskFailure(std::move(error));
  }

  // Destructors cannot throw, and dropping a task failure without a word is
  // exactly the silent-corruption case this whole mechanism exists to avoid.
  void warn_if_unreported(const char* distributor) const noexcept;

 private:
  std::vector<TaskError> m_errors;
};

// Wire form for the tree distributors: "<rank>\n<message>".
//
// A leaf's failure has to reach the root through however many coordinators sit
// in between, and each hop only knows the child it received from -- so the
// originating rank travels in the payload rather than being read off
// MPI_SOURCE. The flat distributors, where the reporting rank is the
// originating rank, send the bare message instead.
inline std::string encode_task_error(const TaskError& error) {
  return std::to_string(error.worker_rank) + "\n" + error.message;
}

inline TaskError decode_task_error(const std::string& payload) {
  const size_t split = payload.find('\n');
  if (split == std::string::npos) return TaskError{-1, payload};
  TaskError error;
  try {
    error.worker_rank = std::stoi(payload.substr(0, split));
  } catch (const std::exception&) {  // LCOV_EXCL_LINE -- only a corrupted payload gets here
    error.worker_rank = -1;          // LCOV_EXCL_LINE
  }
  error.message = payload.substr(split + 1);
  return error;
}

inline std::string truncate_task_error_message(const char* what) {
  std::string message = what == nullptr ? std::string("unknown exception") : std::string(what);
  if (message.size() > kMaxTaskErrorMessage) {
    message.resize(kMaxTaskErrorMessage - 3);
    message += "...";
  }
  return message;
}

// Runs worker_function(task), writing its result into `out`. Returns
// std::nullopt on success, or the failure message if the task threw.
//
// A throwing task must not take the job down with it, and it must not leave the
// protocol short one result either: every distributor's completion accounting
// (contiguous result prefixes, sent-vs-received counters, completion-log
// ranges) assumes exactly one result per dispatched task. So a failed task
// yields a default-constructed ResultT as a placeholder and the message travels
// to the manager out of band.
inline void TaskErrorLog::warn_if_unreported(const char* distributor) const noexcept {
  if (m_errors.empty()) return;
  std::fprintf(
      stderr, "dynampi: %s destroyed with %zu unreported task error(s); first was on rank %d: %s\n",
      distributor, m_errors.size(), m_errors.front().worker_rank, m_errors.front().message.c_str());
}

template <typename ResultT, typename F, typename TaskT>
std::optional<std::string> run_task_guarded(F& worker_function, TaskT&& task, ResultT& out) {
  try {
    out = worker_function(std::forward<TaskT>(task));
    return std::nullopt;
  } catch (const std::exception& e) {
    out = ResultT{};
    return truncate_task_error_message(e.what());
  } catch (...) {
    out = ResultT{};
    return std::string("unknown exception (not derived from std::exception)");
  }
}

}  // namespace detail

}  // namespace dynampi
