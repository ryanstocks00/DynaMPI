/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cassert>
#include <chrono>

namespace dynampi {

class Timer {
  std::optional<std::chrono::time_point<std::chrono::high_resolution_clock>> _start_time;
  std::chrono::duration<double> _elapsed_time{0.0};

 public:
  enum class AutoStart { Yes, No };

  Timer(AutoStart auto_start = AutoStart::Yes) {
    if (auto_start == AutoStart::Yes) {
      start();
    }
  }

  void start() {
    assert(!_start_time.has_value() && "Timer already started");
    _start_time = std::chrono::high_resolution_clock::now();
  }

  std::chrono::duration<double> stop() {
    assert(_start_time.has_value() && "Timer not started");
    auto end_time = std::chrono::high_resolution_clock::now();
    _elapsed_time += end_time - _start_time.value();
    _start_time.reset();
    return _elapsed_time;
  }

  void reset(AutoStart auto_start = AutoStart::Yes) {
    _start_time.reset();
    _elapsed_time = std::chrono::duration<double>(0.0);
    if (auto_start == AutoStart::Yes) {
      start();
    }
  }

  [[nodiscard]] std::chrono::duration<double> elapsed() const {
    if (_start_time.has_value()) {
      return _elapsed_time + (std::chrono::high_resolution_clock::now() - _start_time.value());
    }
    return _elapsed_time;
  }

  friend std::ostream& operator<<(std::ostream& os, const Timer& timer) {
    return os << timer.elapsed().count() << " seconds";
  }
};

}  // namespace dynampi
