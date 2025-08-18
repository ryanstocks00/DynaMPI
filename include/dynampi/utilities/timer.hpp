/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <chrono>

class Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> _start_time;

 public:
  Timer() : _start_time(std::chrono::high_resolution_clock::now()) {}

  void reset() { _start_time = std::chrono::high_resolution_clock::now(); }
  std::chrono::duration<double> elapsed() const {
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - _start_time);
  }

  friend std::ostream& operator<<(std::ostream& os, const Timer& timer) {
    return os << timer.elapsed().count() << " seconds";
  }
};
