/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <dynampi/utilities/timer.hpp>
#include <iostream>
#include <numeric>
#include <vector>

template <typename Clock>
void test_clock_resolution(const char* name) {
  using Duration = typename Clock::duration;
  using Period = typename Duration::period;

  std::cout << "\n" << name << ":\n";
  std::cout << "  Period: " << Period::num;
  if constexpr (Period::den != 1) {
    std::cout << "/" << Period::den;
  }
  std::cout << " seconds\n";

  // Test actual resolution by measuring smallest non-zero difference
  std::vector<double> deltas;
  const int iterations = 10000;

  for (int i = 0; i < iterations; ++i) {
    auto t1 = Clock::now();
    auto t2 = Clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    if (delta > 0) {
      deltas.push_back(static_cast<double>(delta));
    }
  }

  if (deltas.empty()) {
    std::cout << "  Measured resolution: < 1 ns (no measurable difference in " << iterations
              << " iterations)\n";
  } else {
    std::sort(deltas.begin(), deltas.end());
    double min_delta = deltas[0];
    double median_delta = deltas.size() % 2 == 0
                              ? (deltas[deltas.size() / 2 - 1] + deltas[deltas.size() / 2]) / 2.0
                              : deltas[deltas.size() / 2];
    double mean_delta = std::accumulate(deltas.begin(), deltas.end(), 0.0) / deltas.size();

    std::cout << "  Measured resolution (min): " << min_delta << " ns\n";
    std::cout << "  Measured resolution (median): " << median_delta << " ns\n";
    std::cout << "  Measured resolution (mean): " << mean_delta << " ns\n";
    std::cout << "  Non-zero measurements: " << deltas.size() << "/" << iterations << "\n";
  }

  // Test if clock is steady
  bool is_steady = Clock::is_steady;
  std::cout << "  Is steady: " << (is_steady ? "yes" : "no") << "\n";
}

void test_timer_resolution() {
  std::cout << "\nDynaMPI Timer:\n";

  // Test Timer class resolution
  std::vector<double> deltas;
  const int iterations = 10000;

  for (int i = 0; i < iterations; ++i) {
    dynampi::Timer timer(dynampi::Timer::AutoStart::No);
    timer.start();
    auto elapsed1 = timer.elapsed().count();
    auto elapsed2 = timer.elapsed().count();
    double delta = (elapsed2 - elapsed1) * 1e9;  // Convert to nanoseconds
    if (delta > 0) {
      deltas.push_back(delta);
    }
  }

  if (deltas.empty()) {
    std::cout << "  Measured resolution: < 1 ns (no measurable difference in " << iterations
              << " iterations)\n";
  } else {
    std::sort(deltas.begin(), deltas.end());
    double min_delta = deltas[0];
    double median_delta = deltas.size() % 2 == 0
                              ? (deltas[deltas.size() / 2 - 1] + deltas[deltas.size() / 2]) / 2.0
                              : deltas[deltas.size() / 2];
    double mean_delta = std::accumulate(deltas.begin(), deltas.end(), 0.0) / deltas.size();

    std::cout << "  Measured resolution (min): " << min_delta << " ns\n";
    std::cout << "  Measured resolution (median): " << median_delta << " ns\n";
    std::cout << "  Measured resolution (mean): " << mean_delta << " ns\n";
    std::cout << "  Non-zero measurements: " << deltas.size() << "/" << iterations << "\n";
  }
}

int main() {
  std::cout << "Timer Resolution Test\n";
  std::cout << "====================\n";

  test_clock_resolution<std::chrono::high_resolution_clock>("high_resolution_clock");
  test_clock_resolution<std::chrono::steady_clock>("steady_clock");
  test_clock_resolution<std::chrono::system_clock>("system_clock");
  test_timer_resolution();

  std::cout << "\n";
  return 0;
}
