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

void print_resolution_stats(std::vector<double>& deltas, int iterations) {
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

template <typename GetTimePoint>
std::vector<double> measure_resolution(GetTimePoint&& get_time_point, int iterations) {
  std::vector<double> deltas;

  for (int i = 0; i < iterations; ++i) {
    auto t1 = get_time_point();
    auto t2 = get_time_point();
    // Wait for time to advance
    while (t2 <= t1) {
      t2 = get_time_point();
    }
    auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    if (delta > 0) {
      deltas.push_back(static_cast<double>(delta));
    }
  }

  return deltas;
}

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

  const int iterations = 10000;
  auto deltas = measure_resolution([]() { return Clock::now(); }, iterations);
  print_resolution_stats(deltas, iterations);

  // Test if clock is steady
  bool is_steady = Clock::is_steady;
  std::cout << "  Is steady: " << (is_steady ? "yes" : "no") << "\n";
}

void test_timer_resolution() {
  std::cout << "\nDynaMPI Timer:\n";

  const int iterations = 10000;
  dynampi::Timer timer(dynampi::Timer::AutoStart::No);
  timer.start();
  auto deltas = measure_resolution([&timer]() { return timer.elapsed(); }, iterations);
  print_resolution_stats(deltas, iterations);
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
