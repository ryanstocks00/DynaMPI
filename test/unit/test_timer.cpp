/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <dynampi/utilities/timer.hpp>
#include <thread>

TEST(Timer, BasicFunctionality) {
  using namespace dynampi;
  using namespace std::chrono_literals;

  Timer timer1(Timer::AutoStart::No);
  EXPECT_EQ(timer1.elapsed().count(), 0.0);
  Timer timer2;
  timer1.start();
  std::this_thread::sleep_for(1ms);
  auto elapsed1 = timer1.stop();
  EXPECT_GT(elapsed1.count(), 0.0);
  EXPECT_EQ(timer1.elapsed(), elapsed1);
  EXPECT_GE(timer2.elapsed(), elapsed1);
  EXPECT_LT(timer2.elapsed(), 0.1s);
  timer1.reset();
  std::this_thread::sleep_for(1ms);
  EXPECT_GT(timer1.elapsed().count(), 0.0);
  timer1.reset(Timer::AutoStart::No);
  std::this_thread::sleep_for(1ms);
  EXPECT_EQ(timer1.elapsed().count(), 0.0);
  std::stringstream oss;
  oss << timer1;
  EXPECT_EQ(oss.str(), "0 seconds");
}
