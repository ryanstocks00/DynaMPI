/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include <stdexcept>

#ifndef NDEBUG
// Redirect calls to our stub function
#define MPI_Comm_rank DYNAMPI_TEST_MPI_Comm_rank
static inline int DYNAMPI_TEST_MPI_Comm_rank(MPI_Comm /*comm*/, int* rank) {
  if (rank) *rank = 0;
  return MPI_SUCCESS;
}
#endif

// Include the header-under-test AFTER redefining MPI_Comm_rank.
#include <dynampi/utilities/assert.hpp>

using namespace dynampi;

// Helper to capture stderr and detect if a throw occurred.
namespace {
template <typename F>
std::pair<std::string, bool> CaptureStderrAndDidThrow(F&& f) {
  testing::internal::CaptureStderr();
  bool threw = false;
  try {
    f();
  } catch (...) {
    threw = true;
  }
  std::string out = testing::internal::GetCapturedStderr();
  return {out, threw};
}
}  // namespace

TEST(OptionalString, NoArgsReturnsNullopt) {
#ifndef NDEBUG
  auto s = OptionalString();
  EXPECT_FALSE(s.has_value());
#else
  GTEST_SKIP() << "OptionalString only exists in non-NDEBUG builds.";
#endif
}

TEST(OptionalString, ConcatsMultipleArgs) {
#ifndef NDEBUG
  auto s = OptionalString("Value=", 42, ", ok");
  ASSERT_TRUE(s.has_value());
  EXPECT_EQ(*s, "Value=42, ok");
#else
  GTEST_SKIP() << "OptionalString only exists in non-NDEBUG builds.";
#endif
}

TEST(DynaMPIAssert, TrueConditionDoesNotThrow) {
#ifndef NDEBUG
  int a1 = 1, b1 = 1;
  EXPECT_NO_THROW({ DYNAMPI_ASSERT(a1 == b1, "should not throw"); });
#else
  EXPECT_NO_THROW({ DYNAMPI_ASSERT(false, "no-op in NDEBUG"); });
#endif
}

TEST(DynaMPIAssert, NoAssertInDestructorDuringThrow) {
#ifndef NDEBUG
  class Test {
   public:
    ~Test() { DYNAMPI_ASSERT(false, "Destructor should not assert if already throwing"); }
  };

  EXPECT_THROW(
      {
        Test t;
        throw std::logic_error("Test exception");
      },
      std::logic_error);
#else
  GTEST_SKIP() << "DYNAMPI_ASSERT in destructor is a no-op in NDEBUG builds.";
#endif
}

TEST(DynaMPIAssert, FalseConditionThrowsAndPrints) {
#ifndef NDEBUG
  int a1 = 1, b2 = 2;
  auto [msg, threw] = CaptureStderrAndDidThrow([=] { DYNAMPI_ASSERT(a1 == b2, "custom message"); });
  EXPECT_TRUE(threw);
  EXPECT_NE(msg.find("DynaMPI assertion failed"), std::string::npos);
  EXPECT_NE(msg.find("a1 == b2"), std::string::npos);        // condition text
  EXPECT_NE(msg.find("custom message"), std::string::npos);  // user message
  EXPECT_NE(msg.find("rank "), std::string::npos);           // our stub sets rank 0
#else
  GTEST_SKIP() << "DYNAMPI_ASSERT is a no-op in NDEBUG builds.";
#endif
}

// ---------- Binary-op helpers ----------
TEST(DynaMPIAssertBinOp, EqFailureShowsValuesAndNegatedOp) {
#ifndef NDEBUG
  int a1 = 1, b2 = 2;
  auto [msg, threw] = CaptureStderrAndDidThrow([=] { DYNAMPI_ASSERT_EQ(a1, b2, "boom"); });
  EXPECT_TRUE(threw);
  EXPECT_NE(msg.find("1 != 2"), std::string::npos);  // comes from _DYNAMPI_FAILBinOp
  EXPECT_NE(msg.find("boom"), std::string::npos);
  EXPECT_NE(msg.find("DynaMPI assertion failed"), std::string::npos);
#else
  GTEST_SKIP() << "DYNAMPI_ASSERT_* are no-ops in NDEBUG builds.";
#endif
}

TEST(DynaMPIAssertBinOp, GeAndLtPassWithoutThrow) {
#ifndef NDEBUG
  int a5 = 5, b5 = 5, c4 = 4;
  EXPECT_NO_THROW({ DYNAMPI_ASSERT_GE(a5, b5, "ok"); });
  EXPECT_NO_THROW({ DYNAMPI_ASSERT_LT(c4, a5, "ok"); });
#else
  EXPECT_NO_THROW({ DYNAMPI_ASSERT_GE(c4, a5, "no-op"); });
  EXPECT_NO_THROW({ DYNAMPI_ASSERT_LT(can write, anything here); });
#endif
}
