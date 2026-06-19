/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <dynampi/utilities/printing.hpp>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <tuple>
#include <vector>

using namespace dynampi;

namespace {
template <typename T>
std::string to_str(const T& v) {
  std::ostringstream oss;
  oss << v;
  return oss.str();
}

}  // namespace

TEST(ByteOstreamTest, PrintsAsUnsignedInteger) {
  std::byte b1{static_cast<unsigned char>(0)};
  std::byte b2{static_cast<unsigned char>(255)};
  std::byte b3{static_cast<unsigned char>(42)};

  EXPECT_EQ(to_str(b1), "0");
  EXPECT_EQ(to_str(b2), "255");
  EXPECT_EQ(to_str(b3), "42");
}

TEST(SpanOstreamTest, EmptySpanPrintsBrackets) {
  std::vector<int> v;
  std::span<const int> sp(v.data(), v.size());
  EXPECT_EQ(to_str(sp), "[]");
}

TEST(SpanOstreamTest, PrintsCommaSeparatedWithBrackets) {
  int data[] = {1, 2, 3};
  std::span<const int> sp(data, 3);
  EXPECT_EQ(to_str(sp), "[1, 2, 3]");
}

TEST(VectorOstreamTest, DelegatesToSpanFormat) {
  std::vector<int> v{4, 5, 6};
  EXPECT_EQ(to_str(v), "[4, 5, 6]");
}

TEST(ArrayOstreamTest, PrintsCommaSeparatedWithBrackets) {
  std::array<int, 3> a{7, 8, 9};
  EXPECT_EQ(to_str(a), "[7, 8, 9]");
}

TEST(SetOstreamTest, PrintsInAscendingOrderWithBraces) {
  std::set<int> s{5, 1, 3};
  EXPECT_EQ(to_str(s), "{1, 3, 5}");
}

TEST(PairOstreamTest, PrintsInParensWithCommaSpace) {
  std::pair<int, std::string> p{10, "foo"};
  EXPECT_EQ(to_str(p), "(10, foo)");
}

TEST(OptionalOstreamTest, PrintsSome) {
  std::optional<int> o = 123;
  EXPECT_EQ(to_str(o), "Some(123)");
}

TEST(OptionalOstreamTest, PrintsNone) {
  std::optional<int> o;
  EXPECT_EQ(to_str(o), "None");
}

TEST(NestedContainersTest, WorksRecursively) {
  std::vector<std::set<int>> v{{2, 1}, {4, 3}};
  // vector prints as span -> each set prints with braces
  EXPECT_EQ(to_str(v), "[{1, 2}, {3, 4}]");
}

TEST(SpanOfBytesTest, PrintsNumericBytes) {
  std::array<std::byte, 4> bytes{std::byte{0x00}, std::byte{0x0A}, std::byte{0x7F},
                                 std::byte{0xFF}};
  std::span<const std::byte> sp(bytes.data(), bytes.size());
  EXPECT_EQ(to_str(sp), "[0, 10, 127, 255]");
}

TEST(TupleOstreamTest, PrintsCommaSeparatedInParensNoTrailingComma) {
  auto t = std::make_tuple(1, std::string("x"), 3);
  EXPECT_EQ(to_str(t), "(1, x, 3)");
}
