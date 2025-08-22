/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <mpi.h>

#include <exception>
#ifndef _MSC_VER
#define DYNAMPI_HAS_BUILTIN(x) __has_builtin(x)
#else
#define DYNAMPI_HAS_BUILTIN(x) 0
#endif

#ifndef NDEBUG
#include <iostream>
#include <optional>

#if __has_include(<source_location>)
#include <source_location>
#if defined(__cpp_lib_source_location)
#define DYNAMPI_HAS_SOURCE_LOCATION
#endif
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
namespace std {
using source_location = std::experimental::source_location;
}
#define DYNAMPI_HAS_SOURCE_LOCATION
#endif

#include <sstream>
#include <string>

#include "printing.hpp"
#endif

namespace dynampi {

#ifndef NDEBUG
template <typename... Args>
std::optional<std::string> OptionalString(Args &&...args) {
  if constexpr (sizeof...(args) == 0) {
    return std::nullopt;
  } else {
    std::stringstream ss;
    (ss << ... << args);
    return ss.str();
  }
}

#define DYNAMPI_ASSERT(condition, ...)                                                 \
  do {                                                                                 \
    if (!(condition))                                                                  \
      dynampi::_DYNAMPI_FAIL_ASSERT(#condition, dynampi::OptionalString(__VA_ARGS__)); \
  } while (false)

inline void _DYNAMPI_FAIL_ASSERT(const std::string &condition_str,
                                 const std::optional<std::string> &message
#ifdef DYNAMPI_HAS_SOURCE_LOCATION
                                 ,
                                 const std::source_location &loc = std::source_location::current()
#endif
) {
  if (!std::uncaught_exceptions()) {
    std::stringstream ss;
#ifdef DYNAMPI_HAS_SOURCE_LOCATION
    std::string_view s = loc.file_name();
    std::string_view filename = s.substr(s.find_last_of('/') + 1);
#endif
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    ss << "DynaMPI assertion failed on rank " << rank << ": " << condition_str
       << (message ? " " + *message : "")

#ifdef DYNAMPI_HAS_SOURCE_LOCATION
       << "\n in " << loc.function_name() << " at " << filename << ":" << loc.line()
#endif
       << std::endl;
    std::cerr << ss.str();
    throw std::runtime_error(ss.str());
  }
}

#define DYNAMPI_ASSERT_BIN_OP(a, b, op, nop, ...)                        \
  do {                                                                   \
    const auto A = a;                                                    \
    const auto B = b;                                                    \
    if (!((A)op(B)))                                                     \
      dynampi::_DYNAMPI_FAILBinOp((A), (B), (#a), (#b), (#nop),          \
                                  dynampi::OptionalString(__VA_ARGS__)); \
  } while (false)

template <typename A, typename B>
inline void _DYNAMPI_FAILBinOp(const A &a, const B &b, const std::string &a_str,
                               const std::string &b_str, const std::string &nop,
                               const std::optional<std::string> &message
#ifdef DYNAMPI_HAS_SOURCE_LOCATION
                               ,
                               const std::source_location &loc = std::source_location::current()
#endif
) {
  std::stringstream ss;
  ss << a << " " << nop << " " << b;
  dynampi::_DYNAMPI_FAIL_ASSERT(a_str + " " + nop + " " + b_str,
                                message ? (ss.str() + " " + *message) : ss.str()
#ifdef DYNAMPI_HAS_SOURCE_LOCATION
                                    ,
                                loc
#endif
  );
}

#else
#define DYNAMPI_ASSERT(condition, ...) \
  do {                                 \
  } while (false)
#define DYNAMPI_ASSERT_BIN_OP(a, b, op, nop, ...) \
  do {                                            \
  } while (false)
#endif

#define DYNAMPI_FAIL(...)             \
  DYNAMPI_ASSERT(false, __VA_ARGS__); \
  UNREACHABLE()  // LCOV_EXCL_LINE

#define DYNAMPI_UNIMPLEMENTED(...) DYNAMPI_FAIL("DYNAMPI_UNIMPLEMENTED")

#define DYNAMPI_ASSERT_GE(expr, val, ...) DYNAMPI_ASSERT_BIN_OP(expr, val, >=, <, __VA_ARGS__)
#define DYNAMPI_ASSERT_LE(expr, val, ...) DYNAMPI_ASSERT_BIN_OP(expr, val, <=, >, __VA_ARGS__)
#define DYNAMPI_ASSERT_GT(expr, val, ...) DYNAMPI_ASSERT_BIN_OP(expr, val, >, <=, __VA_ARGS__)
#define DYNAMPI_ASSERT_LT(expr, val, ...) DYNAMPI_ASSERT_BIN_OP(expr, val, <, >=, __VA_ARGS__)
#define DYNAMPI_ASSERT_EQ(expr, val, ...) DYNAMPI_ASSERT_BIN_OP(expr, val, ==, !=, __VA_ARGS__)
#define DYNAMPI_ASSERT_NE(expr, val, ...) DYNAMPI_ASSERT_BIN_OP(expr, val, !=, ==, __VA_ARGS__)

#if defined(_MSC_VER) && !defined(__clang__)  // MSVC
#define UNREACHABLE() __assume(false)
#else  // GCC, Clang
#define UNREACHABLE() __builtin_unreachable()
#endif

}  // namespace dynampi
