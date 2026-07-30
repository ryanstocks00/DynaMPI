/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string_view>
#include <tuple>

// Version macros are normally supplied by the `dynampi` CMake target. The
// fallbacks below keep every header usable with a plain `-I include` build (no
// CMake, no -D flags). The top-level CMakeLists.txt parses these three #define
// lines to derive project(DynaMPI VERSION ...), so the fallbacks are the single
// source of truth and cannot drift from the CMake version -- keep the literal
// `#define DYNAMPI_VERSION_<part> <integer>` shape when bumping them.
#ifndef DYNAMPI_VERSION_MAJOR
#define DYNAMPI_VERSION_MAJOR 0
#endif
#ifndef DYNAMPI_VERSION_MINOR
#define DYNAMPI_VERSION_MINOR 0
#endif
#ifndef DYNAMPI_VERSION_PATCH
#define DYNAMPI_VERSION_PATCH 1
#endif

// Set by CMake from `git rev-parse HEAD` (with a `-dirty` suffix when the tree
// has uncommitted changes). Builds without CMake have no way to know it.
#ifndef DYNAMPI_COMMIT_HASH
#define DYNAMPI_COMMIT_HASH "unknown"
#endif

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

}  // namespace dynampi
