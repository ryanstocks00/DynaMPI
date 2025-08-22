/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <vector>

namespace dynampi {

inline std::ostream& operator<<(std::ostream& os, const std::byte& b) {
  return os << static_cast<uint32_t>(b);
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::span<T>& vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    os << vec[i];
    if (i < vec.size() - 1) {
      os << ", ";
    }
  }
  return os << "]";
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
  return os << std::span<const T>(vec);
}

template <typename T, size_t N>
inline std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
  os << "[";
  for (size_t i = 0; i < arr.size(); i++) {
    os << arr[i];
    if (i < arr.size() - 1) {
      os << ", ";
    }
  }
  return os << "]";
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::set<T>& set) {
  os << "{";
  for (const T& elem : set) {
    os << elem << ", ";
  }
  return os << "}";
}

template <typename T, typename U>
inline std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& pair) {
  return os << "(" << pair.first << ", " << pair.second << ")";
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::optional<T>& op) {
  if (op.has_value()) {
    return os << "Some(" << op.value() << ")";
  }
  return os << "None";
}

template <typename... Args>
inline std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& tup) {
  std::apply([&os](const Args&... args) { ((os << args << ", "), ...); }, tup);
  return os;
}
}  // namespace dynampi
