/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>
#include <set>
#include <span>
#include <tuple>
#include <vector>

namespace dynampi {

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::set<T>& set);
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec);
template <typename T, std::size_t N>
inline std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr);
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::span<T>& vec);
template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::optional<T>& op);
template <typename... Args>
inline std::ostream& operator<<(std::ostream& os, const std::tuple<Args...>& tup);
template <typename T, typename U>
inline std::ostream& operator<<(std::ostream& os, const std::pair<T, U>& pair);
inline std::ostream& operator<<(std::ostream& os, const std::byte& b);

// --------------- IMPLEMENTATIONS ---------------

inline std::ostream& operator<<(std::ostream& os, const std::byte& b) {
  return os << static_cast<uint32_t>(b);
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::span<T>& vec) {
  os << "[";
  for (std::size_t i = 0; i < vec.size(); i++) {
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

template <typename T, std::size_t N>
inline std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
  os << "[";
  for (std::size_t i = 0; i < arr.size(); i++) {
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
  auto it = set.begin();
  while (it != set.end()) {
    os << *it;
    ++it;
    if (it != set.end()) {
      os << ", ";
    }
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
  os << "(";
  std::apply(
      [&os](const Args&... args) {
        std::size_t i = 0;
        ((os << args << (++i < sizeof...(Args) ? ", " : "")), ...);
      },
      tup);
  return os << ")";
}

}  // namespace dynampi
