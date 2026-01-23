/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

namespace dynampi {

template <typename T, typename = void>
struct MPI_Type {
  static_assert(sizeof(T) == 0,
                "dynampi::MPI_Type<T> is not defined for this T. "
                "Provide a specialization or use a supported primitive.");
};

#define DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(type, mpi_type)         \
  template <>                                                     \
  struct MPI_Type<type, void> {                                   \
    inline static const MPI_Datatype value = mpi_type;            \
    inline static const bool resize_required = false;             \
    static int count(const type&) noexcept { return 1; }          \
    static void resize(type&, int new_size) noexcept {            \
      (void)new_size;                                             \
      assert(new_size == 1);                                      \
    }                                                             \
    static void* ptr(type& t) noexcept { return &t; }             \
    static const void* ptr(const type& t) noexcept { return &t; } \
  }

// Primitives
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(char, MPI_CHAR);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(std::byte, MPI_BYTE);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(signed char, MPI_SIGNED_CHAR);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(unsigned char, MPI_UNSIGNED_CHAR);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(short, MPI_SHORT);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(unsigned short, MPI_UNSIGNED_SHORT);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(int, MPI_INT);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(unsigned int, MPI_UNSIGNED);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(long, MPI_LONG);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(unsigned long, MPI_UNSIGNED_LONG);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(long long, MPI_LONG_LONG_INT);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(unsigned long long, MPI_UNSIGNED_LONG_LONG);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(float, MPI_FLOAT);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(double, MPI_DOUBLE);
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(long double, MPI_LONG_DOUBLE);
#if defined(MPI_CXX_BOOL)
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(bool, MPI_CXX_BOOL);
#else
// Fallback for when MPI_CXX_BOOL is not available (e.g. Microsoft-MPI)
DYNAMPI_DEFINE_PRIMITIVE_MPI_TYPE(bool, MPI_C_BOOL);
#endif

template <>
struct MPI_Type<std::nullptr_t> {
  inline static const MPI_Datatype value = MPI_PACKED;
  inline static const bool resize_required = false;

  static int count(const std::nullptr_t&) noexcept { return 0; }
  static void resize(std::nullptr_t&, int new_size) noexcept {
    (void)new_size;  // No-op, nullptr cannot be resized
  }
  static void* ptr(std::nullptr_t&) noexcept { return nullptr; }
  static const void* ptr(const std::nullptr_t&) noexcept { return nullptr; }
};

// Helper trait: is there a dynampi::MPI_Type<U> specialization?
template <typename, typename = void>
struct has_dynampi_mpi_type : std::false_type {};
template <typename U>
struct has_dynampi_mpi_type<U, std::void_t<decltype(MPI_Type<U>::value)>> : std::true_type {};

// std::vector<T> specialization (contiguous storage). Excludes vector<bool>.
template <typename T>
struct MPI_Type<std::vector<T>, std::enable_if_t<has_dynampi_mpi_type<T>::value>> {
  inline static const MPI_Datatype value = MPI_Type<T>::value;
  inline static const bool resize_required = true;

  static int count(const std::vector<T>& vec) {
    // Traditional MPI calls take 'int' counts; very large vectors require MPI-4 large-count APIs.
    // Caller responsibility if vec.size() exceeds INT_MAX.
    return static_cast<int>(vec.size());
  }
  static void resize(std::vector<T>& vec, int new_size) {
    vec.resize(static_cast<size_t>(new_size));
  }
  static void* ptr(std::vector<T>& vec) noexcept { return vec.data(); }
  static const void* ptr(const std::vector<T>& vec) noexcept { return vec.data(); }

  static_assert(!std::is_same_v<bool, T>,
                "dynampi::MPI_Type<std::vector<bool>> is not supported: "
                "std::vector<bool> is bit-packed and not contiguous. "
                "Use std::vector<unsigned char> or a custom container.");
};

// std::string specialization
template <>
struct MPI_Type<std::string> {
  inline static const MPI_Datatype value = MPI_CHAR;
  inline static const bool resize_required = true;

  static int count(const std::string& str) { return static_cast<int>(str.size()); }
  static void resize(std::string& str, int new_size) { str.resize(static_cast<size_t>(new_size)); }
  static void* ptr(std::string& str) noexcept { return str.data(); }
  static const void* ptr(const std::string& str) noexcept { return str.data(); }
};

}  // namespace dynampi
