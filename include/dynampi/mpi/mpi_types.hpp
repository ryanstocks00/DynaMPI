/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <stdexcept>
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

// Verifies the contract that `resize_required == false` implies: exactly ONE
// element of `value` covers the whole object, i.e. count() == 1 and
// MPI_Type_size(value) == sizeof(T).
//
// Distributors that reserve fixed-width RMA window slots, or that batch values
// into a single std::vector<T> message, size those buffers as one element per
// value and so depend on this. A type that violates it -- e.g. a struct of
// three doubles declaring MPI_DOUBLE with count() == 3 and
// resize_required == false -- silently truncates or overruns instead of
// failing any existing size check, which is why this throws unconditionally
// rather than asserting: NDEBUG builds are exactly where silent corruption
// does the most damage.
//
// Such a struct should declare resize_required = true with a no-op resize(),
// which asks the distributor to size its buffers from count() instead.
// MPI_Type<T> specializations for genuinely variable-length types (and
// distributors like NaiveMPIWorkDistributor, which sends count() elements
// directly and needs no fixed-width slot) are unaffected.
template <typename T>
inline void check_fixed_size_mpi_type(const char* type_role, const char* distributor_name) {
  if constexpr (!MPI_Type<T>::resize_required) {
    int element_bytes = 0;
    if (MPI_Type_size(MPI_Type<T>::value, &element_bytes) != MPI_SUCCESS) {
      throw std::invalid_argument(std::string(distributor_name) + ": could not query the MPI " +
                                  "datatype of its " + type_role + " type");
    }
    if (static_cast<size_t>(element_bytes) != sizeof(T)) {
      throw std::invalid_argument(
          std::string(distributor_name) + ": the " + type_role +
          " type's MPI_Type specialization declares resize_required == false, which requires one "
          "element of its datatype to cover the whole object (" +
          std::to_string(sizeof(T)) + " bytes), but one element is " +
          std::to_string(element_bytes) +
          " bytes. For a fixed-size struct spanning several elements, declare "
          "resize_required = true with a no-op resize() instead.");
    }
  }
}

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
