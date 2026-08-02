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

// How many elements of MPI_Type<T>::value make up one whole T.
//
// 1 for the primitives, whose datatype already covers the object. For a
// fixed-size aggregate described by its scalar element type -- e.g. a struct of
// three doubles declaring MPI_DOUBLE -- it is 3, and anything that sizes a
// buffer "per value" has to account for that: the fixed-width slots in the RMA
// window tables, and the element count of a std::vector<T> batch message.
//
// Derived from sizeof(T) rather than from count(), so it needs no instance and
// is a stable property of the type. Genuinely variable-length types have no
// such constant and report 1, since they size their buffers from count()
// instead. Queried once per T and cached; MPI_Type_size needs MPI_Init, which
// has necessarily run by the time any value is being moved.
template <typename T>
inline int mpi_elements_per_value() {
  if constexpr (MPI_Type<T>::resize_required) {
    return 1;
  } else {
    static const int elements = [] {
      int element_bytes = 0;
      if (MPI_Type_size(MPI_Type<T>::value, &element_bytes) != MPI_SUCCESS || element_bytes <= 0) {
        return 1;
      }
      const size_t per_value = sizeof(T) / static_cast<size_t>(element_bytes);
      return per_value > 0 ? static_cast<int>(per_value) : 1;
    }();
    return elements;
  }
}

// std::vector<T> specialization (contiguous storage). Excludes vector<bool>.
template <typename T>
struct MPI_Type<std::vector<T>, std::enable_if_t<has_dynampi_mpi_type<T>::value>> {
  inline static const MPI_Datatype value = MPI_Type<T>::value;
  inline static const bool resize_required = true;

  // Counts are in elements of `value`, not in T, so a T spanning several
  // elements (a fixed-size aggregate) contributes all of them -- otherwise a
  // batch message would carry only the first element of each value. count()
  // and resize() are inverses in those units, which is exactly how every call
  // site pairs them: MPI_Get_count -> resize -> recv.
  static int count(const std::vector<T>& vec) {
    // Traditional MPI calls take 'int' counts; very large vectors require MPI-4 large-count APIs.
    // Caller responsibility if vec.size() exceeds INT_MAX.
    return static_cast<int>(vec.size()) * mpi_elements_per_value<T>();
  }
  static void resize(std::vector<T>& vec, int new_size) {
    vec.resize(static_cast<size_t>(new_size) / static_cast<size_t>(mpi_elements_per_value<T>()));
  }
  static void* ptr(std::vector<T>& vec) noexcept { return vec.data(); }
  static const void* ptr(const std::vector<T>& vec) noexcept { return vec.data(); }

  static_assert(!std::is_same_v<bool, T>,
                "dynampi::MPI_Type<std::vector<bool>> is not supported: "
                "std::vector<bool> is bit-packed and not contiguous. "
                "Use std::vector<unsigned char> or a custom container.");
};

// Verifies what a fixed-size (resize_required == false) payload must satisfy
// for mpi_elements_per_value<T>() to describe it: the object has to be a whole
// number of datatype elements, so that per-value buffer sizing is exact.
//
// A struct of three doubles declaring MPI_DOUBLE satisfies this (24 = 3 x 8); a
// struct of a double and an int declaring MPI_DOUBLE does not, and there is no
// element count that would let a fixed-width slot or a batch message describe
// it. Such a type needs a datatype that really covers it (MPI_Type_create_struct
// with count() == 1) or MPI_BYTE with count() == sizeof(T).
//
// Throws rather than asserts: a mismatch here moves truncated data instead of
// failing any existing size check, and NDEBUG builds are exactly where that
// does the most damage.
template <typename T>
inline void check_fixed_size_mpi_type(const char* type_role, const char* distributor_name) {
  if constexpr (!MPI_Type<T>::resize_required) {
    int element_bytes = 0;
    // Check NULL before MPI_Type_size: with the default MPI_ERRORS_ARE_FATAL
    // handler many implementations abort on an invalid datatype rather than
    // returning an error code, so a C++ throw would never be reachable.
    if (MPI_Type<T>::value == MPI_DATATYPE_NULL ||
        MPI_Type_size(MPI_Type<T>::value, &element_bytes) != MPI_SUCCESS || element_bytes <= 0) {
      throw std::invalid_argument(std::string(distributor_name) + ": could not query the MPI " +
                                  "datatype of its " + type_role + " type");
    }
    if (sizeof(T) % static_cast<size_t>(element_bytes) != 0) {
      throw std::invalid_argument(
          std::string(distributor_name) + ": the " + type_role + " type is " +
          std::to_string(sizeof(T)) +
          " bytes, which is not a whole number of its MPI datatype's elements (" +
          std::to_string(element_bytes) +
          " bytes each). Use a datatype that covers the whole object with count() == 1, or "
          "MPI_BYTE "
          "with count() == sizeof(T).");
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
