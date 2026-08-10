/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 *
 * Sending your own struct as the task and result payload, by specialising
 * dynampi::MPI_Type. The result here also carries its task index, which is how
 * you recover task identity from an unordered distributor.
 *
 *   mpirun -n 4 ./04_custom_task_type
 */

#include <cmath>
#include <cstddef>
#include <dynampi/impl/lockfree_rma_distributor.hpp>
#include <iostream>
#include <vector>

// Plain aggregates of same-typed scalars, laid out contiguously.
struct Ray {
  double origin;
  double direction;
  double length;
};

struct Hit {
  double index;     // which task produced this
  double distance;  // the actual answer
  double energy;
};

static_assert(sizeof(Ray) == 3 * sizeof(double), "Ray must be contiguous doubles");
static_assert(sizeof(Hit) == 3 * sizeof(double), "Hit must be contiguous doubles");

// DynaMPI moves a value as `count()` elements of `value`, read through `ptr()`.
// One Ray is three MPI_DOUBLE elements; every distributor accounts for a value
// spanning several elements, so nothing else is needed here.
//
// Two requirements: count() * MPI_Type_size(value) must equal sizeof(T), and
// the storage at ptr() must be contiguous -- hence the static_asserts above.
// See docs/src/api.md#custom-types.
template <>
struct dynampi::MPI_Type<Ray> {
  inline static const MPI_Datatype value = MPI_DOUBLE;
  inline static const bool resize_required = false;  // fixed size

  static int count(const Ray&) noexcept { return 3; }
  static void resize(Ray&, int) noexcept {}
  static void* ptr(Ray& r) noexcept { return &r; }
  static const void* ptr(const Ray& r) noexcept { return &r; }
};

template <>
struct dynampi::MPI_Type<Hit> {
  inline static const MPI_Datatype value = MPI_DOUBLE;
  inline static const bool resize_required = false;  // fixed size

  static int count(const Hit&) noexcept { return 3; }
  static void resize(Hit&, int) noexcept {}
  static void* ptr(Hit& h) noexcept { return &h; }
  static const void* ptr(const Hit& h) noexcept { return &h; }
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    // Custom structs work with every distributor; this one happens to use the
    // flat lock-free RMA distributor.
    using Distributor = dynampi::LockFreeRMAWorkDistributor<Ray, Hit>;

    // The index rides along in the task so the worker can stamp it on the
    // result; this distributor is unordered, so nothing else recovers it.
    auto trace = [](Ray ray) -> Hit {
      return Hit{ray.origin, ray.direction * ray.length, std::sqrt(ray.length)};
    };

    Distributor dist(trace);
    if (dist.is_root_manager()) {
      constexpr size_t kNumRays = 32;
      std::vector<Ray> rays;
      for (size_t i = 0; i < kNumRays; ++i) {
        rays.push_back(Ray{static_cast<double>(i), 2.0, static_cast<double>(i) + 1.0});
      }
      dist.insert_tasks(rays);

      std::vector<double> distance_by_index(kNumRays, -1.0);
      for (const Hit& hit : dist.finish_remaining_tasks()) {
        distance_by_index[static_cast<size_t>(hit.index)] = hit.distance;
      }

      std::cout << "ray 7 -> distance " << distance_by_index[7] << " (expect 16)\n";
    }
  }
  MPI_Finalize();
  return 0;
}
