/*
 * SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
 * SPDX-License-Identifier: MIT
 */

#include "dynampi.h"

#include <cstdlib>
#include <cstring>
#include <dynampi/dynampi.hpp>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

class CDynamicDistributor
    : public dynampi::NaiveMPIWorkDistributor<std::vector<std::byte>, std::vector<std::byte>> {
 public:
  using Base = dynampi::NaiveMPIWorkDistributor<std::vector<std::byte>, std::vector<std::byte>>;
  using WorkerFn = std::function<std::vector<std::byte>(std::vector<std::byte>)>;
  explicit CDynamicDistributor(WorkerFn worker, const dynampi_config_t& cfg)
      : Base(worker,
             {.comm = cfg.comm, .manager_rank = cfg.manager_rank, .auto_run_workers = false}),
        config_(cfg) {}

  dynampi_config_t config_;
};

static std::vector<std::byte> make_bytes(const unsigned char* data, size_t size) {
  std::vector<std::byte> v;
  v.resize(size);
  if (size && data) std::memcpy(v.data(), data, size);
  return v;
}

namespace {

// Adapts a C worker function to a C++ byte-vector based worker
inline CDynamicDistributor::WorkerFn make_worker_adapter(dynampi_worker_t worker_function) {
  return [worker_function](std::vector<std::byte> in) -> std::vector<std::byte> {
    unsigned char* out_data = nullptr;
    size_t out_size = 0;
    const unsigned char* in_data = reinterpret_cast<const unsigned char*>(in.data());
    worker_function(in_data, in.size(), &out_data, &out_size);
    std::vector<std::byte> out;
    if (out_data && out_size) {
      out.resize(out_size);
      std::memcpy(out.data(), out_data, out_size);
      std::free(out_data);
    }
    return out;
  };
}

// Copies a vector of byte buffers into a C ABI dynampi_buffer_t array allocated via malloc
inline dynampi_buffer_t* copy_results_out(const std::vector<std::vector<std::byte>>& results,
                                          size_t* result_count) {
  if (!result_count) return nullptr;
  *result_count = results.size();
  if (results.empty()) return nullptr;

  dynampi_buffer_t* out =
      static_cast<dynampi_buffer_t*>(std::malloc(results.size() * sizeof(dynampi_buffer_t)));
  if (!out) {
    *result_count = 0;
    return nullptr;
  }
  for (size_t i = 0; i < results.size(); ++i) {
    const auto& v = results[i];
    out[i].size = v.size();
    out[i].data = static_cast<unsigned char*>(v.size() ? std::malloc(v.size()) : nullptr);
    if (v.size() && out[i].data) std::memcpy(out[i].data, v.data(), v.size());
  }
  return out;
}

// Inserts a task consisting of a little-endian encoded size_t index
inline void insert_task_index(CDynamicDistributor& distributor, size_t index_value) {
  unsigned char buf[sizeof(size_t)];
  std::memcpy(buf, &index_value, sizeof(size_t));
  distributor.insert_task(make_bytes(buf, sizeof(size_t)));
}

}  // namespace

// C interface implementations
extern "C" {

dynampi_work_distributor_t* dynampi_create_work_distributor(dynampi_worker_t worker_function,
                                                            const dynampi_config_t* config) {
  if (!worker_function) return nullptr;
  dynampi_config_t cfg = config ? *config : dynampi_default_config();
  CDynamicDistributor::WorkerFn worker = make_worker_adapter(worker_function);
  auto* dist = new CDynamicDistributor(worker, cfg);
  return reinterpret_cast<dynampi_work_distributor_t*>(dist);
}

void dynampi_destroy_work_distributor(dynampi_work_distributor_t* distributor) {
  if (distributor) delete reinterpret_cast<CDynamicDistributor*>(distributor);
}

int dynampi_is_manager(const dynampi_work_distributor_t* distributor) {
  if (!distributor) return 0;
  return reinterpret_cast<const CDynamicDistributor*>(distributor)->is_manager() ? 1 : 0;
}

void dynampi_run_worker(dynampi_work_distributor_t* distributor) {
  if (!distributor) return;
  reinterpret_cast<CDynamicDistributor*>(distributor)->run_worker();
}

void dynampi_insert_task(dynampi_work_distributor_t* distributor, const unsigned char* data,
                         size_t size) {
  if (!distributor) return;
  auto* d = reinterpret_cast<CDynamicDistributor*>(distributor);
  d->insert_task(make_bytes(data, size));
}

dynampi_buffer_t* dynampi_finish_remaining_tasks(dynampi_work_distributor_t* distributor,
                                                 size_t* result_count) {
  if (!distributor || !result_count) return nullptr;
  auto* d = reinterpret_cast<CDynamicDistributor*>(distributor);
  if (!d->is_manager()) {
    *result_count = 0;
    return nullptr;
  }
  auto results = d->Base::finish_remaining_tasks();
  return copy_results_out(results, result_count);
}

const char* dynampi_version_string(void) {
  static std::string v = std::string(dynampi::version::string);
  return v.c_str();
}
int dynampi_version_major(void) { return dynampi::version::major; }
int dynampi_version_minor(void) { return dynampi::version::minor; }
int dynampi_version_patch(void) { return dynampi::version::patch; }
const char* dynampi_commit_hash(void) {
  static std::string h = std::string(dynampi::version::commit_hash());
  return h.c_str();
}
const char* dynampi_compile_date(void) {
  static std::string d = std::string(dynampi::version::compile_date());
  return d.c_str();
}

dynampi_config_t dynampi_default_config(void) {
  dynampi_config_t config;
  config.comm = MPI_COMM_WORLD;
  config.manager_rank = 0;
  config.auto_run_workers = 1;
  return config;
}

int dynampi_manager_worker_distribution(size_t n_tasks, dynampi_worker_t worker_function,
                                        dynampi_buffer_t** results, size_t* result_count,
                                        MPI_Comm comm, int manager_rank) {
  if (!worker_function || !results || !result_count) return -1;
  dynampi_config_t cfg;
  cfg.comm = comm;
  cfg.manager_rank = manager_rank;
  cfg.auto_run_workers = 0;
  // Build a bytes-based distributor whose worker adapts the C worker
  CDynamicDistributor::WorkerFn wf = make_worker_adapter(worker_function);
  auto* dist = new CDynamicDistributor(wf, cfg);
  if (dist->is_manager()) {
    for (size_t i = 0; i < n_tasks; ++i) insert_task_index(*dist, i);
    auto results_vec = dist->Base::finish_remaining_tasks();
    *results = copy_results_out(results_vec, result_count);
    delete dist;
    return *results || results_vec.empty() ? 0 : -1;
  } else {
    dist->run_worker();
    delete dist;
    *results = nullptr;
    *result_count = 0;
    return 0;
  }
}

}  // extern "C"
