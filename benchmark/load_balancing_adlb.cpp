/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

// The ADLB arm of the load_balancing sweep: the same batch drain, against
// ADLB/X (Swift/T's `lb`). Its own binary because ADLB owns its ranks for the
// whole process and cannot be re-initialised; workload and CSV schema are
// shared with load_balancing.cpp via load_balancing_workload.hpp.

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cxxopts.hpp>
#include <dynampi/utilities/timer.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "load_balancing_workload.hpp"

// Last, and fenced: adlb-defs.h hides __cplusplus across its own <mpi.h>
// include and uses constructs that are extensions in C++. Quarantined here
// rather than loosening the target's warning flags.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wbuiltin-macro-redefined"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wreserved-user-defined-literal"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wbuiltin-macro-redefined"
#endif
extern "C" {
#include <adlb.h>
}
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

// lb fixes its steal tuning at compile time with no environment override, but
// leaves the values writable. Setting them before any steal happens is
// equivalent to rebuilding with different constants.
extern "C" {
extern double xlb_steal_rate_limit;
extern double xlb_steal_backoff;
}

using load_balancing::DurationMode;
using load_balancing::Result;
using load_balancing::ResultRow;
using load_balancing::Task;
using load_balancing::WorkerFunctor;

namespace {

// ADLB matches Gets to Puts by type: work out to the pool, results back to
// the rank that published the batch.
constexpr int kWorkType = 0;
constexpr int kResultType = 1;
// ADLB rank 0, the counterpart of the DynaMPI distributors' manager.
constexpr int kMasterRank = 0;
// End-of-batch marker, sent as an ordinary work task because ADLB_Get takes
// one type and a worker parked in it cannot listen on a second. Task ids run
// to a few hundred thousand over a whole sweep, so the top of the range is
// free; publish_tasks() asserts they never meet.
constexpr Task kBatchDone = 0xFFFFFFFFu;

struct BenchmarkOptions {
  uint64_t expected_us = 1000;
  DurationMode duration_mode = DurationMode::Fixed;
  double task_duration_cv = 1.0;
  uint64_t min_tasks_per_worker = 1;
  uint64_t max_tasks_per_worker = 10;
  uint64_t repeats = 3;
  uint64_t nodes = 0;
  // Server ranks; 0 means one per node. Must divide the node count.
  int servers = 0;
  // Who publishes: 1 = the master alone, as the DynaMPI drivers' manager
  // does; -1 = every worker publishes its own tasks_per_worker share.
  int producers = 1;
  // Results kept requested at once (ADLB_Amget).
  int result_window = 128;
  // --producers 1 only: round-robin the master's puts over every server.
  // Off by default: measured worse than lb's own untargeted routing at every
  // server count above 1, because it trades node-local puts for remote ones.
  bool spread_puts = false;
  // master: results targeted at the master, so its server collects them all.
  // none: no results at all; completion is counted through a one-sided MPI
  // counter instead, which bounds what removing the return path could buy.
  std::string result_routing = "master";
  // Seconds between a starved server's steal attempts. <= 0 keeps lb's own
  // shipped value (0.05 s, with a 0.1 s backoff after a full round fails).
  double steal_rate_limit = 0.0;
  std::string system;
  std::string output_path;
};

void check(adlb_code code, const char* what) {
  if (code != ADLB_SUCCESS) {
    throw std::runtime_error(std::string("ADLB call failed (") + what + "): code " +
                             std::to_string(static_cast<int>(code)));
  }
}

struct AdlbPlacement {
  MPI_Comm comm = MPI_COMM_NULL;  // MPI_COMM_NULL on a rank ADLB never sees
  uint64_t nodes = 0;
  uint64_t executing_workers = 0;  // every worker but the master
  int servers = 0;
};

// Reorders MPI_COMM_WORLD into the communicator ADLB is initialised on.
// ADLB's layout is fixed -- workers are ranks [0, workers), servers trail
// them, and worker r maps to server r / (workers / servers) -- so ordering
// workers node-major and appending one server per block puts each server in
// the block it serves, and a worker's ADLB_RANK_ANY put reaches its own node.
//
// One rank per node is held out whatever --servers is (the last local rank,
// as load_balancing.cpp's flat_comm drops), fixing the task-executing count
// at world_size - nodes - 1 -- what the hierarchical distributors report, so
// every series drains the same batch. Held-out non-servers sit the run out.
AdlbPlacement build_adlb_comm(MPI_Comm world, int requested_servers) {
  int world_rank = 0, world_size = 0;
  MPI_Comm_rank(world, &world_rank);
  MPI_Comm_size(world, &world_size);

  MPI_Comm node_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
  int local_rank = 0, local_size = 0;
  MPI_Comm_rank(node_comm, &local_rank);
  MPI_Comm_size(node_comm, &local_size);

  // Node index via the leaders' communicator, so this holds under any rank
  // placement rather than assuming contiguous per-node blocks.
  MPI_Comm leader_comm = MPI_COMM_NULL;
  MPI_Comm_split(world, local_rank == 0 ? 0 : MPI_UNDEFINED, world_rank, &leader_comm);
  int node_index = 0, node_count = 0;
  if (leader_comm != MPI_COMM_NULL) {
    MPI_Comm_rank(leader_comm, &node_index);
    MPI_Comm_size(leader_comm, &node_count);
    MPI_Comm_free(&leader_comm);
  }
  MPI_Bcast(&node_index, 1, MPI_INT, 0, node_comm);
  MPI_Bcast(&node_count, 1, MPI_INT, 0, node_comm);
  MPI_Comm_free(&node_comm);

  if (local_size < 2 || node_count * (local_size - 1) < 2) {
    throw std::runtime_error(
        "load_balancing_adlb needs at least 2 ranks per node and 2 workers overall: one rank per "
        "node is held out, and one of the rest is the master.");
  }
  const int servers = requested_servers > 0 ? requested_servers : node_count;
  if (servers > node_count || node_count % servers != 0) {
    throw std::runtime_error("--servers must divide the node count (" + std::to_string(node_count) +
                             ") and be no larger than it; got " + std::to_string(servers) + ".");
  }

  const int workers_per_node = local_size - 1;
  const int total_workers = node_count * workers_per_node;
  // Dividing evenly keeps this the chunk ADLB itself computes, so server j
  // really does own the workers of nodes [j, j + nodes_per_server).
  const int nodes_per_server = node_count / servers;
  const bool held_out = local_rank == local_size - 1;
  const bool is_server = held_out && node_index % nodes_per_server == 0;

  AdlbPlacement placement;
  placement.nodes = static_cast<uint64_t>(node_count);
  placement.servers = servers;
  placement.executing_workers = static_cast<uint64_t>(total_workers - 1);

  if (held_out && !is_server) {
    MPI_Comm_split(world, MPI_UNDEFINED, 0, &placement.comm);
    return placement;
  }
  // Comm_split orders by key, so these keys *are* the ADLB ranks.
  const int key = is_server ? total_workers + node_index / nodes_per_server
                            : node_index * workers_per_node + local_rank;
  MPI_Comm_split(world, 0, key, &placement.comm);
  return placement;
}

// Publishes tasks [first, first + count). One blocking round trip each:
// ADLB_Put waits for a response and the API has no async or batched put.
//
// spread_over == 0 uses ADLB_RANK_ANY, which lb routes to the caller's own
// server -- right for a worker publishing its own share. For the master that
// would pile the batch on one server and leave the rest to steal, so
// spread_over > 0 round-robins a *soft* target over that many worker ranks:
// lb routes each put to the target's server and, for anything but
// STRICT_HARD, also files it in that server's untargeted pool, so the batch
// spreads without pinning any task.
void publish_tasks(uint64_t first, uint64_t count, uint64_t spread_over) {
  if (first + count >= kBatchDone) {
    throw std::runtime_error("task ids reached the end-of-batch marker; shorten the sweep.");
  }
  adlb_put_opts opts = ADLB_DEFAULT_PUT_OPTS;
  if (spread_over > 0) opts.strictness = ADLB_TGT_STRICT_SOFT;
  for (uint64_t i = 0; i < count; ++i) {
    const Task task = static_cast<Task>(first + i);
    // Worker ranks are 1..spread_over, so a full pass covers every server.
    const int target = spread_over > 0 ? static_cast<int>(1 + i % spread_over) : ADLB_RANK_ANY;
    check(ADLB_Put(&task, static_cast<int>(sizeof(Task)), target, kMasterRank, kWorkType, opts),
          "ADLB_Put(work)");
  }
}

// Counts executed tasks without ADLB, for --result_routing none. One
// fire-and-forget MPI_Accumulate per task: no reply, no round trip, and it
// never reaches a server. The master polls its own copy. This measures ADLB's
// task distribution with its return path removed, so those rows are not
// comparable with series that return a result per task.
struct DoneCounter {
  MPI_Win win = MPI_WIN_NULL;
  int64_t* counter = nullptr;  // master only
  bool enabled = false;

  MPI_Comm comm = MPI_COMM_NULL;

  void open(MPI_Comm c, int rank) {
    enabled = true;
    comm = c;
    const MPI_Aint bytes = (rank == kMasterRank) ? static_cast<MPI_Aint>(sizeof(int64_t)) : 0;
    MPI_Win_allocate(bytes, sizeof(int64_t), MPI_INFO_NULL, comm, &counter, &win);
    if (rank == kMasterRank) *counter = 0;
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win);
    MPI_Barrier(comm);
  }
  void bump() {
    const int64_t one = 1;
    MPI_Accumulate(&one, 1, MPI_INT64_T, kMasterRank, 0, 1, MPI_INT64_T, MPI_SUM, win);
    // flush, not flush_local: the latter only frees the origin buffer and
    // leaves the master free never to observe the increment.
    MPI_Win_flush(kMasterRank, win);
  }
  void wait_for(int64_t target) {
    while (true) {
      MPI_Win_sync(win);
      if (*static_cast<volatile int64_t*>(counter) >= target) return;
      // The two-sided engine has to run for remote RMA state to advance; a
      // bare spin starves passive-target progress and livelocks. Same reason
      // dynampi's rma_detail::rma_wait_idle exists.
      int flag = 0;
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, MPI_STATUS_IGNORE);
    }
  }
  void close() {
    if (!enabled) return;
    MPI_Win_unlock_all(win);
    MPI_Win_free(&win);
    enabled = false;
  }
};

// Executes tasks until the master's end-of-batch marker arrives. Returns
// false if ADLB shut the run down instead, which ends the sweep.
bool drain_until_batch_end(WorkerFunctor& worker_function, int my_rank,
                           const std::string& result_routing, DoneCounter& done) {
  Task task = 0;
  while (true) {
    // Buffer and limit are both sizeof(Task), so ADLB never reallocates:
    // an oversized task would come back as ADLB_ERROR instead.
    void* payload = &task;
    int length = static_cast<int>(sizeof(Task));
    int answer = 0, type_recvd = 0;
    MPI_Comm task_comm = MPI_COMM_NULL;
    const adlb_code rc = ADLB_Get(kWorkType, &payload, &length, static_cast<int>(sizeof(Task)),
                                  &answer, &type_recvd, &task_comm);
    if (rc == ADLB_SHUTDOWN) return false;
    check(rc, "ADLB_Get(work)");
    if (task == kBatchDone) return true;

    const Result result = worker_function(task);
    if (result_routing == "none") {
      done.bump();
      continue;
    }
    // Only the master ever requests kResultType, so an untargeted result is
    // still unambiguous -- it just waits on this node's server until the
    // master's server steals it.
    check(ADLB_Put(&result, static_cast<int>(sizeof(Result)), kMasterRank, my_rank, kResultType,
                   ADLB_DEFAULT_PUT_OPTS),
          "ADLB_Put(result)");
  }
}

// Blocks until `count` results are back. ADLB_Amget posts `window` receives
// behind a single request, so collecting a batch costs a handful of messages
// rather than one round trip per result -- but every result is targeted at
// the master, so they all land on the master's server either way.
void collect_results(uint64_t count, size_t window) {
  // Reused across chunks: every request of a chunk is waited on before its
  // slot is posted again.
  std::vector<Result> buffers(window);
  std::vector<adlb_payload_buf> payloads(window);
  std::vector<adlb_get_req> reqs(window);
  for (size_t i = 0; i < window; ++i) {
    payloads[i] = {&buffers[i], static_cast<int>(sizeof(Result))};
  }

  for (uint64_t remaining = count; remaining > 0;) {
    const size_t n = static_cast<size_t>(std::min<uint64_t>(window, remaining));
    // wait=false: posting returns once the request is sent; each result is
    // waited for below.
    check(ADLB_Amget(kResultType, static_cast<int>(n), false, payloads.data(), reqs.data()),
          "ADLB_Amget(result)");
    for (size_t i = 0; i < n; ++i) {
      int length = 0, answer = 0, type_recvd = 0;
      MPI_Comm task_comm = MPI_COMM_NULL;
      const adlb_code rc = ADLB_Aget_wait(&reqs[i], &length, &answer, &type_recvd, &task_comm);
      if (rc == ADLB_SHUTDOWN) {
        throw std::runtime_error(
            "ADLB shut down mid-batch with results outstanding; raise ADLB_EXHAUST_TIME.");
      }
      check(rc, "ADLB_Aget_wait(result)");
    }
    remaining -= n;
  }
}

// Releases every worker from its drain loop, one marker each. Sent after the
// batch timer has stopped: it is harness teardown, not part of the drain.
void end_batch(uint64_t workers) {
  for (uint64_t w = 1; w <= workers; ++w) {
    check(ADLB_Put(&kBatchDone, static_cast<int>(sizeof(Task)), static_cast<int>(w), kMasterRank,
                   kWorkType, ADLB_DEFAULT_PUT_OPTS),
          "ADLB_Put(batch end)");
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int world_rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  cxxopts::Options options(
      "load_balancing_adlb",
      "The ADLB arm of the load_balancing sweep: times ADLB's wall-clock drain of a fixed-size "
      "task batch (tasks_per_worker * workers tasks) across a range of tasks_per_worker values, "
      "writing the same CSV schema load_balancing does.");
  options.add_options()("t,expected_us", "Fixed task duration in microseconds",
                        cxxopts::value<uint64_t>()->default_value("1000"))(
      "duration_mode", "fixed, uniform or lognormal; as load_balancing's option of the same name.",
      cxxopts::value<std::string>()->default_value("fixed"))(
      "task_duration_cv", "duration_mode=lognormal only: coefficient of variation (stddev/mean).",
      cxxopts::value<double>()->default_value("1.0"))(
      "min_tasks_per_worker", "Smallest tasks-per-worker batch size to test",
      cxxopts::value<uint64_t>()->default_value("1"))(
      "max_tasks_per_worker", "Largest tasks-per-worker batch size to test",
      cxxopts::value<uint64_t>()->default_value("10"))(
      "r,repeats", "Number of repeats averaged per tasks_per_worker combo",
      cxxopts::value<uint64_t>()->default_value("3"))(
      "D,distribution", "Accepted for symmetry with load_balancing; must be adlb.",
      cxxopts::value<std::string>()->default_value("adlb"))(
      "n,nodes", "Number of nodes for labeling output (defaults to the job's)",
      cxxopts::value<uint64_t>()->default_value("0"))(
      "servers",
      "ADLB server ranks. Must divide the node count. 0 (default) puts one on every node, the "
      "same per-node overhead the hierarchical distributors pay for a node manager.",
      cxxopts::value<int>()->default_value("0"))(
      "producers",
      "Who publishes the batch: 1 (default) the master alone, as the DynaMPI drivers' manager "
      "does; -1 every worker, each publishing its own tasks_per_worker share to its node's "
      "server. Those are the only two values -- a partial producer set is not a configuration "
      "anyone would run.",
      cxxopts::value<int>()->default_value("1"))(
      "put_routing",
      "--producers 1 only: local (default) is lb's own untargeted routing, all on the master's "
      "own server; spread round-robins a soft target over every server, which measured worse "
      "because a targeted put leaves the node.",
      cxxopts::value<std::string>()->default_value("local"))(
      "result_routing",
      "master (default) targets every result at the master; none returns no results at all and "
      "counts completions through a one-sided MPI counter, which bounds what removing the return "
      "path could buy. Untargeted results are not an option: lb's heap_steal_type() skips heap "
      "indices 0 and 1, so the last two units on a server can never be stolen and results "
      "stranded away from the master's server hang the collection.",
      cxxopts::value<std::string>()->default_value("master"))(
      "steal_rate_limit",
      "Seconds between a starved server's steal attempts, overriding lb's compile-time 0.05 s. "
      "<= 0 (default) leaves it alone. The post-failure backoff is scaled with it.",
      cxxopts::value<double>()->default_value("0"))(
      "result_window", "Results the master requests at once via ADLB_Amget.",
      cxxopts::value<int>()->default_value("128"))(
      "S,system", "System label for plotting (frontier, aurora, ...)",
      cxxopts::value<std::string>()->default_value(""))(
      "o,output", "Append results to CSV file", cxxopts::value<std::string>()->default_value(""))(
      // load_balancing's launch scripts pass these unconditionally; they name
      // knobs on DynaMPI's distributors, so accept and ignore them.
      "max_upper_fanout", "Ignored (DynaMPI hierarchical option)",
      cxxopts::value<int>()->default_value("-1"))("pipeline_depth",
                                                  "Ignored (DynaMPI hierarchical option)",
                                                  cxxopts::value<int>()->default_value("-1"))(
      "max_pending_rounds", "Ignored (DynaMPI hierarchical option)",
      cxxopts::value<int>()->default_value("-1"))("h,help", "Print usage");

  cxxopts::ParseResult args;
  try {
    args = options.parse(argc, argv);
  } catch (const std::exception& e) {
    if (world_rank == 0) {
      std::cerr << "Error parsing options: " << e.what() << "\n" << options.help() << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  if (args.count("help")) {
    if (world_rank == 0) std::cout << options.help() << std::endl;
    MPI_Finalize();
    return 0;
  }

  BenchmarkOptions opts;
  try {
    opts.expected_us = args["expected_us"].as<uint64_t>();
    opts.duration_mode =
        load_balancing::parse_duration_mode(args["duration_mode"].as<std::string>());
    opts.task_duration_cv = args["task_duration_cv"].as<double>();
    opts.min_tasks_per_worker = args["min_tasks_per_worker"].as<uint64_t>();
    opts.max_tasks_per_worker = args["max_tasks_per_worker"].as<uint64_t>();
    opts.repeats = args["repeats"].as<uint64_t>();
    opts.nodes = args["nodes"].as<uint64_t>();
    opts.servers = args["servers"].as<int>();
    opts.producers = args["producers"].as<int>();
    opts.result_window = args["result_window"].as<int>();
    const std::string routing = args["put_routing"].as<std::string>();
    if (routing != "spread" && routing != "local") {
      throw std::runtime_error("--put_routing must be spread or local; got '" + routing + "'.");
    }
    opts.spread_puts = routing == "spread";
    opts.steal_rate_limit = args["steal_rate_limit"].as<double>();
    opts.result_routing = args["result_routing"].as<std::string>();
    if (opts.result_routing != "master" && opts.result_routing != "none") {
      throw std::runtime_error("--result_routing must be master or none; got '" +
                               opts.result_routing + "'.");
    }
    opts.system = args["system"].as<std::string>();
    opts.output_path = args["output"].as<std::string>();

    const std::string distribution = args["distribution"].as<std::string>();
    if (distribution != "adlb") {
      throw std::runtime_error("load_balancing_adlb only runs --distribution adlb (got '" +
                               distribution + "'); use the load_balancing binary for the rest.");
    }
    if (opts.expected_us == 0) throw std::runtime_error("--expected_us must be >= 1 microsecond.");
    if (opts.min_tasks_per_worker > opts.max_tasks_per_worker) {
      throw std::runtime_error("--min_tasks_per_worker must be <= --max_tasks_per_worker.");
    }
    if (opts.repeats == 0) throw std::runtime_error("--repeats must be >= 1.");
    if (opts.result_window < 1) throw std::runtime_error("--result_window must be >= 1.");
    if (opts.producers != 1 && opts.producers != -1) {
      throw std::runtime_error("--producers must be 1 (master alone) or -1 (every worker); got " +
                               std::to_string(opts.producers) + ".");
    }
  } catch (const std::exception& e) {
    if (world_rank == 0) std::cerr << "Error: " << e.what() << std::endl;
    MPI_Finalize();
    return 1;
  }

  try {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const AdlbPlacement placement = build_adlb_comm(MPI_COMM_WORLD, opts.servers);
    if (opts.nodes == 0) opts.nodes = placement.nodes;
    if (placement.comm == MPI_COMM_NULL) {
      // Held out and not a server: takes no part in the run, and only has to
      // stay alive until everyone else is done, which MPI_Finalize handles.
      MPI_Finalize();
      return 0;
    }

    int type_vect[2] = {kWorkType, kResultType};
    int am_server = 0;
    MPI_Comm adlb_comm = placement.comm;
    MPI_Comm worker_comm = MPI_COMM_NULL;

    // ADLB is initialised once per process, not per batch. Reported as
    // construct_s anyway -- it is what standing an ADLB instance up costs,
    // which is the column the other drivers put their setup cost in.
    dynampi::Timer init_timer;
    check(ADLB_Init(placement.servers, 2, type_vect, &am_server, adlb_comm, &worker_comm),
          "ADLB_Init");
    const double construct_s = init_timer.stop().count();

    // After Init so nothing re-initialises them, before ADLB_Server so the
    // server ranks -- the only ones that steal -- pick the values up.
    if (opts.steal_rate_limit > 0.0) {
      xlb_steal_rate_limit = opts.steal_rate_limit;
      xlb_steal_backoff = 2.0 * opts.steal_rate_limit;  // lb ships a 2:1 ratio
    }

    if (am_server) {
      ADLB_Server(1024L * 1024L * 1024L);
    } else {
      int worker_rank = 0;
      MPI_Comm_rank(worker_comm, &worker_rank);
      WorkerFunctor worker_function(world_rank, opts.expected_us, opts.duration_mode,
                                    opts.task_duration_cv);
      const size_t window = static_cast<size_t>(opts.result_window);
      const uint64_t workers = placement.executing_workers;
      const bool distributed = opts.producers < 0;

      // Our own communicator rather than ADLB's, so the per-batch barrier
      // cannot collide with anything the library does on its own. A rank
      // waiting here is outside ADLB's request queue and so counts as busy,
      // which is the safe direction: idle detection cannot fire mid-sweep.
      MPI_Comm bench_comm = MPI_COMM_NULL;
      MPI_Comm_dup(worker_comm, &bench_comm);

      DoneCounter done;
      if (opts.result_routing == "none") done.open(bench_comm, worker_rank);
      int64_t done_target = 0;

      // Both arms run the same protocol -- barrier, publish, drain to the
      // master's end-of-batch marker -- so the only difference between them
      // is which ranks call publish_tasks().
      uint64_t next_task = 0;
      bool running = true;
      for (uint64_t k = opts.min_tasks_per_worker; running && k <= opts.max_tasks_per_worker; ++k) {
        for (uint64_t repeat = 0; running && repeat < opts.repeats; ++repeat) {
          // One task per worker first, drained untimed, as load_balancing.cpp
          // warms each combo; then the batch proper.
          double phase_s[2] = {0.0, 0.0};
          for (int phase = 0; phase < 2 && running; ++phase) {
            const uint64_t per_worker = (phase == 0) ? 1 : k;
            const uint64_t count = per_worker * workers;

            MPI_Barrier(bench_comm);
            done_target += static_cast<int64_t>(count);
            if (worker_rank == kMasterRank) {
              dynampi::Timer timer;
              // Static split: worker w owns [next_task + (w-1)*per_worker, ...).
              // The master publishes the lot only when it is the sole producer.
              if (!distributed) {
                publish_tasks(next_task, count, opts.spread_puts ? workers : 0);
              }
              if (opts.result_routing == "none") {
                done.wait_for(done_target);
              } else {
                collect_results(count, window);
              }
              phase_s[phase] = timer.stop().count();
              end_batch(workers);
            } else {
              if (distributed) {
                // Untargeted: a worker's own server is already the right one.
                publish_tasks(next_task + static_cast<uint64_t>(worker_rank - 1) * per_worker,
                              per_worker, 0);
              }
              running =
                  drain_until_batch_end(worker_function, worker_rank, opts.result_routing, done);
            }
            next_task += count;
          }

          if (worker_rank == kMasterRank) {
            // finalize_s/destruct_s stay 0: ADLB has no per-batch teardown,
            // and its one shutdown is the servers' idle timeout at the end
            // of the process.
            ResultRow row;
            row.system = opts.system;
            // Its own series: same library, a different producer structure,
            // and the figures carry both.
            // Both knobs are in the series name, not just columns: the plot
            // keys series on the distributor, so anything left out of the
            // name silently averages across it.
            row.distributor = distributed ? "adlb_distributed" : "adlb";
            if (opts.result_routing == "none") row.distributor += "_no_results";
            row.adlb_servers = placement.servers;
            row.adlb_producers = distributed ? static_cast<int>(workers) : 1;
            row.adlb_put_routing = distributed ? "n/a" : (opts.spread_puts ? "spread" : "local");
            row.adlb_result_routing = opts.result_routing;
            row.adlb_steal_rate_limit = opts.steal_rate_limit;
            row.nodes = opts.nodes;
            row.world_size = static_cast<uint64_t>(world_size);
            row.workers = workers;
            row.expected_us = opts.expected_us;
            row.duration_mode = opts.duration_mode;
            row.task_duration_cv = opts.task_duration_cv;
            row.tasks_per_worker = k;
            row.repeat = repeat;
            row.total_tasks = k * workers;
            row.elapsed_s = phase_s[1];
            row.construct_s = construct_s;
            row.warmup_s = phase_s[0];
            load_balancing::append_result(opts.output_path, row);
          }
        }
      }

      done.close();
      MPI_Comm_free(&bench_comm);
      // Everyone parks in a Get with no work left, which is the servers'
      // exhaustion condition; it releases the whole run.
      if (running) {
        DoneCounter unused;
        drain_until_batch_end(worker_function, worker_rank, "master", unused);
      }
    }

    ADLB_Finalize();
    MPI_Comm_free(&adlb_comm);
  } catch (const std::exception& e) {
    std::cerr << "Benchmark failed on world rank " << world_rank << ": " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  } catch (...) {
    std::cerr << "Benchmark failed on world rank " << world_rank << ": unknown error" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
