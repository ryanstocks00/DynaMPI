/*
 * SPDX-FileCopyrightText: 2026 Ryan Stocks
 * SPDX-License-Identifier: Apache-2.0
 */

// The ADLB arm of the load_balancing sweep: the same batch-drain measurement,
// made against ADLB/X (shipped as `lb` in Swift/T) so the figures have an
// external work-stealing scheduler to compare against.
//
// A separate binary because ADLB owns its ranks for the whole process --
// ADLB_Init() parks servers in ADLB_Server() and workers in ADLB_Get() until
// its own idle detection ends the run, and it cannot be re-initialised -- so
// it cannot share one with distributors built per batch. The workload and CSV
// schema are shared with load_balancing.cpp via load_balancing_workload.hpp.

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
// Distributed injection only: a producer's instruction to publish one stride
// of the batch. Its own type so a producer can park in ADLB_Get(kInjectType)
// without also matching work -- ADLB_Get takes one type, not a set.
constexpr int kInjectType = 2;
// ADLB rank 0, the counterpart of the DynaMPI distributors' manager.
constexpr int kMasterRank = 0;

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
  // Producer ranks that publish the batch on the master's behalf. 0 keeps the
  // master the sole producer; -1 takes every rank held out and not used as a
  // server. See build_adlb_comm() for where they come from.
  int producers = 0;
  // Results kept requested at once (ADLB_Amget).
  int result_window = 128;
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
  int producers = 0;
  // ADLB rank of producer 0; producer i is producer_base + i.
  int producer_base = 0;
};

// What one producer is told to publish. Carried as a kInjectType task rather
// than over a side communicator: a producer waiting in ADLB_Get is where
// ADLB's idle detection expects an idle worker to be, whereas one parked in
// an MPI_Bcast between batches is invisible to it.
struct InjectDescriptor {
  uint64_t first_task = 0;
  uint64_t count = 0;
};

// Reorders MPI_COMM_WORLD into the communicator ADLB is initialised on.
// ADLB's layout is fixed -- workers are ranks [0, workers), servers are the
// trailing ones, and worker r is served by server r / (workers / servers)
// (lb's xlb_map_to_server) -- so ordering workers node-major and appending
// one server per server-block puts each server inside the block it serves.
// Left alone, the servers would be every rank of the last few nodes.
//
// One rank per node is held out of the worker set whatever --servers is (the
// last local rank, as load_balancing.cpp's flat_comm drops, keeping world
// rank 0 the master). That fixes the task-executing count at
// world_size - nodes - 1, what the hierarchical distributors also report, so
// every series drains the same batch and shares one reference line. Held-out
// ranks that are not servers or producers sit the run out entirely.
//
// Producers come out of that same held-out pool, so turning them on leaves
// the task-executing count untouched and the two ADLB arms stay comparable
// with each other and with the DynaMPI series.
AdlbPlacement build_adlb_comm(MPI_Comm world, int requested_servers, int requested_producers) {
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

  if (local_size < 2) {
    throw std::runtime_error("load_balancing_adlb needs at least 2 ranks per node.");
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

  // One held-out rank per node, minus the ones serving: what is left to draw
  // producers from. Computed from node_index alone so every rank agrees
  // without another collective.
  const int server_nodes_before = (node_index + nodes_per_server - 1) / nodes_per_server;
  const int spare_index = node_index - server_nodes_before;
  const int spare_nodes = node_count - servers;
  const int producers = requested_producers < 0 ? spare_nodes : requested_producers;
  if (producers > spare_nodes) {
    throw std::runtime_error(
        "--producers must be no larger than the held-out ranks left over once --servers has "
        "taken its own (" +
        std::to_string(spare_nodes) + " here); got " + std::to_string(producers) + ".");
  }
  const bool is_producer = held_out && !is_server && spare_index < producers;

  AdlbPlacement placement;
  placement.nodes = static_cast<uint64_t>(node_count);
  placement.servers = servers;
  placement.producers = producers;
  placement.producer_base = total_workers;
  placement.executing_workers = static_cast<uint64_t>(total_workers - 1);

  // Producers extend the worker range, so the block each server owns is
  // (total_workers + producers) / servers rather than total_workers /
  // servers. Uneven division leaves ADLB's own mapping straddling the block
  // boundaries this ordering was built to line up with -- harmless for the
  // canonical single server, worth saying out loud for the sweep.
  if (world_rank == 0 && servers > 1 && (total_workers + producers) % servers != 0) {
    std::cerr << "warning: " << producers << " producers and " << servers
              << " servers do not divide evenly; server affinity is approximate.\n";
  }

  if (held_out && !is_server && !is_producer) {
    MPI_Comm_split(world, MPI_UNDEFINED, 0, &placement.comm);
    return placement;
  }
  // Comm_split orders by key, so these keys *are* the ADLB ranks. Servers
  // have to trail every worker, producers included.
  int key = node_index * workers_per_node + local_rank;
  if (is_producer) {
    key = total_workers + spare_index;
  } else if (is_server) {
    key = total_workers + producers + node_index / nodes_per_server;
  }
  MPI_Comm_split(world, 0, key, &placement.comm);
  return placement;
}

// Drains tasks until ADLB shuts the run down. Results go back as a task
// targeted at the master, which is what makes a batch's completion
// observable there -- the same thing run_tasks() returning gives the
// DynaMPI drivers, through the same one-result-per-task funnel.
void worker_loop(WorkerFunctor& worker_function, int my_rank) {
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
    if (rc == ADLB_SHUTDOWN) break;
    check(rc, "ADLB_Get(work)");

    const Result result = worker_function(task);
    check(ADLB_Put(&result, static_cast<int>(sizeof(Result)), kMasterRank, my_rank, kResultType,
                   ADLB_DEFAULT_PUT_OPTS),
          "ADLB_Put(result)");
  }
}

// Publishes the stride each inject descriptor asks for, until ADLB shuts the
// run down. A producer never executes a task -- it is one of the ranks held
// out of the worker set -- so the executing count is the same whether or not
// distributed injection is on.
void producer_loop() {
  InjectDescriptor descriptor;
  while (true) {
    void* payload = &descriptor;
    int length = static_cast<int>(sizeof(InjectDescriptor));
    int answer = 0, type_recvd = 0;
    MPI_Comm task_comm = MPI_COMM_NULL;
    const adlb_code rc =
        ADLB_Get(kInjectType, &payload, &length, static_cast<int>(sizeof(InjectDescriptor)),
                 &answer, &type_recvd, &task_comm);
    if (rc == ADLB_SHUTDOWN) break;
    check(rc, "ADLB_Get(inject)");

    for (uint64_t i = 0; i < descriptor.count; ++i) {
      const Task task = static_cast<Task>(descriptor.first_task + i);
      check(ADLB_Put(&task, static_cast<int>(sizeof(Task)), ADLB_RANK_ANY, kMasterRank, kWorkType,
                     ADLB_DEFAULT_PUT_OPTS),
            "ADLB_Put(work)");
    }
  }
}

// Publishes `count` tasks and blocks until `count` results are back: the ADLB
// equivalent of insert_tasks() then run_tasks() draining to exhaustion.
//
// Publishing is one blocking round trip per task -- ADLB_Put waits for a
// response and the API has no async or batched put. Collecting does have one,
// so it uses it: ADLB_Amget posts `window` receives behind a single request.
//
// With `producers` > 0 the master publishes one descriptor per producer
// instead, and they publish the tasks concurrently. That trades `count`
// serial round trips for `producers` of them plus a concurrent
// `count / producers` -- minimised near sqrt(count), which is about where a
// producer per node lands for these batch sizes. It is the shape Swift/T
// itself generates work in, and the reason the two arms are plotted
// together: one is ADLB as the DynaMPI drivers are structured, the other is
// ADLB given a producer set it can actually use.
void drain_batch(uint64_t first_task, uint64_t count, size_t window, int producers,
                 int producer_base) {
  if (producers <= 0) {
    for (uint64_t i = 0; i < count; ++i) {
      const Task task = static_cast<Task>(first_task + i);
      check(ADLB_Put(&task, static_cast<int>(sizeof(Task)), ADLB_RANK_ANY, kMasterRank, kWorkType,
                     ADLB_DEFAULT_PUT_OPTS),
            "ADLB_Put(work)");
    }
  } else {
    // Even split, remainder spread over the leading producers. Targeted, so
    // each descriptor lands on the rank meant to act on it.
    const uint64_t per_producer = count / static_cast<uint64_t>(producers);
    const uint64_t remainder = count % static_cast<uint64_t>(producers);
    uint64_t next_task = first_task;
    for (int i = 0; i < producers; ++i) {
      InjectDescriptor descriptor;
      descriptor.first_task = next_task;
      descriptor.count = per_producer + (static_cast<uint64_t>(i) < remainder ? 1 : 0);
      next_task += descriptor.count;
      // Fewer tasks than producers: the tail has nothing to do, and a
      // descriptor asking for none would only cost a round trip.
      if (descriptor.count == 0) continue;
      check(ADLB_Put(&descriptor, static_cast<int>(sizeof(InjectDescriptor)), producer_base + i,
                     kMasterRank, kInjectType, ADLB_DEFAULT_PUT_OPTS),
            "ADLB_Put(inject)");
    }
  }

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
      "Ranks that publish the batch on the master's behalf, drawn from the ranks held out of "
      "the worker set. 0 (default) leaves the master the sole producer, as the DynaMPI drivers' "
      "manager is; -1 uses every held-out rank --servers has not taken.",
      cxxopts::value<int>()->default_value("0"))(
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
  } catch (const std::exception& e) {
    if (world_rank == 0) std::cerr << "Error: " << e.what() << std::endl;
    MPI_Finalize();
    return 1;
  }

  try {
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const AdlbPlacement placement = build_adlb_comm(MPI_COMM_WORLD, opts.servers, opts.producers);
    if (opts.nodes == 0) opts.nodes = placement.nodes;
    if (placement.comm == MPI_COMM_NULL) {
      // Held out and not a server: takes no part in the run, and only has to
      // stay alive until everyone else is done, which MPI_Finalize handles.
      MPI_Finalize();
      return 0;
    }

    int type_vect[3] = {kWorkType, kResultType, kInjectType};
    int am_server = 0;
    MPI_Comm adlb_comm = placement.comm;
    MPI_Comm worker_comm = MPI_COMM_NULL;

    // ADLB is initialised once per process, not per batch. Reported as
    // construct_s anyway -- it is what standing an ADLB instance up costs,
    // which is the column the other drivers put their setup cost in.
    dynampi::Timer init_timer;
    check(ADLB_Init(placement.servers, 3, type_vect, &am_server, adlb_comm, &worker_comm),
          "ADLB_Init");
    const double construct_s = init_timer.stop().count();

    if (am_server) {
      ADLB_Server(1024L * 1024L * 1024L);
    } else {
      int worker_rank = 0;
      MPI_Comm_rank(worker_comm, &worker_rank);
      WorkerFunctor worker_function(world_rank, opts.expected_us, opts.duration_mode,
                                    opts.task_duration_cv);
      const size_t window = static_cast<size_t>(opts.result_window);
      const int producers = placement.producers;

      if (producers > 0 && worker_rank >= placement.producer_base) {
        producer_loop();
      } else if (worker_rank == kMasterRank) {
        const uint64_t workers = placement.executing_workers;
        uint64_t next_task = 0;
        for (uint64_t k = opts.min_tasks_per_worker; k <= opts.max_tasks_per_worker; ++k) {
          for (uint64_t repeat = 0; repeat < opts.repeats; ++repeat) {
            // One task per worker, drained untimed, as load_balancing.cpp
            // warms each combo.
            dynampi::Timer warmup_timer;
            drain_batch(next_task, workers, window, producers, placement.producer_base);
            next_task += workers;
            const double warmup_s = warmup_timer.stop().count();

            const uint64_t total_tasks = k * workers;
            dynampi::Timer batch_timer;
            drain_batch(next_task, total_tasks, window, producers, placement.producer_base);
            const double elapsed_s = batch_timer.stop().count();
            next_task += total_tasks;

            // finalize_s/destruct_s stay 0: ADLB has no per-batch teardown,
            // and its one shutdown is the servers' idle timeout at the end
            // of the process.
            ResultRow row;
            row.system = opts.system;
            // Its own series: same library, a different producer structure,
            // and the figures carry both.
            row.distributor = producers > 0 ? "adlb_distributed" : "adlb";
            row.adlb_servers = placement.servers;
            row.adlb_producers = producers;
            row.nodes = opts.nodes;
            row.world_size = static_cast<uint64_t>(world_size);
            row.workers = workers;
            row.expected_us = opts.expected_us;
            row.duration_mode = opts.duration_mode;
            row.task_duration_cv = opts.task_duration_cv;
            row.tasks_per_worker = k;
            row.repeat = repeat;
            row.total_tasks = total_tasks;
            row.elapsed_s = elapsed_s;
            row.construct_s = construct_s;
            row.warmup_s = warmup_s;
            load_balancing::append_result(opts.output_path, row);
          }
        }

        // The master joins too: once every worker is parked in a Get with
        // no work left, the servers hit their exhaustion condition and
        // release everyone. It is also what keeps the run alive between
        // batches -- while the master is out here, its server counts it as
        // busy.
        worker_loop(worker_function, worker_rank);
      } else {
        worker_loop(worker_function, worker_rank);
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
