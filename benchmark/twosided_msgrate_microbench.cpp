// SPDX-FileCopyrightText: 2026 Ryan Stocks
// SPDX-License-Identifier: Apache-2.0

// Measures the raw ceiling of the two-sided request/reply pattern that
// NaiveWorkDistributor is built on: one root rank serving N workers, each of
// which sends a result and blocks for the next task. No queue, no bookkeeping,
// no task execution -- just the MPI cost of the exchange, so this isolates
// "how fast can one rank serve N peers" from anything about how the
// distributor uses it.
//
// Three modes, differing only in how the root receives:
//   probe    -- MPI_Probe + MPI_Get_count + MPI_Recv, then MPI_Send. This is
//               exactly what NaiveWorkDistributor does today, on both sides.
//   noprobe  -- one MPI_Recv(ANY_SOURCE, ANY_TAG) into a max-sized buffer,
//               reading source and tag from the recv's own status, then
//               MPI_Send. Valid whenever every message has a compile-time size
//               bound, which holds for fixed-size TaskT/ResultT (task-failure
//               strings are already capped at kMaxTaskErrorMessage).
//   oneway   -- workers stream messages to the root with no per-message reply,
//               bounded by a credit window. Gives the pure receive ceiling,
//               without the reply leg, as an upper bound on the other two.
//
// The first two count completed round trips, which is the same unit as the
// distributors' tasks/s, so the numbers are directly comparable to the naive
// plateau in the weak-scaling sweeps. `oneway` counts messages received.

#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

enum Tag : int { TASK = 0, STOP = 1, RESULT = 2, ACK = 3 };

double elapsed_since(const std::chrono::steady_clock::time_point& t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// True on every rank that shares a compute node with `root`. --exclude_root_node
// uses this to drop the root's own node from the worker set, so the measured
// rate is the fabric's alone rather than a mix of fabric round trips and much
// cheaper shared-memory ones.
bool shares_node_with_root(int root) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm node_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
  const int has_root = (rank == root) ? 1 : 0;
  int any_has_root = 0;
  MPI_Allreduce(&has_root, &any_has_root, 1, MPI_INT, MPI_MAX, node_comm);
  MPI_Comm_free(&node_comm);
  return any_has_root == 1;
}

// ---------------------------------------------------------------------------
// Round-trip modes

long long root_probe(int nworkers, int payload, double warmup_s, double duration_s) {
  std::vector<int> rbuf(static_cast<size_t>(payload)), sbuf(static_cast<size_t>(payload), 1);
  auto t0 = std::chrono::steady_clock::now();
  long long completed = 0;
  bool counting = false, stopping = false;
  int stopped = 0;
  while (stopped < nworkers) {
    MPI_Status st;
    MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    int count = 0;
    MPI_Get_count(&st, MPI_INT, &count);
    MPI_Recv(rbuf.data(), count, MPI_INT, st.MPI_SOURCE, st.MPI_TAG, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    const double t = elapsed_since(t0);
    if (!counting && t >= warmup_s) {
      counting = true;
      completed = 0;
    }
    if (!stopping && t >= warmup_s + duration_s) stopping = true;

    if (stopping) {
      MPI_Send(sbuf.data(), 0, MPI_INT, st.MPI_SOURCE, Tag::STOP, MPI_COMM_WORLD);
      stopped++;
    } else {
      MPI_Send(sbuf.data(), payload, MPI_INT, st.MPI_SOURCE, Tag::TASK, MPI_COMM_WORLD);
      if (counting) completed++;
    }
  }
  return completed;
}

long long root_noprobe(int nworkers, int payload, double warmup_s, double duration_s) {
  std::vector<int> rbuf(static_cast<size_t>(payload)), sbuf(static_cast<size_t>(payload), 1);
  auto t0 = std::chrono::steady_clock::now();
  long long completed = 0;
  bool counting = false, stopping = false;
  int stopped = 0;
  while (stopped < nworkers) {
    MPI_Status st;
    MPI_Recv(rbuf.data(), payload, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);

    const double t = elapsed_since(t0);
    if (!counting && t >= warmup_s) {
      counting = true;
      completed = 0;
    }
    if (!stopping && t >= warmup_s + duration_s) stopping = true;

    if (stopping) {
      MPI_Send(sbuf.data(), 0, MPI_INT, st.MPI_SOURCE, Tag::STOP, MPI_COMM_WORLD);
      stopped++;
    } else {
      MPI_Send(sbuf.data(), payload, MPI_INT, st.MPI_SOURCE, Tag::TASK, MPI_COMM_WORLD);
      if (counting) completed++;
    }
  }
  return completed;
}

void worker_roundtrip(int payload, bool use_probe) {
  std::vector<int> sbuf(static_cast<size_t>(payload), 1), rbuf(static_cast<size_t>(payload));
  while (true) {
    MPI_Send(sbuf.data(), payload, MPI_INT, 0, Tag::RESULT, MPI_COMM_WORLD);
    MPI_Status st;
    if (use_probe) {
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
      int count = 0;
      MPI_Get_count(&st, MPI_INT, &count);
      MPI_Recv(rbuf.data(), count, MPI_INT, 0, st.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv(rbuf.data(), payload, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    }
    if (st.MPI_TAG == Tag::STOP) break;
  }
}

// ---------------------------------------------------------------------------
// One-way mode: pure receive ceiling, credit-windowed so the number of
// unmatched eager messages stays bounded at `window` per worker.

long long root_oneway(const std::vector<int>& worker_index, int nworkers, int payload, int window,
                      double warmup_s, double duration_s) {
  std::vector<int> rbuf(static_cast<size_t>(payload));
  std::vector<int> seen(static_cast<size_t>(nworkers), 0);
  auto t0 = std::chrono::steady_clock::now();
  long long received = 0;
  bool counting = false, stopping = false;
  int stopped = 0;
  while (stopped < nworkers) {
    MPI_Status st;
    MPI_Recv(rbuf.data(), payload, MPI_INT, MPI_ANY_SOURCE, Tag::RESULT, MPI_COMM_WORLD, &st);

    const double t = elapsed_since(t0);
    if (!counting && t >= warmup_s) {
      counting = true;
      received = 0;
    }
    if (!stopping && t >= warmup_s + duration_s) stopping = true;
    if (counting) received++;

    const size_t w = static_cast<size_t>(worker_index[st.MPI_SOURCE]);
    if (++seen[w] == window) {
      seen[w] = 0;
      if (stopping) {
        MPI_Send(nullptr, 0, MPI_INT, st.MPI_SOURCE, Tag::STOP, MPI_COMM_WORLD);
        stopped++;
      } else {
        MPI_Send(nullptr, 0, MPI_INT, st.MPI_SOURCE, Tag::ACK, MPI_COMM_WORLD);
      }
    }
  }
  return received;
}

void worker_oneway(int payload, int window) {
  std::vector<int> sbuf(static_cast<size_t>(payload), 1);
  while (true) {
    for (int i = 0; i < window; ++i) {
      MPI_Send(sbuf.data(), payload, MPI_INT, 0, Tag::RESULT, MPI_COMM_WORLD);
    }
    MPI_Status st;
    MPI_Recv(nullptr, 0, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
    if (st.MPI_TAG == Tag::STOP) break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  cxxopts::Options options("twosided_msgrate_microbench",
                           "Root-rank request/reply message-rate ceiling for N workers");
  options.add_options()  //
      ("m,modes", "comma-separated: probe,noprobe,oneway",
       cxxopts::value<std::string>()->default_value("probe,noprobe,oneway"))                    //
      ("d,duration_s", "timed seconds per mode", cxxopts::value<double>()->default_value("5"))  //
      ("w,warmup_s", "untimed seconds before each mode",
       cxxopts::value<double>()->default_value("1"))  //
      ("p,payload_ints", "payload in 4-byte ints, each direction",
       cxxopts::value<int>()->default_value("1"))  //
      ("credit_window", "messages per credit in oneway mode",
       cxxopts::value<int>()->default_value("64"))  //
      ("exclude_root_node", "drop workers sharing the root's node, leaving only off-node workers",
       cxxopts::value<bool>()->default_value("false"))  //
      ("nodes", "node count, recorded in the CSV only",
       cxxopts::value<int>()->default_value("0"))  //
      ("system", "system name, recorded in the CSV only",
       cxxopts::value<std::string>()->default_value("unknown"))  //
      ("o,output", "append a CSV row per mode to this file",
       cxxopts::value<std::string>()->default_value(""))  //
      ("h,help", "print usage");

  auto parsed = options.parse(argc, argv);
  if (parsed.count("help")) {
    if (rank == 0) std::cout << options.help() << std::endl;
    MPI_Finalize();
    return 0;
  }
  if (size < 2) {
    if (rank == 0) std::cerr << "Need at least 2 ranks\n";
    MPI_Finalize();
    return 1;
  }

  const double duration_s = parsed["duration_s"].as<double>();
  const double warmup_s = parsed["warmup_s"].as<double>();
  const int payload = parsed["payload_ints"].as<int>();
  const int window = parsed["credit_window"].as<int>();
  const bool exclude_root_node = parsed["exclude_root_node"].as<bool>();
  const int nodes = parsed["nodes"].as<int>();
  const std::string system = parsed["system"].as<std::string>();
  const std::string output = parsed["output"].as<std::string>();

  std::vector<std::string> modes;
  {
    std::string s = parsed["modes"].as<std::string>();
    size_t pos = 0;
    while (pos < s.size()) {
      size_t comma = s.find(',', pos);
      if (comma == std::string::npos) comma = s.size();
      modes.push_back(s.substr(pos, comma - pos));
      pos = comma + 1;
    }
  }

  // Worker set as explicit world ranks. Normally every rank but the root; with
  // --exclude_root_node, also minus the ranks sharing the root's node, whose
  // round trips go through shared memory instead of the fabric. Called on every
  // rank because MPI_Comm_split_type inside is collective over MPI_COMM_WORLD.
  const bool on_root_node = shares_node_with_root(0);
  std::vector<int> workers;
  std::vector<int> worker_index(static_cast<size_t>(size), -1);
  {
    const int is_worker = (rank != 0 && !(exclude_root_node && on_root_node)) ? 1 : 0;
    std::vector<int> flags(static_cast<size_t>(size), 0);
    MPI_Allgather(&is_worker, 1, MPI_INT, flags.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for (int r = 0; r < size; ++r) {
      if (flags[static_cast<size_t>(r)] != 0) {
        worker_index[static_cast<size_t>(r)] = static_cast<int>(workers.size());
        workers.push_back(r);
      }
    }
  }
  const int nworkers = static_cast<int>(workers.size());
  if (nworkers == 0) {
    if (rank == 0) {
      std::cerr << "No workers left: --exclude_root_node needs more than one node\n";
    }
    MPI_Finalize();
    return 1;
  }
  const bool is_worker = worker_index[static_cast<size_t>(rank)] >= 0;

  std::ofstream csv;
  if (rank == 0 && !output.empty()) {
    const bool exists = std::ifstream(output).good();
    csv.open(output, std::ios::app);
    if (!exists) {
      csv << "system,mode,nodes,world_size,workers,payload_bytes,duration_s,completed,"
             "completed_per_s,completed_per_s_per_worker,exclude_root_node\n";
    }
  }

  for (const std::string& mode : modes) {
    MPI_Barrier(MPI_COMM_WORLD);
    long long completed = 0;
    if (rank == 0) {
      if (mode == "probe") {
        completed = root_probe(nworkers, payload, warmup_s, duration_s);
      } else if (mode == "noprobe") {
        completed = root_noprobe(nworkers, payload, warmup_s, duration_s);
      } else if (mode == "oneway") {
        completed = root_oneway(worker_index, nworkers, payload, window, warmup_s, duration_s);
      } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    } else if (is_worker) {
      if (mode == "oneway") {
        worker_oneway(payload, window);
      } else {
        worker_roundtrip(payload, mode == "probe");
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      const double rate = static_cast<double>(completed) / duration_s;
      std::cout << "RESULT mode=" << mode << " world_size=" << size << " workers=" << nworkers
                << " payload_bytes=" << (payload * 4) << " completed=" << completed
                << " completed_per_s=" << rate
                << " completed_per_s_per_worker=" << (rate / nworkers)
                << " exclude_root_node=" << (exclude_root_node ? 1 : 0) << std::endl;
      if (csv.is_open()) {
        csv << system << "," << mode << "," << nodes << "," << size << "," << nworkers << ","
            << (payload * 4) << "," << duration_s << "," << completed << "," << rate << ","
            << (rate / nworkers) << "," << (exclude_root_node ? 1 : 0) << "\n";
        csv.flush();
      }
    }
  }

  MPI_Finalize();
  return 0;
}
