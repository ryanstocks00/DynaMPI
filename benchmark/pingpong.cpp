// SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
// SPDX-License-Identifier: MIT

// mpi_pair_bench.cpp
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

enum class Method { SEND, ISEND, BSEND, SSEND };

struct Options {
  std::size_t min_bytes = 1;
  std::size_t max_bytes = 1u << 25;  // 32 MiB
  int factor = 2;                    // geometric progression; use 1 for linear
  int warmup = 10;
  int iters = 100;
  int only_rank = -1;           // if >=0, test only pairs involving this rank
  std::vector<Method> methods;  // default: all
  std::string outfile = "mpi_pair_bench.csv";
};

struct PingResult {
  double avg_rtt_s;          // average round-trip time (per message)
  double send_call_total_s;  // total time spent inside send() calls across timed iterations
};

static void die(int rank, const std::string &msg) {
  if (rank == 0) std::cerr << "Error: " << msg << std::endl;
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static std::string method_name(Method m) {
  switch (m) {
    case Method::SEND:
      return "send";
    case Method::ISEND:
      return "isend";
    case Method::BSEND:
      return "bsend";
    case Method::SSEND:
      return "ssend";
  }
  return "?";
}

static std::optional<Method> parse_method(const std::string &s) {
  if (s == "send") {
    return Method::SEND;
  }
  if (s == "isend") {
    return Method::ISEND;
  }
  if (s == "bsend") {
    return Method::BSEND;
  }
  if (s == "ssend") {
    return Method::SSEND;
  }
  return std::nullopt;
}

static Options parse_args(int argc, char **argv, int rank) {
  Options opt;
  bool methods_specified = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char *name) {
      if (i + 1 >= argc) die(rank, std::string("missing value for ") + name);
    };
    if (a == "--min-bytes") {
      need("--min-bytes");
      try {
        opt.min_bytes = std::stoull(argv[++i]);
      } catch (const std::invalid_argument &e) {
        die(rank, "invalid value for --min-bytes: " + std::string(argv[i]) + ": " + e.what());
      } catch (const std::out_of_range &e) {
        die(rank, "invalid value for --min-bytes: " + std::string(argv[i]) + ": " + e.what());
      }
    } else if (a == "--max-bytes") {
      need("--max-bytes");
      try {
        opt.max_bytes = std::stoull(argv[++i]);
      } catch (const std::invalid_argument &e) {
        die(rank, "invalid value for --max-bytes: " + std::string(argv[i]) + ": " + e.what());
      } catch (const std::out_of_range &e) {
        die(rank, "invalid value for --max-bytes: " + std::string(argv[i]) + ": " + e.what());
      }
    } else if (a == "--factor") {
      need("--factor");
      try {
        opt.factor = std::stoi(argv[++i]);
      } catch (const std::invalid_argument &e) {
        die(rank, "invalid value for --factor: " + std::string(argv[i]) + ": " + e.what());
      } catch (const std::out_of_range &e) {
        die(rank, "invalid value for --factor: " + std::string(argv[i]) + ": " + e.what());
      }
    } else if (a == "--warmup") {
      need("--warmup");
      try {
        opt.warmup = std::stoi(argv[++i]);
      } catch (const std::invalid_argument &e) {
        die(rank, "invalid value for --warmup: " + std::string(argv[i]) + ": " + e.what());
      } catch (const std::out_of_range &e) {
        die(rank, "invalid value for --warmup: " + std::string(argv[i]) + ": " + e.what());
      }
    } else if (a == "--iters") {
      need("--iters");
      try {
        opt.iters = std::stoi(argv[++i]);
      } catch (const std::invalid_argument &e) {
        die(rank, "invalid value for --iters: " + std::string(argv[i]) + ": " + e.what());
      } catch (const std::out_of_range &e) {
        die(rank, "invalid value for --iters: " + std::string(argv[i]) + ": " + e.what());
      }
    } else if (a == "--outfile") {
      need("--outfile");
      opt.outfile = argv[++i];
    } else if (a == "--only-rank") {
      need("--only-rank");
      try {
        opt.only_rank = std::stoi(argv[++i]);
      } catch (const std::invalid_argument &e) {
        die(rank, "invalid value for --only-rank: " + std::string(argv[i]) + ": " + e.what());
      } catch (const std::out_of_range &e) {
        die(rank, "invalid value for --only-rank: " + std::string(argv[i]) + ": " + e.what());
      }
    } else if (a == "--methods") {
      need("--methods");
      methods_specified = true;
      opt.methods.clear();
      std::string list = argv[++i];
      size_t start = 0;
      while (start <= list.size()) {
        size_t comma = list.find(',', start);
        std::string tok =
            (comma == std::string::npos) ? list.substr(start) : list.substr(start, comma - start);
        auto m = parse_method(tok);
        if (!m) die(rank, "unknown method in --methods: " + tok);
        opt.methods.push_back(*m);
        if (comma == std::string::npos) break;
        start = comma + 1;
      }
    } else if (a == "-h" || a == "--help") {
      if (rank == 0) {
        std::cout
            << "MPI pairwise bandwidth/latency benchmark\n\n"
               "Usage: mpirun -n <P> ./mpi_pair_bench [options]\n\n"
               "Options:\n"
               "  --min-bytes N        starting message size (default 1)\n"
               "  --max-bytes N        maximum message size (default 33554432 = 32 MiB)\n"
               "  --factor K           size multiplier per step (default 2; use 1 for linear)\n"
               "  --warmup W           warmup iterations per size (default 10)\n"
               "  --iters I            timed iterations per size (default 100)\n"
               "  --methods LIST       subset of: send,isend,bsend,ssend (default: all)\n"
               "  --only-rank R        only test pairs involving rank R (default: all pairs)\n"
               "  --outfile PATH       CSV output file (default mpi_pair_bench.csv)\n";
      }
      MPI_Finalize();
      std::exit(0);
    } else {
      die(rank, "unknown argument: " + a);
    }
  }

  if (!methods_specified) {
    opt.methods = {Method::SEND, Method::ISEND, Method::BSEND, Method::SSEND};
  }
  if (opt.min_bytes == 0) die(rank, "--min-bytes must be >= 1");
  if (opt.max_bytes < opt.min_bytes) die(rank, "--max-bytes must be >= --min-bytes");
  if (opt.max_bytes > INT_MAX) die(rank, "--max-bytes must be <= INT_MAX");
  if (opt.factor < 1) die(rank, "--factor must be >= 1");
  if (opt.iters <= 0 || opt.warmup < 0) die(rank, "iterations must be positive");
  return opt;
}

// Measure one direction using the unified pattern:
// sender:   for i: send(); recv();   then if isend -> Waitall
// receiver: for i: recv(); send();   then if isend -> Waitall
static PingResult ping_once(int sender, int receiver, int me, std::size_t bytes, int warmup,
                            int iters, std::vector<char> &buf, Method method) {
  const int tag = 42424;

  // ---- Warmup (no timing) ----
  if (warmup > 0) {
    if (me == sender) {
      std::vector<MPI_Request> sreq;
      sreq.reserve(method == Method::ISEND ? warmup : 0);
      for (int w = 0; w < warmup; ++w) {
        if (method == Method::ISEND) {
          MPI_Request r;
          MPI_Isend(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD, &r);
          sreq.push_back(r);
        } else if (method == Method::SEND) {
          MPI_Send(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD);
        } else if (method == Method::BSEND) {
          MPI_Bsend(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD);
        } else /* SSEND */ {
          MPI_Ssend(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD);
        }
        MPI_Recv(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
      }
      if (method == Method::ISEND && !sreq.empty())
        MPI_Waitall((int)sreq.size(), sreq.data(), MPI_STATUSES_IGNORE);
    } else if (me == receiver) {
      std::vector<MPI_Request> sreq;
      sreq.reserve(method == Method::ISEND ? warmup : 0);
      for (int w = 0; w < warmup; ++w) {
        MPI_Recv(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (method == Method::ISEND) {
          MPI_Request r;
          MPI_Isend(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD, &r);
          sreq.push_back(r);
        } else if (method == Method::SEND) {
          MPI_Send(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD);
        } else if (method == Method::BSEND) {
          MPI_Bsend(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD);
        } else /* SSEND */ {
          MPI_Ssend(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD);
        }
      }
      if (method == Method::ISEND && !sreq.empty())
        MPI_Waitall((int)sreq.size(), sreq.data(), MPI_STATUSES_IGNORE);
    }
  }

  // ---- Timed phase ----
  PingResult res{-1.0, -1.0};
  if (me == sender) {
    std::vector<MPI_Request> sreq;
    sreq.reserve(method == Method::ISEND ? iters : 0);
    double send_call_total = 0.0;

    double t0 = MPI_Wtime();
    for (int i = 0; i < iters; ++i) {
      if (method == Method::ISEND) {
        double c0 = MPI_Wtime();
        MPI_Request r;
        MPI_Isend(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD, &r);
        double c1 = MPI_Wtime();
        send_call_total += (c1 - c0);
        sreq.push_back(r);
      } else {
        double c0 = MPI_Wtime();
        if (method == Method::SEND)
          MPI_Send(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD);
        else if (method == Method::BSEND)
          MPI_Bsend(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD);
        else /* SSEND */
          MPI_Ssend(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD);
        double c1 = MPI_Wtime();
        send_call_total += (c1 - c0);
      }

      MPI_Recv(buf.data(), (int)bytes, MPI_CHAR, receiver, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (method == Method::ISEND && !sreq.empty())
      MPI_Waitall((int)sreq.size(), sreq.data(), MPI_STATUSES_IGNORE);
    double t1 = MPI_Wtime();

    res.avg_rtt_s = (t1 - t0) / (double)iters;
    res.send_call_total_s = send_call_total;
  } else if (me == receiver) {
    std::vector<MPI_Request> sreq;
    sreq.reserve(method == Method::ISEND ? iters : 0);
    for (int i = 0; i < iters; ++i) {
      MPI_Recv(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      if (method == Method::ISEND) {
        MPI_Request r;
        MPI_Isend(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD, &r);
        sreq.push_back(r);
      } else {
        if (method == Method::SEND)
          MPI_Send(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD);
        else if (method == Method::BSEND)
          MPI_Bsend(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD);
        else /* SSEND */
          MPI_Ssend(buf.data(), (int)bytes, MPI_CHAR, sender, tag, MPI_COMM_WORLD);
      }
    }
    if (method == Method::ISEND && !sreq.empty())
      MPI_Waitall((int)sreq.size(), sreq.data(), MPI_STATUSES_IGNORE);
  }
  return res;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int world, me;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  if (world < 2) {
    if (me == 0) std::cerr << "Run with at least 2 ranks.\n";
    MPI_Finalize();
    return 1;
  }

  Options opt = parse_args(argc, argv, me);

  // Gather processor names for locality classification
  char myname[MPI_MAX_PROCESSOR_NAME] = {};
  int mylen = 0;
  MPI_Get_processor_name(myname, &mylen);
  std::vector<char> allnames(world * MPI_MAX_PROCESSOR_NAME, 0);
  MPI_Allgather(myname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, allnames.data(), MPI_MAX_PROCESSOR_NAME,
                MPI_CHAR, MPI_COMM_WORLD);
  auto rank_name = [&](int r) -> std::string {
    const char *p = &allnames[r * MPI_MAX_PROCESSOR_NAME];
    return std::string(p);  // buffer is zero-padded
  };

  // Prepare message sizes
  std::vector<std::size_t> sizes;
  {
    std::size_t s = opt.min_bytes;
    if (opt.factor == 1) {
      for (; s <= opt.max_bytes; ++s) sizes.push_back(s);
    } else {
      while (s <= opt.max_bytes) {
        sizes.push_back(s);
        if (s > opt.max_bytes / (std::size_t)opt.factor) break;
        s *= (std::size_t)opt.factor;
      }
    }
  }

  // Reusable buffer
  std::vector<char> buffer(opt.max_bytes, 0);

  // Attach a Bsend buffer if BSEND is in use (supports one outstanding bsend at a time)
  std::vector<char> bsend_storage;
  bool have_bsend =
      std::find(opt.methods.begin(), opt.methods.end(), Method::BSEND) != opt.methods.end();
  if (have_bsend) {
    int pack_max = 0;
    MPI_Pack_size((int)opt.max_bytes, MPI_CHAR, MPI_COMM_WORLD, &pack_max);
    int bsz = pack_max + MPI_BSEND_OVERHEAD;
    bsend_storage.resize((size_t)bsz);
    if (MPI_Buffer_attach(bsend_storage.data(), bsz) != MPI_SUCCESS) {
      die(me, "MPI_Buffer_attach failed");
    }
  }

  // CSV accumulation (only lower rank logs)
  std::ostringstream local_csv;
  auto add_line = [&](int src, int dst, const char *direction, const char *locality,
                      std::size_t bytes, int iters, const PingResult &res, Method method) {
    double latency_s = res.avg_rtt_s / 2.0;
    double bw_MBps = (2.0 * (double)bytes / res.avg_rtt_s) / 1.0e6;  // MB/s (1e6)
    local_csv << src << ',' << dst << ',' << method_name(method) << ',' << direction << ','
              << locality << ',' << bytes << ',' << iters << ',' << std::setprecision(12)
              << res.avg_rtt_s << ',' << std::setprecision(12) << latency_s << ','
              << std::setprecision(12) << bw_MBps << ',' << std::setprecision(12)
              << res.send_call_total_s << '\n';
  };

  auto pair_is_enabled = [&](int a, int b) -> bool {
    if (opt.only_rank < 0) return true;
    return (a == opt.only_rank) || (b == opt.only_rank);
  };

  const int TAG_B_TO_A_RESULT = 88001;

  // Main sweep: pairs × sizes × methods
  for (int a = 0; a < world; ++a) {
    for (int b = a + 1; b < world; ++b) {
      if (!pair_is_enabled(a, b)) continue;

      const bool same_node = (rank_name(a) == rank_name(b));
      const char *locality = same_node ? "intranode" : "internode";

      for (std::size_t bytes : sizes) {
        for (Method m : opt.methods) {
          // a->b
          MPI_Barrier(MPI_COMM_WORLD);
          PingResult rtt_ab = ping_once(a, b, me, bytes, opt.warmup, opt.iters, buffer, m);

          // b->a
          MPI_Barrier(MPI_COMM_WORLD);
          PingResult rtt_b_to_a = ping_once(b, a, me, bytes, opt.warmup, opt.iters, buffer, m);

          // Ship b->a sender's measurement to logger (rank a)
          if (me == b) {
            double payload[2] = {rtt_b_to_a.avg_rtt_s, rtt_b_to_a.send_call_total_s};
            MPI_Send(payload, 2, MPI_DOUBLE, a, TAG_B_TO_A_RESULT, MPI_COMM_WORLD);
          }

          if (me == a) {
            double payload[2];
            MPI_Recv(payload, 2, MPI_DOUBLE, b, TAG_B_TO_A_RESULT, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            PingResult rtt_b_to_a_from_b{payload[0], payload[1]};
            add_line(a, b, "a->b", locality, bytes, opt.iters, rtt_ab, m);  // measured by a
            add_line(a, b, "b->a", locality, bytes, opt.iters, rtt_b_to_a_from_b,
                     m);  // measured by b
          }

          MPI_Barrier(MPI_COMM_WORLD);
        }
      }
    }
  }

  if (have_bsend) {
    void *bufptr = nullptr;
    int size = 0;
    MPI_Buffer_detach(&bufptr, &size);
  }

  // Gather CSV chunks to rank 0
  std::string chunk = local_csv.str();
  long long local_len = (long long)chunk.size();
  std::vector<long long> all_lens(world, 0);
  MPI_Gather(&local_len, 1, MPI_LONG_LONG, all_lens.data(), 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  std::vector<int> recvcounts, displs;
  std::vector<char> recvbuf;
  if (me == 0) {
    recvcounts.resize(world);
    displs.resize(world);
    int offset = 0;
    for (int r = 0; r < world; ++r) {
      if (all_lens[r] > INT_MAX) {
        die(0, "CSV output too large for MPI_Gatherv");
      }
      recvcounts[r] = (int)all_lens[r];
      displs[r] = offset;
      offset += recvcounts[r];
    }
    recvbuf.resize(offset);
  }

  MPI_Gatherv(chunk.data(), (int)local_len, MPI_CHAR, recvbuf.data(), recvcounts.data(),
              displs.data(), MPI_CHAR, 0, MPI_COMM_WORLD);

  if (me == 0) {
    std::string_view header =
        "src_rank,dst_rank,method,direction,locality,msg_bytes,iters,avg_rtt_seconds,latency_"
        "seconds,bandwidth_MBps,send_call_total_seconds\n";
    FILE *fp = std::fopen(opt.outfile.c_str(), "wb");
    if (!fp) {
      std::cerr << "Failed to open output file: " << opt.outfile << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    bool write_error = false;
    if (std::fwrite(header.data(), 1, header.size(), fp) != header.size()) {
      std::cerr << "Failed to write header to " << opt.outfile << std::endl;
      write_error = true;
    }
    if (!write_error && !recvbuf.empty()) {
      if (std::fwrite(recvbuf.data(), 1, recvbuf.size(), fp) != recvbuf.size()) {
        std::cerr << "Failed to write data to " << opt.outfile << std::endl;
        write_error = true;
      }
    }
    if (!write_error && std::fflush(fp) != 0) {
      std::cerr << "Failed to flush " << opt.outfile << std::endl;
      write_error = true;
    }
    if (std::fclose(fp) != 0) {
      std::cerr << "Failed to close " << opt.outfile << std::endl;
      write_error = true;
    }
    if (write_error) {
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    std::cout << "Wrote results to " << opt.outfile << std::endl;
  }

  MPI_Finalize();
  return 0;
}
