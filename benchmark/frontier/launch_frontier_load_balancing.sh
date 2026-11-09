#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Sweeps the load_balancing driver, which times each distributor's wall-clock
# drain of a fixed-size batch (tasks_per_worker * workers tasks) over a range
# of tasks_per_worker values.
#
# Example usage:
#   sbatch --nodes=128 --time=02:00:00 launch_frontier_load_balancing.sh
#   NODE_LIST=128 TASK_US_LIST="1000" DURATION_MODES="uniform" \
#     ./benchmark/frontier/launch_frontier_load_balancing.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="${SYSTEM:-frontier}"
CSV_SUFFIX="${CSV_SUFFIX:-}"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-128}"
IFS=' ' read -r -a RANKS_PER_NODE_LIST <<< "${RANKS_PER_NODE_LIST:-9}"
IFS=' ' read -r -a TASK_US_LIST <<< "${TASK_US_LIST:-1000 10000 100000}"
# fixed / uniform / lognormal, all with mean expected_us.
IFS=' ' read -r -a DURATION_MODES <<< "${DURATION_MODES:-fixed uniform lognormal}"
# Parallel to TASK_US_LIST: either one value for every task size, or one per
# entry. Long tasks want a smaller k so the sweep still fits its walltime --
# e.g. TASK_US_LIST="1000 10000 100000" MAX_TASKS_PER_WORKER_LIST="20 20 10".
IFS=' ' read -r -a MAX_TASKS_PER_WORKER_LIST <<< \
  "${MAX_TASKS_PER_WORKER_LIST:-${MAX_TASKS_PER_WORKER:-20}}"
MIN_TASKS_PER_WORKER="${MIN_TASKS_PER_WORKER:-1}"
REPEATS="${REPEATS:-3}"
# Comma-separated, or "all", or "adlb".
DISTRIBUTIONS="${DISTRIBUTIONS:-all}"
# ADLB server ranks; each must divide the node count, empty takes the binary's
# default. Space-separated sweeps one run per count.
IFS=' ' read -r -a ADLB_SERVER_LIST <<< "${ADLB_SERVERS:-}"
if [[ "${#ADLB_SERVER_LIST[@]}" -eq 0 ]]; then
  ADLB_SERVER_LIST=("")
fi
# Who publishes: 1 the master alone, -1 every worker publishing its own share.
# Only those two; "1 -1" sweeps both, putting both ADLB series on one figure.
IFS=' ' read -r -a ADLB_PRODUCER_LIST <<< "${ADLB_PRODUCERS:-}"
if [[ "${#ADLB_PRODUCER_LIST[@]}" -eq 0 ]]; then
  ADLB_PRODUCER_LIST=("")
fi
# --producers 1 only: spread round-robins a soft target over every server;
# local is lb's default, all on the master's own. Space separated sweeps both.
IFS=' ' read -r -a ADLB_ROUTING_LIST <<< "${ADLB_PUT_ROUTING:-}"
if [[ "${#ADLB_ROUTING_LIST[@]}" -eq 0 ]]; then
  ADLB_ROUTING_LIST=("")
fi
# master (default) targets results at the master; local puts them on the
# worker's own server and lets the master's server steal them in batches.
IFS=' ' read -r -a ADLB_RESULT_LIST <<< "${ADLB_RESULT_ROUTING:-}"
if [[ "${#ADLB_RESULT_LIST[@]}" -eq 0 ]]; then
  ADLB_RESULT_LIST=("")
fi
# Seconds between a starved server's steal attempts, overriding lb's
# compile-time 0.05; backoffs.c's own comment derives 0.0005.
IFS=' ' read -r -a ADLB_STEAL_LIST <<< "${ADLB_STEAL_RATE_LIMIT:-}"
if [[ "${#ADLB_STEAL_LIST[@]}" -eq 0 ]]; then
  ADLB_STEAL_LIST=("")
fi
# Idle time before ADLB's servers declare the problem exhausted. Reached once,
# after the last batch, so it is paid per run rather than per batch.
ADLB_EXHAUST_TIME="${ADLB_EXHAUST_TIME:-1.0}"
TASK_DURATION_CV="${TASK_DURATION_CV:-1.0}"
# Negative leaves the library default in place for each of these.
MAX_UPPER_FANOUT="${MAX_UPPER_FANOUT:--1}"
PIPELINE_DEPTH="${PIPELINE_DEPTH:--1}"
MAX_PENDING_ROUNDS="${MAX_PENDING_ROUNDS:--1}"
# Optional per-run wall-clock guard. A distributor bug can hang the whole
# allocation otherwise; empty (the default) means no timeout.
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-}"

# ADLB is a separate binary but takes the same flags and writes the same CSV.
if [[ -z "${APP:-}" ]]; then
  if [[ "${DISTRIBUTIONS}" == "adlb" ]]; then
    APP="${BUILD_DIR}/benchmark/load_balancing_adlb"
  else
    APP="${BUILD_DIR}/benchmark/load_balancing"
  fi
fi
if [[ "${DISTRIBUTIONS}" == *adlb* && "${DISTRIBUTIONS}" != "adlb" ]]; then
  echo "DISTRIBUTIONS=adlb must be on its own: ADLB cannot share a run with the" \
       "DynaMPI distributors." >&2
  exit 1
fi

if [[ "${#MAX_TASKS_PER_WORKER_LIST[@]}" -ne 1 &&
      "${#MAX_TASKS_PER_WORKER_LIST[@]}" -ne "${#TASK_US_LIST[@]}" ]]; then
  echo "MAX_TASKS_PER_WORKER_LIST must have 1 entry or one per TASK_US_LIST entry" \
       "(got ${#MAX_TASKS_PER_WORKER_LIST[@]} for ${#TASK_US_LIST[@]} task sizes)." >&2
  exit 1
fi

LAUNCHER="${LAUNCHER:-}"
IFS=' ' read -r -a LAUNCHER_ARGS <<< "${LAUNCHER_ARGS:-}"
if [[ -z "${LAUNCHER}" ]]; then
  if command -v srun >/dev/null 2>&1; then
    LAUNCHER="srun"
  elif command -v mpiexec >/dev/null 2>&1; then
    LAUNCHER="mpiexec"
  elif command -v mpirun >/dev/null 2>&1; then
    LAUNCHER="mpirun"
  else
    echo "No launcher found. Install srun, mpiexec, or mpirun." >&2
    exit 1
  fi
fi

if [[ ! -x "${APP}" ]]; then
  echo "Benchmark binary not found or not executable: ${APP}" >&2
  echo "Build it with: cmake -B build -DDYNAMPI_BUILD_BENCHMARKS=ON && cmake --build build -j" >&2
  if [[ "${DISTRIBUTIONS}" == "adlb" ]]; then
    echo "load_balancing_adlb additionally needs -DADLB_ROOT=<prefix> -DCUTILS_ROOT=<prefix>" \
         "pointing at a build of ADLB/X (swift-t's lb) and ExM c-utils." >&2
  fi
  exit 1
fi

export ADLB_EXHAUST_TIME

mkdir -p "${OUTPUT_DIR}"
CSV="${OUTPUT_DIR}/load_balancing_${SYSTEM}${CSV_SUFFIX}.csv"

failures=0
for nodes in "${NODE_LIST[@]}"; do
  for ranks_per_node in "${RANKS_PER_NODE_LIST[@]}"; do
    total_ranks=$((nodes * ranks_per_node))
    for i in "${!TASK_US_LIST[@]}"; do
      expected_us="${TASK_US_LIST[${i}]}"
      if [[ "${#MAX_TASKS_PER_WORKER_LIST[@]}" -eq 1 ]]; then
        max_k="${MAX_TASKS_PER_WORKER_LIST[0]}"
      else
        max_k="${MAX_TASKS_PER_WORKER_LIST[${i}]}"
      fi
      for duration_mode in "${DURATION_MODES[@]}"; do
        for adlb_servers in "${ADLB_SERVER_LIST[@]}"; do
         for adlb_producers in "${ADLB_PRODUCER_LIST[@]}"; do
          for adlb_routing in "${ADLB_ROUTING_LIST[@]}"; do
           for adlb_steal in "${ADLB_STEAL_LIST[@]}"; do
            for adlb_result in "${ADLB_RESULT_LIST[@]}"; do
          echo "Running ${SYSTEM} nodes=${nodes} ranks_per_node=${ranks_per_node}" \
               "dist=${DISTRIBUTIONS} duration_mode=${duration_mode}" \
               "expected_us=${expected_us} k=${MIN_TASKS_PER_WORKER}-${max_k}" \
               "${adlb_servers:+adlb_servers=${adlb_servers}}" \
               "${adlb_producers:+adlb_producers=${adlb_producers}}" \
               "${adlb_routing:+put_routing=${adlb_routing}}" \
               "${adlb_steal:+steal_rate_limit=${adlb_steal}}" \
               "${adlb_result:+result_routing=${adlb_result}}"
          run_cmd=()
          if [[ -n "${RUN_TIMEOUT_S}" ]]; then
            run_cmd=(timeout "${RUN_TIMEOUT_S}")
          fi
          launcher_base="$(basename "${LAUNCHER}")"
          if [[ "${launcher_base}" == mpiexec || "${launcher_base}" == mpirun ]]; then
            run_cmd+=("${LAUNCHER}" ${LAUNCHER_ARGS[@]+"${LAUNCHER_ARGS[@]}"}
                      -n "${total_ranks}" --ppn "${ranks_per_node}")
          else
            run_cmd+=("${LAUNCHER}" ${LAUNCHER_ARGS[@]+"${LAUNCHER_ARGS[@]}"}
                      -N "${nodes}" -n "${total_ranks}" --ntasks-per-node="${ranks_per_node}")
          fi
          servers_arg=()
          if [[ "${DISTRIBUTIONS}" == "adlb" && -n "${adlb_servers}" ]]; then
            servers_arg=(--servers "${adlb_servers}")
          fi
          producers_arg=()
          if [[ "${DISTRIBUTIONS}" == "adlb" && -n "${adlb_producers}" ]]; then
            producers_arg=(--producers "${adlb_producers}")
          fi
          if [[ "${DISTRIBUTIONS}" == "adlb" && -n "${adlb_routing}" ]]; then
            producers_arg+=(--put_routing "${adlb_routing}")
          fi
          if [[ "${DISTRIBUTIONS}" == "adlb" && -n "${adlb_steal}" ]]; then
            producers_arg+=(--steal_rate_limit "${adlb_steal}")
          fi
          if [[ "${DISTRIBUTIONS}" == "adlb" && -n "${adlb_result}" ]]; then
            producers_arg+=(--result_routing "${adlb_result}")
          fi
          run_cmd+=("${APP}"
                    --distribution "${DISTRIBUTIONS}"
                    --expected_us "${expected_us}"
                    --duration_mode "${duration_mode}"
                    --task_duration_cv "${TASK_DURATION_CV}"
                    --min_tasks_per_worker "${MIN_TASKS_PER_WORKER}"
                    --max_tasks_per_worker "${max_k}"
                    --repeats "${REPEATS}"
                    --max_upper_fanout "${MAX_UPPER_FANOUT}"
                    --pipeline_depth "${PIPELINE_DEPTH}"
                    --max_pending_rounds "${MAX_PENDING_ROUNDS}"
                    --nodes "${nodes}"
                    ${servers_arg[@]+"${servers_arg[@]}"}
                    ${producers_arg[@]+"${producers_arg[@]}"}
                    --system "${SYSTEM}"
                    --output "${CSV}")
          # Unlike weak_scaling_distribution_rate, load_balancing finalizes and
          # returns 0 on success, so a non-zero exit is a real failure. Keep
          # going anyway -- one bad combo shouldn't discard the rest of a sweep
          # that already holds the allocation -- but report it at the end.
          if ! "${run_cmd[@]}"; then
            echo "FAILED nodes=${nodes} ranks_per_node=${ranks_per_node}" \
                 "expected_us=${expected_us} duration_mode=${duration_mode}" \
                 "${adlb_servers:+adlb_servers=${adlb_servers}}" \
                 "${adlb_producers:+adlb_producers=${adlb_producers}}" \
                 "${adlb_routing:+put_routing=${adlb_routing}}" \
                 "${adlb_steal:+steal_rate_limit=${adlb_steal}}" \
                 "${adlb_result:+result_routing=${adlb_result}}" >&2
            failures=$((failures + 1))
          fi
            done
           done
          done
         done
        done
      done
    done
  done
done

if [[ "${failures}" -gt 0 ]]; then
  echo "${failures} combination(s) failed; see above." >&2
  exit 1
fi
echo "Wrote ${CSV}"
