#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage:
#   sbatch --nodes=2048 --time=02:00:00 launch_frontier_weak_scaling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/weak_scaling_distribution_rate}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="frontier"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512 1024 2048}"
IFS=' ' read -r -a TASK_US_LIST <<< "${TASK_US_LIST:-10 100 1000 10000 100000 1000000}"
IFS=' ' read -r -a DISTRIBUTIONS <<< \
  "${DISTRIBUTIONS:-naive hierarchical lockfree_rma hierarchical_lockfree_rma}"
# Paper workload is uniform on [0, 2T]. Override with MODES="uniform fixed".
IFS=' ' read -r -a MODES <<< "${MODES:-uniform}"
DURATION_S="${DURATION_S:-20}"
# Hierarchical distributors only. Negative = auto; 0 = one unbounded manager
# level. Extra values sweep fan-out; other distributors ignore this list.
IFS=' ' read -r -a MAX_UPPER_FANOUT_LIST <<< "${MAX_UPPER_FANOUT_LIST:-${MAX_UPPER_FANOUT:--1}}"
# RMA ring capacity and this benchmark's publication budget. Empty keeps the
# 500M default (~19 GiB), enough for a 20 s window.
MAX_TASKS="${MAX_TASKS:-}"
MAX_TASKS_ARGS=()
if [[ -n "${MAX_TASKS}" ]]; then
  MAX_TASKS_ARGS=(--max_tasks "${MAX_TASKS}")
fi
IFS=' ' read -r -a RANKS_PER_NODE_LIST <<< "${RANKS_PER_NODE_LIST:-9}"
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

export FI_CXI_RX_MATCH_MODE="${FI_CXI_RX_MATCH_MODE:-software}"

mkdir -p "${OUTPUT_DIR}"
CSV="${OUTPUT_DIR}/weak_scaling_${SYSTEM}.csv"

for nodes in "${NODE_LIST[@]}"; do
  for rpn in "${RANKS_PER_NODE_LIST[@]}"; do
    if [[ "${rpn}" == "core" || "${rpn}" == "cores" ]]; then
      if [[ -n "${SLURM_JOB_CPUS_PER_NODE:-}" ]]; then
        ranks_per_node="${SLURM_JOB_CPUS_PER_NODE%%(*}"
        ranks_per_node="${ranks_per_node%%,*}"
      else
        ranks_per_node="${CORES_PER_NODE:-56}"
      fi
    else
      ranks_per_node="${rpn}"
    fi
    total_ranks=$((nodes * ranks_per_node))
    for dist in "${DISTRIBUTIONS[@]}"; do
      # Only hierarchical and hierarchical_lockfree_rma's behavior
      # depends on max_upper_fanout; every other distributor would just
      # repeat identical runs, so collapse its fanout list down to one (the
      # first) value.
      if [[ "${dist}" == "hierarchical_lockfree_rma" || "${dist}" == "hierarchical" ]]; then
        fanouts=("${MAX_UPPER_FANOUT_LIST[@]}")
      else
        fanouts=("${MAX_UPPER_FANOUT_LIST[0]}")
      fi
      for fanout in "${fanouts[@]}"; do
      for mode in "${MODES[@]}"; do
      for expected_us in "${TASK_US_LIST[@]}"; do
        echo "Running ${SYSTEM} nodes=${nodes} ranks_per_node=${ranks_per_node} dist=${dist} mode=${mode} expected_us=${expected_us} max_upper_fanout=${fanout}"
        launcher_base="$(basename "${LAUNCHER}")"
        # `|| true`: weak_scaling_distribution_rate prints its RESULT line
        # and then calls MPI_Abort itself instead of returning/finalizing
        # normally, so every run -- success or not -- exits non-zero. Without
        # `|| true`, this script's `set -e` would kill the whole node-count's
        # remaining combos after just the first one (see
        # launch_aurora_weak_scaling.sh for the same fix).
        if [[ "${launcher_base}" == mpiexec || "${launcher_base}" == mpirun ]]; then
          "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -n "${total_ranks}" --ppn "${ranks_per_node}" \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_us "${expected_us}" \
            --duration_s "${DURATION_S}" \
            --nodes "${nodes}" \
            --system "${SYSTEM}" \
            --max_upper_fanout "${fanout}" \
            ${MAX_TASKS_ARGS[@]+"${MAX_TASKS_ARGS[@]}"} \
            --output "${CSV}" || true
        else
          "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -N "${nodes}" -n "${total_ranks}" \
            --ntasks-per-node="${ranks_per_node}" \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_us "${expected_us}" \
            --duration_s "${DURATION_S}" \
            --nodes "${nodes}" \
            --system "${SYSTEM}" \
            --max_upper_fanout "${fanout}" \
            ${MAX_TASKS_ARGS[@]+"${MAX_TASKS_ARGS[@]}"} \
            --output "${CSV}" || true
        fi
        done
      done
      done
    done
  done
done
