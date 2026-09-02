#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage:
#   sbatch --nodes=512 --time=00:15:00 launch_frontier_naive_shutdown.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/shutdown_time}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="frontier"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192}"
IFS=' ' read -r -a DISTRIBUTIONS <<< "${DISTRIBUTIONS:-naive hierarchical}"
# hierarchical and hierarchical_lockfree_rma only: see the matching
# comment in launch_frontier_weak_scaling.sh / launch_aurora_weak_scaling.sh.
IFS=' ' read -r -a MAX_UPPER_FANOUT_LIST <<< "${MAX_UPPER_FANOUT_LIST:-${MAX_UPPER_FANOUT:--1}}"
IFS=' ' read -r -a RANKS_PER_NODE_LIST <<< "${RANKS_PER_NODE_LIST:-core}"
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

mkdir -p "${OUTPUT_DIR}"
CSV="${OUTPUT_DIR}/shutdown_${SYSTEM}.csv"

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
      if [[ "${dist}" == "hierarchical_lockfree_rma" || "${dist}" == "hierarchical" ]]; then
        fanouts=("${MAX_UPPER_FANOUT_LIST[@]}")
      else
        fanouts=("${MAX_UPPER_FANOUT_LIST[0]}")
      fi
      for fanout in "${fanouts[@]}"; do
      echo "Running ${SYSTEM} nodes=${nodes} ranks_per_node=${ranks_per_node} dist=${dist} max_upper_fanout=${fanout}"
      launcher_base="$(basename "${LAUNCHER}")"
      if [[ "${launcher_base}" == mpiexec || "${launcher_base}" == mpirun ]]; then
        "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -n "${total_ranks}" --ppn "${ranks_per_node}" \
          "${APP}" \
          --distribution "${dist}" \
          --nodes "${nodes}" \
          --system "${SYSTEM}" \
          --max_upper_fanout "${fanout}" \
          --output "${CSV}"
      else
        "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -N "${nodes}" -n "${total_ranks}" \
          --ntasks-per-node="${ranks_per_node}" \
          "${APP}" \
          --distribution "${dist}" \
          --nodes "${nodes}" \
          --system "${SYSTEM}" \
          --max_upper_fanout "${fanout}" \
          --output "${CSV}"
      fi
      done
    done
  done
done

echo "Results written to ${CSV}"
