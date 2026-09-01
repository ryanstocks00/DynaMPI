#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage:
#   ./benchmark/scripts/launch_local_weak_scaling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/weak_scaling_distribution_rate}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="local"

IFS=' ' read -r -a RANK_LIST <<< "${RANK_LIST:-1 2 4 8 12}"
IFS=' ' read -r -a TASK_US_LIST <<< "${TASK_US_LIST:-10 100 1000 10000 100000 1000000}"
IFS=' ' read -r -a DISTRIBUTIONS <<< \
  "${DISTRIBUTIONS:-naive hierarchical lockfree_rma hierarchical_lockfree_rma}"
# Paper workload is uniform on [0, 2T]. Override with MODES="uniform fixed".
IFS=' ' read -r -a MODES <<< "${MODES:-uniform}"
DURATION_S="${DURATION_S:-20}"
# Smaller than the compute-node 500M default: 25M slots (~1 GiB) cover the
# default 12-rank, 20 s local sweep. Raise MAX_TASKS for larger RANK_LIST.
MAX_TASKS="${MAX_TASKS:-25000000}"
MAX_TASKS_ARGS=()
if [[ -n "${MAX_TASKS}" ]]; then
  MAX_TASKS_ARGS=(--max_tasks "${MAX_TASKS}")
fi
LAUNCHER="${LAUNCHER:-}"
IFS=' ' read -r -a LAUNCHER_ARGS <<< "${LAUNCHER_ARGS:-}"

if [[ -z "${LAUNCHER}" ]]; then
  if command -v mpirun >/dev/null 2>&1; then
    LAUNCHER="mpirun"
  elif command -v mpiexec >/dev/null 2>&1; then
    LAUNCHER="mpiexec"
  else
    echo "No launcher found. Install mpirun or mpiexec." >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"
CSV="${OUTPUT_DIR}/weak_scaling_${SYSTEM}.csv"

for ranks in "${RANK_LIST[@]}"; do
  for dist in "${DISTRIBUTIONS[@]}"; do
    for mode in "${MODES[@]}"; do
      for expected_us in "${TASK_US_LIST[@]}"; do
        echo "Running ${SYSTEM} ranks=${ranks} dist=${dist} mode=${mode} expected_us=${expected_us}"
        launcher_base="$(basename "${LAUNCHER}")"
        # A successful measurement prints its row and intentionally calls
        # MPI_Abort, so launcher exit status is not a benchmark success signal.
        if [[ "${launcher_base}" == mpiexec ]]; then
          "${LAUNCHER}" ${LAUNCHER_ARGS[@]+"${LAUNCHER_ARGS[@]}"} -n "${ranks}" \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_us "${expected_us}" \
            --duration_s "${DURATION_S}" \
            --nodes 1 \
            --system "${SYSTEM}" \
            ${MAX_TASKS_ARGS[@]+"${MAX_TASKS_ARGS[@]}"} \
            --output "${CSV}" || true
        else
          "${LAUNCHER}" ${LAUNCHER_ARGS[@]+"${LAUNCHER_ARGS[@]}"} -np "${ranks}" \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_us "${expected_us}" \
            --duration_s "${DURATION_S}" \
            --nodes 1 \
            --system "${SYSTEM}" \
            ${MAX_TASKS_ARGS[@]+"${MAX_TASKS_ARGS[@]}"} \
            --output "${CSV}" || true
        fi
      done
    done
  done
done

echo "Results written to ${CSV}"
