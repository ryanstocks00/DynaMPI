#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage (PBS, qsub):
#   qsub -l select=1:ncpus=102:mpiprocs=102 -l walltime=02:00:00 launch_aurora_weak_scaling.sh
# Or use the submit script: ./benchmark/scripts/submit_aurora_weak_scaling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/weak_scaling_distribution_rate}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="aurora"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512}"
IFS=' ' read -r -a TASK_US_LIST <<< "${TASK_US_LIST:-1 10 100 1000 10000 100000 1000000}"
IFS=' ' read -r -a DISTRIBUTIONS <<< "${DISTRIBUTIONS:-naive hierarchical}"
IFS=' ' read -r -a MODES <<< "${MODES:-fixed random}"
DURATION_S="${DURATION_S:-10}"
# hierarchical_lockfree_rma only: forwarded as --max_upper_fanout (ignored by
# every other distributor, which is run exactly once regardless of how many values
# are listed here). Negative (default) = auto, picking a fanout from manager
# count -- see HierarchicalLockFreeRMAWorkDistributor's setup_upper_chain().
# 0 = single unbounded manager level. Set multiple space-separated values to
# sweep hierarchy branching factor within one job.
IFS=' ' read -r -a MAX_UPPER_FANOUT_LIST <<< "${MAX_UPPER_FANOUT_LIST:-${MAX_UPPER_FANOUT:--1}}"
# Lifetime task capacity of each preallocated RMA window (lockfree_rma and
# hierarchical_lockfree_rma only). Empty keeps the binary's default, which is
# sized for a compute node (~19GiB on the manager rank) -- lower it to run the
# RMA distributors where that much memory is not available, at the cost of a
# shorter measured window if the run exhausts the table before duration_s.
MAX_TASKS="${MAX_TASKS:-}"
MAX_TASKS_ARGS=()
if [[ -n "${MAX_TASKS}" ]]; then
  MAX_TASKS_ARGS=(--max_tasks "${MAX_TASKS}")
fi
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

get_allocated_cores_per_node() {
  if [[ -n "${PBS_NCPUS:-}" ]]; then
    echo "${PBS_NCPUS}"
    return
  fi
  if [[ -n "${CORES_PER_NODE:-}" ]]; then
    echo "${CORES_PER_NODE}"
    return
  fi
  if [[ -n "${NCPUS_PER_NODE:-}" ]]; then
    echo "${NCPUS_PER_NODE}"
    return
  fi
  echo 102
}

ALLOC_CORES_PER_NODE="$(get_allocated_cores_per_node)"
echo "Allocated cores per node: ${ALLOC_CORES_PER_NODE}"

export FI_CXI_RX_MATCH_MODE=software

mkdir -p "${OUTPUT_DIR}"
CSV="${OUTPUT_DIR}/weak_scaling_${SYSTEM}.csv"

for nodes in "${NODE_LIST[@]}"; do
  for rpn in "${RANKS_PER_NODE_LIST[@]}"; do
    if [[ "${rpn}" == "core" || "${rpn}" == "cores" ]]; then
      ranks_per_node="${CORES_PER_NODE:-102}"
    else
      ranks_per_node="${rpn}"
    fi
    if [[ "${ranks_per_node}" -gt "${ALLOC_CORES_PER_NODE}" ]]; then
      echo "Requested ranks_per_node=${ranks_per_node} exceeds allocation ${ALLOC_CORES_PER_NODE}" >&2
      exit 1
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
        # `|| true` on both launches below: weak_scaling_distribution_rate
        # now prints its RESULT line and then calls MPI_Abort itself (see
        # print_result_and_abort's comment) instead of returning/finalizing
        # normally, so every run -- success or not -- now exits non-zero.
        # Without `|| true`, this script's `set -e` (line 4) would treat
        # that as a fatal error and kill the whole node-count's remaining
        # combos after just the first one (confirmed: three real jobs each
        # stopped after exactly 1 of 36 combos this way). Exit code is no
        # longer a meaningful success signal for this benchmark either way
        # -- check for a RESULT line in the CSV/log instead.
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
