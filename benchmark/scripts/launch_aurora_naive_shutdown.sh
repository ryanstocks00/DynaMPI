#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage:
#   qsub --nodes=512 --time=00:15:00 launch_aurora_naive_shutdown.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/naive_shutdown_time}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="aurora"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512 1024 2048}"
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
CSV="${OUTPUT_DIR}/naive_shutdown_${SYSTEM}.csv"

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
    echo "Running ${SYSTEM} nodes=${nodes} ranks_per_node=${ranks_per_node}"
    launcher_base="$(basename "${LAUNCHER}")"
    if [[ "${launcher_base}" == mpiexec || "${launcher_base}" == mpirun ]]; then
      "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -n "${total_ranks}" --ppn "${ranks_per_node}" \
        "${APP}" \
        --nodes "${nodes}" \
        --system "${SYSTEM}" \
        --output "${CSV}"
    else
      "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -N "${nodes}" -n "${total_ranks}" \
        --ntasks-per-node="${ranks_per_node}" \
        "${APP}" \
        --nodes "${nodes}" \
        --system "${SYSTEM}" \
        --output "${CSV}"
    fi
  done
done

echo "Results written to ${CSV}"
