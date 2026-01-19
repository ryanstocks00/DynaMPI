#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage:
#   sbatch --nodes=8096 --time=02:00:00 launch_aurora_strong_scaling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/strong_scaling_distribution_rate}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="aurora"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512}"
TASK_NS_LIST=()
if [[ -n "${TASK_NS_LIST:-}" ]]; then
  IFS=' ' read -r -a TASK_NS_LIST <<< "${TASK_NS_LIST}"
elif [[ -n "${TASK_US_LIST:-}" ]]; then
  IFS=' ' read -r -a TASK_US_LIST <<< "${TASK_US_LIST}"
  for us in "${TASK_US_LIST[@]}"; do
    TASK_NS_LIST+=("$((us * 1000))")
  done
else
  IFS=' ' read -r -a TASK_NS_LIST <<< "10 100 1000 10000 100000 1000000 10000000 100000000 1000000000"
fi
IFS=' ' read -r -a DISTRIBUTIONS <<< "${DISTRIBUTIONS:-naive hierarchical}"
IFS=' ' read -r -a MODES <<< "${MODES:-fixed poisson}"
DURATION_S="${DURATION_S:-10}"
BUNDLE_TARGET_MS="${BUNDLE_TARGET_MS:-10}"
ROUND_TARGET_MS="${ROUND_TARGET_MS:-200}"
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
CSV="${OUTPUT_DIR}/strong_scaling_${SYSTEM}.csv"

for nodes in "${NODE_LIST[@]}"; do
  for rpn in "${RANKS_PER_NODE_LIST[@]}"; do
    if [[ "${rpn}" == "core" || "${rpn}" == "cores" ]]; then
      ranks_per_node="${CORES_PER_NODE:-102}"
    else
      ranks_per_node="${rpn}"
    fi
    total_ranks=$((nodes * ranks_per_node))
    for dist in "${DISTRIBUTIONS[@]}"; do
      for mode in "${MODES[@]}"; do
        for expected_ns in "${TASK_NS_LIST[@]}"; do
          echo "Running ${SYSTEM} nodes=${nodes} ranks_per_node=${ranks_per_node} dist=${dist} mode=${mode} expected_ns=${expected_ns}"
        launcher_base="$(basename "${LAUNCHER}")"
        if [[ "${launcher_base}" == mpiexec || "${launcher_base}" == mpirun ]]; then
          "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -n "${total_ranks}" --ppn "${ranks_per_node}" \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_ns "${expected_ns}" \
            --duration_s "${DURATION_S}" \
            --bundle_target_ms "${BUNDLE_TARGET_MS}" \
            --round_target_ms "${ROUND_TARGET_MS}" \
            --nodes "${nodes}" \
            --system "${SYSTEM}" \
            --output "${CSV}"
        else
          "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -N "${nodes}" -n "${total_ranks}" \
            --ntasks-per-node="${ranks_per_node}" \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_ns "${expected_ns}" \
            --duration_s "${DURATION_S}" \
            --bundle_target_ms "${BUNDLE_TARGET_MS}" \
            --round_target_ms "${ROUND_TARGET_MS}" \
            --nodes "${nodes}" \
            --system "${SYSTEM}" \
            --output "${CSV}"
        fi
        done
      done
    done
  done
done
