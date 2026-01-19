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

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8096}"
IFS=' ' read -r -a TASK_US_LIST <<< "${TASK_US_LIST:-1 10 100 1000 10000 100000 1000000}"
IFS=' ' read -r -a DISTRIBUTIONS <<< "${DISTRIBUTIONS:-naive hierarchical}"
IFS=' ' read -r -a MODES <<< "${MODES:-fixed poisson}"
DURATION_S="${DURATION_S:-60}"
BUNDLE_TARGET_MS="${BUNDLE_TARGET_MS:-10}"
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
  for dist in "${DISTRIBUTIONS[@]}"; do
    for mode in "${MODES[@]}"; do
      for expected_us in "${TASK_US_LIST[@]}"; do
        echo "Running ${SYSTEM} nodes=${nodes} dist=${dist} mode=${mode} expected_us=${expected_us}"
        launcher_base="$(basename "${LAUNCHER}")"
        if [[ "${launcher_base}" == mpiexec || "${launcher_base}" == mpirun ]]; then
          "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -n "${nodes}" --ppn 1 \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_us "${expected_us}" \
            --duration_s "${DURATION_S}" \
            --bundle_target_ms "${BUNDLE_TARGET_MS}" \
            --nodes "${nodes}" \
            --system "${SYSTEM}" \
            --output "${CSV}"
        else
          "${LAUNCHER}" "${LAUNCHER_ARGS[@]}" -N "${nodes}" -n "${nodes}" --ntasks-per-node=1 \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_us "${expected_us}" \
            --duration_s "${DURATION_S}" \
            --bundle_target_ms "${BUNDLE_TARGET_MS}" \
            --nodes "${nodes}" \
            --system "${SYSTEM}" \
            --output "${CSV}"
        fi
      done
    done
  done
done
