#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage:
#   ./benchmark/scripts/launch_local_strong_scaling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/strong_scaling_distribution_rate}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="local"

IFS=' ' read -r -a RANK_LIST <<< "${RANK_LIST:-1 2 4 8 12}"
IFS=' ' read -r -a TASK_US_LIST <<< "${TASK_US_LIST:-1 10 100 1000 10000 100000 1000000}"
IFS=' ' read -r -a DISTRIBUTIONS <<< "${DISTRIBUTIONS:-naive hierarchical}"
IFS=' ' read -r -a MODES <<< "${MODES:-fixed random}"
DURATION_S="${DURATION_S:-10}"
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
CSV="${OUTPUT_DIR}/strong_scaling_${SYSTEM}.csv"

for ranks in "${RANK_LIST[@]}"; do
  for dist in "${DISTRIBUTIONS[@]}"; do
    for mode in "${MODES[@]}"; do
      for expected_us in "${TASK_US_LIST[@]}"; do
        echo "Running ${SYSTEM} ranks=${ranks} dist=${dist} mode=${mode} expected_us=${expected_us}"
        launcher_base="$(basename "${LAUNCHER}")"
        if [[ "${launcher_base}" == mpiexec ]]; then
          "${LAUNCHER}" ${LAUNCHER_ARGS[@]+"${LAUNCHER_ARGS[@]}"} -n "${ranks}" \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_us "${expected_us}" \
            --duration_s "${DURATION_S}" \
            --nodes 1 \
            --system "${SYSTEM}" \
            --output "${CSV}"
        else
          "${LAUNCHER}" ${LAUNCHER_ARGS[@]+"${LAUNCHER_ARGS[@]}"} -np "${ranks}" \
            "${APP}" \
            --distribution "${dist}" \
            --mode "${mode}" \
            --expected_us "${expected_us}" \
            --duration_s "${DURATION_S}" \
            --nodes 1 \
            --system "${SYSTEM}" \
            --output "${CSV}"
        fi
      done
    done
  done
done

echo "Results written to ${CSV}"
