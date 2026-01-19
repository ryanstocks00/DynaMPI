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

NODE_LIST=(${NODE_LIST:-1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8096})
TASK_US_LIST=(${TASK_US_LIST:-1 10 100 1000 10000 100000 1000000})
DISTRIBUTIONS=(${DISTRIBUTIONS:-naive hierarchical})
MODES=(${MODES:-fixed poisson})
DURATION_S="${DURATION_S:-60}"
BUNDLE_TARGET_MS="${BUNDLE_TARGET_MS:-10}"
LAUNCHER="${LAUNCHER:-srun}"

mkdir -p "${OUTPUT_DIR}"
CSV="${OUTPUT_DIR}/strong_scaling_${SYSTEM}.csv"

for nodes in "${NODE_LIST[@]}"; do
  for dist in "${DISTRIBUTIONS[@]}"; do
    for mode in "${MODES[@]}"; do
      for expected_us in "${TASK_US_LIST[@]}"; do
        echo "Running ${SYSTEM} nodes=${nodes} dist=${dist} mode=${mode} expected_us=${expected_us}"
        "${LAUNCHER}" -N "${nodes}" -n "${nodes}" --ntasks-per-node=1 \
          "${APP}" \
          --distribution "${dist}" \
          --mode "${mode}" \
          --expected_us "${expected_us}" \
          --duration_s "${DURATION_S}" \
          --bundle_target_ms "${BUNDLE_TARGET_MS}" \
          --nodes "${nodes}" \
          --system "${SYSTEM}" \
          --output "${CSV}"
      done
    done
  done
done
