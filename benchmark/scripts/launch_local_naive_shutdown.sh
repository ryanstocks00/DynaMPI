#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage:
#   ./benchmark/scripts/launch_local_naive_shutdown.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/naive_shutdown_time}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="local"

IFS=' ' read -r -a RANK_LIST <<< "${RANK_LIST:-1 2 4 8 12 16 20 24}"
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
CSV="${OUTPUT_DIR}/naive_shutdown_${SYSTEM}.csv"

for ranks in "${RANK_LIST[@]}"; do
  echo "Running ${SYSTEM} ranks=${ranks}"
  launcher_base="$(basename "${LAUNCHER}")"
  if [[ "${launcher_base}" == mpiexec ]]; then
    "${LAUNCHER}" ${LAUNCHER_ARGS[@]+"${LAUNCHER_ARGS[@]}"} -n "${ranks}" \
      "${APP}" \
      --nodes 1 \
      --system "${SYSTEM}" \
      --output "${CSV}"
  else
    "${LAUNCHER}" ${LAUNCHER_ARGS[@]+"${LAUNCHER_ARGS[@]}"} -np "${ranks}" \
      "${APP}" \
      --nodes 1 \
      --system "${SYSTEM}" \
      --output "${CSV}"
  fi
done

echo "Results written to ${CSV}"
