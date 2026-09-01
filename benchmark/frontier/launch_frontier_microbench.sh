#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Runs the two ceiling microbenchmarks inside one allocation:
#   twosided_msgrate_microbench -- root-rank request/reply ceiling, with and
#                                  without the per-receive MPI_Probe
#   rma_atomic_microbench       -- single-counter Fetch_and_op ceiling
#
# Both write CSV via --output, appending one row per mode/phase, so a whole
# sweep lands in one file per job.
#
# Example (inside an salloc/sbatch allocation):
#   NODE_LIST="1 2 4 8" ./benchmark/frontier/launch_frontier_microbench.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
TWOSIDED_APP="${TWOSIDED_APP:-${BUILD_DIR}/benchmark/twosided_msgrate_microbench}"
RMA_APP="${RMA_APP:-${BUILD_DIR}/benchmark/rma_atomic_microbench}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"
SYSTEM="${SYSTEM:-frontier}"
# Keeps an off-node sweep from appending into the all-workers CSVs.
CSV_SUFFIX="${CSV_SUFFIX:-}"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512 1024 2048}"
# 9 = the GPU-mapped layout the paper's weak-scaling sweeps use on Frontier, so
# these ceilings can be overlaid directly on those curves. 56 = all cores.
IFS=' ' read -r -a RANKS_PER_NODE_LIST <<< "${RANKS_PER_NODE_LIST:-9}"
# 4-byte payload matches the synthetic workload; the wider list is only worth
# running at a few node counts (see PAYLOAD_NODE_LIST below).
IFS=' ' read -r -a PAYLOAD_INTS_LIST <<< "${PAYLOAD_INTS_LIST:-1}"
MODES="${MODES:-probe,noprobe,oneway}"
DURATION_S="${DURATION_S:-5}"
WARMUP_S="${WARMUP_S:-1}"
CREDIT_WINDOW="${CREDIT_WINDOW:-64}"
RMA_DURATION_S="${RMA_DURATION_S:-3}"
RMA_PIPELINE_DEPTHS="${RMA_PIPELINE_DEPTHS:-1,8,32,128}"
RUN_RMA="${RUN_RMA:-1}"
RUN_TWOSIDED="${RUN_TWOSIDED:-1}"
# 1 drops workers sharing the root's node (8 at the default 9 ranks/node), so
# every measured exchange crosses the fabric. Needs at least 2 nodes.
EXCLUDE_ROOT_NODE="${EXCLUDE_ROOT_NODE:-0}"

LAUNCHER="${LAUNCHER:-}"
if [[ -z "${LAUNCHER}" ]]; then
  if command -v srun >/dev/null 2>&1; then
    LAUNCHER="srun"
  elif command -v mpiexec >/dev/null 2>&1; then
    LAUNCHER="mpiexec"
  else
    echo "No launcher found. Install srun or mpiexec." >&2
    exit 1
  fi
fi

EXCLUDE_ARGS=()
if [[ "${EXCLUDE_ROOT_NODE}" == "1" ]]; then
  EXCLUDE_ARGS+=(--exclude_root_node=true)
fi

mkdir -p "${OUTPUT_DIR}"
TS_CSV="${OUTPUT_DIR}/twosided_msgrate_${SYSTEM}${CSV_SUFFIX}.csv"
RMA_CSV="${OUTPUT_DIR}/rma_atomic_${SYSTEM}${CSV_SUFFIX}.csv"

for nodes in "${NODE_LIST[@]}"; do
  for rpn in "${RANKS_PER_NODE_LIST[@]}"; do
    total_ranks=$((nodes * rpn))

    if [[ "${RUN_TWOSIDED}" == "1" ]]; then
      for payload in "${PAYLOAD_INTS_LIST[@]}"; do
        echo "== twosided nodes=${nodes} rpn=${rpn} ranks=${total_ranks} payload_ints=${payload}"
        "${LAUNCHER}" -N "${nodes}" -n "${total_ranks}" --ntasks-per-node="${rpn}" \
          "${TWOSIDED_APP}" \
          --modes "${MODES}" \
          --duration_s "${DURATION_S}" \
          --warmup_s "${WARMUP_S}" \
          --payload_ints "${payload}" \
          --credit_window "${CREDIT_WINDOW}" \
          --nodes "${nodes}" \
          --system "${SYSTEM}" \
          ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"} \
          --output "${TS_CSV}" || echo "  !! twosided failed at nodes=${nodes} payload=${payload}"
      done
    fi

    if [[ "${RUN_RMA}" == "1" ]]; then
      echo "== rma nodes=${nodes} rpn=${rpn} ranks=${total_ranks}"
      "${LAUNCHER}" -N "${nodes}" -n "${total_ranks}" --ntasks-per-node="${rpn}" \
        "${RMA_APP}" \
        --duration_s "${RMA_DURATION_S}" \
        --pipeline_depths "${RMA_PIPELINE_DEPTHS}" \
        --nodes "${nodes}" \
        --system "${SYSTEM}" \
        ${EXCLUDE_ARGS[@]+"${EXCLUDE_ARGS[@]}"} \
        --output "${RMA_CSV}" || echo "  !! rma failed at nodes=${nodes}"
    fi
  done
done

echo "Wrote:"
echo "  ${TS_CSV}"
echo "  ${RMA_CSV}"
