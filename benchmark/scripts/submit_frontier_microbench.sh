#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# One Slurm job per node count, mirroring submit_frontier_strong_scaling.sh.
# Example:
#   ./benchmark/scripts/submit_frontier_microbench.sh
#   NODE_LIST="1 8 64" WALLTIME=00:10:00 ./benchmark/scripts/submit_frontier_microbench.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSTEM="frontier"
SCRIPT="${ROOT_DIR}/benchmark/scripts/launch_frontier_microbench.sh"

if [[ -z "${NODE_LIST:-}" ]]; then
  NODE_LIST="1 2 4 8 16 32 64 128 256 512 1024 2048"
fi
IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST}"
IFS=' ' read -r -a SBATCH_ARGS <<< "${SBATCH_ARGS:-}"
ACCOUNT="${ACCOUNT:-chm213}"
WALLTIME="${WALLTIME:-00:10:00}"

RANKS_PER_NODE_LIST="${RANKS_PER_NODE_LIST:-}"
PAYLOAD_INTS_LIST="${PAYLOAD_INTS_LIST:-}"
MODES="${MODES:-}"
DURATION_S="${DURATION_S:-}"
WARMUP_S="${WARMUP_S:-}"
RMA_DURATION_S="${RMA_DURATION_S:-}"
RMA_PIPELINE_DEPTHS="${RMA_PIPELINE_DEPTHS:-}"
RUN_RMA="${RUN_RMA:-}"
RUN_TWOSIDED="${RUN_TWOSIDED:-}"
BUILD_DIR="${BUILD_DIR:-}"
OUTPUT_BASE="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"

for nodes in "${NODE_LIST[@]}"; do
  job_name="dynampi_micro_${SYSTEM}_${nodes}"
  submit_args=(${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"})
  if [[ -n "${ACCOUNT}" ]]; then
    submit_args+=(--account="${ACCOUNT}")
  fi
  job_script="#!/usr/bin/env bash
set -euo pipefail
cd \"${ROOT_DIR}\"
export NODE_LIST=\"${nodes}\"
export RANKS_PER_NODE_LIST=\"${RANKS_PER_NODE_LIST}\"
export PAYLOAD_INTS_LIST=\"${PAYLOAD_INTS_LIST}\"
export MODES=\"${MODES}\"
export DURATION_S=\"${DURATION_S}\"
export WARMUP_S=\"${WARMUP_S}\"
export RMA_DURATION_S=\"${RMA_DURATION_S}\"
export RMA_PIPELINE_DEPTHS=\"${RMA_PIPELINE_DEPTHS}\"
export RUN_RMA=\"${RUN_RMA}\"
export RUN_TWOSIDED=\"${RUN_TWOSIDED}\"
export BUILD_DIR=\"${BUILD_DIR}\"
export OUTPUT_DIR=\"${OUTPUT_BASE}/${SYSTEM}/${nodes}-${job_name}-\${SLURM_JOB_ID:-manual}\"
${SCRIPT}
"
  echo "sbatch ${submit_args[*]} --job-name=${job_name} --nodes=${nodes} --time=${WALLTIME}"
  sbatch "${submit_args[@]}" \
    --job-name="${job_name}" \
    --nodes="${nodes}" \
    --time="${WALLTIME}" <<< "${job_script}"
done
