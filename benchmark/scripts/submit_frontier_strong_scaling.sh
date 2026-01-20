#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Submit one Slurm job per node count to avoid long serial waits.
# Example:
#   ./benchmark/scripts/submit_frontier_strong_scaling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSTEM="frontier"
SCRIPT="${ROOT_DIR}/benchmark/scripts/launch_frontier_strong_scaling.sh"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512}"
IFS=' ' read -r -a SBATCH_ARGS <<< "${SBATCH_ARGS:-}"
ACCOUNT="${ACCOUNT:-chm213}"

WALLTIME="${WALLTIME:-00:15:00}"
LAUNCHER="${LAUNCHER:-}"
LAUNCHER_ARGS="${LAUNCHER_ARGS:-}"
OUTPUT_BASE="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"

for nodes in "${NODE_LIST[@]}"; do
  job_name="dynampi_ss_${SYSTEM}_${nodes}"
  submit_args=("${SBATCH_ARGS[@]}")
  if [[ -n "${ACCOUNT}" ]]; then
    submit_args+=(--account="${ACCOUNT}")
  fi
  sbatch "${submit_args[@]}" \
    --job-name="${job_name}" \
    --nodes="${nodes}" \
    --time="${WALLTIME}" \
    --export=ALL,NODE_LIST="${nodes}",LAUNCHER="${LAUNCHER}",LAUNCHER_ARGS="${LAUNCHER_ARGS}" \
    --wrap="cd ${ROOT_DIR} && OUTPUT_DIR=\"${OUTPUT_BASE}/${SYSTEM}/${nodes}-${job_name}-${SLURM_JOB_ID:-manual}\" ${SCRIPT}"
done
