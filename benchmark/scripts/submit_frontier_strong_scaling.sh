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

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32}"
IFS=' ' read -r -a SBATCH_ARGS <<< "${SBATCH_ARGS:-}"

WALLTIME="${WALLTIME:-01:00:00}"
LAUNCHER="${LAUNCHER:-srun}"
LAUNCHER_ARGS="${LAUNCHER_ARGS:-}"

for nodes in "${NODE_LIST[@]}"; do
  job_name="dynampi_ss_${SYSTEM}_${nodes}"
  sbatch "${SBATCH_ARGS[@]}" \
    --job-name="${job_name}" \
    --nodes="${nodes}" \
    --time="${WALLTIME}" \
    --export=ALL,NODE_LIST="${nodes}",LAUNCHER="${LAUNCHER}",LAUNCHER_ARGS="${LAUNCHER_ARGS}" \
    --wrap="cd ${ROOT_DIR} && ${SCRIPT}"
done
