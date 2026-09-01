#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# One Slurm job per node count, mirroring submit_frontier_weak_scaling.sh.
# Example:
#   ./benchmark/frontier/submit_frontier_load_balancing.sh
#   NODE_LIST=128 TASK_US_LIST="1000 10000 100000" \
#     MAX_TASKS_PER_WORKER_LIST="20 20 10" \
#     ./benchmark/frontier/submit_frontier_load_balancing.sh
#   NODE_LIST=8 TASK_US_LIST=1000 DURATION_MODES=fixed MAX_TASKS_PER_WORKER=2 \
#     WALLTIME=00:10:00 CSV_SUFFIX=_smoke JOB_TAG=_smoke \
#     ./benchmark/frontier/submit_frontier_load_balancing.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSTEM="frontier"
SCRIPT="${ROOT_DIR}/benchmark/frontier/launch_frontier_load_balancing.sh"

if [[ -z "${NODE_LIST:-}" ]]; then
  NODE_LIST="128"
fi
IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST}"
IFS=' ' read -r -a SBATCH_ARGS <<< "${SBATCH_ARGS:-}"
ACCOUNT="${ACCOUNT:-chm213}"
WALLTIME="${WALLTIME:-02:00:00}"

LAUNCHER="${LAUNCHER:-}"
LAUNCHER_ARGS="${LAUNCHER_ARGS:-}"
RANKS_PER_NODE_LIST="${RANKS_PER_NODE_LIST:-}"
TASK_US_LIST="${TASK_US_LIST:-}"
DURATION_MODES="${DURATION_MODES:-}"
MIN_TASKS_PER_WORKER="${MIN_TASKS_PER_WORKER:-}"
MAX_TASKS_PER_WORKER="${MAX_TASKS_PER_WORKER:-}"
MAX_TASKS_PER_WORKER_LIST="${MAX_TASKS_PER_WORKER_LIST:-}"
REPEATS="${REPEATS:-}"
DISTRIBUTIONS="${DISTRIBUTIONS:-}"
TASK_DURATION_CV="${TASK_DURATION_CV:-}"
MAX_UPPER_FANOUT="${MAX_UPPER_FANOUT:-}"
PIPELINE_DEPTH="${PIPELINE_DEPTH:-}"
MAX_PENDING_ROUNDS="${MAX_PENDING_ROUNDS:-}"
RUN_TIMEOUT_S="${RUN_TIMEOUT_S:-}"
CSV_SUFFIX="${CSV_SUFFIX:-}"
JOB_TAG="${JOB_TAG:-}"
BUILD_DIR="${BUILD_DIR:-}"
OUTPUT_BASE="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"

for nodes in "${NODE_LIST[@]}"; do
  job_name="dynampi_lb${JOB_TAG}_${SYSTEM}_${nodes}"
  submit_args=(${SBATCH_ARGS[@]+"${SBATCH_ARGS[@]}"})
  if [[ -n "${ACCOUNT}" ]]; then
    submit_args+=(--account="${ACCOUNT}")
  fi
  job_script="#!/usr/bin/env bash
set -euo pipefail
cd \"${ROOT_DIR}\"
export NODE_LIST=\"${nodes}\"
export LAUNCHER=\"${LAUNCHER}\"
export LAUNCHER_ARGS=\"${LAUNCHER_ARGS}\"
export RANKS_PER_NODE_LIST=\"${RANKS_PER_NODE_LIST}\"
export TASK_US_LIST=\"${TASK_US_LIST}\"
export DURATION_MODES=\"${DURATION_MODES}\"
export MIN_TASKS_PER_WORKER=\"${MIN_TASKS_PER_WORKER}\"
export MAX_TASKS_PER_WORKER=\"${MAX_TASKS_PER_WORKER}\"
export MAX_TASKS_PER_WORKER_LIST=\"${MAX_TASKS_PER_WORKER_LIST}\"
export REPEATS=\"${REPEATS}\"
export DISTRIBUTIONS=\"${DISTRIBUTIONS}\"
export TASK_DURATION_CV=\"${TASK_DURATION_CV}\"
export MAX_UPPER_FANOUT=\"${MAX_UPPER_FANOUT}\"
export PIPELINE_DEPTH=\"${PIPELINE_DEPTH}\"
export MAX_PENDING_ROUNDS=\"${MAX_PENDING_ROUNDS}\"
export RUN_TIMEOUT_S=\"${RUN_TIMEOUT_S}\"
export CSV_SUFFIX=\"${CSV_SUFFIX}\"
export BUILD_DIR=\"${BUILD_DIR}\"
export OUTPUT_DIR=\"${OUTPUT_BASE}/${SYSTEM}/${nodes}-${job_name}-\${SLURM_JOB_ID:-manual}\"
${SCRIPT}
"
  echo "sbatch ${submit_args[*]} --job-name=${job_name} --nodes=${nodes} --time=${WALLTIME}"
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "${job_script}"
    continue
  fi
  sbatch "${submit_args[@]}" \
    --job-name="${job_name}" \
    --nodes="${nodes}" \
    --time="${WALLTIME}" <<< "${job_script}"
done
