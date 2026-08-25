#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Submit one PBS job per node count to avoid long serial waits.
# Example:
#   ./benchmark/scripts/submit_aurora_weak_scaling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=benchmark/scripts/aurora_queue_utils.sh
source "${SCRIPT_DIR}/aurora_queue_utils.sh"
SYSTEM="aurora"
SCRIPT="${ROOT_DIR}/benchmark/scripts/launch_aurora_weak_scaling.sh"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32 64 128 256 512}"
IFS=' ' read -r -a QSUB_ARGS <<< "${QSUB_ARGS:-}"
ACCOUNT="${ACCOUNT:-DynaMPI}"
FILESYSTEMS="${FILESYSTEMS:-flare}"
NCPUS_PER_NODE="${NCPUS_PER_NODE:-102}"

WALLTIME="${WALLTIME:-00:15:00}"
DURATION_S="${DURATION_S:-}"
LAUNCHER="${LAUNCHER:-}"
LAUNCHER_ARGS="${LAUNCHER_ARGS:-}"
DISTRIBUTIONS="${DISTRIBUTIONS:-}"
TASK_US_LIST="${TASK_US_LIST:-}"
MODES="${MODES:-}"
MAX_UPPER_FANOUT="${MAX_UPPER_FANOUT:-}"
MAX_UPPER_FANOUT_LIST="${MAX_UPPER_FANOUT_LIST:-}"
MAX_TASKS="${MAX_TASKS:-}"
OUTPUT_BASE="${OUTPUT_DIR:-${ROOT_DIR}/benchmark/results}"

for nodes in "${NODE_LIST[@]}"; do
  job_name="dynampi_ws_${SYSTEM}_${nodes}"
  submit_args=("${QSUB_ARGS[@]}")
  if [[ -n "${ACCOUNT}" ]]; then
    submit_args+=(-A "${ACCOUNT}")
  fi
  if [[ "${nodes}" -lt 256 ]]; then
    submit_args+=(-q "debug-scaling")
  else
    submit_args+=(-q "prod")
  fi
  wait_for_aurora_queue_space "${nodes}"
  job_script="#!/usr/bin/env bash
#PBS -j oe
set -euo pipefail
cd \"${ROOT_DIR}\"
export NODE_LIST=\"${nodes}\"
export LAUNCHER=\"${LAUNCHER}\"
export LAUNCHER_ARGS=\"${LAUNCHER_ARGS}\"
export CORES_PER_NODE=\"${NCPUS_PER_NODE}\"
export DISTRIBUTIONS=\"${DISTRIBUTIONS}\"
export TASK_US_LIST=\"${TASK_US_LIST}\"
export MODES=\"${MODES}\"
export DURATION_S=\"${DURATION_S}\"
export MAX_UPPER_FANOUT=\"${MAX_UPPER_FANOUT}\"
export MAX_UPPER_FANOUT_LIST=\"${MAX_UPPER_FANOUT_LIST}\"
export MAX_TASKS=\"${MAX_TASKS}\"
export OUTPUT_DIR=\"${OUTPUT_BASE}/${SYSTEM}/${nodes}-${job_name}-\${PBS_JOBID_SHORT:-manual}\"
${SCRIPT}
"
  echo "qsub ${submit_args[*]} -N \"${job_name}\" -l \"select=${nodes}:ncpus=${NCPUS_PER_NODE}:mpiprocs=${NCPUS_PER_NODE}\" -l \"walltime=${WALLTIME}\" -l \"filesystems=${FILESYSTEMS}\" <<'QSUBEOF'"
  echo "${job_script}"
  echo "QSUBEOF"
  # The account can have OTHER unrelated jobs (different projects) sitting in
  # the queue that count against a per-user "jobs in Q state" limit enforced
  # by PBS itself (independent of wait_for_aurora_queue_space's running/
  # debug-scaling checks, which only look at running-count and debug-scaling
  # occupancy). qsub fails outright when that limit is hit -- retry with
  # backoff for a bounded number of attempts (permanent errors like bad
  # resource requests should not block the sweep forever).
  qsub_attempts=0
  qsub_max_attempts="${AURORA_QSUB_MAX_ATTEMPTS:-30}"
  until qsub "${submit_args[@]}" -N "${job_name}" -l "select=${nodes}:ncpus=${NCPUS_PER_NODE}:mpiprocs=${NCPUS_PER_NODE}" \
    -l "walltime=${WALLTIME}" -l "filesystems=${FILESYSTEMS}" <<< "${job_script}"; do
    qsub_attempts=$((qsub_attempts + 1))
    if (( qsub_attempts >= qsub_max_attempts )); then
      echo "qsub failed permanently for nodes=${nodes} after ${qsub_attempts} attempts" >&2
      exit 1
    fi
    echo "qsub failed (queue limit?) for nodes=${nodes}; retrying in ${AURORA_QUEUE_POLL_INTERVAL}s (${qsub_attempts}/${qsub_max_attempts}) ..." >&2
    sleep "${AURORA_QUEUE_POLL_INTERVAL}"
  done
done
