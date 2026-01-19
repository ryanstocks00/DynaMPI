#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Submit one PBS job per node count to avoid long serial waits.
# Example:
#   ./benchmark/scripts/submit_aurora_strong_scaling.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SYSTEM="aurora"
SCRIPT="${ROOT_DIR}/benchmark/scripts/launch_aurora_strong_scaling.sh"

IFS=' ' read -r -a NODE_LIST <<< "${NODE_LIST:-1 2 4 8 16 32}"
IFS=' ' read -r -a QSUB_ARGS <<< "${QSUB_ARGS:-}"
ACCOUNT="${ACCOUNT:-DynaMPI}"
FILESYSTEMS="${FILESYSTEMS:-flare}"

WALLTIME="${WALLTIME:-00:05:00}"
LAUNCHER="${LAUNCHER:-}"
LAUNCHER_ARGS="${LAUNCHER_ARGS:-}"

for nodes in "${NODE_LIST[@]}"; do
  job_name="dynampi_ss_${SYSTEM}_${nodes}"
  submit_args=("${QSUB_ARGS[@]}")
  if [[ -n "${ACCOUNT}" ]]; then
    submit_args+=(-A "${ACCOUNT}")
  fi
  qsub "${submit_args[@]}" -N "${job_name}" -l "select=${nodes}" -l "walltime=${WALLTIME}" \
    -l "filesystems=${FILESYSTEMS}" <<EOF
#!/usr/bin/env bash
#PBS -j oe
set -euo pipefail
cd "${ROOT_DIR}"
export NODE_LIST="${nodes}"
  export LAUNCHER="${LAUNCHER}"
export LAUNCHER_ARGS="${LAUNCHER_ARGS}"
${SCRIPT}
EOF
done
