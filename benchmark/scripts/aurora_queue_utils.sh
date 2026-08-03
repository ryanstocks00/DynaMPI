#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Aurora PBS queue helpers: enforce "only 1 job <256 nodes in queue" and "at most 2 running".
# Source this from submit_aurora_*.sh. Set SKIP_QUEUE_POLL=1 to disable waiting.

# Poll interval in seconds. Override with AURORA_QUEUE_POLL_INTERVAL.
AURORA_QUEUE_POLL_INTERVAL="${AURORA_QUEUE_POLL_INTERVAL:-60}"

# Job name prefix used by all of this repo's submit scripts (see
# submit_aurora_strong_scaling.sh's job_name="dynampi_ss_${SYSTEM}_${nodes}"
# and the shutdown-benchmark equivalent) -- lets the checks below ignore
# other, unrelated jobs under the same account (e.g. a different project's
# runs) instead of deferring to them.
AURORA_JOB_NAME_PREFIX="${AURORA_JOB_NAME_PREFIX:-dynampi_}"

# Count my OWN jobs (by name prefix): running (state R).
# qstat -u columns: JobID Username Queue Jobname ... S Time
_aurora_running_count() {
  qstat -u "${USER}" 2>/dev/null | awk -v prefix="${AURORA_JOB_NAME_PREFIX}" '
    NR > 5 && NF >= 10 && index($4, prefix) == 1 && $10 == "R" { n++ }
    END { print 0 + n }
  '
}

# Count my OWN jobs in debug-scaling (queued + running).
_aurora_debug_scaling_count() {
  qstat -u "${USER}" 2>/dev/null | awk -v prefix="${AURORA_JOB_NAME_PREFIX}" '
    NR > 5 && NF >= 4 && index($4, prefix) == 1 && $3 == "debug-scaling" { n++ }
    END { print 0 + n }
  '
}

# Block until we are allowed to submit a job with this many nodes.
# Rules (debug-scaling jobs, i.e. <256 nodes, only): only 1 job in debug-scaling
# at a time; at most 2 of our own jobs running. Neither rule applies at 256+
# nodes (prod queue) -- prod is a separate resource pool from debug-scaling, so
# there's no reason to throttle our own submission rate into it; qsub/PBS's own
# scheduling and per-account limits are the real gate there.
wait_for_aurora_queue_space() {
  local nodes="${1:?}"
  if [[ -n "${SKIP_QUEUE_POLL:-}" ]]; then
    return 0
  fi
  if [[ "${nodes}" -ge 256 ]]; then
    return 0
  fi
  while true; do
    local running
    running="$(_aurora_running_count)"
    if [[ "${running}" -ge 2 ]]; then
      echo "Aurora: ${running} jobs running (max 2); waiting ${AURORA_QUEUE_POLL_INTERVAL}s ..."
      sleep "${AURORA_QUEUE_POLL_INTERVAL}"
      continue
    fi
    local in_debug
    in_debug="$(_aurora_debug_scaling_count)"
    if [[ "${in_debug}" -ge 1 ]]; then
      echo "Aurora: ${in_debug} job(s) already in debug-scaling (max 1); waiting ${AURORA_QUEUE_POLL_INTERVAL}s ..."
      sleep "${AURORA_QUEUE_POLL_INTERVAL}"
      continue
    fi
    return 0
  done
}
