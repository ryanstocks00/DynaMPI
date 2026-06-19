#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
#
# Aurora PBS queue helpers: enforce "only 1 job <256 nodes in queue" and "at most 2 running".
# Source this from submit_aurora_*.sh. Set SKIP_QUEUE_POLL=1 to disable waiting.

# Poll interval in seconds. Override with AURORA_QUEUE_POLL_INTERVAL.
AURORA_QUEUE_POLL_INTERVAL="${AURORA_QUEUE_POLL_INTERVAL:-60}"

# Count my jobs: running (state R). Assumes qstat -u output has state as second-to-last column.
_aurora_running_count() {
  qstat -u "${USER}" 2>/dev/null | awk '
    NR > 5 && NF >= 2 && $(NF-1) == "R" { n++ }
    END { print 0 + n }
  '
}

# Count my jobs in debug-scaling (queued + running). Queue name is last column.
_aurora_debug_scaling_count() {
  qstat -u "${USER}" 2>/dev/null | awk '
    NR > 5 && NF >= 2 && $NF == "debug-scaling" { n++ }
    END { print 0 + n }
  '
}

# Block until we are allowed to submit a job with this many nodes.
# Rules: only 1 job <256 nodes (debug-scaling) at a time; at most 2 jobs running.
wait_for_aurora_queue_space() {
  local nodes="${1:?}"
  if [[ -n "${SKIP_QUEUE_POLL:-}" ]]; then
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
    if [[ "${nodes}" -lt 256 ]]; then
      local in_debug
      in_debug="$(_aurora_debug_scaling_count)"
      if [[ "${in_debug}" -ge 1 ]]; then
        echo "Aurora: ${in_debug} job(s) already in debug-scaling (max 1); waiting ${AURORA_QUEUE_POLL_INTERVAL}s ..."
        sleep "${AURORA_QUEUE_POLL_INTERVAL}"
        continue
      fi
    fi
    return 0
  done
}
