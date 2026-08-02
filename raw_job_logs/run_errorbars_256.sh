#!/bin/bash
cd "$HOME/DynaMPI" || exit 1
NTRIALS=5
JOBIDS=()

debug_scaling_count() {
  qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $3 ~ /^debug-s/ { n++ } END { print 0+n }'
}

for i in $(seq 1 $NTRIALS); do
  while [ "$(debug_scaling_count)" -ge 2 ]; do
    echo "$(date +%T) waiting for debug-scaling slot (trial $i) ..."
    sleep 30
  done
  out=$(qsub -N "dynampi_errbar_256_t${i}" errorbars_256.pbs 2>&1)
  echo "$(date +%T) trial $i submit: $out"
  jobid=$(echo "$out" | grep -oE '^[0-9]+')
  if [ -n "$jobid" ]; then
    JOBIDS+=("$jobid")
  else
    echo "$(date +%T) trial $i submission FAILED, retrying in 30s"
    sleep 30
    # retry once more, same slot-wait loop will re-check next outer iteration
    i=$((i - 1))
  fi
done

echo "SUBMITTED_JOBIDS: ${JOBIDS[*]}"

# Wait for all to finish
declare -A DONE=()
remaining=${#JOBIDS[@]}
while [ "$remaining" -gt 0 ]; do
  sleep 30
  for jobid in "${JOBIDS[@]}"; do
    [ -n "${DONE[$jobid]:-}" ] && continue
    state=$(qstat -xf "$jobid" 2>/dev/null | grep -m1 "job_state" | awk '{print $3}')
    if [ -z "$state" ]; then
      sleep 10
      state=$(qstat -xf "$jobid" 2>/dev/null | grep -m1 "job_state" | awk '{print $3}')
    fi
    if [ "$state" = "F" ]; then
      DONE[$jobid]=1
      remaining=$((remaining - 1))
      echo "$(date +%T) job $jobid finished ($remaining remaining)"
    fi
  done
done

echo "ALL TRIALS FINISHED"
for jobid in "${JOBIDS[@]}"; do
  outfile=$(find "$HOME/DynaMPI" -maxdepth 1 -name "dynampi_errbar_256_t*.o${jobid}" 2>/dev/null | head -1)
  echo "=== job $jobid output: $outfile ==="
  if [ -n "$outfile" ]; then cat "$outfile"; fi
done
