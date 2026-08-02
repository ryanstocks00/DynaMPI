#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/DynaMPI" || exit 1
export FI_CXI_RX_MATCH_MODE=software
HOSTFILE="$HOME/nodefile_8689537.txt"
CSV="$HOME/DynaMPI/benchmark/results/strong_scaling_128node.csv"
mkdir -p "$HOME/DynaMPI/benchmark/results"
: > "$CSV"
echo "distributor,gather_mode,mode,expected_us,nodes,world_size,total_tasks,elapsed_s,throughput_tasks_per_s" > "$CSV"

RPN=8
DURATION=5
NODE_LIST="1 2 4 8 16 32 64 128"
US_LIST="100 1000"

for nodes in $NODE_LIST; do
  ranks=$((nodes * RPN))
  for us in $US_LIST; do
    for dist in naive hierarchical lockfree; do
      echo "=== nodes=$nodes dist=$dist expected_us=$us ==="
      out=$(timeout $((DURATION+20)) mpiexec --hostfile "$HOSTFILE" -n "$ranks" --ppn "$RPN" \
        ./build/benchmark/strong_scaling_distribution_rate \
        --distribution "$dist" --mode fixed --expected_us "$us" --duration_s "$DURATION" \
        --nodes "$nodes" --system aurora128 2>&1)
      echo "$out" | tail -2
      line=$(echo "$out" | grep "^RESULT" | tail -1)
      tt=$(echo "$line" | grep -oP 'total_tasks=\K[0-9]+')
      el=$(echo "$line" | grep -oP 'elapsed_s=\K[0-9.]+')
      tp=$(echo "$line" | grep -oP 'throughput_tasks_per_s=\K[0-9.]+')
      echo "$dist,n/a,fixed,$us,$nodes,$ranks,${tt:-NA},${el:-NA},${tp:-NA}" >> "$CSV"
    done
    # lockfree gather_mode=final
    echo "=== nodes=$nodes dist=lockfree gather_mode=final expected_us=$us ==="
    out=$(timeout $((DURATION+20)) mpiexec --hostfile "$HOSTFILE" -n "$ranks" --ppn "$RPN" \
      ./build/benchmark/strong_scaling_distribution_rate \
      --distribution lockfree --gather_mode final --mode fixed --expected_us "$us" --duration_s "$DURATION" \
      --nodes "$nodes" --system aurora128 2>&1)
    echo "$out" | tail -2
    line=$(echo "$out" | grep "^RESULT" | tail -1)
    tt=$(echo "$line" | grep -oP 'total_tasks=\K[0-9]+')
    el=$(echo "$line" | grep -oP 'elapsed_s=\K[0-9.]+')
    tp=$(echo "$line" | grep -oP 'throughput_tasks_per_s=\K[0-9.]+')
    echo "lockfree,final,fixed,$us,$nodes,$ranks,${tt:-NA},${el:-NA},${tp:-NA}" >> "$CSV"
  done
done
echo "SWEEP COMPLETE"
