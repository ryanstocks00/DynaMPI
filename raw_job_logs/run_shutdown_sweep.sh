#!/usr/bin/env bash
set -uo pipefail
cd "$HOME/DynaMPI" || exit 1
export FI_CXI_RX_MATCH_MODE=software
HOSTFILE="$HOME/nodefile_8689537.txt"
CSV="$HOME/DynaMPI/benchmark/results/shutdown_128node.csv"
echo "distributor,nodes,world_size,workers,time_per_shutdown_us,iterations" > "$CSV"

RPN=8
NODE_LIST="1 2 4 8 16 32 64 128"

for nodes in $NODE_LIST; do
  ranks=$((nodes * RPN))
  for dist in naive hierarchical lockfree; do
    echo "=== nodes=$nodes dist=$dist ==="
    out=$(timeout 30 mpiexec --hostfile "$HOSTFILE" -n "$ranks" --ppn "$RPN" \
      ./build/benchmark/shutdown_time --distribution "$dist" --nodes "$nodes" --system aurora128 2>&1)
    echo "$out" | tail -2
    line=$(echo "$out" | grep "^RESULT" | tail -1)
    ws=$(echo "$line" | grep -oP 'world_size=\K[0-9]+')
    wk=$(echo "$line" | grep -oP 'workers=\K[0-9]+')
    tpu=$(echo "$line" | grep -oP 'time_per_shutdown_us=\K[0-9.eE+-]+')
    it=$(echo "$line" | grep -oP 'iterations=\K[0-9]+')
    echo "$dist,$nodes,${ws:-NA},${wk:-NA},${tpu:-NA},${it:-NA}" >> "$CSV"
  done
done
echo "SHUTDOWN SWEEP COMPLETE"
