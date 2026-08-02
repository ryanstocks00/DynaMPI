#!/bin/bash
cd "$HOME/DynaMPI" || exit 1

NODE_GROUPS=("1 2" "4 8" "16 32" "64 128")
RPNS=("102" "7")

debug_scaling_count() {
  qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $3 ~ /^debug-s/ { n++ } END { print 0+n }'
}

for rpn in "${RPNS[@]}"; do
  for group in "${NODE_GROUPS[@]}"; do
    while [ "$(debug_scaling_count)" -ge 1 ]; do
      echo "$(date +%T) waiting for debug-scaling slot (rpn=$rpn group=[$group]) ..."
      sleep 60
    done
    max_nodes=$(echo "$group" | tr ' ' '\n' | sort -n | tail -1)
    job_name="dynampi_shutdownsmall_${rpn}_$(echo "$group" | tr ' ' '_')"
    job_script="#!/bin/bash
#PBS -l select=${max_nodes}:ncpus=102:mpiprocs=102
#PBS -l walltime=00:35:00
#PBS -q debug-scaling
#PBS -A DynaMPI
#PBS -l filesystems=flare
#PBS -N ${job_name}
#PBS -j oe

cd \"\$HOME/DynaMPI\" || exit 1
export FI_CXI_RX_MATCH_MODE=software
SD=\"\$HOME/DynaMPI/build/benchmark/shutdown_time\"

run_one() {
  local dist=\"\$1\" fanout=\"\$2\" nodes=\"\$3\"
  local ranks=\$((nodes * ${rpn}))
  echo \"##### shutdown dist=\$dist fanout=\$fanout nodes=\$nodes rpn=${rpn} #####\"
  timeout 60 mpiexec -n \"\$ranks\" --ppn ${rpn} \"\$SD\" --distribution \"\$dist\" --nodes \"\$nodes\" \\
    --max_upper_fanout \"\$fanout\" --system aurora
  echo \"exit=\$?\"
}

for nodes in ${group}; do
  run_one naive -1 \"\$nodes\"
  run_one hierarchical 0 \"\$nodes\"
  run_one hierarchical -1 \"\$nodes\"
  run_one async_put_lockfree -1 \"\$nodes\"
  run_one hierarchical_async_put_lockfree 0 \"\$nodes\"
  run_one hierarchical_async_put_lockfree -1 \"\$nodes\"
done
echo \"ALL DONE\"
"
    jobid=""
    until [ -n "$jobid" ]; do
      out=$(qsub <<< "$job_script" 2>&1)
      jobid=$(echo "$out" | grep -oE '^[0-9]+')
      if [ -n "$jobid" ]; then
        echo "$(date +%T) submitted rpn=$rpn group=[$group] -> job $jobid"
      else
        echo "$(date +%T) submit failed rpn=$rpn group=[$group]: $out -- retrying in 60s"
        sleep 60
      fi
    done
  done
done
echo "SHUTDOWN SMALL-GROUP SUBMISSION DONE"
