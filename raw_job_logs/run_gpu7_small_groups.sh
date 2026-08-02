#!/bin/bash
cd "$HOME/DynaMPI" || exit 1

NODE_GROUPS=("1 2" "4 8" "16 32" "64 128")
MODES=("random" "fixed")

debug_scaling_count() {
  qstat -u "$USER" 2>/dev/null | awk 'NR>5 && $3 ~ /^debug-s/ { n++ } END { print 0+n }'
}

for mode in "${MODES[@]}"; do
  for group in "${NODE_GROUPS[@]}"; do
    while [ "$(debug_scaling_count)" -ge 1 ]; do
      echo "$(date +%T) waiting for debug-scaling slot (mode=$mode group=[$group]) ..."
      sleep 60
    done
    max_nodes=$(echo "$group" | tr ' ' '\n' | sort -n | tail -1)
    job_name="dynampi_gpu7small_${mode}_$(echo "$group" | tr ' ' '_')"
    job_script="#!/bin/bash
#PBS -l select=${max_nodes}:ncpus=102:mpiprocs=102
#PBS -l walltime=00:55:00
#PBS -q debug-scaling
#PBS -A DynaMPI
#PBS -l filesystems=flare
#PBS -N ${job_name}
#PBS -j oe

cd \"\$HOME/DynaMPI\" || exit 1
export FI_CXI_RX_MATCH_MODE=software
SS=\"\$HOME/DynaMPI/build/benchmark/strong_scaling_distribution_rate\"

EXPECTED_US_LIST=\"10 100 1000 10000 100000 1000000\"

run_one() {
  local dist=\"\$1\" fanout=\"\$2\" nodes=\"\$3\" us=\"\$4\"
  local ranks=\$((nodes * 7))
  echo \"##### dist=\$dist fanout=\$fanout nodes=\$nodes mode=${mode} expected_us=\$us rpn=7 #####\"
  timeout 90 mpiexec -n \"\$ranks\" --ppn 7 \"\$SS\" --distribution \"\$dist\" --mode ${mode} \\
    --expected_us \"\$us\" --duration_s 20 --nodes \"\$nodes\" --max_upper_fanout \"\$fanout\" --system aurora
  echo \"exit=\$?\"
}

for nodes in ${group}; do
  for us in \$EXPECTED_US_LIST; do
    run_one naive -1 \"\$nodes\" \"\$us\"
    run_one hierarchical 0 \"\$nodes\" \"\$us\"
    run_one hierarchical -1 \"\$nodes\" \"\$us\"
    run_one async_put_lockfree -1 \"\$nodes\" \"\$us\"
    run_one hierarchical_async_put_lockfree 0 \"\$nodes\" \"\$us\"
    run_one hierarchical_async_put_lockfree -1 \"\$nodes\" \"\$us\"
  done
done
echo \"ALL DONE\"
"
    jobid=""
    until [ -n "$jobid" ]; do
      out=$(qsub <<< "$job_script" 2>&1)
      jobid=$(echo "$out" | grep -oE '^[0-9]+')
      if [ -n "$jobid" ]; then
        echo "$(date +%T) submitted mode=$mode group=[$group] -> job $jobid"
      else
        echo "$(date +%T) submit failed mode=$mode group=[$group]: $out -- retrying in 60s"
        sleep 60
      fi
    done
  done
done
echo "GPU7 SMALL-GROUP SUBMISSION DONE"
