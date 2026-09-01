#!/bin/bash
#SBATCH -A chm213
set -euo pipefail
REPO_ROOT="/lustre/orion/chm213/scratch/ryan/exess"
source "$REPO_ROOT/tools/frontier_env_setup_dynampi_rocm713_mpich910.sh"
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
export AMD_DIRECT_DISPATCH=1
export MBE_NGPUS=1
export MPICH_OFI_NIC_POLICY=GPU
export EXESS_RECORDS_PATH="$REPO_ROOT"
export MPICH_GPU_SUPPORT_ENABLED=0

cd "$SLURM_SUBMIT_DIR/run"
date +%s > start_epoch.txt
srun --ntasks=$((SLURM_JOB_NUM_NODES * 17)) --ntasks-per-node=17 --cpus-per-task=3 "$REPO_ROOT/exess_rocm713_mpich910" input.json
date +%s > end_epoch.txt
echo "wall_seconds=$(( $(cat end_epoch.txt) - $(cat start_epoch.txt) ))" | tee wall_seconds.txt
