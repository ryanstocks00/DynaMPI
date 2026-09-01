#!/bin/bash
#SBATCH -A chm213
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 00:05:00
#SBATCH -J dynampi_capture_env
#SBATCH -o dynampi_capture_env.o%j

# Captures the Frontier compute-node environment behind this branch's results.
# Frontier counterpart of aurora_results' capture_env.pbs. Run on a live
# allocation so hardware/driver detail that only exists on compute nodes
# (GPUs, HSN NICs, Lustre mounts) is recorded as the jobs actually saw it.

set -x
echo "=== hostname ==="
srun -N 1 -n 1 hostname -f
date

echo "=== OS / kernel ==="
srun -N 1 -n 1 cat /etc/os-release
srun -N 1 -n 1 uname -a

echo "=== lscpu ==="
srun -N 1 -n 1 lscpu

echo "=== NUMA layout ==="
srun -N 1 -n 1 numactl -H 2>&1 || echo "numactl not available"

echo "=== memory ==="
srun -N 1 -n 1 free -h
srun -N 1 -n 1 head -5 /proc/meminfo

echo "=== GPU: rocm-smi ==="
srun -N 1 -n 1 rocm-smi 2>&1 || echo "rocm-smi not available"

echo "=== GPU: rocm-smi showhw ==="
srun -N 1 -n 1 rocm-smi --showhw 2>&1 || echo "rocm-smi --showhw not available"

echo "=== GPU: rocminfo (agents) ==="
srun -N 1 -n 1 bash -c 'rocminfo 2>&1 | head -80' || echo "rocminfo not available"

echo "=== loaded modules ==="
module list 2>&1

echo "=== module versions (key ones) ==="
for m in PrgEnv-cray cce craype cray-mpich libfabric rocm cmake craype-accel-amd-gfx90a; do
  echo "--- module show $m ---"
  module show "$m" 2>&1
done

echo "=== compilers ==="
cc --version 2>&1 || echo "cc not available"
CC --version 2>&1 || echo "CC not available"
gcc --version 2>&1 || echo "gcc not available"
craycc --version 2>&1 || echo "craycc not available"
hipcc --version 2>&1 || echo "hipcc not available"

echo "=== MPI info ==="
srun --version 2>&1 || echo "srun --version not available"
mpichversion 2>&1 || echo "mpichversion not available"
echo "--- cray-mpich via pkg-config/ldd ---"
ls -la "${CRAY_MPICH_DIR:-/nonexistent}/lib" 2>&1 | head -10

echo "=== libfabric providers ==="
srun -N 1 -n 1 fi_info --version 2>&1 || echo "fi_info not available"
srun -N 1 -n 1 bash -c 'fi_info -l' 2>&1 || echo "fi_info -l not available"

echo "=== network / HSN ==="
srun -N 1 -n 1 ip -brief addr 2>&1
srun -N 1 -n 1 bash -c 'cat /sys/class/cxi/cxi0/device/properties 2>/dev/null' 2>&1 || echo "no cxi properties"
srun -N 1 -n 1 bash -c 'ls /sys/class/cxi 2>/dev/null' 2>&1 || echo "no cxi devices"

echo "=== filesystems ==="
srun -N 1 -n 1 df -h
srun -N 1 -n 1 bash -c 'mount | grep -E "lustre|orion|home|nfs"' 2>&1

echo "=== scheduler ==="
scontrol --version 2>&1
sinfo --version 2>&1
scontrol show node "$SLURMD_NODENAME" 2>&1 | head -20

echo "=== cmake ==="
cmake --version 2>&1 || echo "cmake not loaded"

echo "=== relevant env vars ==="
env | grep -E "FI_CXI|MPICH|LIBFABRIC|CRAY|ROCM|HIP|SLURM_JOB" | sort

echo "=== ulimits ==="
srun -N 1 -n 1 bash -c 'ulimit -a'

echo "ALL DONE"
