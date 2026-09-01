#!/bin/bash
module purge
module load Core/25.03
module load PrgEnv-gnu
module load gcc-native/13.2
module load rocm/7.13.0
module load craype-accel-amd-gfx90a
module load cray-mpich/9.1.0
module load cray-hdf5/1.14.3.1
module load cray-libsci/24.07.0
module load cmake/3.30.5
module load julia/1.11.0

export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
export MPI_ROOT=$MPICH_DIR
export MPICH_GPU_SUPPORT_ENABLED=1
