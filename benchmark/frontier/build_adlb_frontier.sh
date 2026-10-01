#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Builds ADLB/X (Swift/T's `lb`) and its c-utils dependency for
# benchmark/load_balancing_adlb.cpp, printing the CMake flags to use after.
# Run from a login node: clones over https, needs autoconf/automake.

PREFIX="${PREFIX:-/lustre/orion/${ACCOUNT:-chm213}/scratch/${USER}/adlb}"
SRC_DIR="${SRC_DIR:-${PREFIX}/swift-t}"
REPO="${REPO:-https://github.com/swift-lang/swift-t.git}"

if [[ -z "${MPICH_DIR:-}" ]]; then
  echo "MPICH_DIR is unset: load a cray-mpich module first." >&2
  exit 1
fi

mkdir -p "${PREFIX}"
if [[ ! -d "${SRC_DIR}" ]]; then
  git clone --depth 1 "${REPO}" "${SRC_DIR}"
fi

# lb generates dependencies with a second compiler (DEPCC, gcc by default) run
# with -MG, which emits headers it cannot find as bare filenames -- giving an
# unsatisfiable "No rule to make target 'mpi.h'". Point it at mpi.h too.
DEPCC="gcc -I${MPICH_DIR}/include"

echo "=== c-utils"
pushd "${SRC_DIR}/c-utils/code" > /dev/null
./bootstrap
CC=cc ./configure --prefix="${PREFIX}/install/c-utils" --enable-shared=no
make -j 8
make install
popd > /dev/null

echo "=== lb (ADLB/X)"
pushd "${SRC_DIR}/lb/code" > /dev/null
./bootstrap
# --enable-fast is lb's own production setting (no logging or asserts). It
# leaves the backoff/steal tuning at lb's BACKOFF_FAST defaults either way.
CC=cc CFLAGS="-I${MPICH_DIR}/include -O2" ./configure \
  --with-c-utils="${PREFIX}/install/c-utils" \
  --prefix="${PREFIX}/install/lb" \
  --enable-fast --enable-shared=no
make -j 8 DEPCC="${DEPCC}"
make install DEPCC="${DEPCC}"
popd > /dev/null

echo
echo "ADLB installed. Configure DynaMPI with:"
echo "  cmake -B build -DDYNAMPI_BUILD_BENCHMARKS=ON \\"
echo "        -DADLB_ROOT=${PREFIX}/install/lb -DCUTILS_ROOT=${PREFIX}/install/c-utils"
