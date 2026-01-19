#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

module load cmake

# Default values
BUILD_DIR="build"
BUILD_TYPE="Release"

usage() {
    echo "Usage: $0 [-d build_dir] [-t build_type] [--clean]"
    echo
    echo "Options:"
    echo "  -d DIR    Build directory (default: build)"
    echo "  -t TYPE   Build type: Release, Debug, RelWithDebInfo, MinSizeRel (default: Release)"
    echo "  --clean   Remove build directory before configuring"
    exit 1
}

# Parse args
CLEAN=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) BUILD_DIR="$2"; shift 2 ;;
        -t) BUILD_TYPE="$2"; shift 2 ;;
        --clean) CLEAN=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ $CLEAN -eq 1 && -d "$BUILD_DIR" ]]; then
    echo "Cleaning $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

echo "Configuring with CMake..."
cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -B "$BUILD_DIR"

echo "Building..."
cmake --build "$BUILD_DIR" -- -j"$(nproc)"

echo "✅ Build finished in $BUILD_DIR ($BUILD_TYPE)"
