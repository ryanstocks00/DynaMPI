#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Example usage:
#   ./benchmark/scripts/check_timer_resolution.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP="${APP:-${ROOT_DIR}/build/benchmark/timer_resolution}"

if [[ ! -f "${APP}" ]]; then
    echo "Error: ${APP} not found. Please build the benchmark first." >&2
    echo "Run: cmake --build build --target timer_resolution" >&2
    exit 1
fi

"${APP}"
