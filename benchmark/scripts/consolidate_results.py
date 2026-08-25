#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
"""Fold the per-job result directories into one CSV per sweep type.

The benchmark launchers write one directory per Slurm/PBS job, so a sweep
that was submitted as twelve jobs and rerun twice lands as thirty-six
directories holding the same handful of columns. This collapses each sweep
into a single file and records where every row came from.

One provenance column, ``job_id``, is added: the scheduler job id pulled
off the end of the source directory name.

Rows are reduced to what the plot scripts actually draw: one row per plot
configuration, keeping the highest job id. That deliberately discards
measurements a plot would never reach -- an older short-window run of a
configuration that was later re-measured, for instance -- so the committed
CSVs are exactly the data behind the figures rather than a superset of it.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable, Sequence
from typing import Any

# Columns that are measurements rather than settings. Everything else in a
# row identifies the configuration that produced it. Used for the sweeps
# whose plots keep every distinct setting, which is all of them bar the two
# with an explicit key below.
MEASURED: dict[str, frozenset[str]] = {
    "weak_scaling": frozenset({"total_tasks", "elapsed_s", "throughput_tasks_per_s"}),
    "shutdown": frozenset({"time_per_shutdown_us", "iterations"}),
    "test_load_balancing": frozenset(
        {"total_tasks", "elapsed_s", "construct_s", "warmup_s", "finalize_s", "destruct_s"}
    ),
}


def _int(value: str, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _ranks_per_node(row: dict[str, Any]) -> int:
    nodes = _int(row.get("nodes", ""))
    return round(_int(row.get("world_size", "")) / nodes) if nodes else 0


def _fanout(row: dict[str, Any]) -> int:
    """Fan-out only distinguishes hierarchical topologies, as in the plots."""
    distributor = (row.get("distributor") or "").strip()
    if "hierarchical" not in distributor:
        return -1
    return _int(row.get("max_upper_fanout", ""), -1)


# Sweeps whose plots collapse several settings into one drawn point. Keying on
# the raw columns would keep rows no figure can reach -- most visibly the
# 10-second windows superseded by 20-second reruns of the same configuration.
CONFIG_KEY = {
    "weak_scaling": lambda r: (
        r.get("system", ""),
        (r.get("distributor") or "").strip(),
        "random" if r.get("mode") == "poisson" else r.get("mode", ""),
        _int(r.get("expected_us", "")),
        _ranks_per_node(r),
        _fanout(r),
        _int(r.get("nodes", "")),
    ),
    "shutdown": lambda r: (
        r.get("system", ""),
        (r.get("distributor") or "").strip() or "naive",
        _fanout(r),
        _ranks_per_node(r),
        _int(r.get("nodes", "")),
    ),
}


def is_plottable(sweep: str, row: dict[str, Any]) -> bool:
    """Drop rows every plot rejects, so a rejected row cannot win its key.

    Only the shutdown sweep has any: runs above the plotted node range, and
    sub-microsecond hierarchical timings that are teardown no-ops rather than
    measurements.
    """
    if sweep != "shutdown":
        return True
    if _int(row.get("nodes", "")) > 2048:
        return False
    distributor = (row.get("distributor") or "").strip() or "naive"
    try:
        elapsed = float(row.get("time_per_shutdown_us") or 0.0)
    except ValueError:
        return False
    return not ("hierarchical" in distributor and elapsed < 1.0)

PROVENANCE = ("job_id",)


def source_runs(results_dir: str, sweep: str) -> Iterable[tuple[str, str]]:
    """Yield (csv_path, source_run) for every per-job CSV of one sweep."""
    for system in sorted(os.listdir(results_dir)):
        system_dir = os.path.join(results_dir, system)
        if not os.path.isdir(system_dir):
            continue
        for run in sorted(os.listdir(system_dir)):
            run_dir = os.path.join(system_dir, run)
            if not os.path.isdir(run_dir):
                continue
            for name in sorted(os.listdir(run_dir)):
                if name.startswith(f"{sweep}_") and name.endswith(".csv"):
                    yield os.path.join(run_dir, name), run


def job_id_of(source_run: str) -> str:
    """The trailing scheduler id, e.g. ``...-8725895`` or ``...-8720430.aurora``."""
    tail = source_run.rsplit("-", 1)[-1].split(".")[0]
    return tail if tail.isdigit() else ""


def consolidate(results_dir: str, sweep: str) -> tuple[list[str], list[dict[str, Any]], int]:
    measured = MEASURED[sweep]
    config_key = CONFIG_KEY.get(sweep)
    fieldnames: list[str] = []
    newest: dict[tuple[Any, ...], tuple[Any, dict[str, Any]]] = {}
    seen = 0

    for path, run in source_runs(results_dir, sweep):
        recency = (int(job_id_of(run) or 0), os.path.getmtime(path))
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seen += 1
                if not fieldnames:
                    fieldnames = [c for c in row if c is not None]
                if not is_plottable(sweep, row):
                    continue
                row = dict(row)
                row["job_id"] = job_id_of(run)
                if config_key is not None:
                    key: tuple[Any, ...] = config_key(row)
                else:
                    key = tuple(row.get(c, "") for c in fieldnames if c not in measured)
                if key not in newest or recency > newest[key][0]:
                    newest[key] = (recency, row)

    rows = [row for _, row in newest.values()]
    rows.sort(key=lambda r: tuple(str(r.get(c, "")) for c in fieldnames))
    return fieldnames + list(PROVENANCE), rows, seen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fold the per-job result directories into one CSV per sweep type."
    )
    parser.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"),
        help="Directory holding the per-system, per-job result directories",
    )
    parser.add_argument(
        "--sweeps", nargs="+", default=sorted(MEASURED), choices=sorted(MEASURED)
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would be written without writing it",
    )
    args = parser.parse_args(argv)
    results_dir = os.path.normpath(args.results_dir)

    for sweep in args.sweeps:
        fieldnames, rows, seen = consolidate(results_dir, sweep)
        if not rows:
            print(f"{sweep}: no source CSVs found")
            continue
        out = os.path.join(results_dir, f"{sweep}.csv")
        print(f"{sweep}: {seen} rows in -> {len(rows)} out ({seen - len(rows)} superseded) -> {out}")
        if args.check:
            continue
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
