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

Rows are deduplicated on every column except the measured outputs, keeping
the highest job id. Rows that merely *look* duplicated to a plot (same
configuration, different ``duration_s``) are kept, so no measurement is
discarded here; the plot scripts resolve those from ``job_id``.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable, Sequence
from typing import Any

# Columns that are measurements rather than settings. Everything else in a
# row identifies the configuration that produced it.
MEASURED: dict[str, frozenset[str]] = {
    "weak_scaling": frozenset({"total_tasks", "elapsed_s", "throughput_tasks_per_s"}),
    "shutdown": frozenset({"time_per_shutdown_us", "iterations"}),
    "test_load_balancing": frozenset(
        {"total_tasks", "elapsed_s", "construct_s", "warmup_s", "finalize_s", "destruct_s"}
    ),
}

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
    fieldnames: list[str] = []
    newest: dict[tuple[str, ...], tuple[Any, dict[str, Any]]] = {}
    seen = 0

    for path, run in source_runs(results_dir, sweep):
        recency = (int(job_id_of(run) or 0), os.path.getmtime(path))
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                seen += 1
                if not fieldnames:
                    fieldnames = [c for c in row if c is not None]
                row = dict(row)
                row["job_id"] = job_id_of(run)
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
