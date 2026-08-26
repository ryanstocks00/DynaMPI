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

For repeat and provenance analysis, ``--all-runs-dir`` additionally writes one
audit CSV per sweep. It retains every source row and marks whether the default
latest-run selection kept, superseded, or rejected it. The default outputs and
selection rules are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

# Columns that are measurements rather than settings. Everything else in a
# row identifies the configuration that produced it. Used for the sweeps
# whose plots keep every distinct setting, which is all of them bar the two
# with an explicit key below.
MEASURED: dict[str, frozenset[str]] = {
    "weak_scaling": frozenset({"total_tasks", "elapsed_s", "throughput_tasks_per_s"}),
    "shutdown": frozenset({"time_per_shutdown_us", "iterations"}),
    "load_balancing": frozenset(
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
        r.get("mode", ""),
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
AUDIT_PROVENANCE = (
    "source_run",
    "source_path",
    "source_row",
    "source_mtime_ns",
    "selected",
    "selection_reason",
)
Recency = tuple[int, float]


@dataclass
class SourceRecord:
    row: dict[str, Any]
    recency: Recency
    key: tuple[Any, ...] | None
    source_run: str
    source_path: str
    source_row: int
    source_mtime_ns: int
    plottable: bool
    selected: bool = False
    selection_reason: str = ""


@dataclass
class Consolidation:
    fieldnames: list[str]
    all_fieldnames: list[str]
    selected_rows: list[dict[str, Any]]
    records: list[SourceRecord]
    seen: int


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


def _append_unique(fieldnames: Sequence[str], additions: Sequence[str]) -> list[str]:
    output = list(fieldnames)
    output.extend(name for name in additions if name not in output)
    return output


def build_consolidation(results_dir: str, sweep: str) -> Consolidation:
    """Read every source row and apply the default latest-run selection."""
    measured = MEASURED[sweep]
    config_key = CONFIG_KEY.get(sweep)
    fieldnames: list[str] = []
    all_fieldnames: list[str] = []
    records: list[SourceRecord] = []
    newest: dict[tuple[Any, ...], SourceRecord] = {}
    seen = 0

    for path, run in source_runs(results_dir, sweep):
        stat = os.stat(path)
        recency = (int(job_id_of(run) or 0), stat.st_mtime)
        relative_path = os.path.relpath(path, results_dir).replace(os.sep, "/")
        with open(path, newline="", encoding="utf-8") as handle:
            for source_row, row in enumerate(csv.DictReader(handle), start=2):
                seen += 1
                if not fieldnames:
                    fieldnames = [c for c in row if c is not None]
                all_fieldnames = _append_unique(
                    all_fieldnames, [c for c in row if c is not None]
                )
                plottable = is_plottable(sweep, row)
                row = dict(row)
                row["job_id"] = job_id_of(run)
                if config_key is not None:
                    key: tuple[Any, ...] = config_key(row)
                else:
                    key = tuple(row.get(c, "") for c in fieldnames if c not in measured)
                record = SourceRecord(
                    row=row,
                    recency=recency,
                    key=key if plottable else None,
                    source_run=run,
                    source_path=relative_path,
                    source_row=source_row,
                    source_mtime_ns=stat.st_mtime_ns,
                    plottable=plottable,
                )
                records.append(record)
                if plottable and (key not in newest or recency > newest[key].recency):
                    newest[key] = record

    for record in records:
        if not record.plottable:
            record.selection_reason = "excluded_not_plottable"
        elif record.key is not None and newest[record.key] is record:
            record.selected = True
            record.selection_reason = "selected_latest"
        elif record.key is not None and newest[record.key].recency == record.recency:
            record.selection_reason = "superseded_by_same_run_first_row"
        else:
            record.selection_reason = "superseded_by_newer_run"

    rows = [record.row for record in records if record.selected]
    rows.sort(key=lambda r: tuple(str(r.get(c, "")) for c in fieldnames))
    return Consolidation(fieldnames, all_fieldnames, rows, records, seen)


def consolidate(results_dir: str, sweep: str) -> tuple[list[str], list[dict[str, Any]], int]:
    """Return the unchanged default latest-run output."""
    result = build_consolidation(results_dir, sweep)
    return _append_unique(result.fieldnames, PROVENANCE), result.selected_rows, result.seen


def all_runs_rows(result: Consolidation) -> tuple[list[str], list[dict[str, Any]]]:
    """Return every source row plus provenance and its selection disposition."""
    rows: list[dict[str, Any]] = []
    for record in result.records:
        row = dict(record.row)
        row.update(
            {
                "source_run": record.source_run,
                "source_path": record.source_path,
                "source_row": record.source_row,
                "source_mtime_ns": record.source_mtime_ns,
                "selected": "1" if record.selected else "0",
                "selection_reason": record.selection_reason,
            }
        )
        rows.append(row)
    rows.sort(key=lambda row: (str(row["source_path"]), int(row["source_row"])))
    fieldnames = _append_unique(result.all_fieldnames, (*PROVENANCE, *AUDIT_PROVENANCE))
    return fieldnames, rows


def write_csv(path: str, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


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
        "--output-dir",
        help="Directory for the selected latest-run CSVs (default: --results-dir)",
    )
    parser.add_argument(
        "--all-runs-dir",
        help="Optional directory for <sweep>_all_runs.csv audit files. These retain "
        "every repeat and source row with selection provenance.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report every selected and all-runs output without creating or writing files",
    )
    args = parser.parse_args(argv)
    results_dir = os.path.normpath(args.results_dir)
    output_dir = os.path.normpath(args.output_dir or results_dir)
    all_runs_dir = os.path.normpath(args.all_runs_dir) if args.all_runs_dir else None

    for sweep in args.sweeps:
        result = build_consolidation(results_dir, sweep)
        if not result.records:
            print(f"{sweep}: no source CSVs found")
            continue
        selected = len(result.selected_rows)
        rejected = sum(not record.plottable for record in result.records)
        superseded = result.seen - selected - rejected
        out = os.path.join(output_dir, f"{sweep}.csv")
        print(
            f"{sweep}: {result.seen} rows in -> {selected} selected "
            f"({superseded} superseded, {rejected} excluded) -> {out}"
        )
        if not args.check:
            if result.selected_rows:
                os.makedirs(output_dir, exist_ok=True)
                write_csv(
                    out,
                    _append_unique(result.fieldnames, PROVENANCE),
                    result.selected_rows,
                )
            else:
                print(f"{sweep}: no selected rows, leaving {out} untouched")

        if all_runs_dir is not None:
            audit_out = os.path.join(all_runs_dir, f"{sweep}_all_runs.csv")
            print(
                f"{sweep}: {result.seen} provenance-preserving rows "
                f"(selection report) -> {audit_out}"
            )
            if not args.check:
                os.makedirs(all_runs_dir, exist_ok=True)
                audit_fieldnames, audit_rows = all_runs_rows(result)
                write_csv(audit_out, audit_fieldnames, audit_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
