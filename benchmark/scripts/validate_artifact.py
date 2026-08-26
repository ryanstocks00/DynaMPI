# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
"""Validate the committed benchmark artifact and its paper-facing claims.

The default validation uses only committed consolidated CSVs and EXESS logs.
Pass one or more raw result directories or ``*_all_runs.csv`` files with
``--variability-input`` to additionally recompute the archived repeatability
summary without copying raw result branches into this tree.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path


DISTRIBUTORS = {
    "naive",
    "hierarchical",
    "lockfree_rma",
    "hierarchical_lockfree_rma",
}

RATE_COLUMNS = {
    "system",
    "distributor",
    "mode",
    "expected_us",
    "duration_s",
    "nodes",
    "world_size",
    "workers",
    "max_upper_fanout",
    "total_tasks",
    "elapsed_s",
    "throughput_tasks_per_s",
    "job_id",
}
REQUIRED_COLUMNS = {
    "weak_scaling": RATE_COLUMNS,
    "large_scale": RATE_COLUMNS,
    "shutdown": {
        "system",
        "distributor",
        "nodes",
        "max_upper_fanout",
        "world_size",
        "workers",
        "time_per_shutdown_us",
        "iterations",
        "job_id",
    },
    "load_balancing": {
        "system",
        "distributor",
        "nodes",
        "world_size",
        "workers",
        "max_upper_fanout",
        "pipeline_depth",
        "max_pending_rounds",
        "expected_us",
        "duration_mode",
        "task_duration_cv",
        "tasks_per_worker",
        "repeat",
        "total_tasks",
        "elapsed_s",
        "construct_s",
        "warmup_s",
        "finalize_s",
        "destruct_s",
        "job_id",
    },
}

HARVEST_RE = re.compile(r"Result harvesting .* took ([0-9]+(?:\.[0-9]+)?)s")
ENERGY_RE = re.compile(r"Total energy:.*\bE_total:\s*(-?[0-9]+(?:\.[0-9]+)?)")
FLOPS_RE = re.compile(r"Total flops:.*\bDP:\s*([0-9]+(?:\.[0-9]+)?)\s+PFLOPS")


class Reporter:
    def __init__(self) -> None:
        self.passes = 0
        self.failures = 0

    def check(self, condition: bool, label: str, detail: str = "") -> bool:
        prefix = "PASS" if condition else "FAIL"
        suffix = f" -- {detail}" if detail else ""
        print(f"{prefix}: {label}{suffix}")
        if condition:
            self.passes += 1
        else:
            self.failures += 1
        return condition

    @staticmethod
    def info(label: str, detail: str = "") -> None:
        suffix = f" -- {detail}" if detail else ""
        print(f"INFO: {label}{suffix}")


def as_int(row: dict[str, str], name: str) -> int:
    return int(float(row[name]))


def as_float(row: dict[str, str], name: str) -> float:
    return float(row[name])


def read_csv(
    path: Path, required: set[str], reporter: Reporter
) -> list[dict[str, str]]:
    if not reporter.check(path.is_file(), f"{path.name} exists", str(path)):
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = sorted(required - columns)
            reporter.check(
                not missing,
                f"{path.name} required columns",
                "complete" if not missing else f"missing {', '.join(missing)}",
            )
            rows = list(reader)
    except (OSError, csv.Error) as error:
        reporter.check(False, f"{path.name} readable", str(error))
        return []
    reporter.check(bool(rows), f"{path.name} has data", f"{len(rows)} rows")
    return rows


def validate_coverage(
    name: str,
    rows: Sequence[dict[str, str]],
    expected_systems: set[str],
    expected_distributors: set[str],
    reporter: Reporter,
) -> None:
    systems = {row.get("system", "").strip() for row in rows}
    distributors = {row.get("distributor", "").strip() for row in rows}
    reporter.check(
        expected_systems <= systems,
        f"{name} system coverage",
        f"found {', '.join(sorted(systems))}",
    )
    reporter.check(
        expected_distributors <= distributors,
        f"{name} distributor coverage",
        f"found {', '.join(sorted(distributors))}",
    )


def row_matches(
    row: dict[str, str],
    *,
    system: str,
    distributor: str,
    mode: str,
    expected_us: int,
    nodes: int,
    ranks_per_node: int,
) -> bool:
    try:
        return (
            row.get("system") == system
            and row.get("distributor") == distributor
            and row.get("mode") == mode
            and as_int(row, "expected_us") == expected_us
            and as_int(row, "duration_s") == 20
            and as_int(row, "nodes") == nodes
            and as_int(row, "world_size") == nodes * ranks_per_node
            and as_int(row, "max_upper_fanout") < 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def select_one(
    rows: Sequence[dict[str, str]],
    predicate: Callable[[dict[str, str]], bool],
    label: str,
    reporter: Reporter,
) -> dict[str, str] | None:
    matches = [row for row in rows if predicate(row)]
    reporter.check(
        len(matches) == 1,
        f"{label} row is unique",
        f"found {len(matches)}",
    )
    return matches[0] if len(matches) == 1 else None


def near(value: float, target: float, relative_tolerance: float) -> bool:
    return math.isclose(value, target, rel_tol=relative_tolerance, abs_tol=0.0)


def task_executing_workers(row: dict[str, str]) -> int:
    """Worker model used by the paper's hierarchical efficiency lines."""
    nodes = as_int(row, "nodes")
    world_size = as_int(row, "world_size")
    ranks_per_node = world_size // nodes
    return max(1, nodes * (ranks_per_node - 1) - 1)


def validate_weak_scaling_claims(
    rows: Sequence[dict[str, str]], reporter: Reporter
) -> None:
    systems = (
        ("aurora", 7, 1.99e7, 0.83),
        ("frontier", 9, 2.16e7, 0.79),
    )
    for system, ranks_per_node, rate_target, efficiency_target in systems:
        rate_row = select_one(
            rows,
            lambda row, s=system, r=ranks_per_node: row_matches(
                row,
                system=s,
                distributor="hierarchical_lockfree_rma",
                mode="uniform",
                expected_us=10,
                nodes=2048,
                ranks_per_node=r,
            ),
            f"{system} 2048-node 10 us default-fanout",
            reporter,
        )
        if rate_row is not None:
            rate = as_float(rate_row, "throughput_tasks_per_s")
            reporter.check(
                near(rate, rate_target, 0.01),
                f"{system} 2048-node 10 us rate",
                f"{rate:.6g} tasks/s (target about {rate_target:.3g})",
            )

        efficiency_row = select_one(
            rows,
            lambda row, s=system, r=ranks_per_node: row_matches(
                row,
                system=s,
                distributor="hierarchical_lockfree_rma",
                mode="uniform",
                expected_us=1000,
                nodes=2048,
                ranks_per_node=r,
            ),
            f"{system} 2048-node 1 ms default-fanout",
            reporter,
        )
        if efficiency_row is not None:
            rate = as_float(efficiency_row, "throughput_tasks_per_s")
            workers = task_executing_workers(efficiency_row)
            ideal_rate = workers / 0.001
            efficiency = rate / ideal_rate
            reporter.check(
                math.isclose(efficiency, efficiency_target, abs_tol=0.01),
                f"{system} 2048-node 1 ms efficiency",
                f"{efficiency:.1%} from {workers} task-executing workers",
            )


def validate_large_scale_claim(
    rows: Sequence[dict[str, str]], reporter: Reporter
) -> None:
    row = select_one(
        rows,
        lambda candidate: row_matches(
            candidate,
            system="frontier",
            distributor="hierarchical_lockfree_rma",
            mode="uniform",
            expected_us=1000,
            nodes=8192,
            ranks_per_node=9,
        ),
        "Frontier 8192-node 1 ms default-fanout",
        reporter,
    )
    if row is None:
        return
    rate = as_float(row, "throughput_tasks_per_s")
    workers = task_executing_workers(row)
    efficiency = rate / (workers / 0.001)
    reporter.check(
        near(rate, 2.93e7, 0.01),
        "Frontier 8192-node 1 ms rate",
        f"{rate:.6g} tasks/s",
    )
    reporter.check(
        math.isclose(efficiency, 0.44, abs_tol=0.01),
        "Frontier 8192-node 1 ms efficiency",
        f"{efficiency:.1%} from {workers} task-executing workers",
    )


def validate_load_balancing_repeats(
    rows: Sequence[dict[str, str]], reporter: Reporter
) -> None:
    measured = {
        "repeat",
        "total_tasks",
        "elapsed_s",
        "construct_s",
        "warmup_s",
        "finalize_s",
        "destruct_s",
        "job_id",
    }
    key_columns = sorted(REQUIRED_COLUMNS["load_balancing"] - measured)
    repeats: dict[tuple[str, ...], Counter[int]] = defaultdict(Counter)
    malformed = 0
    for row in rows:
        try:
            key = tuple(row.get(column, "") for column in key_columns)
            repeats[key][as_int(row, "repeat")] += 1
        except (KeyError, TypeError, ValueError):
            malformed += 1
    expected = {0, 1, 2}
    incomplete = [
        key
        for key, counts in repeats.items()
        if set(counts) != expected or any(count != 1 for count in counts.values())
    ]
    reporter.check(
        malformed == 0,
        "load-balancing repeat values parse",
        f"{malformed} malformed rows",
    )
    reporter.check(
        bool(repeats) and not incomplete,
        "load-balancing repeat coverage",
        f"{len(repeats)} configurations each contain repeats 0, 1, and 2; "
        f"{len(incomplete)} incomplete",
    )


def parse_exess_log(path: Path) -> tuple[float, float, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    harvest = HARVEST_RE.search(text)
    energy = ENERGY_RE.search(text)
    flops = FLOPS_RE.search(text)
    if harvest is None or energy is None or flops is None:
        missing = []
        if harvest is None:
            missing.append("harvesting time")
        if energy is None:
            missing.append("total energy")
        if flops is None:
            missing.append("total floating-point work")
        raise ValueError(f"missing {' and '.join(missing)}")
    return float(harvest.group(1)), float(energy.group(1)), float(flops.group(1))


def validate_exess(exess_dir: Path, reporter: Reporter) -> None:
    expected_times = {
        "naive": 100.3,
        "hierarchical": 19.9,
        "hierarchical_rma": 7.4,
    }
    measurements: dict[str, tuple[float, float, float]] = {}
    for implementation, target in expected_times.items():
        logs = sorted((exess_dir / implementation).glob("slurm-*.out"))
        if not reporter.check(
            len(logs) == 1,
            f"EXESS {implementation} log coverage",
            f"found {len(logs)} logs",
        ):
            continue
        try:
            harvest, energy, flops = parse_exess_log(logs[0])
        except (OSError, ValueError) as error:
            reporter.check(False, f"EXESS {implementation} log parses", str(error))
            continue
        measurements[implementation] = (harvest, energy, flops)
        reporter.check(
            near(harvest, target, 0.02),
            f"EXESS {implementation} harvesting time",
            f"{harvest:.6f} s (target about {target:.1f} s)",
        )

    if set(measurements) != set(expected_times):
        return
    naive = measurements["naive"][0]
    hierarchical = measurements["hierarchical"][0]
    hierarchical_rma = measurements["hierarchical_rma"][0]
    reporter.check(
        near(naive / hierarchical, 5.05, 0.02),
        "EXESS hierarchical speedup",
        f"{naive / hierarchical:.2f}x",
    )
    reporter.check(
        near(naive / hierarchical_rma, 13.6, 0.02),
        "EXESS hierarchical RMA speedup",
        f"{naive / hierarchical_rma:.2f}x",
    )
    energies = [measurement[1] for measurement in measurements.values()]
    energy_spread = max(energies) - min(energies)
    reporter.check(
        energy_spread <= 1.0e-4,
        "EXESS energies are consistent",
        f"range {min(energies):.8f} to {max(energies):.8f} Eh "
        f"(spread {energy_spread:.2g} Eh)",
    )
    flops = [measurement[2] for measurement in measurements.values()]
    flops_spread = max(flops) - min(flops)
    reporter.check(
        flops_spread <= 2.0e-6,
        "EXESS floating-point work is consistent",
        f"range {min(flops):.6f} to {max(flops):.6f} PFLOP "
        f"(spread {flops_spread:.2g} PFLOP)",
    )


def variability_csv_paths(inputs: Sequence[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for entry in inputs:
        candidates = [entry] if entry.is_file() else sorted(entry.rglob("*.csv"))
        for candidate in candidates:
            if "weak_scaling" not in candidate.name.lower():
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield candidate


def read_variability_rows(
    inputs: Sequence[Path], reporter: Reporter
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = list(variability_csv_paths(inputs))
    reporter.check(
        bool(paths),
        "variability input coverage",
        f"found {len(paths)} weak-scaling CSVs",
    )
    for path in paths:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        except (OSError, csv.Error) as error:
            reporter.check(False, f"variability input {path} readable", str(error))
    return rows


VARIABILITY_EXPECTED_US = {10, 100, 1000, 10000, 100000, 1000000}
VARIABILITY_KEY_COLUMNS = (
    "system",
    "distributor",
    "mode",
    "expected_us",
    "duration_s",
    "nodes",
    "world_size",
    "workers",
    "max_upper_fanout",
)


def variability_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(column, "").strip() for column in VARIABILITY_KEY_COLUMNS)


def is_known_aurora_variability_row(row: dict[str, str]) -> bool:
    try:
        return (
            row.get("system") == "aurora"
            and row.get("distributor") == "hierarchical_lockfree_rma"
            and row.get("mode") in {"fixed", "uniform"}
            and as_int(row, "expected_us") in VARIABILITY_EXPECTED_US
            and as_int(row, "duration_s") == 20
            and as_int(row, "nodes") == 64
            and as_int(row, "world_size") == 6528
            and as_int(row, "workers") == 6527
            and as_int(row, "max_upper_fanout") < 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_variability(
    inputs: Sequence[Path], reporter: Reporter
) -> None:
    rows = read_variability_rows(inputs, reporter)
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    malformed = 0
    for row in rows:
        if not is_known_aurora_variability_row(row):
            continue
        try:
            groups[variability_key(row)].append(as_float(row, "throughput_tasks_per_s"))
        except (KeyError, TypeError, ValueError):
            malformed += 1

    repeated = {key: values for key, values in groups.items() if len(values) >= 2}
    reporter.check(
        malformed == 0,
        "Aurora variability throughput values parse",
        f"{malformed} malformed rows",
    )
    reporter.check(
        len(repeated) == 12,
        "Aurora 64-node repeat configuration coverage",
        f"{len(repeated)} of 12 configurations have at least two runs",
    )
    if len(repeated) == 12:
        cvs = [
            statistics.stdev(values) / statistics.mean(values) * 100.0
            for values in repeated.values()
        ]
        median_cv = statistics.median(cvs)
        max_cv = max(cvs)
        reporter.check(
            math.isclose(median_cv, 4.2, abs_tol=0.2),
            "Aurora repeat throughput median CV",
            f"{median_cv:.3f}%",
        )
        reporter.check(
            math.isclose(max_cv, 10.7, abs_tol=0.2),
            "Aurora repeat throughput maximum CV",
            f"{max_cv:.3f}%",
        )

    frontier: dict[tuple[str, ...], int] = Counter(
        variability_key(row) for row in rows if row.get("system") == "frontier"
    )
    frontier_repeated = sum(count >= 2 for count in frontier.values())
    if frontier:
        reporter.check(
            frontier_repeated == 0,
            "Frontier repeat coverage",
            f"{frontier_repeated} repeated configurations",
        )
    else:
        reporter.info(
            "Frontier repeat coverage",
            "no Frontier rows supplied; the artifact has no archived Frontier repeats",
        )


def main(argv: Sequence[str] | None = None) -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Validate committed DynaMPI artifact data and paper-facing claims."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir.parent / "results",
        help="Directory containing committed consolidated CSVs",
    )
    parser.add_argument(
        "--exess-dir",
        type=Path,
        default=script_dir.parent / "exess",
        help="Directory containing the three committed EXESS runs",
    )
    parser.add_argument(
        "--variability-input",
        type=Path,
        action="append",
        default=[],
        help="Raw result directory or weak-scaling all-runs CSV. Repeat to combine inputs.",
    )
    args = parser.parse_args(argv)

    reporter = Reporter()
    rows: dict[str, list[dict[str, str]]] = {}
    for name, columns in REQUIRED_COLUMNS.items():
        rows[name] = read_csv(args.results_dir / f"{name}.csv", columns, reporter)

    if rows["weak_scaling"]:
        validate_coverage(
            "weak_scaling",
            rows["weak_scaling"],
            {"aurora", "frontier"},
            DISTRIBUTORS,
            reporter,
        )
        validate_weak_scaling_claims(rows["weak_scaling"], reporter)
    if rows["shutdown"]:
        validate_coverage(
            "shutdown",
            rows["shutdown"],
            {"aurora", "frontier"},
            DISTRIBUTORS,
            reporter,
        )
    if rows["load_balancing"]:
        validate_coverage(
            "load_balancing",
            rows["load_balancing"],
            {"frontier"},
            DISTRIBUTORS,
            reporter,
        )
        validate_load_balancing_repeats(rows["load_balancing"], reporter)
    if rows["large_scale"]:
        validate_coverage(
            "large_scale",
            rows["large_scale"],
            {"frontier"},
            {"hierarchical_lockfree_rma"},
            reporter,
        )
        validate_large_scale_claim(rows["large_scale"], reporter)

    validate_exess(args.exess_dir, reporter)
    if args.variability_input:
        validate_variability(args.variability_input, reporter)
    else:
        reporter.info(
            "variability validation skipped",
            "pass --variability-input with raw directories or an all-runs CSV",
        )

    print(
        f"\nArtifact validation: {reporter.passes} passed, "
        f"{reporter.failures} failed."
    )
    return 1 if reporter.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
