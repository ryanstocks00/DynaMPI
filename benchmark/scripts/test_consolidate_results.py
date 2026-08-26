# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
"""Tests for provenance-preserving result consolidation."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from consolidate_results import all_runs_rows, build_consolidation, consolidate, main


FIELDS = [
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
]


class ConsolidateResultsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.results_dir = Path(self.temp_dir.name) / "results"
        self._write_run("1000001", [self._row("10", "1.0")])
        self._write_run(
            "1000002",
            [
                self._row("20", "2.0"),
                self._row("20", "2.1"),
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _row(duration_s: str, throughput: str) -> dict[str, str]:
        return {
            "system": "aurora",
            "distributor": "hierarchical_lockfree_rma",
            "mode": "uniform",
            "expected_us": "10",
            "duration_s": duration_s,
            "nodes": "64",
            "world_size": "448",
            "workers": "447",
            "max_upper_fanout": "-1",
            "total_tasks": "100",
            "elapsed_s": duration_s,
            "throughput_tasks_per_s": throughput,
        }

    def _write_run(self, job_id: str, rows: list[dict[str, str]]) -> None:
        run_dir = self.results_dir / "aurora" / f"64-dynampi_ws_aurora_64-{job_id}"
        run_dir.mkdir(parents=True)
        with (run_dir / "weak_scaling_aurora.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_default_selection_is_latest_and_deterministic(self) -> None:
        first = consolidate(str(self.results_dir), "weak_scaling")
        second = consolidate(str(self.results_dir), "weak_scaling")
        self.assertEqual(first, second)
        fieldnames, rows, seen = first
        self.assertEqual(seen, 3)
        self.assertIn("job_id", fieldnames)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["job_id"], "1000002")
        self.assertEqual(rows[0]["throughput_tasks_per_s"], "2.0")

    def test_all_runs_is_a_selection_report(self) -> None:
        result = build_consolidation(str(self.results_dir), "weak_scaling")
        fieldnames, rows = all_runs_rows(result)
        self.assertEqual(len(rows), 3)
        self.assertIn("source_path", fieldnames)
        self.assertIn("source_row", fieldnames)
        self.assertEqual([row["selected"] for row in rows], ["0", "1", "0"])
        self.assertEqual(
            [row["selection_reason"] for row in rows],
            [
                "superseded_by_newer_run",
                "selected_latest",
                "superseded_by_same_run_first_row",
            ],
        )

    def test_check_mode_creates_nothing(self) -> None:
        output_dir = Path(self.temp_dir.name) / "selected"
        audit_dir = Path(self.temp_dir.name) / "audit"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--results-dir",
                    str(self.results_dir),
                    "--output-dir",
                    str(output_dir),
                    "--all-runs-dir",
                    str(audit_dir),
                    "--sweeps",
                    "weak_scaling",
                    "--check",
                ]
            )
        self.assertEqual(result, 0)
        self.assertFalse(output_dir.exists())
        self.assertFalse(audit_dir.exists())

    def test_explicit_output_dirs_write_selected_and_all_runs(self) -> None:
        output_dir = Path(self.temp_dir.name) / "selected"
        audit_dir = Path(self.temp_dir.name) / "audit"
        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--results-dir",
                    str(self.results_dir),
                    "--output-dir",
                    str(output_dir),
                    "--all-runs-dir",
                    str(audit_dir),
                    "--sweeps",
                    "weak_scaling",
                ]
            )
        self.assertEqual(result, 0)
        with (output_dir / "weak_scaling.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            selected = list(csv.DictReader(handle))
        with (audit_dir / "weak_scaling_all_runs.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            all_runs = list(csv.DictReader(handle))
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["job_id"], "1000002")
        self.assertEqual(len(all_runs), 3)
        self.assertEqual(sum(row["selected"] == "1" for row in all_runs), 1)

    def test_no_selected_rows_leaves_existing_output_untouched(self) -> None:
        results_dir = Path(self.temp_dir.name) / "shutdown_results"
        run_dir = results_dir / "aurora" / "64-dynampi_shutdown_aurora_64-1000003"
        run_dir.mkdir(parents=True)
        shutdown_fields = ["system", "distributor", "nodes", "time_per_shutdown_us", "iterations"]
        with (run_dir / "shutdown_aurora.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=shutdown_fields, lineterminator="\n")
            writer.writeheader()
            # nodes above the is_plottable cutoff (2048): filtered out entirely.
            writer.writerow(
                {
                    "system": "aurora",
                    "distributor": "naive",
                    "nodes": "4096",
                    "time_per_shutdown_us": "5.0",
                    "iterations": "10",
                }
            )

        output_dir = Path(self.temp_dir.name) / "existing_selected"
        output_dir.mkdir()
        existing_output = output_dir / "shutdown.csv"
        with existing_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[*shutdown_fields, "job_id"], lineterminator="\n"
            )
            writer.writeheader()
            writer.writerow(
                {
                    "system": "aurora",
                    "distributor": "naive",
                    "nodes": "2",
                    "time_per_shutdown_us": "1.0",
                    "iterations": "10",
                    "job_id": "999",
                }
            )
        before = existing_output.read_text(encoding="utf-8")

        with redirect_stdout(io.StringIO()):
            result = main(
                [
                    "--results-dir",
                    str(results_dir),
                    "--output-dir",
                    str(output_dir),
                    "--sweeps",
                    "shutdown",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(existing_output.read_text(encoding="utf-8"), before)


if __name__ == "__main__":
    unittest.main()
