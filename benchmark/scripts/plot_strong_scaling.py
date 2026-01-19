#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "system": row.get("system", "").strip() or "unknown",
                    "distributor": row.get("distributor", "").strip(),
                    "mode": row.get("mode", "").strip(),
                    "expected_us": int(float(row.get("expected_us", 0))),
                    "nodes": int(float(row.get("nodes", 0))),
                    "throughput": float(row.get("throughput_tasks_per_s", 0.0)),
                }
            )
    return rows


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["system"], row["distributor"], row["mode"], row["expected_us"])
        grouped[key].append((row["nodes"], row["throughput"]))
    return grouped


def plot_distributor(system, distributor, grouped, output_dir, image_format):
    modes = ["fixed", "poisson"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f"Strong Scaling Distribution Rate ({distributor}) - {system}")

    for ax, mode in zip(axes, modes):
        ax.set_title(mode.capitalize())
        series = []
        for (sys_name, dist, mode_name, expected_us), points in grouped.items():
            if sys_name != system or dist != distributor or mode_name != mode:
                continue
            points_sorted = sorted(points, key=lambda x: x[0])
            nodes = [p[0] for p in points_sorted]
            throughput = [p[1] for p in points_sorted]
            series.append((expected_us, nodes, throughput))

        for expected_us, nodes, throughput in sorted(series, key=lambda x: x[0]):
            label = f"{expected_us} us"
            ax.plot(nodes, throughput, marker="o", label=label)

        ax.set_xlabel("Nodes")
        ax.set_ylabel("Tasks per second")
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()

    filename = f"strong_scaling_{system}_{distributor}.{image_format}"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot strong scaling distribution throughput.")
    parser.add_argument("--input", required=True, help="CSV file with benchmark results")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write output plots"
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg"], help="Output image format"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = parse_rows(args.input)
    grouped = group_rows(rows)

    systems = sorted({row["system"] for row in rows})
    distributors = sorted({row["distributor"] for row in rows})
    for system in systems:
        for distributor in distributors:
            plot_distributor(system, distributor, grouped, args.output_dir, args.format)


if __name__ == "__main__":
    main()
