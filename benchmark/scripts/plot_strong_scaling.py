#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def format_duration(expected_ns):
    if expected_ns <= 0:
        return "0 ns"
    if expected_ns >= 1_000_000_000:
        return f"{expected_ns / 1_000_000_000:g} s"
    if expected_ns >= 1_000_000:
        return f"{expected_ns / 1_000_000:g} ms"
    if expected_ns >= 1_000:
        return f"{expected_ns / 1_000:g} us"
    return f"{expected_ns:g} ns"


def collect_csv_paths(inputs):
    paths = []
    for raw in inputs:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if os.path.isdir(entry):
                for root, _, files in os.walk(entry):
                    for name in files:
                        if name.endswith(".csv"):
                            paths.append(os.path.join(root, name))
            else:
                paths.append(entry)
    return paths


def parse_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                expected_ns_raw = row.get("expected_ns", "").strip()
                expected_us_raw = row.get("expected_us", "").strip()
                if expected_ns_raw:
                    expected_ns = int(float(expected_ns_raw))
                elif expected_us_raw:
                    expected_ns = int(float(expected_us_raw) * 1000)
                else:
                    expected_ns = 0
                nodes = int(float(row.get("nodes", 0)))
                world_size = int(float(row.get("world_size", 0)))
                ranks_per_node = int(round(world_size / nodes)) if nodes else 0
                rows.append(
                    {
                        "system": row.get("system", "").strip() or "unknown",
                        "distributor": row.get("distributor", "").strip(),
                        "mode": row.get("mode", "").strip(),
                        "expected_ns": expected_ns,
                        "nodes": nodes,
                        "ranks_per_node": ranks_per_node,
                        "throughput": float(row.get("throughput_tasks_per_s", 0.0)),
                    }
                )
    return rows


def group_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["system"],
            row["distributor"],
            row["mode"],
            row["expected_ns"],
            row["ranks_per_node"],
        )
        grouped[key].append((row["nodes"], row["throughput"]))
    return grouped


def plot_distributor(system, distributor, grouped, output_dir, image_format):
    modes = ["fixed", "random"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle(f"Strong Scaling Distribution Rate ({distributor}) - {system}")

    for ax, mode in zip(axes, modes):
        ax.set_title(mode.capitalize())
        series = []
        for (
            sys_name,
            dist,
            mode_name,
            expected_ns,
            ranks_per_node,
        ), points in grouped.items():
            # Handle backward compatibility: treat "poisson" as "random"
            normalized_mode = "random" if mode_name == "poisson" else mode_name
            if sys_name != system or dist != distributor or normalized_mode != mode:
                continue
            points_sorted = sorted(points, key=lambda x: x[0])
            nodes = [p[0] for p in points_sorted]
            throughput = [p[1] for p in points_sorted]
            series.append((expected_ns, ranks_per_node, nodes, throughput))

        for expected_ns, ranks_per_node, nodes, throughput in sorted(
            series, key=lambda x: (x[0], x[1])
        ):
            rpn_label = f"{ranks_per_node} rpn" if ranks_per_node else "rpn=?"
            label = f"{format_duration(expected_ns)}, {rpn_label}"
            ax.plot(nodes, throughput, marker="o", label=label)

        ax.set_xlabel("Nodes")
        ax.set_ylabel("Tasks per second")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend()

    filename = f"strong_scaling_{system}_{distributor}.{image_format}"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot strong scaling distribution throughput.")
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        help="CSV file or directory (can be passed multiple times)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write output plots"
    )
    parser.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg"], help="Output image format"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_paths = collect_csv_paths(args.input)
    rows = parse_rows(input_paths)
    grouped = group_rows(rows)

    systems = sorted({row["system"] for row in rows})
    distributors = sorted({row["distributor"] for row in rows})
    for system in systems:
        for distributor in distributors:
            plot_distributor(system, distributor, grouped, args.output_dir, args.format)


if __name__ == "__main__":
    main()
