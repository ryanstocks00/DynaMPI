#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 QDX Technologies. Authored by Ryan Stocks <ryan.stocks00@gmail.com>
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter
import scienceplots

# IEEE styling parameters
IEEE_FIG_WIDTH = 3.5  # Single column width in inches
IEEE_FIG_HEIGHT = 3.5  # Height in inches

# Hollow marker shapes for different series
MARKER_SHAPES = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '<', '>', 'd']


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
        file_mtime = os.path.getmtime(path)
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                nodes = int(float(row.get("nodes", 0)))
                world_size = int(float(row.get("world_size", 0)))
                workers = int(float(row.get("workers", 0)))
                time_per_shutdown_us = float(row.get("time_per_shutdown_us", 0.0))
                rows.append(
                    {
                        "system": row.get("system", "").strip() or "unknown",
                        "nodes": nodes,
                        "world_size": world_size,
                        "workers": workers,
                        "time_per_shutdown_us": time_per_shutdown_us,
                        "file_mtime": file_mtime,
                    }
                )
    return rows


def group_rows(rows):
    # First, filter to keep only newest results for each unique configuration
    # Key: (system, nodes)
    # Value: (time_per_shutdown_us, file_mtime)
    newest_by_config = {}
    for row in rows:
        config_key = (
            row["system"],
            row["nodes"],
        )
        if config_key not in newest_by_config:
            newest_by_config[config_key] = (row["time_per_shutdown_us"], row["file_mtime"])
        else:
            # Keep the one from the newest file
            _, existing_mtime = newest_by_config[config_key]
            if row["file_mtime"] > existing_mtime:
                newest_by_config[config_key] = (row["time_per_shutdown_us"], row["file_mtime"])

    # Now group by system for plotting
    grouped = defaultdict(list)
    for (system, nodes), (time_per_shutdown_us, _) in newest_by_config.items():
        grouped[system].append((nodes, time_per_shutdown_us))
    return grouped


def plot_all_systems(grouped, output_dir, image_format):
    # Use scienceplots IEEE style
    with plt.style.context(['science', 'ieee']):
        fig, ax = plt.subplots(figsize=(IEEE_FIG_WIDTH, IEEE_FIG_HEIGHT))

        # Filter out "local" system
        systems = sorted([s for s in grouped.keys() if s != "local"])
        all_nodes = set()
        handles = []
        labels = []

        # Plot each system with different markers/colors
        for idx, system in enumerate(systems):
            points = grouped[system]
            points_sorted = sorted(points, key=lambda x: x[0])
            nodes = [p[0] for p in points_sorted]
            time_per_shutdown_us = [p[1] for p in points_sorted]
            # Convert microseconds to seconds
            time_per_shutdown_s = [t / 1_000_000.0 for t in time_per_shutdown_us]

            all_nodes.update(nodes)

            marker = MARKER_SHAPES[idx % len(MARKER_SHAPES)]
            color = plt.cm.tab10(idx % 10)

            # Plot data
            line, = ax.plot(nodes, time_per_shutdown_s, marker=marker, fillstyle='none',
                           markeredgewidth=1.0, linewidth=1.0, color=color, label=system.capitalize())
            handles.append(line)
            labels.append(system.capitalize())

        ax.set_xlabel("Nodes")
        ax.set_ylabel("Shutdown time (s)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")

        # Show actual node counts (1, 2, 4, 8, 16, ...) rather than 2^n formatting.
        # Keep the log2 spacing but format ticks as plain integers.
        if all_nodes:
            node_ticks = sorted(all_nodes)
            ax.xaxis.set_major_locator(FixedLocator(node_ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

        # Add very light grey underlying grid
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, color='lightgrey', alpha=0.5, zorder=0)

        # Add legend
        ax.legend(handles, labels, frameon=False, loc='best')

        filename = f"shutdown_time_combined.{image_format}"
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot shutdown time vs number of nodes.")
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

    # Plot all systems on the same figure
    plot_all_systems(grouped, args.output_dir, args.format)


if __name__ == "__main__":
    main()
