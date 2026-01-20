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
IEEE_FIG_HEIGHT = 3.5  # Height in inches (increased for bottom legend)

# Hollow marker shapes for different series
MARKER_SHAPES = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '<', '>', 'd']


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
        file_mtime = os.path.getmtime(path)
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
                        "file_mtime": file_mtime,
                    }
                )
    return rows


def group_rows(rows):
    # First, filter to keep only newest results for each unique configuration
    # Key: (system, distributor, mode, expected_ns, ranks_per_node, nodes)
    # Value: (throughput, file_mtime)
    # Note: Normalize mode for backward compatibility (poisson -> random)
    newest_by_config = {}
    for row in rows:
        # Handle backward compatibility: treat "poisson" as "random"
        normalized_mode = "random" if row["mode"] == "poisson" else row["mode"]
        config_key = (
            row["system"],
            row["distributor"],
            normalized_mode,
            row["expected_ns"],
            row["ranks_per_node"],
            row["nodes"],
        )
        if config_key not in newest_by_config:
            newest_by_config[config_key] = (row["throughput"], row["file_mtime"])
        else:
            # Keep the one from the newest file
            _, existing_mtime = newest_by_config[config_key]
            if row["file_mtime"] > existing_mtime:
                newest_by_config[config_key] = (row["throughput"], row["file_mtime"])

    # Now group by (system, distributor, mode, expected_ns, ranks_per_node) for plotting
    grouped = defaultdict(list)
    for (system, distributor, mode, expected_ns, ranks_per_node, nodes), (throughput, _) in newest_by_config.items():
        key = (system, distributor, mode, expected_ns, ranks_per_node)
        grouped[key].append((nodes, throughput))
    return grouped


def plot_distributor(system, distributor, grouped, output_dir, image_format):
    modes = ["fixed", "random"]

    # Create separate plots for each mode
    for mode in modes:
        # Use scienceplots IEEE style
        plt.style.use(['science', 'ieee'])

        fig, ax = plt.subplots(figsize=(IEEE_FIG_WIDTH, IEEE_FIG_HEIGHT))

        series = []
        all_nodes = set()
        ranks_per_node_value = None
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
            all_nodes.update(nodes)
            if ranks_per_node_value is None:
                ranks_per_node_value = ranks_per_node
            series.append((expected_ns, ranks_per_node, nodes, throughput))

        # Sort series by expected_ns (duration) to ensure proper ordering
        series_sorted = sorted(series, key=lambda x: x[0])  # Sort by expected_ns only
        handles = []
        labels = []

        for idx, (expected_ns, ranks_per_node, nodes, throughput) in enumerate(series_sorted):
            # Remove rpn from legend label, only show duration
            label = format_duration(expected_ns)
            marker = MARKER_SHAPES[idx % len(MARKER_SHAPES)]
            # Use matplotlib's default color cycle for different colors
            line, = ax.plot(nodes, throughput, marker=marker, label=label,
                           fillstyle='none', markeredgewidth=1.0,
                           color=plt.cm.tab10(idx % 10))
            handles.append(line)
            labels.append(label)

        ax.set_xlabel("Nodes")
        ax.set_ylabel("Tasks per second")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        # Show actual node counts (2, 4, 8, 16, ...) rather than 2^n formatting.
        # Keep the log2 spacing but format ticks as plain integers.
        if all_nodes:
            node_ticks = sorted(all_nodes)
            ax.xaxis.set_major_locator(FixedLocator(node_ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

        # Add very light grey underlying grid
        ax.grid(True, which="both", linestyle="-", linewidth=0.5, color='lightgrey', alpha=0.5, zorder=0)

        # Reorder handles and labels to go across columns first (row-major)
        # Matplotlib's legend with ncol fills column-major (down columns first),
        # so we need to transpose the order to get row-major display
        ncol = 4
        n_items = len(handles)
        n_rows = (n_items + ncol - 1) // ncol  # Ceiling division

        # Create reordered lists: transpose so matplotlib's column-major fill gives row-major display
        reordered_handles = []
        reordered_labels = []
        for col in range(ncol):
            for row in range(n_rows):
                idx = row * ncol + col
                if idx < n_items:
                    reordered_handles.append(handles[idx])
                    reordered_labels.append(labels[idx])

        # Compact legend with increased column spacing - 4 columns at bottom, no border
        # Items ordered by duration (1us, 10us, 100us, ...) going across columns first
        legend = ax.legend(reordered_handles, reordered_labels,
                          frameon=False,
                          ncol=ncol, columnspacing=0.8,
                          loc='upper center', bbox_to_anchor=(0.5, -0.15))

        # Add rpn to filename
        rpn_str = f"_{ranks_per_node_value}rpn" if ranks_per_node_value else ""
        filename = f"strong_scaling_{system}_{distributor}_{mode}{rpn_str}.{image_format}"
        fig.tight_layout(rect=[0, 0.12, 1, 1])  # Leave space at bottom for legend
        fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
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
