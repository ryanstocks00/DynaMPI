#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import TypedDict

import matplotlib.pyplot as plt

from plot_common import (
    IEEE_FIG_HEIGHT,
    IEEE_FIG_WIDTH,
    add_light_grid,
    add_plot_cli_args,
    collect_csv_paths,
    dedupe_newest,
    save_figure,
    series_color,
    series_marker,
    set_log_node_axes,
)


class ShutdownRow(TypedDict):
    system: str
    nodes: int
    world_size: int
    workers: int
    time_per_shutdown_us: float
    file_mtime: float


def parse_rows(paths: Sequence[str]) -> list[ShutdownRow]:
    rows: list[ShutdownRow] = []
    for path in paths:
        file_mtime = os.path.getmtime(path)
        with open(path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if "time_per_shutdown_us" not in row:
                    continue
                time_per_shutdown_us = float(row.get("time_per_shutdown_us", 0.0))
                if time_per_shutdown_us <= 0.0:
                    continue
                rows.append(
                    {
                        "system": row.get("system", "").strip() or "unknown",
                        "nodes": int(float(row.get("nodes", 0))),
                        "world_size": int(float(row.get("world_size", 0))),
                        "workers": int(float(row.get("workers", 0))),
                        "time_per_shutdown_us": time_per_shutdown_us,
                        "file_mtime": file_mtime,
                    }
                )
    return rows


def group_rows(rows: Sequence[ShutdownRow]) -> dict[str, list[tuple[int, float]]]:
    newest = dedupe_newest(rows, lambda row: (row["system"], row["nodes"]), "time_per_shutdown_us")
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (system, nodes), (time_per_shutdown_us, _) in newest.items():
        grouped[system].append((nodes, time_per_shutdown_us))
    return grouped


def plot_all_systems(
    grouped: dict[str, list[tuple[int, float]]],
    output_dir: str,
    image_format: str,
) -> None:
    with plt.style.context(['science', 'ieee']):
        fig, ax = plt.subplots(figsize=(IEEE_FIG_WIDTH, IEEE_FIG_HEIGHT))

        systems = sorted(system for system in grouped if system != "local")
        all_nodes: set[int] = set()
        handles = []
        labels = []

        for idx, system in enumerate(systems):
            points_sorted = sorted(grouped[system], key=lambda point: point[0])
            nodes = [point[0] for point in points_sorted]
            time_per_shutdown_s = [point[1] / 1_000_000.0 for point in points_sorted]
            all_nodes.update(nodes)

            line, = ax.plot(
                nodes,
                time_per_shutdown_s,
                marker=series_marker(idx),
                fillstyle='none',
                markeredgewidth=1.0,
                linewidth=1.0,
                color=series_color(idx),
                label=system.capitalize(),
            )
            handles.append(line)
            labels.append(system.capitalize())

        ax.set_xlabel("Nodes")
        ax.set_ylabel("Shutdown time (s)")
        set_log_node_axes(ax, all_nodes)
        add_light_grid(ax)
        ax.legend(handles, labels, frameon=False, loc='best')

        save_figure(fig, output_dir, f"shutdown_time_combined.{image_format}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot shutdown time vs number of nodes.")
    add_plot_cli_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    input_paths = collect_csv_paths(args.input, "shutdown")
    grouped = group_rows(parse_rows(input_paths))
    plot_all_systems(grouped, args.output_dir, args.format)


if __name__ == "__main__":
    main()
