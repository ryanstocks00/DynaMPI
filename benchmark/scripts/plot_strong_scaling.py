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
    normalize_mode,
    save_figure,
    series_color,
    series_marker,
    set_log_node_axes,
)


class StrongScalingRow(TypedDict):
    system: str
    distributor: str
    mode: str
    expected_ns: int
    nodes: int
    ranks_per_node: int
    throughput: float
    file_mtime: float


def format_duration(expected_ns: int | float) -> str:
    if expected_ns <= 0:
        return "0 ns"
    if expected_ns >= 1_000_000_000:
        return f"{expected_ns / 1_000_000_000:g} s"
    if expected_ns >= 1_000_000:
        return f"{expected_ns / 1_000_000:g} ms"
    if expected_ns >= 1_000:
        return f"{expected_ns / 1_000:g} us"
    return f"{expected_ns:g} ns"


def parse_rows(paths: Sequence[str]) -> list[StrongScalingRow]:
    rows: list[StrongScalingRow] = []
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


def group_rows(
    rows: Sequence[StrongScalingRow],
) -> dict[tuple[str, str, str, int, int], list[tuple[int, float]]]:
    newest = dedupe_newest(
        rows,
        lambda row: (
            row["system"],
            row["distributor"],
            normalize_mode(row["mode"]),
            row["expected_ns"],
            row["ranks_per_node"],
            row["nodes"],
        ),
        "throughput",
    )
    grouped: dict[tuple[str, str, str, int, int], list[tuple[int, float]]] = defaultdict(list)
    for (
        system,
        distributor,
        mode,
        expected_ns,
        ranks_per_node,
        nodes,
    ), (throughput, _) in newest.items():
        grouped[(system, distributor, mode, expected_ns, ranks_per_node)].append((nodes, throughput))
    return grouped


def plot_distributor(
    system: str,
    distributor: str,
    grouped: dict[tuple[str, str, str, int, int], list[tuple[int, float]]],
    output_dir: str,
    image_format: str,
) -> None:
    for mode in ("fixed", "random"):
        with plt.style.context(['science', 'ieee']):
            fig, ax = plt.subplots(figsize=(IEEE_FIG_WIDTH, IEEE_FIG_HEIGHT))

            series = []
            all_nodes: set[int] = set()
            ranks_per_node_value: int | None = None
            for (
                sys_name,
                dist,
                mode_name,
                expected_ns,
                ranks_per_node,
            ), points in grouped.items():
                if sys_name != system or dist != distributor or normalize_mode(mode_name) != mode:
                    continue
                points_sorted = sorted(points, key=lambda point: point[0])
                nodes = [point[0] for point in points_sorted]
                throughput = [point[1] for point in points_sorted]
                all_nodes.update(nodes)
                if ranks_per_node_value is None:
                    ranks_per_node_value = ranks_per_node
                series.append((expected_ns, ranks_per_node, nodes, throughput))

            if not series:
                plt.close(fig)
                continue

            series_sorted = sorted(series, key=lambda item: item[0])
            handles = []
            labels = []

            for idx, (expected_ns, ranks_per_node, nodes, throughput) in enumerate(series_sorted):
                label = format_duration(expected_ns)
                line, = ax.plot(
                    nodes,
                    throughput,
                    marker=series_marker(idx),
                    label=label,
                    fillstyle='none',
                    markeredgewidth=1.0,
                    color=series_color(idx),
                )
                handles.append(line)
                labels.append(label)

            ax.set_xlabel("Nodes")
            ax.set_ylabel("Tasks per second")
            set_log_node_axes(ax, all_nodes)
            add_light_grid(ax)

            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for idx, (expected_ns, ranks_per_node, _nodes, _throughput) in enumerate(series_sorted):
                if all_nodes:
                    ideal_nodes = sorted(all_nodes)
                    ideal_throughput = [n * ranks_per_node * 1e9 / expected_ns for n in ideal_nodes]
                    ax.plot(
                        ideal_nodes,
                        ideal_throughput,
                        linestyle='--',
                        color=series_color(idx),
                        linewidth=1.0,
                        alpha=0.5,
                        zorder=0,
                    )
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

            ncol = 4
            n_items = len(handles)
            n_rows = (n_items + ncol - 1) // ncol
            reordered_handles = []
            reordered_labels = []
            for col in range(ncol):
                for row in range(n_rows):
                    idx = row * ncol + col
                    if idx < n_items:
                        reordered_handles.append(handles[idx])
                        reordered_labels.append(labels[idx])

            ax.legend(
                reordered_handles,
                reordered_labels,
                frameon=False,
                ncol=ncol,
                columnspacing=0.8,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.15),
            )

            rpn_str = f"_{ranks_per_node_value}rpn" if ranks_per_node_value is not None else ""
            filename = f"strong_scaling_{system}_{distributor}_{mode}{rpn_str}.{image_format}"
            save_figure(
                fig,
                output_dir,
                filename,
                tight_layout_rect=(0, 0.12, 1, 1),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot strong scaling distribution throughput.")
    add_plot_cli_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = parse_rows(collect_csv_paths(args.input, "strong_scaling"))
    grouped = group_rows(rows)

    systems = sorted({row["system"] for row in rows})
    distributors = sorted({row["distributor"] for row in rows if row["distributor"].strip()})
    for system in systems:
        for distributor in distributors:
            plot_distributor(system, distributor, grouped, args.output_dir, args.format)


if __name__ == "__main__":
    main()
