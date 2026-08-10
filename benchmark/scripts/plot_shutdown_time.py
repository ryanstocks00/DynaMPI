#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import TypedDict

from plot_common import (
    Recency,
    add_plot_cli_args,
    collect_csv_paths,
    dedupe_newest,
    distributor_series_index,
    filter_ranks_per_node,
    filter_systems,
    finish_compact_node_plot,
    format_distributor_label,
    ieee_figure,
    iter_csv_rows,
    plot_node_series,
    save_figure,
    sorted_series_xy,
)


class ShutdownRow(TypedDict):
    system: str
    distributor: str
    fanout: int
    ranks_per_node: int
    nodes: int
    world_size: int
    workers: int
    time_per_shutdown_us: float
    file_mtime: float
    path: str
    recency: Recency


def parse_rows(paths: Sequence[str]) -> list[ShutdownRow]:
    rows: list[ShutdownRow] = []
    for row, path, file_mtime, recency in iter_csv_rows(paths):
        if "time_per_shutdown_us" not in row:
            continue
        time_per_shutdown_us = float(row.get("time_per_shutdown_us", 0.0))
        if time_per_shutdown_us <= 0.0:
            continue
        nodes = int(float(row.get("nodes", 0)))
        if nodes > 2048:
            continue
        world_size = int(float(row.get("world_size", 0)))
        ranks_per_node = int(round(world_size / nodes)) if nodes else 0
        distributor = row.get("distributor", "").strip() or "naive"
        fanout = int(float(row.get("max_upper_fanout", -1) or -1))
        # Fanout only distinguishes hierarchical topologies.
        if "hierarchical" not in distributor:
            fanout = -1
        # Sanity-drop any leftover sub-microsecond hierarchical no-ops.
        if "hierarchical" in distributor and time_per_shutdown_us < 1.0:
            continue
        rows.append(
            {
                "system": row.get("system", "").strip() or "unknown",
                "distributor": distributor,
                "fanout": fanout,
                "ranks_per_node": ranks_per_node,
                "nodes": nodes,
                "world_size": world_size,
                "workers": int(float(row.get("workers", 0))),
                "time_per_shutdown_us": time_per_shutdown_us,
                "file_mtime": file_mtime,
                "path": path,
                "recency": recency,
            }
        )
    return rows


def group_rows(
    rows: Sequence[ShutdownRow],
) -> dict[tuple[str, str, int, int], list[tuple[int, float]]]:
    newest = dedupe_newest(
        rows,
        lambda row: (
            row["system"],
            row["distributor"],
            row["fanout"],
            row["ranks_per_node"],
            row["nodes"],
        ),
        "time_per_shutdown_us",
    )
    grouped: dict[tuple[str, str, int, int], list[tuple[int, float]]] = defaultdict(list)
    for (system, distributor, fanout, ranks_per_node, nodes), (
        time_per_shutdown_us,
        _,
    ) in newest.items():
        grouped[(system, distributor, fanout, ranks_per_node)].append(
            (nodes, time_per_shutdown_us)
        )
    return grouped


def plot_system_rpn(
    system: str,
    ranks_per_node: int,
    grouped: dict[tuple[str, str, int, int], list[tuple[int, float]]],
    output_dir: str,
    image_format: str,
) -> None:
    # Legend order matches the ranking a reader wants: quickest teardown
    # first at the largest node count the series reach.
    def final_shutdown(key: tuple[str, str, int, int]) -> tuple[int, float]:
        points = sorted(grouped[key], key=lambda point: point[0])
        return (-points[-1][0], points[-1][1])

    series = sorted(
        (
            key
            for key in grouped
            if key[0] == system
            and key[3] == ranks_per_node
            # Keep three-layer (-1) and two-layer (0); skip explicit fanout sweeps.
            and (key[2] in (-1, 0) or "hierarchical" not in key[1])
        ),
        key=final_shutdown,
    )
    if not series:
        return

    with ieee_figure() as (fig, ax):
        all_nodes: set[int] = set()
        handles = []
        labels = []

        for key in series:
            _, distributor, fanout, _rpn = key
            nodes, times_us = sorted_series_xy(grouped[key])
            time_per_shutdown_s = [t / 1_000_000.0 for t in times_us]
            all_nodes.update(nodes)

            label = format_distributor_label(distributor, fanout)
            color_idx = distributor_series_index(distributor, fanout)
            line = plot_node_series(ax, color_idx, nodes, time_per_shutdown_s, label, linewidth=1.0)
            handles.append(line)
            labels.append(label)

        finish_compact_node_plot(ax, all_nodes, handles, labels, "Shutdown time (s)")

        save_figure(
            fig,
            output_dir,
            f"shutdown_time_{system}_{ranks_per_node}rpn.{image_format}",
        )


# Comparable Aurora/Frontier ranks-per-node pairings (gpu-sparse and denser).
SYSTEM_RPN_PAIRS: list[tuple[tuple[str, int], tuple[str, int]]] = [
    (("aurora", 7), ("frontier", 9)),
    (("aurora", 102), ("frontier", 56)),
]


def plot_cross_system_by_distributor(
    grouped: dict[tuple[str, str, int, int], list[tuple[int, float]]],
    output_dir: str,
    image_format: str,
) -> None:
    """One Aurora-vs-Frontier plot per distributor (and hierarchical fanout)."""
    configs = sorted(
        {
            (distributor, fanout)
            for (_system, distributor, fanout, _rpn) in grouped
            if fanout in (-1, 0) or "hierarchical" not in distributor
        }
    )
    for distributor, fanout in configs:
        for (sys_a, rpn_a), (sys_b, rpn_b) in SYSTEM_RPN_PAIRS:
            key_a = (sys_a, distributor, fanout, rpn_a)
            key_b = (sys_b, distributor, fanout, rpn_b)
            if key_a not in grouped or key_b not in grouped:
                continue

            with ieee_figure() as (fig, ax):
                all_nodes: set[int] = set()
                handles = []
                labels = []
                for idx, (system, rpn, key) in enumerate(
                    ((sys_a, rpn_a, key_a), (sys_b, rpn_b, key_b))
                ):
                    nodes, times_us = sorted_series_xy(grouped[key])
                    time_per_shutdown_s = [t / 1_000_000.0 for t in times_us]
                    all_nodes.update(nodes)
                    label = f"{system.capitalize()} ({rpn} rpn)"
                    line = plot_node_series(
                        ax, idx, nodes, time_per_shutdown_s, label, linewidth=1.0
                    )
                    handles.append(line)
                    labels.append(label)

                finish_compact_node_plot(ax, all_nodes, handles, labels, "Shutdown time (s)")

                fanout_str = ""
                if "hierarchical" in distributor:
                    fanout_str = f"_fanout{fanout}"
                filename = (
                    f"shutdown_compare_{distributor}{fanout_str}"
                    f"_{rpn_a}v{rpn_b}rpn.{image_format}"
                )
                save_figure(fig, output_dir, filename)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot shutdown time vs number of nodes.")
    add_plot_cli_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = parse_rows(collect_csv_paths(args.input, "shutdown"))
    rows = filter_systems(rows, args.exclude_system)
    rows = filter_ranks_per_node(rows, args.ranks_per_node)
    grouped = group_rows(rows)

    configs = sorted({(row["system"], row["ranks_per_node"]) for row in rows})
    for system, ranks_per_node in configs:
        plot_system_rpn(system, ranks_per_node, grouped, args.output_dir, args.format)
    plot_cross_system_by_distributor(grouped, args.output_dir, args.format)


if __name__ == "__main__":
    main()
