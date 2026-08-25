#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
"""Plot weak-scaling throughput against the hierarchy's upper fan-out.

The hierarchical distributors size their upper coordination layer from
``resolve_upper_fanout``: flat below ~32 node managers, otherwise the next
power of two at least sqrt(managers). These figures are the evidence for that
rule -- they hold everything else fixed and vary only ``max_upper_fanout``,
so the cost of picking it badly is visible directly.

One figure per (system, distributor, ranks-per-node, task duration), with one
curve per fan-out setting, drawn only where at least two settings were
measured. ``default`` is the auto rule and ``two-layer`` disables the upper
layer entirely; both are plotted alongside the explicit values they are being
justified against.
"""

import argparse
import os
from collections import defaultdict
from collections.abc import Sequence

from plot_common import (
    add_plot_cli_args,
    collect_csv_paths,
    dedupe_newest,
    filter_ranks_per_node,
    filter_systems,
    finish_compact_node_plot,
    format_distributor_label,
    format_fanout,
    ieee_figure,
    normalize_mode,
    plot_node_series,
    save_figure,
    set_output_formats,
    sorted_series_xy,
)
from plot_weak_scaling import WeakScalingRow, format_duration, parse_rows

# (system, distributor, ranks_per_node, mode, expected_ns) -> fanout -> [(nodes, rate)]
PlotKey = tuple[str, str, int, str, int]
Grouped = dict[PlotKey, dict[int, list[tuple[int, float]]]]


def group_rows(rows: Sequence[WeakScalingRow]) -> Grouped:
    newest = dedupe_newest(
        rows,
        lambda row: (
            row["system"],
            row["distributor"],
            normalize_mode(row["mode"]),
            row["expected_ns"],
            row["ranks_per_node"],
            row["fanout"],
            row["nodes"],
        ),
        "throughput",
    )
    grouped: Grouped = defaultdict(lambda: defaultdict(list))
    for key, (throughput, _recency) in newest.items():
        system, distributor, mode, expected_ns, ranks_per_node, fanout, nodes = key
        if "hierarchical" not in distributor:
            continue
        plot_key: PlotKey = (system, distributor, ranks_per_node, mode, expected_ns)
        grouped[plot_key][fanout].append((nodes, throughput))
    return grouped


# Legend and colour order: the two named settings first, then the explicit
# fan-outs ascending, so the rule being justified reads before the sweep.
def fanout_sort_key(fanout: int) -> tuple[int, int]:
    if fanout < 0:
        return (0, 0)
    if fanout == 0:
        return (1, 0)
    return (2, fanout)


def plot_config(key: PlotKey, series: dict[int, list[tuple[int, float]]],
                output_dir: str, image_format: str) -> None:
    system, distributor, ranks_per_node, mode, expected_ns = key
    with ieee_figure() as (fig, ax):
        all_nodes: set[int] = set()
        handles, labels = [], []
        for idx, fanout in enumerate(sorted(series, key=fanout_sort_key)):
            nodes, rates = sorted_series_xy(series[fanout])
            all_nodes.update(nodes)
            label = format_fanout(fanout)
            # Where only one node count was measured -- the Aurora 128-node
            # sweep's shape -- the marker alone carries the datum.
            line = plot_node_series(ax, idx, nodes, rates, label, linewidth=1.0)
            handles.append(line)
            labels.append(label)
        ax.set_yscale("log")
        ax.set_title(
            f"{format_distributor_label(distributor)}, {format_duration(expected_ns)} tasks",
            fontsize=8,
        )
        finish_compact_node_plot(ax, all_nodes, handles, labels, "Tasks/s")
        mode_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        save_figure(
            fig,
            mode_dir,
            f"fanout_{system}_{distributor}_{ranks_per_node}rpn_"
            f"{format_duration(expected_ns).replace(' ', '')}.{image_format}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot weak-scaling throughput against hierarchy upper fan-out."
    )
    add_plot_cli_args(parser)
    args = parser.parse_args()
    set_output_formats(args.format)

    rows = parse_rows(collect_csv_paths(args.input, "weak_scaling"))
    rows = filter_systems(rows, args.exclude_system)
    rows = filter_ranks_per_node(rows, args.ranks_per_node)
    grouped = group_rows(rows)

    drawn = 0
    for key, series in sorted(grouped.items()):
        # Nothing to justify unless an explicit fan-out was measured against
        # at least one alternative.
        if len(series) < 2 or not any(fanout > 0 for fanout in series):
            continue
        plot_config(key, series, args.output_dir, args.format[0])
        drawn += 1
    print(f"wrote {drawn} fan-out figures to {args.output_dir}")


if __name__ == "__main__":
    main()
