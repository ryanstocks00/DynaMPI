#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
"""Plot weak-scaling task-distribution throughput.

These sweeps are weak scaling: the manager keeps every worker supplied for a
fixed wall-clock window, so the work offered grows in proportion to the node
count and the ideal curve is linear in nodes. The driver, launch scripts and
result CSVs still carry the older ``strong_scaling`` name, so
``collect_csv_paths`` accepts either spelling as input; only the plots this
script writes are named ``weak_scaling_*``.
"""

import argparse
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import TypedDict

import matplotlib.pyplot as plt

from plot_common import (
    COMPACT_LEGEND_STYLE,
    PANEL_HSPACE,
    Recency,
    add_light_grid,
    add_plot_cli_args,
    collect_csv_paths,
    dedupe_newest,
    distributor_series_index,
    filter_ranks_per_node,
    filter_systems,
    finish_compact_node_plot,
    format_distributor_label,
    ieee_figure,
    ieee_panel_figure,
    iter_csv_rows,
    legend_avoiding_data,
    log_padded_limits,
    normalize_mode,
    plot_node_series,
    save_figure,
    series_color,
    set_log_node_axes,
    sorted_series_xy,
)


class WeakScalingRow(TypedDict):
    system: str
    distributor: str
    mode: str
    expected_ns: int
    nodes: int
    ranks_per_node: int
    fanout: int
    throughput: float
    file_mtime: float
    path: str
    recency: Recency


def format_duration(expected_ns: int | float) -> str:
    """ASCII form, used to build filenames (``..._compare_100us_7rpn.pdf``)."""
    if expected_ns <= 0:
        return "0 ns"
    if expected_ns >= 1_000_000_000:
        return f"{expected_ns / 1_000_000_000:g} s"
    if expected_ns >= 1_000_000:
        return f"{expected_ns / 1_000_000:g} ms"
    if expected_ns >= 1_000:
        return f"{expected_ns / 1_000:g} us"
    return f"{expected_ns:g} ns"


def format_duration_label(expected_ns: int | float) -> str:
    """Display form for legends: real mu, to match the prose in the paper.

    Kept separate from format_duration() because that one's output ends up in
    filenames, where the math escape would be worse than useless.
    """
    return format_duration(expected_ns).replace("us", r"$\mu$s")


def parse_rows(paths: Sequence[str]) -> list[WeakScalingRow]:
    rows: list[WeakScalingRow] = []
    for row, path, file_mtime, recency in iter_csv_rows(paths):
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
        distributor = row.get("distributor", "").strip()
        fanout = int(float(row.get("max_upper_fanout", -1) or -1))
        if "hierarchical" not in distributor:
            fanout = -1
        rows.append(
            {
                "system": row.get("system", "").strip() or "unknown",
                "distributor": distributor,
                "mode": row.get("mode", "").strip(),
                "expected_ns": expected_ns,
                "nodes": nodes,
                "ranks_per_node": ranks_per_node,
                "fanout": fanout,
                "throughput": float(row.get("throughput_tasks_per_s", 0.0)),
                "file_mtime": file_mtime,
                "path": path,
                "recency": recency,
            }
        )
    return rows


def group_rows(
    rows: Sequence[WeakScalingRow],
) -> dict[tuple[str, str, str, int, int, int], list[tuple[int, float]]]:
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
    grouped: dict[tuple[str, str, str, int, int, int], list[tuple[int, float]]] = defaultdict(
        list
    )
    for (
        system,
        distributor,
        mode,
        expected_ns,
        ranks_per_node,
        fanout,
        nodes,
    ), (throughput, _) in newest.items():
        grouped[(system, distributor, mode, expected_ns, ranks_per_node, fanout)].append(
            (nodes, throughput)
        )
    return grouped


def comparison_series_kept(distributor: str, fanout: int) -> bool:
    """Whether a (distributor, fanout) pair appears in the comparison plots.

    Only the auto and flat hierarchical topologies are drawn there; the
    explicit fan-out sweeps would crowd the figure. Shared with the y-limit
    pass so the range is computed over exactly the curves that get plotted.
    """
    return "hierarchical" not in distributor or fanout in (-1, 0)


def per_distributor_ylimits(
    configs: Sequence[tuple[str, str, int, int]],
    grouped: dict[tuple[str, str, str, int, int, int], list[tuple[int, float]]],
) -> dict[tuple[str, int, str], tuple[float, float]]:
    """One shared y-range per (system, ranks/node, mode).

    The per-distributor figures for one layout are the four designs the paper
    puts side by side and compares in prose ("about three times the naive
    ceiling"); autoscaling each one separately spreads them over axes with
    different top decades and makes that comparison impossible to see. The
    range covers measured throughput only -- the ideal lines run orders of
    magnitude above the data at the fine grains and are drawn clipped.
    """
    plotted = set(configs)
    buckets: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (
        system,
        distributor,
        mode,
        _expected_ns,
        ranks_per_node,
        fanout,
    ), points in grouped.items():
        if (system, distributor, ranks_per_node, fanout) not in plotted:
            continue
        key = (system, ranks_per_node, normalize_mode(mode))
        buckets[key].extend(throughput for _, throughput in points)
    return {
        key: limits
        for key, values in buckets.items()
        if (limits := log_padded_limits(values)) is not None
    }


# The cross-machine comparisons are made at matched layouts, not matched rank
# counts: one worker per GPU (6 + node manager on Aurora, 8 + node manager on
# Frontier), or every core filled. Each pair is read as a pair and shares an
# axis; the two pairs are separate comparisons and must not, or the GPU-mapped
# figures get sized for core-filled rates they never plot.
LAYOUT_CLASSES = {7: "gpu", 9: "gpu", 102: "core", 56: "core"}


def layout_class(ranks_per_node: int) -> str:
    """Which cross-machine comparison a ranks-per-node value belongs to.

    An unrecognized layout gets a class of its own rather than being pooled
    with a sweep it was never meant to be compared against.
    """
    return LAYOUT_CLASSES.get(ranks_per_node, f"rpn{ranks_per_node}")


def comparison_ylimits(
    grouped: dict[tuple[str, str, str, int, int, int], list[tuple[int, float]]],
) -> dict[tuple[str, int, str], tuple[float, float]]:
    """One shared y-range per (mode, task duration, layout class).

    The comparison plots are read as a matched cross-machine pair at a single
    granularity, so Aurora and Frontier need the same axis for the eye to do
    the comparison the text claims -- but only against the machine they are
    paired with.
    """
    buckets: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for (
        _system,
        distributor,
        mode,
        expected_ns,
        ranks_per_node,
        fanout,
    ), points in grouped.items():
        if not comparison_series_kept(distributor, fanout):
            continue
        key = (normalize_mode(mode), expected_ns, layout_class(ranks_per_node))
        buckets[key].extend(throughput for _, throughput in points)
    return {
        key: limits
        for key, values in buckets.items()
        if (limits := log_padded_limits(values)) is not None
    }


def plot_distributor(
    system: str,
    distributor: str,
    ranks_per_node: int,
    fanout: int,
    grouped: dict[tuple[str, str, str, int, int, int], list[tuple[int, float]]],
    output_dir: str,
    image_format: str,
    ylimits: dict[tuple[str, int, str], tuple[float, float]] | None = None,
) -> None:
    for mode in ("fixed", "random"):
        with ieee_figure() as (fig, ax):
            series = []
            all_nodes: set[int] = set()
            for (
                sys_name,
                dist,
                mode_name,
                expected_ns,
                rpn,
                series_fanout,
            ), points in grouped.items():
                if (
                    sys_name != system
                    or dist != distributor
                    or normalize_mode(mode_name) != mode
                    or rpn != ranks_per_node
                    or series_fanout != fanout
                ):
                    continue
                nodes, throughput = sorted_series_xy(points)
                all_nodes.update(nodes)
                series.append((expected_ns, nodes, throughput))

            if not series:
                plt.close(fig)
                continue

            series_sorted = sorted(series, key=lambda item: item[0])
            handles = []
            labels = []

            for idx, (expected_ns, nodes, throughput) in enumerate(series_sorted):
                label = format_duration_label(expected_ns)
                line = plot_node_series(ax, idx, nodes, throughput, label)
                handles.append(line)
                labels.append(label)

            ax.set_xlabel("Nodes")
            ax.set_ylabel("Tasks per second")
            set_log_node_axes(ax, all_nodes)
            shared = (ylimits or {}).get((system, ranks_per_node, mode))
            if shared is not None:
                ax.set_ylim(shared)
            add_light_grid(ax)

            # Hierarchical distributors default to manager_per_node, so one
            # rank/node is a manager rather than a worker (7 rpn -> 6
            # workers/node, 9 rpn -> 8 workers/node).
            ideal_workers_per_node = (
                ranks_per_node - 1
                if "hierarchical" in distributor and ranks_per_node > 1
                else ranks_per_node
            )

            # The root manager is one further rank in both families: it never
            # executes a task, and it is excluded from its own node's local
            # group, so that node contributes one fewer worker than the rest.
            # Worth 0.1% at 128 nodes but an eighth of the ideal at a single
            # node, which is why every curve otherwise appears to start around
            # 85% efficiency rather than tracking its reference line.
            def ideal_worker_count(
                nodes: int, ideal_workers_per_node: int = ideal_workers_per_node
            ) -> int:
                return max(1, nodes * ideal_workers_per_node - 1)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            for idx, (expected_ns, _nodes, _throughput) in enumerate(series_sorted):
                if all_nodes and expected_ns > 0:
                    ideal_nodes = sorted(all_nodes)
                    ideal_throughput = [
                        ideal_worker_count(n) * 1e9 / expected_ns for n in ideal_nodes
                    ]
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

            # Entries stay in task-duration order here: within one
            # distributor the curves converge at the largest node count, so
            # ranking them by throughput would scramble a legend whose
            # natural reading order is 10 us -> 1 s.
            legend_avoiding_data(
                ax,
                handles,
                labels,
                locations=("lower right", "upper left", "lower center", "center left"),
                frameon=False,
                ncol=2,
                fontsize=8,
                handlelength=1.5,
                columnspacing=0.8,
                borderpad=0.3,
                labelspacing=0.3,
            )

            fanout_str = ""
            if "hierarchical" in distributor:
                fanout_str = f"_fanout{fanout}"
            mode_dir = os.path.join(output_dir, mode)
            os.makedirs(mode_dir, exist_ok=True)
            filename = (
                f"weak_scaling_{system}_{distributor}"
                f"_{ranks_per_node}rpn{fanout_str}.{image_format}"
            )
            save_figure(fig, mode_dir, filename)


def plot_distributor_comparison(
    system: str,
    mode: str,
    expected_ns: int,
    ranks_per_node: int,
    grouped: dict[tuple[str, str, str, int, int, int], list[tuple[int, float]]],
    output_dir: str,
    image_format: str,
    ylimits: dict[tuple[str, int, str], tuple[float, float]] | None = None,
) -> None:
    """Compare distributors at one workload duration (cross-distributor view)."""
    with ieee_figure() as (fig, ax):
        series = []
        all_nodes: set[int] = set()
        for (
            sys_name,
            dist,
            mode_name,
            series_expected_ns,
            rpn,
            fanout,
        ), points in grouped.items():
            if (
                sys_name != system
                or normalize_mode(mode_name) != mode
                or series_expected_ns != expected_ns
                or rpn != ranks_per_node
            ):
                continue
            if not comparison_series_kept(dist, fanout):
                continue
            nodes, throughput = sorted_series_xy(points)
            all_nodes.update(nodes)
            series.append((dist, fanout, nodes, throughput))

        if len(series) < 2:
            plt.close(fig)
            return

        # Legend order follows the ranking a reader is looking for: fastest
        # first at the largest node count the series reach, so the entries
        # read top-to-bottom in the same order the curves finish.
        def final_throughput(item: tuple[str, int, list[int], list[float]]) -> tuple[int, float]:
            _, _, nodes, throughput = item
            largest = max(range(len(nodes)), key=lambda i: nodes[i])
            return (nodes[largest], throughput[largest])

        series_sorted = sorted(series, key=final_throughput, reverse=True)
        handles = []
        labels = []
        for dist, fanout, nodes, throughput in series_sorted:
            label = format_distributor_label(dist, fanout)
            color_idx = distributor_series_index(dist, fanout)
            line = plot_node_series(ax, color_idx, nodes, throughput, label)
            handles.append(line)
            labels.append(label)

        finish_compact_node_plot(
            ax,
            all_nodes,
            handles,
            labels,
            "Tasks per second",
            ylim=(ylimits or {}).get((mode, expected_ns, layout_class(ranks_per_node))),
        )

        duration = format_duration(expected_ns).replace(" ", "")
        mode_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        filename = (
            f"weak_scaling_{system}_compare_{duration}_{ranks_per_node}rpn."
            f"{image_format}"
        )
        save_figure(fig, mode_dir, filename)


def comparison_panel_series(
    system: str,
    mode: str,
    expected_ns: int,
    ranks_per_node: int,
    grouped: dict[tuple[str, str, str, int, int, int], list[tuple[int, float]]],
) -> list[tuple[str, int, list[int], list[float]]]:
    """The curves one comparison panel draws, fastest at the largest node count first."""
    series = []
    for (sys_name, dist, mode_name, series_ns, rpn, fanout), points in grouped.items():
        if (
            sys_name != system
            or normalize_mode(mode_name) != mode
            or series_ns != expected_ns
            or rpn != ranks_per_node
            or not comparison_series_kept(dist, fanout)
        ):
            continue
        nodes, throughput = sorted_series_xy(points)
        series.append((dist, fanout, nodes, throughput))

    def final_throughput(item: tuple[str, int, list[int], list[float]]) -> tuple[int, float]:
        _, _, nodes, throughput = item
        largest = max(range(len(nodes)), key=lambda i: nodes[i])
        return (nodes[largest], throughput[largest])

    return sorted(series, key=final_throughput, reverse=True)


def plot_comparison_panels(
    mode: str,
    expected_ns: int,
    layouts: Sequence[tuple[str, int]],
    grouped: dict[tuple[str, str, str, int, int, int], list[tuple[int, float]]],
    output_dir: str,
    image_format: str,
    ylimits: dict[tuple[str, int, str], tuple[float, float]] | None = None,
) -> None:
    """The cross-machine comparison as stacked panels over one shared node axis.

    The pair is read as a single comparison, so it is drawn as a single
    figure: sharing the x axis drops a duplicate label and tick row, and the
    shared y range puts both machines on one scale. The legend goes in the
    first panel only, since every panel plots the same series.
    """
    panels = [
        (system, rpn, comparison_panel_series(system, mode, expected_ns, rpn, grouped))
        for system, rpn in layouts
    ]
    panels = [panel for panel in panels if len(panel[2]) >= 2]
    if len(panels) < 2:
        return

    with ieee_panel_figure(len(panels)) as (fig, axes):
        for index, ((system, ranks_per_node, series), ax) in enumerate(zip(panels, axes)):
            all_nodes: set[int] = set()
            handles = []
            labels = []
            for dist, fanout, nodes, throughput in series:
                all_nodes.update(nodes)
                label = format_distributor_label(dist, fanout)
                line = plot_node_series(ax, distributor_series_index(dist, fanout), nodes,
                                        throughput, label)
                handles.append(line)
                labels.append(label)

            ax.set_ylabel("Tasks per second")
            set_log_node_axes(ax, all_nodes)
            limits = (ylimits or {}).get((mode, expected_ns, layout_class(ranks_per_node)))
            if limits is not None:
                ax.set_ylim(limits)
            add_light_grid(ax)
            # Named in the panel rather than a sub-caption: the machine is
            # what the reader needs while looking at the curves.
            ax.annotate(
                f"({chr(ord('a') + index)}) {system.capitalize()}, {ranks_per_node} ranks/node",
                (0.98, 0.03),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8,
            )
            if index == 0:
                legend_avoiding_data(
                    ax, handles, labels, locations=("upper left",), **COMPACT_LEGEND_STYLE
                )
            if index == len(panels) - 1:
                ax.set_xlabel("Nodes")

        duration = format_duration(expected_ns).replace(" ", "")
        mode_dir = os.path.join(output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        save_figure(
            fig,
            mode_dir,
            f"weak_scaling_compare_panels_{duration}.{image_format}",
            hspace=PANEL_HSPACE,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot weak scaling distribution throughput.")
    add_plot_cli_args(parser)
    parser.add_argument(
        "--compare-distributors",
        action="store_true",
        help="Also emit cross-distributor comparison plots per workload duration",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["fixed", "random"],
        default=["fixed", "random"],
        help="Task-time modes to plot into separate subdirectories (default: both)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = parse_rows(collect_csv_paths(args.input, ("weak_scaling", "strong_scaling")))
    rows = filter_systems(rows, args.exclude_system)
    rows = filter_ranks_per_node(rows, args.ranks_per_node)
    modes = {normalize_mode(mode) for mode in args.modes}
    rows = [row for row in rows if normalize_mode(row["mode"]) in modes]
    grouped = group_rows(rows)

    configs = sorted(
        {
            (row["system"], row["distributor"], row["ranks_per_node"], row["fanout"])
            for row in rows
            if row["distributor"].strip()
            and (
                "hierarchical" not in row["distributor"]
                or row["fanout"] in (-1, 0)
            )
        }
    )
    distributor_ylimits = per_distributor_ylimits(configs, grouped)
    for system, distributor, ranks_per_node, fanout in configs:
        plot_distributor(
            system,
            distributor,
            ranks_per_node,
            fanout,
            grouped,
            args.output_dir,
            args.format,
            ylimits=distributor_ylimits,
        )

    if args.compare_distributors:
        compare_ylimits = comparison_ylimits(grouped)
        compare_keys = sorted(
            {
                (
                    row["system"],
                    normalize_mode(row["mode"]),
                    row["expected_ns"],
                    row["ranks_per_node"],
                )
                for row in rows
                if row["distributor"].strip() and row["system"] != "unknown"
            }
        )
        for system, mode, expected_ns, ranks_per_node in compare_keys:
            plot_distributor_comparison(
                system,
                mode,
                expected_ns,
                ranks_per_node,
                grouped,
                args.output_dir,
                args.format,
                ylimits=compare_ylimits,
            )

        # One stacked figure per (mode, duration, layout class): the matched
        # cross-machine pair the prose reads together.
        panel_groups: dict[tuple[str, int, str], list[tuple[str, int]]] = defaultdict(list)
        for system, mode, expected_ns, ranks_per_node in compare_keys:
            key = (mode, expected_ns, layout_class(ranks_per_node))
            if (system, ranks_per_node) not in panel_groups[key]:
                panel_groups[key].append((system, ranks_per_node))
        for (mode, expected_ns, _layout), layouts in sorted(panel_groups.items()):
            plot_comparison_panels(
                mode,
                expected_ns,
                layouts,
                grouped,
                args.output_dir,
                args.format,
                ylimits=compare_ylimits,
            )


if __name__ == "__main__":
    main()
