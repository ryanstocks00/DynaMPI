#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
"""Plot makespan against batch size for the load_balancing_makespan benchmark.

Unlike the weak-scaling sweeps, which hold the node count's worth of workers
busy for a fixed window and report a rate, this benchmark drains a batch of
exactly ``tasks_per_worker * workers`` tasks and reports the wall clock it
took. At one task per worker there is no slack for a distributor to balance
against, so the makespan is set by the slowest task plus dispatch; the gap to
the ideal line closes as the batch grows.
"""

import argparse
import heapq
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from statistics import mean
from typing import TypedDict

import numpy as np
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter, LogLocator

from plot_common import (
    Recency,
    COMPACT_LEGEND_STYLE,
    add_light_grid,
    add_plot_cli_args,
    collect_csv_paths,
    distributor_series_index,
    filter_ranks_per_node,
    filter_systems,
    format_distributor_label,
    ieee_figure,
    iter_csv_rows,
    legend_avoiding_data,
    path_recency,
    plot_node_series,
    save_figure,
    series_color,
)


class MakespanRow(TypedDict):
    system: str
    distributor: str
    fanout: int
    nodes: int
    world_size: int
    ranks_per_node: int
    duration_mode: str
    task_duration_cv: float
    expected_us: int
    workers: int
    tasks_per_worker: int
    phases: dict[str, float]
    file_mtime: float
    path: str
    recency: Recency


def parse_rows(paths: Sequence[str]) -> list[MakespanRow]:
    rows: list[MakespanRow] = []
    for row, path, file_mtime, recency in iter_csv_rows(paths):
        # Only this benchmark's CSVs carry tasks_per_worker; the weak-scaling
        # and shutdown files share these directories.
        if "tasks_per_worker" not in row:
            continue
        phases = {name: float(row.get(name, 0.0) or 0.0) for name in PHASE_COLUMNS}
        tasks_per_worker = int(float(row.get("tasks_per_worker", 0)))
        # k = 0 leaves the timed batch block unexecuted, so elapsed_s is
        # clock-read overhead rather than a measurement; it is meaningful
        # only for the variants that add a setup or teardown phase.
        if tasks_per_worker < 0 or (tasks_per_worker == 0 and phases["construct_s"] <= 0.0):
            continue
        nodes = int(float(row.get("nodes", 0)))
        world_size = int(float(row.get("world_size", 0)))
        distributor = row.get("distributor", "").strip()
        fanout = int(float(row.get("max_upper_fanout", -1) or -1))
        if "hierarchical" not in distributor:
            fanout = -1
        rows.append(
            {
                "system": row.get("system", "").strip() or "unknown",
                "distributor": distributor,
                "fanout": fanout,
                "nodes": nodes,
                "world_size": world_size,
                # Provisional; rewritten by launch_ranks_per_node().
                "ranks_per_node": int(round(world_size / nodes)) if nodes else 0,
                # CSVs written before --duration_mode existed are all uniform:
                # that was the only behaviour then, and it is what the
                # published 64/128/256-node sweeps used.
                "duration_mode": (row.get("duration_mode") or "uniform").strip(),
                "task_duration_cv": float(row.get("task_duration_cv") or 0.0),
                "expected_us": int(float(row.get("expected_us", 0))),
                "workers": int(float(row.get("workers", 0))),
                "tasks_per_worker": tasks_per_worker,
                "phases": phases,
                "file_mtime": file_mtime,
                "path": path,
                "recency": recency,
            }
        )
    return rows


def launch_ranks_per_node(rows: list[MakespanRow]) -> list[MakespanRow]:
    """Key every row to the geometry its job was *launched* with.

    naive/lockfree_rma run on a comm with one rank per node dropped (see
    test_load_balancing.cpp's flat_comm), so they report a smaller world_size
    than the hierarchical pair. Per-row rpn would split one sweep across two
    plot keys, and the --ranks-per-node filter would then drop the flat pair
    entirely. The largest world_size in a file is that job's --ntasks-per-node.
    """
    launch: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row["path"], row["nodes"])
        launch[key] = max(launch.get(key, 0), row["world_size"])
    for row in rows:
        nodes = row["nodes"]
        if nodes:
            row["ranks_per_node"] = int(round(launch[(row["path"], nodes)] / nodes))
    return rows


def duration_sampler(mode: str, mean_s: float, cv: float):
    """Draw n task durations in seconds, mirroring test_load_balancing.cpp.

    The reference curves are properties of the duration distribution, not of
    any distributor, so both are resampled per mode. With fixed durations the
    two coincide exactly -- every rank's share is identical, so a static split
    already is the greedy optimum and no dynamic scheme can win.
    """
    if mode == "fixed":
        return lambda rng, n: np.full(n, mean_s)
    if mode == "uniform":
        # uniform_int_distribution(0, 2 * expected_us).
        return lambda rng, n: rng.uniform(0.0, 2.0 * mean_s, size=n)
    if mode == "lognormal":
        # WorkerFunctor picks (mu, sigma) so the lognormal's own mean and cv
        # match expected_us and task_duration_cv, then rounds to whole
        # microseconds and clamps the near-zero left tail at 1 us.
        sigma = math.sqrt(math.log(1.0 + cv * cv))
        mu = math.log(mean_s * 1e6) - sigma * sigma / 2.0
        return lambda rng, n: np.maximum(1.0, np.rint(rng.lognormal(mu, sigma, n))) / 1e6
    raise SystemExit(f"unknown duration mode {mode!r}")


def ideal_greedy_times(
    mode: str, cv: float, workers: int, batches: Sequence[int], mean_s: float
) -> list[float]:
    """Best total time a work-conserving dynamic scheduler could achieve.

    ``k`` mean durations is a lower bound but not a reachable one: tasks are
    atomic, so the last one handed out still runs to completion after the
    rest of the machine has drained. Simulating greedy scheduling -- every
    rank takes the next task the moment it frees up, which is what all four
    distributors do -- gives the floor they are actually competing against.
    Seeded, so a rerun redraws the same line.
    """
    return list(_ideal_greedy_cached(mode, cv, workers, tuple(batches), mean_s))


# Both reference curves depend only on (mode, cv, workers, batches, mean), but
# plot_config() is called once per phase set, so without a cache each one is
# simulated three times over for the same figure.
@lru_cache(maxsize=None)
def _ideal_greedy_cached(
    mode: str, cv: float, workers: int, batches: tuple[int, ...], mean_s: float
) -> tuple[float, ...]:
    rng = np.random.default_rng(20260814)
    draw = duration_sampler(mode, mean_s, cv)
    times = []
    for k in batches:
        runs = []
        for _ in range(_IDEAL_TRIALS):
            durations = draw(rng, k * workers)
            finish = [0.0] * workers
            heapq.heapify(finish)
            for duration in durations:
                soonest = heapq.heappop(finish)
                heapq.heappush(finish, soonest + float(duration))
            runs.append(max(finish))
        times.append(mean(runs))
    return tuple(times)


def static_split_times(
    mode: str, cv: float, workers: int, batches: Sequence[int], mean_s: float
) -> list[float]:
    """Makespan of a static split: k tasks pinned to each rank before the batch runs.

    The greedy floor says how well a distributor *could* do; this says what it
    has to beat to be worth its cost at all. Tasks are assigned round-robin
    with no knowledge of their durations -- the only assignment available
    when durations are unpredictable -- so the makespan is the slowest rank's
    own total rather than the point at which the machine drains.

    At k = 1 the two references coincide whatever the distribution: every rank
    runs exactly one task either way. Above that the gap grows with the spread
    of the durations, and vanishes entirely in fixed mode.
    """
    return list(_static_split_cached(mode, cv, workers, tuple(batches), mean_s))


@lru_cache(maxsize=None)
def _static_split_cached(
    mode: str, cv: float, workers: int, batches: tuple[int, ...], mean_s: float
) -> tuple[float, ...]:
    # Vectorized rather than looped: at _STATIC_TRIALS this is tens of
    # millions of draws per figure, which costs a minute in pure Python and
    # milliseconds here. The greedy simulation above stays looped because its
    # heap is inherently sequential.
    rng = np.random.default_rng(20260815)
    draw = duration_sampler(mode, mean_s, cv)
    times = []
    for k in batches:
        # (trials, workers) -> each rank's own k durations summed, then the
        # slowest rank in each trial.
        totals = draw(rng, _STATIC_TRIALS * workers * k)
        totals = totals.reshape(_STATIC_TRIALS, workers, k).sum(axis=2)
        times.append(float(totals.max(axis=1).mean()))
    return tuple(times)


PHASE_COLUMNS = ("construct_s", "elapsed_s", "finalize_s", "destruct_s")

# The figures this script writes, each summing a different set of phases.
# Draining the batch is the load-balancing measurement proper; adding
# finalize, and then construction, shows how much a caller pays for a
# distributor that exists only for one batch -- the regime where a phase
# boundary forces a fresh distributor rather than one kept alive across a
# run. k = 0 is only plotted where a phase other than the batch is included.
PHASE_SETS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ("elapsed_s",)),
    ("_with_finalize", ("elapsed_s", "finalize_s")),
    ("_with_construct_finalize", ("construct_s", "elapsed_s", "finalize_s")),
)


# Per-label vertical nudges, in points, for the few places where the generic
# above/below rule leaves a label crowding its neighbour. Positive is up.
LABEL_NUDGES: dict[tuple[str, int], float] = {}

# Distributors that must break the alternating above/below rule below. The
# rule assumes each curve's nearest neighbour is the one it alternates
# against, which fails where two curves converge: the flat RMA distributor
# runs close above the hierarchical RMA one over the lower half of the
# range, so the generic choice puts the flat one's labels below its curve
# and the hierarchical one's above its curve -- both into the same gap.
LABEL_ABOVE = frozenset({"lockfree_rma"})

# Where a curve is too close to a neighbour to label cleanly, annotate only
# the part of the range where the two have separated.
LABEL_SKIP: dict[str, frozenset[int]] = {
    "lockfree_rma": frozenset({5, 10}),
}

# Batch sizes at which each curve is annotated with its efficiency against
# the simulated floor. Sparse on purpose: one label per curve per point is
# already four labels, and every batch size would bury the curves.
ANNOTATE_AT = (5, 10, 15, 20)

# Enough to settle the simulated offset to a few parts in a thousand, which
# is far below the spread of the measurements it is drawn against.
_IDEAL_TRIALS = 20

# The static makespan needs an order of magnitude more. It is a max of sums,
# which concentrates far more weakly than a greedy makespan: at 20 trials the
# seed-to-seed spread is 1-3%, enough to move a curve's crossing point by a
# whole batch size. 200 brings it under 0.5% for a few milliseconds of work.
_STATIC_TRIALS = 200


# system, nodes, ranks_per_node, expected_us, duration_mode, task_duration_cv.
# The distribution is part of the identity: merging two modes into one curve
# would average measurements taken against different workloads.
PlotKey = tuple[str, int, int, int, str, float]
SeriesKey = tuple[str, int]  # distributor, fanout
PhaseTimes = dict[int, dict[str, float]]  # tasks_per_worker -> phase -> seconds


def group_rows(
    rows: Sequence[MakespanRow],
) -> tuple[dict[PlotKey, dict[SeriesKey, PhaseTimes]], dict[PlotKey, int]]:
    """Average the repeats of each (config, batch size), keeping the newest run.

    The benchmark writes one row per repeat, so unlike the other scripts the
    dedupe is per *file generation* rather than per row: rows from an older
    job must not be averaged in with a rerun's.
    """
    best: dict[tuple[PlotKey, SeriesKey, int], tuple[Recency, list[dict[str, float]]]] = {}
    # The benchmark sizes every distributor's batch off the hierarchical
    # worker count, so all four drain an identical task list and one reference
    # line fits every series. Take the min: hierarchical_lockfree_rma reports
    # world_size - 1 even though its non-leaf ranks relay rather than execute,
    # overcounting its real workers by one per node.
    workers: dict[PlotKey, int] = {}
    for row in rows:
        plot_key: PlotKey = (
            row["system"],
            row["nodes"],
            row["ranks_per_node"],
            row["expected_us"],
            row["duration_mode"],
            row["task_duration_cv"],
        )
        series_key: SeriesKey = (row["distributor"], row["fanout"])
        prev = workers.get(plot_key)
        workers[plot_key] = row["workers"] if prev is None else min(prev, row["workers"])
        key = (plot_key, series_key, row["tasks_per_worker"])
        recency = row["recency"] or path_recency(row["path"], row["file_mtime"])
        if key not in best or recency > best[key][0]:
            best[key] = (recency, [row["phases"]])
        elif recency == best[key][0]:
            best[key][1].append(row["phases"])

    grouped: dict[PlotKey, dict[SeriesKey, PhaseTimes]] = defaultdict(lambda: defaultdict(dict))
    for (plot_key, series_key, tasks_per_worker), (_recency, values) in best.items():
        grouped[plot_key][series_key][tasks_per_worker] = {
            name: mean([v[name] for v in values]) for name in PHASE_COLUMNS
        }
    return grouped, workers


def plot_config(
    plot_key: PlotKey,
    series: dict[SeriesKey, PhaseTimes],
    workers: int,
    phase_suffix: str,
    phases: Sequence[str],
    output_dir: str,
    image_format: str,
) -> None:
    system, nodes, ranks_per_node, expected_us, duration_mode, cv = plot_key

    def total_ms(points: dict[str, float]) -> float:
        return 1000.0 * sum(points[name] for name in phases)

    # The batch-only figure has nothing to show at k = 0; the others do.
    include_zero = tuple(phases) != ("elapsed_s",)
    totals: dict[SeriesKey, dict[int, float]] = {
        key: {k: total_ms(v) for k, v in points.items() if k >= (0 if include_zero else 1)}
        for key, points in series.items()
    }
    totals = {key: v for key, v in totals.items() if v}
    if not totals:
        return

    # Rank the legend by total time at the largest batch, worst first, so it
    # reads top-to-bottom in the same order as the curves land.
    ordered = sorted(totals, key=lambda key: -totals[key][max(totals[key])])

    with ieee_figure() as (fig, ax):
        handles = []
        labels = []
        all_batches: set[int] = set()
        for points in totals.values():
            all_batches.update(points)
        batches_sorted = sorted(all_batches)
        # The floor is zero work at k = 0, which a log axis cannot show, so
        # the reference line starts at one task per rank.
        ideal_batches = [k for k in batches_sorted if k >= 1]

        # Computed before the curves so each can be annotated with its
        # efficiency against it.
        ideal_ms: dict[int, float] = {}
        if ideal_batches and expected_us > 0 and workers > 0:
            ideal_ms = {
                k: t * 1000.0
                for k, t in zip(
                    ideal_batches,
                    ideal_greedy_times(
                        duration_mode, cv, workers, ideal_batches, expected_us / 1e6
                    ),
                )
            }

        for rank, series_key in enumerate(ordered):
            distributor, fanout = series_key
            points = totals[series_key]
            batches = sorted(points)
            idx = distributor_series_index(distributor, fanout)
            label = format_distributor_label(distributor, fanout)
            line = plot_node_series(
                ax, idx, batches, [points[b] for b in batches], label, linewidth=1.0
            )
            handles.append(line)
            labels.append(label)

            # Efficiency against the floor, in the curve's own color so it
            # needs no leader line, over a white halo so it stays readable
            # where it crosses a neighbouring curve. `ordered` runs
            # slowest-first, i.e. top to bottom on the axes, so alternating
            # the side puts every label on the face away from its nearest
            # neighbour; with all four on one side the upper curve's labels
            # land on the lower curve.
            offset = 5.0 if rank % 2 == 0 else -10.0
            if distributor in LABEL_ABOVE:
                offset = 5.0
            skipped = LABEL_SKIP.get(distributor, frozenset())
            for k in ANNOTATE_AT:
                if k not in points or k not in ideal_ms or k in skipped:
                    continue
                # The percent sign needs escaping: the science/ieee style
                # renders text through LaTeX.
                ax.annotate(
                    rf"{100.0 * ideal_ms[k] / points[k]:.0f}\%",
                    (k, points[k]),
                    textcoords="offset points",
                    xytext=(1, offset + LABEL_NUDGES.get((distributor, k), 0.0)),
                    ha="center",
                    fontsize=7,
                    color=series_color(idx),
                    zorder=5,
                    path_effects=[withStroke(linewidth=1.8, foreground="white")],
                )

        ideal_handle = None
        if ideal_ms:
            (ideal_handle,) = ax.plot(
                ideal_batches,
                [ideal_ms[k] for k in ideal_batches],
                linestyle='--',
                color='0.4',
                linewidth=1.0,
                zorder=0,
                label="Ideal greedy",
            )
            handles.append(ideal_handle)
            labels.append("Ideal greedy")

            # The bar a dynamic distributor has to clear, drawn dotted against
            # the greedy line's dash: a second *dashed* grey line reads as one
            # line with it at this figure width.
            (static_handle,) = ax.plot(
                ideal_batches,
                [
                    t * 1000.0
                    for t in static_split_times(
                        duration_mode, cv, workers, ideal_batches, expected_us / 1e6
                    )
                ],
                linestyle=(0, (1, 1.6)),
                color='0.4',
                linewidth=1.0,
                zorder=0,
                label="Static split",
            )
            handles.append(static_handle)
            labels.append("Static split")

        ax.set_xlabel("Tasks per worker")
        ax.set_ylabel("Total time (ms)")
        ax.set_yscale('log')
        # A decade-only locator labels a single tick over the range these
        # sweeps cover, which reads as an unscaled axis. Ticking the 1-2-5
        # points of each decade gives three or four labels here, and the same
        # rule still works for the coarser sweeps, where it lands on
        # 200/500/1000 rather than 5/10/20/50.
        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        # One tick every five batches: labelling all twenty crowds the axis at
        # this figure width.
        ax.set_xticks([b for b in batches_sorted if b == 1 or b % 5 == 0])
        # A little room at the edges so the first and last efficiency labels
        # are not clipped by the frame.
        ax.set_xlim(batches_sorted[0] - 0.8, batches_sorted[-1] + 0.8)
        add_light_grid(ax)
        legend_avoiding_data(
            ax, handles, labels, locations=("upper left", "lower right"), **COMPACT_LEGEND_STYLE
        )

        mode_dir = os.path.join(output_dir, duration_mode)
        os.makedirs(mode_dir, exist_ok=True)
        save_figure(
            fig,
            mode_dir,
            f"load_balancing{phase_suffix}_{system}_{nodes}n_{ranks_per_node}rpn"
            f"_{expected_us}us.{image_format}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_cli_args(parser)
    args = parser.parse_args()

    paths = collect_csv_paths(args.input, ["load_balancing", "makespan"])
    rows = parse_rows(paths)
    rows = launch_ranks_per_node(rows)
    rows = filter_ranks_per_node(rows, args.ranks_per_node)
    rows = filter_systems(rows, args.exclude_system)
    if not rows:
        raise SystemExit("No load-balancing makespan rows found in the given inputs")

    # Own subdirectory, as plot_weak_scaling.py does for its modes: one
    # results tree feeds several benchmarks, and a flat output directory
    # mixes their figures together.
    output_dir = os.path.join(args.output_dir, "load_balancing")
    os.makedirs(output_dir, exist_ok=True)
    grouped, workers = group_rows(rows)
    for plot_key, series in grouped.items():
        for phase_suffix, phases in PHASE_SETS:
            plot_config(
                plot_key,
                series,
                workers[plot_key],
                phase_suffix,
                phases,
                output_dir,
                args.format,
            )


if __name__ == "__main__":
    main()
