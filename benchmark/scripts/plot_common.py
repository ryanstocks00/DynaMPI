# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for DynaMPI benchmark CSV plotting scripts."""
from __future__ import annotations

import argparse
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import Bbox
import scienceplots  # noqa: F401  # registers matplotlib styles

IEEE_FIG_WIDTH = 3.5
IEEE_FIG_HEIGHT = 3.5

MARKER_SHAPES = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '<', '>', 'd']

# Trailing PBS/Slurm job id in result directory names, e.g. ...-8725895 or
# ...-8720430.aurora. Used as a recency tie-breaker when git checkouts share
# identical file mtimes.
_JOB_ID_RE = re.compile(r"(?:^|[-_/])(\d{6,})(?:\.[A-Za-z]+)?(?:/|$)")

RowT = TypeVar("RowT", bound=Mapping[str, Any])
KeyT = TypeVar("KeyT")
Recency = tuple[int, int, float]


def collect_csv_paths(inputs: Sequence[str], name_substring: str | Sequence[str]) -> list[str]:
    """Collect CSVs whose basename contains any of the given substrings.

    Accepts several needles because the throughput results predate the
    strong- to weak-scaling rename: the benchmark driver and launch scripts
    still emit ``strong_scaling_<system>.csv``, while the plots they feed are
    named ``weak_scaling_*``.
    """
    needles = [name_substring.lower()] if isinstance(name_substring, str) else [
        needle.lower() for needle in name_substring
    ]

    def matches(name: str) -> bool:
        lowered = name.lower()
        return any(needle in lowered for needle in needles)

    paths: list[str] = []
    for raw in inputs:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if os.path.isdir(entry):
                for root, _, files in os.walk(entry):
                    for name in files:
                        if name.endswith(".csv") and matches(name):
                            paths.append(os.path.join(root, name))
            elif matches(os.path.basename(entry)):
                paths.append(entry)
    return paths


def path_recency(path: str, file_mtime: float) -> Recency:
    """Return a comparable recency key: (final_boost, job_id, mtime).

    Job id dominates mtime so a fresh git checkout's near-identical timestamps
    cannot prefer an older PBS result over a later one.
    """
    normalized = path.replace("\\", "/")
    final_boost = 1 if "/final-" in normalized or normalized.startswith("final-") else 0
    job_id = 0
    for match in _JOB_ID_RE.finditer(normalized):
        job_id = max(job_id, int(match.group(1)))
    return (final_boost, job_id, file_mtime)


def dedupe_newest(
    rows: Sequence[RowT],
    config_key: Callable[[RowT], KeyT],
    value_key: str,
) -> dict[KeyT, tuple[Any, Recency]]:
    """Keep the newest row per configuration key.

    Prefer ``final-*`` result dirs, then the highest embedded job id, then
    later file mtimes.
    """
    newest: dict[KeyT, tuple[Any, Recency]] = {}
    for row in rows:
        key = config_key(row)
        recency = row.get("recency")
        if recency is None:
            recency = path_recency(str(row.get("path", "")), float(row["file_mtime"]))
        value = (row[value_key], recency)
        if key not in newest or recency > newest[key][1]:
            newest[key] = value
    return newest


def normalize_mode(mode: str) -> str:
    return "random" if mode == "poisson" else mode


def format_fanout(fanout: int) -> str:
    """Name a hierarchy depth by its number of coordination layers.

    ``0`` disables upper grouping, leaving manager -> node coordinators:
    two layers. The default (negative, auto) inserts a grouping level above
    the node coordinators once they outnumber ~32, giving manager -> group
    leaders -> node coordinators: three layers. Below that threshold auto
    resolves to the same flat topology as ``0``.
    """
    if fanout < 0:
        return "three-layer"
    if fanout == 0:
        return "two-layer"
    return f"fanout={fanout}"


# Display names for the four distributor implementations. The CSV carries the
# driver's --distribution values; these are what the paper calls them.
DISTRIBUTOR_DISPLAY_NAMES = {
    "naive": "Naive",
    "hierarchical": "Hierarchical",
    "lockfree_rma": "Lock-Free RMA",
    "hierarchical_lockfree_rma": "Hierarchical Lock-Free RMA",
}


def format_distributor_label(distributor: str, fanout: int | None = None) -> str:
    label = DISTRIBUTOR_DISPLAY_NAMES.get(distributor, distributor.replace("_", "-"))
    if fanout is None or "hierarchical" not in distributor:
        return label
    return f"{label} ({format_fanout(fanout)})"


# A legend spanning most of the axes width has to clear the curves out at
# the far end of the sweep, which on these diagonal log-log plots costs a
# lot of axis range (~3 decades at the default sizing) and shrinks the data
# to a corner. Trimming the font and the internal padding pulls the legend's
# far edge back over a much lower part of the curves, cutting that cost to
# under a decade.
COMPACT_LEGEND_STYLE: dict[str, Any] = {
    "frameon": False,
    "fontsize": 7,
    "handlelength": 1.2,
    "handletextpad": 0.4,
    "labelspacing": 0.2,
    "columnspacing": 0.8,
    "borderpad": 0.25,
    "borderaxespad": 0.3,
}


def _drawn_points(ax: Axes) -> list[np.ndarray]:
    """Display-space points along every drawn line, segments included.

    Matplotlib joins a Line2D's vertices with straight segments in display
    space, so interpolating there is exact: a legend can sit between two
    markers and still cover the line running between them.
    """
    sampled: list[np.ndarray] = []
    for line in ax.get_lines():
        xy = np.asarray(line.get_xydata(), dtype=float)
        if xy.size == 0:
            continue
        points = ax.transData.transform(xy)
        sampled.append(points)
        if len(points) > 1:
            starts, ends = points[:-1], points[1:]
            steps = np.linspace(0.0, 1.0, 16)[1:-1][:, None, None]
            sampled.append((starts + steps * (ends - starts)).reshape(-1, 2))
    return sampled


def _legend_ink_boxes(legend: Legend, renderer: Any, pad: float) -> list[Bbox]:
    """Padded display-space boxes around what the legend actually draws.

    The legend's own extent is one rectangle as wide as its longest entry
    and padded by ``borderpad``, none of which is visible when
    ``frameon=False``. Testing the label text and handle boxes instead lets
    curves run through the blank space beside a short entry, which is what
    keeps a wide legend from having to sit far above the data.
    """
    handles = getattr(legend, "legend_handles", None)
    if handles is None:  # matplotlib < 3.7
        handles = getattr(legend, "legendHandles", [])
    boxes: list[Bbox] = []
    for artist in [*legend.get_texts(), *handles]:
        try:
            extent = artist.get_window_extent(renderer)
        except TypeError:
            extent = artist.get_window_extent()
        except (AttributeError, NotImplementedError, ValueError, RuntimeError):
            # A handle type that cannot report an extent -- skip it from the
            # ink-box calculation rather than fail the whole legend placement.
            continue
        if extent.width <= 0 or extent.height <= 0:
            continue
        boxes.append(
            Bbox.from_extents(
                extent.x0 - pad, extent.y0 - pad, extent.x1 + pad, extent.y1 + pad
            )
        )
    if not boxes:  # nothing measurable: fall back to the whole legend
        extent = legend.get_window_extent(renderer)
        boxes.append(
            Bbox.from_extents(
                extent.x0 - pad, extent.y0 - pad, extent.x1 + pad, extent.y1 + pad
            )
        )
    return boxes


def _legend_hits_data(ax: Axes, legend: Legend, pad_points: float) -> bool:
    figure = ax.figure
    figure.canvas.draw()
    pad = pad_points * figure.dpi / 72.0
    canvas = cast(FigureCanvasAgg, figure.canvas)
    boxes = _legend_ink_boxes(legend, canvas.get_renderer(), pad)
    for points in _drawn_points(ax):
        for box in boxes:
            inside = (
                (points[:, 0] >= box.x0)
                & (points[:, 0] <= box.x1)
                & (points[:, 1] >= box.y0)
                & (points[:, 1] <= box.y1)
            )
            if inside.any():
                return True
    return False


def legend_avoiding_data(
    ax: Axes,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    locations: Sequence[str] = ("upper left", "lower right", "center left", "lower center"),
    pad_points: float = 3.0,
    expand_step_decades: float = 0.1,
    max_expand_decades: float = 3.0,
    **legend_kwargs: Any,
) -> Legend:
    """Draw a legend that covers none of the plotted curves.

    Tries each candidate corner in order and keeps the first that is already
    clear. If every corner is blocked -- these sweeps run diagonally across
    the axes, so a legend spanning most of the width usually is -- the axis
    range is opened up a tenth of a decade at a time until one frees up:
    upward for a top-anchored legend, downward for a bottom-anchored one.
    Whichever placement needs the least added range wins, which keeps the
    curves as large as the legend allows.

    ``tight_layout`` runs first because ``save_figure`` applies it after the
    fact: it resizes the axes rectangle (here, shorter and wider), and a
    shorter axes means a fixed-height legend covers more of the data, so a
    placement verified against the pre-layout geometry can be wrong in the
    saved figure.
    """
    cast(Figure, ax.figure).tight_layout()
    legend = ax.legend(handles, labels, loc=cast(Any, locations[0]), **legend_kwargs)
    for location in locations:
        legend.remove()
        legend = ax.legend(handles, labels, loc=cast(Any, location), **legend_kwargs)
        if not _legend_hits_data(ax, legend, pad_points):
            return legend

    original = ax.get_ylim()
    max_steps = max(1, int(round(max_expand_decades / expand_step_decades)))
    factor = 10**expand_step_decades
    best: tuple[int, str, tuple[float, float]] | None = None
    for location in locations[:2]:
        downward = location.startswith("lower")
        bottom, top = original
        for steps in range(1, max_steps + 1):
            if downward:
                bottom /= factor
            else:
                top *= factor
            ax.set_ylim(bottom, top)
            legend.remove()
            legend = ax.legend(handles, labels, loc=cast(Any, location), **legend_kwargs)
            if not _legend_hits_data(ax, legend, pad_points):
                if best is None:
                    best = (steps, location, (bottom, top))
                elif steps < best[0]:
                    best = (steps, location, (bottom, top))
                break
        ax.set_ylim(original)

    if best is None:  # nothing fits: fall back to maximum headroom, top-anchored
        best = (max_steps, locations[0], (original[0], original[1] * factor**max_steps))
    _, location, ylim = best
    ax.set_ylim(ylim)
    legend.remove()
    return ax.legend(handles, labels, loc=cast(Any, location), **legend_kwargs)


def series_marker(index: int) -> str:
    return MARKER_SHAPES[index % len(MARKER_SHAPES)]


def series_color(index: int) -> Any:
    return colormaps['tab10'](index % 10)


def format_node_tick(nodes: int) -> str:
    """Compact labels for dense large-node ticks (1024→1k, 2048→2k)."""
    if nodes >= 1024 and nodes % 1024 == 0:
        return f"{nodes // 1024}k"
    return f"{nodes}"


def set_log_node_axes(ax: Axes, all_nodes: set[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # A little more than matplotlib's default 5% so that markers at the ends
    # of a sweep clear the frame instead of sitting on it.
    ax.margins(x=0.07, y=0.09)
    if all_nodes:
        node_ticks = sorted(all_nodes)
        ax.xaxis.set_major_locator(FixedLocator(node_ticks))
        ax.set_xticklabels([format_node_tick(n) for n in node_ticks])


def add_light_grid(ax: Axes) -> None:
    ax.grid(True, which="both", linestyle="-", linewidth=0.5, color='lightgrey', alpha=0.5, zorder=0)


def add_plot_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        help="CSV file or directory (can be passed multiple times)",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write output plots")
    parser.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg"], help="Output image format"
    )
    parser.add_argument(
        "--ranks-per-node",
        type=int,
        nargs="+",
        # Aurora gpu7/full-node and Frontier GPU/CPU-style sweeps.
        default=[7, 9, 56, 102],
        help="Only keep rows whose world_size/nodes matches one of these "
        "(default: 7 9 56 102). Pass 0 to disable filtering.",
    )
    parser.add_argument(
        "--exclude-system",
        action="append",
        default=[],
        help="System label to omit (can be passed multiple times)",
    )


def filter_ranks_per_node(rows: Sequence[RowT], ranks_per_node: Sequence[int]) -> list[RowT]:
    """Drop rows whose ranks-per-node is outside the requested set.

    A single 0 in ranks_per_node disables filtering.
    """
    if not ranks_per_node or 0 in ranks_per_node:
        return list(rows)
    allowed = set(ranks_per_node)
    return [row for row in rows if int(row["ranks_per_node"]) in allowed]


def filter_systems(rows: Sequence[RowT], exclude_systems: Sequence[str]) -> list[RowT]:
    excluded = {name.lower() for name in exclude_systems}
    if not excluded:
        return list(rows)
    return [row for row in rows if str(row["system"]).lower() not in excluded]


@contextmanager
def ieee_figure() -> Iterator[tuple[Figure, Axes]]:
    """Open a (fig, ax) pair styled and sized for the paper's IEEE figures.

    Applies the style/rcParams/figsize shared by every per-series plot in
    these scripts, so each caller only has to write what differs: the data,
    the labels, the legend.
    """
    with plt.style.context(['science', 'ieee']):
        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.labelsize": 10,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 8,
            }
        )
        fig, ax = plt.subplots(figsize=(IEEE_FIG_WIDTH, IEEE_FIG_HEIGHT * 0.7))
        yield fig, ax


def save_figure(
    fig: Figure,
    output_dir: str,
    filename: str,
    *,
    tight_layout_rect: tuple[float, float, float, float] | None = None,
) -> None:
    if tight_layout_rect is not None:
        fig.tight_layout(rect=tight_layout_rect)
    else:
        fig.tight_layout()
    # Crop hard to the drawn content: any border baked into the image becomes
    # a gap between the figure and its caption once the file is included in
    # the paper, on top of whatever the document class already adds.
    fig.savefig(
        os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0.01
    )
    plt.close(fig)
