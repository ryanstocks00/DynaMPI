# SPDX-FileCopyrightText: 2026 Ryan Stocks
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for DynaMPI benchmark CSV plotting scripts."""
from __future__ import annotations

import argparse
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter
import scienceplots  # noqa: F401  # registers matplotlib styles

IEEE_FIG_WIDTH = 3.5
IEEE_FIG_HEIGHT = 3.5

MARKER_SHAPES = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', 'X', '<', '>', 'd']

RowT = TypeVar("RowT", bound=Mapping[str, Any])
KeyT = TypeVar("KeyT")


def collect_csv_paths(inputs: Sequence[str], name_substring: str) -> list[str]:
    needle = name_substring.lower()
    paths: list[str] = []
    for raw in inputs:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if os.path.isdir(entry):
                for root, _, files in os.walk(entry):
                    for name in files:
                        if name.endswith(".csv") and needle in name.lower():
                            paths.append(os.path.join(root, name))
            elif needle in os.path.basename(entry).lower():
                paths.append(entry)
    return paths


def dedupe_newest(
    rows: Sequence[RowT],
    config_key: Callable[[RowT], KeyT],
    value_key: str,
) -> dict[KeyT, tuple[Any, float]]:
    """Keep the newest row per configuration key, using each row's file_mtime."""
    newest: dict[KeyT, tuple[Any, float]] = {}
    for row in rows:
        key = config_key(row)
        value = (row[value_key], row["file_mtime"])
        if key not in newest or row["file_mtime"] > newest[key][1]:
            newest[key] = value
    return newest


def normalize_mode(mode: str) -> str:
    return "random" if mode == "poisson" else mode


def series_marker(index: int) -> str:
    return MARKER_SHAPES[index % len(MARKER_SHAPES)]


def series_color(index: int) -> Any:
    return colormaps['tab10'](index % 10)


def set_log_node_axes(ax: Axes, all_nodes: set[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    if all_nodes:
        node_ticks = sorted(all_nodes)
        ax.xaxis.set_major_locator(FixedLocator(node_ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))


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
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
