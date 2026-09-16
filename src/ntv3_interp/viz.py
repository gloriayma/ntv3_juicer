"""Shared plotting parameters.

Palette values come from the validated reference instance in the dataviz skill
(`references/palette.md`). Categorical slots are used in fixed order, never cycled.

Note on scatter plots: the validated palette supports at most three categorical hues
under the all-pairs pairlist that scatter requires. Since this analysis has ~15
modalities and >100 tissues, categorical coloring is not an option -- every scatter
uses small multiples that highlight one group against a recessive gray, which keeps
identity legible without relying on hue separation at all.
"""

from __future__ import annotations

import matplotlib as mpl

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
GRID = "#e6e5e1"

# Categorical slots, fixed order (light mode).
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

HIGHLIGHT = SERIES[0]
RECESSIVE = "#d8d7d2"

# Sequential blue ramp, light -> dark.
SEQ_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

# Diverging blue <-> red with a neutral gray midpoint (never a hue at the midpoint).
DIVERGING = ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a6a5", "#e34948", "#8f2322"]


def sequential_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def diverging_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("div_br", DIVERGING)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
