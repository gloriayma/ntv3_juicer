"""Shared figure style and the validated palette.

Palette values come from the data-viz reference instance and were checked with its
validator: the three categorical slots used here pass the lightness band, chroma
floor, all-pairs CVD separation (worst dE 9.2) and the normal-vision floor (worst
dE 24.0) on the light surface. Aqua carries a sub-3:1 contrast warning, so wherever
it appears the series is also direct-labelled (the relief rule).

Colour is assigned by *job*, not taste:

* identity -> the categorical slots, in fixed order, never cycled
* magnitude (cosine similarity) -> one blue hue, light to dark
* polarity (delta, signed around zero) -> the blue<->red diverging pair with a
  neutral gray midpoint
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8982"
GRID = "#e6e5e1"

# categorical slots, fixed order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# sequential: single blue hue, light -> dark (magnitude)
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
             "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)

# diverging: blue <-> red with neutral gray midpoint (polarity)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec",
     "#f7c9c9", "#e9908f", "#e34948", "#a51f1e"],
)

# Modality gets a stable colour everywhere it appears, so identity never depends
# on how many categories a given figure happens to show.
MODALITY_COLORS = {
    "RNA": SERIES[0],
    "Accessibility": SERIES[1],
    "CAGE": SERIES[2],
    "HistoneChIP": SERIES[3],
    "TFChIP": SERIES[4],
}


def use_style() -> None:
    """Apply a recessive, print-safe style: thin marks, muted grid, no chartjunk."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "font.size": 9,
        "figure.dpi": 130,
        "savefig.bbox": "tight",
    })


def diverging_norm(vmin: float, vmax: float, center: float = 0.0,
                   symmetric: bool = True) -> TwoSlopeNorm:
    """Diverging norm anchored at ``center`` (usually zero).

    ``symmetric`` gives both arms the same data span, so colour intensity means the
    same magnitude either side of zero. Without it a tiny negative value saturates
    the cool arm and reads as visually equal to a large positive one.
    """
    if symmetric:
        m = max(abs(vmin - center), abs(vmax - center)) or 1e-9
        return TwoSlopeNorm(vmin=center - m, vcenter=center, vmax=center + m)
    lo = min(vmin, center - 1e-9)
    hi = max(vmax, center + 1e-9)
    return TwoSlopeNorm(vmin=lo, vcenter=center, vmax=hi)


def annotate(ax, text: str) -> None:
    """Small explanatory note between the axes and its title.

    Re-sets the existing title with extra padding so the note sits below the title
    rather than colliding with it.
    """
    # use_style() sets axes.titlelocation="left", so set_title writes the LEFT title
    # artist while get_title() defaults to reading the CENTER one (which is empty).
    # Read and rewrite the title at its actual location or the pad is silently lost.
    loc = mpl.rcParams.get("axes.titlelocation", "center")
    title = ax.get_title(loc=loc)
    if title:
        ax.set_title(title, loc=loc, pad=26)
    # offset in points, not axes fraction, so the gap is the same regardless of
    # how tall the axes happens to be
    ax.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction",
                xytext=(0, 4), textcoords="offset points",
                fontsize=8, color=TEXT_MUTED, ha="left", va="bottom")


def save(fig, path) -> None:
    fig.savefig(path)
    plt.close(fig)
