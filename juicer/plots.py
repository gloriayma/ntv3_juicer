"""Figures for Analyses 1 and 3.

Note on the UMAP figures: a scatter puts every pair of categories on screen at once,
and the validated categorical palette only clears the all-pairs colour-vision floors
for three slots. With five coarse modalities, colouring one scatter by modality would
break that. So category views are drawn as **small multiples** -- one panel per
category, that category in its fixed colour against the full cloud in recessive gray.
That also reads better than a five-colour overplot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import viz


def plot_umap_facets(emb: np.ndarray, labels: np.ndarray, title: str, note: str,
                     path, colors: dict | None = None, max_panels: int = 8,
                     order: list[str] | None = None) -> None:
    """Small multiples: one panel per category, highlighted against the full cloud."""
    viz.use_style()
    cats = order or list(pd.Series(labels).value_counts().index[:max_panels])
    n = len(cats)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.9 * nrow),
                             squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for i, c in enumerate(cats):
        ax = axes[i // ncol][i % ncol]
        ax.set_visible(True)
        ax.grid(False)
        ax.scatter(emb[:, 0], emb[:, 1], s=2, c="#dcdbd6", linewidths=0,
                   rasterized=True)
        m = labels == c
        # When no stable category->colour mapping is supplied, every panel uses the
        # same highlight colour: identity is carried by the panel title, so cycling
        # hues here would imply a colour code that does not exist.
        col = colors[c] if colors and c in colors else viz.SERIES[0]
        ax.scatter(emb[m, 0], emb[m, 1], s=4, c=col, linewidths=0, rasterized=True)
        # direct label on the panel satisfies the relief rule for low-contrast hues
        ax.set_title(f"{c}  (n={int(m.sum())})", fontsize=9, color=col)
        ax.set_xticks([])
        ax.set_yticks([])
    # title first, note on its own line beneath it
    fig.suptitle(title, x=0.008, y=0.995, ha="left", va="top", fontsize=11,
                 color=viz.TEXT_PRIMARY, weight="bold")
    fig.text(0.008, 0.955, note, ha="left", va="top", fontsize=8,
             color=viz.TEXT_MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    viz.save(fig, path)


def plot_predictability(df: pd.DataFrame, title: str, note: str, path) -> None:
    """Balanced accuracy vs number of PCs, per label, against permuted baselines.

    Series identity is carried by direct labels at the right-hand end; the legend
    explains the solid/dotted distinction rather than repeating the colours, and
    sits below the axes so it never covers the data.
    """
    from matplotlib.lines import Line2D

    viz.use_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ks = sorted(df["k_pcs"].unique())
    labels = list(dict.fromkeys(df["label"]))
    for i, lab in enumerate(labels):
        sub = df[df["label"] == lab].sort_values("k_pcs")
        col = viz.SERIES[i % len(viz.SERIES)]
        ax.plot(sub["k_pcs"], sub["balanced_acc"], color=col, marker="o")
        ax.plot(sub["k_pcs"], sub["permuted_baseline"], color=col, linestyle=":",
                linewidth=1.4, alpha=0.75)
        last = sub.iloc[-1]
        ax.annotate(f" {lab}", (last["k_pcs"], last["balanced_acc"]),
                    textcoords="offset points", xytext=(7, 0), fontsize=8,
                    color=col, va="center", ha="left")

    ax.set_xscale("log")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.xaxis.set_minor_locator(plt.NullLocator())     # kill 3x10^0 clutter
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(ks[0] * 0.82, ks[-1] * 2.9)           # room for direct labels
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("balanced accuracy (grouped CV)")
    ax.set_title(title)
    viz.annotate(ax, note)
    ax.legend(handles=[
        Line2D([], [], color=viz.TEXT_SECONDARY, lw=2.0, label="real labels"),
        Line2D([], [], color=viz.TEXT_SECONDARY, lw=1.4, ls=":",
               label="permuted labels (chance)"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    viz.save(fig, path)


AXIS_LABELS = {
    "w_norm": "‖w_t‖₂   (weight-row norm)",
    "w_gain_norm": "‖w_t ⊙ γ‖₂   (gain-folded norm)",
    "bias": "b_t   (raw head bias)",
    "bias_effective": "w_t·β + b_t   (effective offset)",
}


def plot_norms(tbl: pd.DataFrame, value: str, title: str, note: str, path,
               by: str = "assay") -> None:
    """Distribution of a per-track scalar by category (magnitude, single hue)."""
    viz.use_style()
    cats = list(tbl.groupby(by)[value].median().sort_values().index)
    data = [tbl.loc[tbl[by] == c, value].to_numpy() for c in cats]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(cats) + 2.0))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.6,
                    showfliers=False, medianprops=dict(color=viz.TEXT_PRIMARY,
                                                       linewidth=1.4))
    shades = viz.SEQ(np.linspace(0.30, 0.85, len(cats)))
    for patch, col in zip(bp["boxes"], shades):
        patch.set_facecolor(col)
        patch.set_edgecolor(viz.SURFACE)
        patch.set_linewidth(1.2)
    for part in ("whiskers", "caps"):
        for artist in bp[part]:
            artist.set_color(viz.TEXT_MUTED)
            artist.set_linewidth(1.0)
    ax.set_yticks(range(1, len(cats) + 1), cats)
    ax.set_xlabel(AXIS_LABELS.get(value, value))
    ax.set_title(title)
    viz.annotate(ax, note)
    ax.grid(axis="y", visible=False)
    viz.save(fig, path)


def plot_delta_by_pair(by_pair: pd.DataFrame, title: str, note: str, path) -> None:
    """Per-modality-pair delta. Signed quantity -> diverging colour, zero anchored."""
    viz.use_style()
    d = by_pair.sort_values("delta")
    fig, ax = plt.subplots(figsize=(6.6, 0.40 * len(d) + 2.0))
    norm = viz.diverging_norm(float(d["delta"].min()), float(d["delta"].max()))
    colors = viz.DIV(norm(d["delta"].to_numpy()))
    y = np.arange(len(d))
    ax.barh(y, d["delta"], color=colors, height=0.68,
            edgecolor=viz.SURFACE, linewidth=1.2)
    ax.axvline(0, color=viz.TEXT_SECONDARY, linewidth=1.0)
    ax.set_yticks(y, d["modality_pair"])
    ax.set_xlabel("Δ  =  mean cos(same tissue) − mean cos(different tissue)")
    ax.set_title(title)
    viz.annotate(ax, note)
    ax.grid(axis="y", visible=False)
    for yi, (val, nc) in enumerate(zip(d["delta"], d["n_cells"])):
        off = 6 if val >= 0 else -6
        ha = "left" if val >= 0 else "right"
        ax.annotate(f"{val:+.3f}  ({nc} tissues)", (val, yi),
                    textcoords="offset points", xytext=(off, 0), fontsize=7.5,
                    color=viz.TEXT_SECONDARY, va="center", ha=ha)
    # leave room for the value labels, which sit outside the bar ends
    lo, hi = float(d["delta"].min()), float(d["delta"].max())
    span = (hi - lo) or 1.0
    ax.set_xlim(lo - (0.62 * span if lo < 0 else 0.04 * span), hi + 0.34 * span)
    viz.save(fig, path)


def plot_null(observed: float, null_mean: float, null_sd: float, nulls: np.ndarray,
              title: str, note: str, path) -> None:
    """Permutation null distribution with the observed statistic marked."""
    viz.use_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(nulls, bins=40, color=viz.SEQ_STEPS[2], edgecolor=viz.SURFACE,
            linewidth=0.8)
    ax.axvline(observed, color="#e34948", linewidth=2.0)
    ax.annotate(f"observed Δ = {observed:+.4f}", (observed, ax.get_ylim()[1] * 0.92),
                textcoords="offset points", xytext=(-8, 0), fontsize=8.5,
                color="#a51f1e", ha="right", va="top")
    ax.set_xlabel("aggregate Δ under permuted tissue labels")
    ax.set_ylabel("permutations")
    ax.set_title(title)
    viz.annotate(ax, note)
    viz.save(fig, path)


def plot_delta_hist(cells: pd.DataFrame, title: str, note: str, path) -> None:
    """Distribution of per-cell deltas. Signed -> diverging fill around zero."""
    viz.use_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    vals = cells["delta"].to_numpy()
    counts, edges = np.histogram(vals, bins=46)
    centers = (edges[:-1] + edges[1:]) / 2
    norm = viz.diverging_norm(float(centers.min()), float(centers.max()))
    ax.bar(centers, counts, width=np.diff(edges), color=viz.DIV(norm(centers)),
           edgecolor=viz.SURFACE, linewidth=0.6)
    ax.axvline(0, color=viz.TEXT_SECONDARY, linewidth=1.0)
    frac = float((vals > 0).mean())
    ax.set_xlabel("Δ per (tissue × modality-pair) cell")
    ax.set_ylabel("cells")
    ax.set_title(title)
    viz.annotate(ax, note)
    ax.annotate(f"{frac:.0%} of cells positive", (0.98, 0.92),
                xycoords="axes fraction", ha="right", fontsize=8.5,
                color=viz.TEXT_SECONDARY)
    viz.save(fig, path)
