"""Analysis 2: the cosine similarity matrix, shown under different orderings.

An explicit caveat carried into the report: reordering rows and columns does not
change the matrix, so a block pattern appearing under one ordering and not another
is a *visual* cue only -- it is not a statistical null. The null lives in
:mod:`juicer.tissue_test`.

The full 7362x7362 image is included for completeness, but the block-mean matrices
(mean cosine per assay x assay, per tissue x tissue) are what can actually be read,
so those are the primary figures here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import viz


def ordering(md: pd.DataFrame, by: list[str]) -> np.ndarray:
    """Row order sorted by the given label columns in sequence."""
    return md.sort_values(by, kind="stable").index.to_numpy()


def block_means(S: np.ndarray, labels: np.ndarray,
                categories: list[str] | None = None) -> pd.DataFrame:
    """Mean cosine per pair of label blocks, excluding self-comparisons."""
    cats = categories or sorted(set(labels))
    idx = {c: np.where(labels == c)[0] for c in cats}
    out = pd.DataFrame(index=cats, columns=cats, dtype=float)
    for a in cats:
        for b in cats:
            block = S[np.ix_(idx[a], idx[b])]
            if a == b:
                n = len(idx[a])
                out.loc[a, b] = (block[np.triu_indices(n, k=1)].mean()
                                 if n > 1 else np.nan)
            else:
                out.loc[a, b] = block.mean()
    return out


def plot_block_matrix(M: pd.DataFrame, title: str, note: str, path,
                      annot: bool = True) -> None:
    """Heatmap of a block-mean matrix. Magnitude -> one blue hue, light to dark."""
    viz.use_style()
    n = len(M)
    fig, ax = plt.subplots(figsize=(max(4.5, 0.62 * n + 2.6),
                                    max(3.8, 0.62 * n + 1.9)))
    ax.grid(False)
    vals = M.values.astype(float)
    im = ax.imshow(vals, cmap=viz.SEQ, vmin=0.0,
                   vmax=float(np.nanmax(vals)), aspect="auto")
    ax.set_xticks(range(n), M.columns, rotation=45, ha="right")
    ax.set_yticks(range(n), M.index)
    ax.set_title(title)
    viz.annotate(ax, note)
    if annot and n <= 12:
        mid = float(np.nanmax(vals)) * 0.55
        for i in range(n):
            for j in range(n):
                v = vals[i, j]
                if np.isnan(v):
                    continue
                # value labels in ink, light on dark cells for legibility
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color=("#ffffff" if v > mid else viz.TEXT_PRIMARY))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("mean cosine", color=viz.TEXT_SECONDARY, fontsize=8)
    cb.outline.set_visible(False)
    viz.save(fig, path)


def plot_full_matrix(S: np.ndarray, md: pd.DataFrame, by: list[str], title: str,
                     path, max_n: int = 3000, seed: int = 0) -> None:
    """Full similarity matrix under one ordering, with label bands in the margin."""
    viz.use_style()
    order = ordering(md, by)
    if len(order) > max_n:
        rng = np.random.default_rng(seed)
        # keep the ordering, thin it evenly so block structure is preserved
        sel = np.sort(rng.choice(len(order), size=max_n, replace=False))
        order = order[sel]
    Sub = S[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.grid(False)
    im = ax.imshow(Sub, cmap=viz.SEQ, vmin=0.0, vmax=0.6,
                   interpolation="nearest", aspect="equal")
    ax.set_title(title)
    viz.annotate(ax, f"ordered by {' -> '.join(by)}; n={len(order)} tracks "
                     "(reordering is a visual cue, not a null)")
    ax.set_xticks([])
    ax.set_yticks([])

    # boundaries of the primary grouping
    prim = md.loc[order, by[0]].to_numpy()
    edges = np.where(prim[1:] != prim[:-1])[0] + 1
    for e in edges:
        ax.axhline(e, color=viz.SURFACE, linewidth=0.8)
        ax.axvline(e, color=viz.SURFACE, linewidth=0.8)
    # label each primary block on the y axis
    bounds = np.concatenate([[0], edges, [len(order)]])
    ticks, names = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > len(order) * 0.02:
            ticks.append((a + b) / 2)
            names.append(str(prim[a]))
    ax.set_yticks(ticks, names, fontsize=7)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("cosine", color=viz.TEXT_SECONDARY, fontsize=8)
    cb.outline.set_visible(False)
    viz.save(fig, path)
