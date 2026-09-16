"""A2: similarity structure under different orderings, plus the modality-removal control.

Two things here:
  1. The SAME cosine matrix rendered under modality-first and tissue-first orderings.
     Reordering is a visual aid, not a test -- S itself is unchanged by permuting its rows.
  2. The principled way to "remove modality": subtract each modality's centroid from its
     members (regressing out a categorical), then re-run the A3 delta test. Deleting PC1
     would be the wrong move -- modality is categorical and occupies several dimensions.

Run: uv run python scripts/a2_grids.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ntv3_interp import viz  # noqa: E402
from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)
from ntv3_interp.tissue_test import cross_modality_auroc, run_delta_test  # noqa: E402

viz.apply_style()
Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200

head = load_head("data/head_NTv3_650M_post.npz")
tracks = pd.read_parquet("data/tracks.parquet")
V = effective_vectors(head)
alive = dead_track_mask(head)

df = tracks.copy()
df["alive"] = alive
df = df[df.alive & df.modality.notna() & df.term_id.notna()]
vc = df.modality.value_counts()
df = df[df.modality.isin(vc[vc >= 10].index)]
npm = df.groupby("term_id").modality.nunique()
df = df[df.term_id.isin(npm[npm >= 2].index)].reset_index(drop=True)

idx = df["index"].to_numpy()
U = l2_normalize(V[idx])
print(f"{len(df)} tracks, {df.term_id.nunique()} tissues, {df.modality.nunique()} modalities")

S = U @ U.T

# ---------------------------------------------------------------- ordered grids
def ordered_grid(keys: list[str], title: str, subtitle: str, fname: str) -> None:
    order = df.sort_values(keys).index.to_numpy()
    M = S[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    vmax = 0.6
    im = ax.imshow(
        M, cmap=viz.diverging_cmap(), vmin=-vmax, vmax=vmax, interpolation="nearest"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=11.5, color=viz.TEXT_PRIMARY, pad=14)
    ax.text(
        0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8.5, color=viz.TEXT_SECONDARY,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("cosine similarity", fontsize=8.5, color=viz.TEXT_SECONDARY)
    cb.outline.set_visible(False)

    # Mark the boundaries of the primary sort key.
    prim = df.loc[order, keys[0]].to_numpy()
    bounds = np.flatnonzero(prim[1:] != prim[:-1]) + 1
    for b in bounds:
        ax.axhline(b, color=viz.SURFACE, lw=0.4)
        ax.axvline(b, color=viz.SURFACE, lw=0.4)
    fig.savefig(f"figures/{fname}")
    plt.close(fig)


ordered_grid(
    ["modality", "term_id"],
    "Ordered by modality, then tissue",
    "Sharp blocks on the diagonal: modality is the dominant organizing structure",
    "a2_grid_modality_first.png",
)
ordered_grid(
    ["term_id", "modality"],
    "Same matrix, ordered by tissue, then modality",
    "No comparable block structure — tissue is not the dominant axis",
    "a2_grid_tissue_first.png",
)

# ---------------------------------------------------------------- modality block means
mods = df.modality.value_counts().index.tolist()
mi = {m: k for k, m in enumerate(mods)}
code = df.modality.map(mi).to_numpy()
nm = len(mods)
B = np.zeros((nm, nm))
for a in range(nm):
    ia = code == a
    for b in range(nm):
        ib = code == b
        blk = S[np.ix_(ia, ib)]
        if a == b:
            n = ia.sum()
            B[a, b] = (blk.sum() - np.trace(blk)) / max(n * (n - 1), 1)
        else:
            B[a, b] = blk.mean()

fig, ax = plt.subplots(figsize=(7.4, 6.4))
v = np.abs(B).max()
im = ax.imshow(B, cmap=viz.diverging_cmap(), vmin=-v, vmax=v)
ax.set_xticks(range(nm))
ax.set_yticks(range(nm))
ax.set_xticklabels(mods, rotation=90, fontsize=7)
ax.set_yticklabels(mods, fontsize=7)
ax.grid(False)
ax.set_title(
    "Mean cosine between assay types", fontsize=11.5, color=viz.TEXT_PRIMARY, pad=12
)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.set_label("mean cosine", fontsize=8.5, color=viz.TEXT_SECONDARY)
cb.outline.set_visible(False)
fig.savefig("figures/a2_modality_blocks.png")
plt.close(fig)
np.save("results/a2_modality_blocks.npy", B)
pd.Series(mods).to_csv("results/a2_modality_order.csv", index=False, header=["modality"])
del S

# ---------------------------------------------------------------- modality removal control
print("\nmodality-removal control: subtract each modality centroid, then re-test tissue")
Vsub = V[idx].copy()
for m in df.modality.unique():
    sel = (df.modality == m).to_numpy()
    Vsub[sel] -= Vsub[sel].mean(axis=0, keepdims=True)
U2 = l2_normalize(Vsub).astype(np.float32)

tissue_codes, tissue_uniques = pd.factorize(df.term_id)
mod_codes, mod_uniques = pd.factorize(df.modality)

out = {}
for name, Umat in [("raw", U.astype(np.float32)), ("modality_removed", U2)]:
    auroc = cross_modality_auroc(Umat, tissue_codes, mod_codes)
    res = run_delta_test(
        Umat,
        tissue_codes.astype(np.int64),
        mod_codes.astype(np.int64),
        n_tissue=len(tissue_uniques),
        n_mod=len(mod_uniques),
        n_perm=N_PERM,
    )
    out[name] = {
        "observed_delta": res["observed_delta"],
        "null_mean": res["null_mean"],
        "null_sd": res["null_sd"],
        "z": res["z"],
        "p_perm": res["p_perm"],
        "stratified_auroc": auroc.get("stratified_auroc"),
        "pooled_auroc": auroc.get("pooled_auroc"),
        "frac_positive": float(np.nanmean(res["per_stratum"] > 0)),
    }
    print(
        f"  {name:18s} delta={res['observed_delta']:+.5f} z={res['z']:+.1f} "
        f"AUROC={auroc.get('stratified_auroc'):.4f} "
        f"pos={out[name]['frac_positive']:.1%}"
    )

Path("results/a2_modality_removed.json").write_text(json.dumps(out, indent=2))
print("\nsaved results/a2_*.json and figures/a2_*.png")
