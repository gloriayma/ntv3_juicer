"""A11: family-level versions of the similarity grids and the lab/batch confound.

Everything here repeats an earlier analysis with assay FAMILY in place of the fine
assay:target modality label, so that every cross-group comparison is between genuinely
different measurement types.

Run: uv run python scripts/a11_family_rerun.py [n_perm]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ntv3_interp import viz  # noqa: E402
from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)
from ntv3_interp.tissue_test import cross_modality_auroc, run_delta_test  # noqa: E402

viz.apply_style()
N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out: dict = {}

head = load_head("data/head_NTv3_650M_post.npz")
tracks = pd.read_parquet("data/tracks.parquet")
V = effective_vectors(head)
alive = dead_track_mask(head)

df = tracks.copy()
df["alive"] = alive
df = df[df.alive & df.assay_family.notna() & df.term_id.notna()]
npf = df.groupby("term_id").assay_family.nunique()
df = df[df.term_id.isin(npf[npf >= 2].index)].reset_index(drop=True)
U = l2_normalize(V[df["index"].to_numpy()])
print(f"{len(df)} tracks | {df.term_id.nunique()} tissues | "
      f"{df.assay_family.nunique()} families")

S = U @ U.T

# ------------------------------------------------------------- grids
def grid(keys, title, subtitle, fname):
    order = df.sort_values(keys).index.to_numpy()
    M = S[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(M, cmap=viz.diverging_cmap(), vmin=-0.6, vmax=0.6, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(title, fontsize=11.5, color=viz.TEXT_PRIMARY, pad=14)
    ax.text(0.5, 1.012, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=viz.TEXT_SECONDARY)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("cosine similarity", fontsize=8.5, color=viz.TEXT_SECONDARY)
    cb.outline.set_visible(False)
    prim = df.loc[order, keys[0]].to_numpy()
    for b in np.flatnonzero(prim[1:] != prim[:-1]) + 1:
        ax.axhline(b, color=viz.SURFACE, lw=0.5)
        ax.axvline(b, color=viz.SURFACE, lw=0.5)
    fig.savefig(f"figures/{fname}")
    plt.close(fig)


grid(["assay_family", "term_id"], "Ordered by assay family, then tissue",
     "Four blocks: assay family is the dominant structure", "a11_grid_family_first.png")
grid(["term_id", "assay_family"], "Same matrix, ordered by tissue, then assay family",
     "No comparable block structure at tissue level", "a11_grid_tissue_first.png")

# family x family block means
fams = df.assay_family.value_counts().index.tolist()
code = df.assay_family.map({f: i for i, f in enumerate(fams)}).to_numpy()
nf = len(fams)
B = np.zeros((nf, nf))
for a in range(nf):
    ia = code == a
    for b in range(nf):
        ib = code == b
        blk = S[np.ix_(ia, ib)]
        B[a, b] = ((blk.sum() - np.trace(blk)) / max(ia.sum() * (ia.sum() - 1), 1)
                   if a == b else blk.mean())

fig, ax = plt.subplots(figsize=(5.6, 4.8))
v = np.abs(B).max()
im = ax.imshow(B, cmap=viz.diverging_cmap(), vmin=-v, vmax=v)
ax.set_xticks(range(nf)); ax.set_yticks(range(nf))
ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=8.5)
ax.set_yticklabels(fams, fontsize=8.5)
ax.grid(False)
for a in range(nf):
    for b in range(nf):
        ax.text(b, a, f"{B[a, b]:.2f}", ha="center", va="center", fontsize=8.5,
                color=viz.SURFACE if abs(B[a, b]) > v * 0.55 else viz.TEXT_PRIMARY)
ax.set_title("Mean cosine between assay families", fontsize=11.5,
             color=viz.TEXT_PRIMARY, pad=12)
fig.savefig("figures/a11_family_blocks.png")
plt.close(fig)
out["family_blocks"] = {"families": fams, "matrix": B.tolist()}
del S

# ------------------------------------------------------------- lab confound, family level
print("\nlab/batch confound at FAMILY level")
d2 = df[df.lab.notna()].reset_index(drop=True)
U2 = l2_normalize(V[d2["index"].to_numpy()]).astype(np.float32)
S2 = (U2 @ U2.T).astype(np.float32)
iu = np.triu_indices(len(d2), k=1)
i, j = iu[0].astype(np.int32), iu[1].astype(np.int32)
famc = pd.factorize(d2.assay_family)[0]
tisc = pd.factorize(d2.term_id)[0]
labc = pd.factorize(d2.lab)[0]
xf = famc[i] != famc[j]
i, j = i[xf], j[xf]
sims = S2[i, j]
del S2
same_t = tisc[i] == tisc[j]
same_l = labc[i] == labc[j]

res = {
    "auroc_tissue": float(roc_auc_score(same_t, sims)),
    "auroc_lab": float(roc_auc_score(same_l, sims)),
    "pct_same_tissue_share_lab": float(same_l[same_t].mean()),
    "pct_diff_tissue_share_lab": float(same_l[~same_t].mean()),
    "n_cross_family_pairs": int(i.size),
}
xl = ~same_l
res["auroc_tissue_cross_lab_only"] = float(roc_auc_score(same_t[xl], sims[xl]))
res["n_cross_lab_same_tissue"] = int(same_t[xl].sum())
for k, v in res.items():
    print(f"  {k:34s} {v}")
out["lab_confound_family"] = res

# ------------------------------------------------------------- delta, family x lab
tis, tis_u = pd.factorize(d2.term_id)
fam_i, fam_u = pd.factorize(d2.assay_family)
modlab = pd.factorize(d2.assay_family.astype(str) + "||" + d2.lab.astype(str))[0]
auroc = cross_modality_auroc(U2, tis, fam_i)
print(f"\n  cross-family stratified AUROC: {auroc['stratified_auroc']:.4f}")
for label, grp in [("within family", None), ("within family x lab", modlab.astype(np.int64))]:
    r = run_delta_test(U2, tis.astype(np.int64), fam_i.astype(np.int64),
                       n_tissue=len(tis_u), n_mod=len(fam_u), n_perm=N_PERM, perm_group=grp)
    print(f"  permute {label:22s} delta {r['observed_delta']:+.5f} "
          f"null {r['null_mean']:+.6f} z {r['z']:+.1f} p {r['p_perm']:.5f}")
    out[f"delta_{label.replace(' ', '_')}"] = {
        k: r[k] for k in ("observed_delta", "null_mean", "null_sd", "z", "p_perm")
    }

Path("results").mkdir(exist_ok=True)
Path("results/a11_family_rerun.json").write_text(json.dumps(out, indent=2, default=float))
print("\nsaved results/a11_family_rerun.json and figures/a11_*.png")
