"""Final figure set, all at assay-GROUP level."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ntv3_interp import viz  # noqa: E402
from ntv3_interp.grouping import add_assay_group  # noqa: E402
from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)

viz.apply_style()
Path("figures").mkdir(exist_ok=True)
F = json.loads(Path("results/final_merged_rna.json").read_text())
A3 = json.loads(Path("results/a3_summary.json").read_text())
A7 = json.loads(Path("results/a7_family_level.json").read_text())

K = "NTv3_650M_post::all_biosamples"

SHORT = {
    "ChIP-seq": "ChIP",
    "RNA-seq (total + polyA)": "RNA",
    "DNase-seq": "DNase",
    "CAGE": "CAGE",
}


def short(s: str) -> str:
    for k, v in SHORT.items():
        s = s.replace(k, v)
    return s

# ---------------------------------------------------------------- A. strictness ladder
levels = [
    ("Cross fine-modality\n(mostly ChIP target vs ChIP target)",
     A3[K]["observed_delta"], A3[K]["z"], A3[K]["stratified_auroc"]),
    ("Cross assay family\n(RNA-seq and polyA kept apart)",
     A7[K]["within family"]["observed_delta"], A7[K]["within family"]["z"],
     A7[K]["stratified_auroc"]),
    ("Cross assay group\n(RNA assays merged — strictest)",
     F["delta"][K]["within_group"]["observed_delta"], F["delta"][K]["within_group"]["z"],
     F["delta"][K]["stratified_auroc"]),
]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
y = np.arange(len(levels))[::-1]
d = [v[1] for v in levels]
z = [v[2] for v in levels]
au = [v[3] for v in levels]

axes[0].barh(y, d, color=viz.SERIES[0], height=0.55, zorder=3)
for yi, (dv, zv) in zip(y, zip(d, z)):
    axes[0].text(dv + 0.0016, yi, f"Δ {dv:+.4f}   z {zv:.0f}", va="center",
                 fontsize=8.8, color=viz.TEXT_PRIMARY, fontweight="bold")
axes[0].set_yticks(y)
axes[0].set_yticklabels([v[0] for v in levels], fontsize=8.2)
axes[0].set_xlabel("observed Δ (same-tissue − different-tissue cosine)")
axes[0].set_xlim(0, max(d) * 1.42)
axes[0].grid(axis="y", visible=False)
axes[0].set_title("Effect size shrinks as the comparison tightens", fontsize=10.5)

axes[1].barh(y, [a - 0.5 for a in au], left=0.5, color=viz.SERIES[0], height=0.55, zorder=3)
axes[1].axvline(0.5, color=viz.TEXT_MUTED, lw=1.2, zorder=5)
for yi, a in zip(y, au):
    axes[1].text(a + 0.003, yi, f"{a:.3f}", va="center", fontsize=8.8,
                 color=viz.TEXT_PRIMARY, fontweight="bold")
axes[1].set_yticks(y)
axes[1].set_yticklabels([])
axes[1].set_xlabel("cross-group AUROC")
axes[1].set_xlim(0.49, 0.65)
axes[1].grid(axis="y", visible=False)
axes[1].set_title("…but never reaches chance", fontsize=10.5)
fig.suptitle("The tissue effect survives every tightening of 'different modality'",
             y=1.05, fontsize=11.5, color=viz.TEXT_PRIMARY)
fig.tight_layout()
fig.savefig("figures/f_strictness_ladder.png")
plt.close(fig)

# ---------------------------------------------------------------- B. probes
p = F["probes"]
wg = pd.DataFrame(p["within_group"]).sort_values("n", ascending=False)
labels = ["Tissue from\nGROUP LABEL only", "Tissue from\nhead vectors"]
vals = [p["tissue_from_group_label_only"], p["tissue_from_vectors"]]
chs = [p["tissue_chance"], p["tissue_chance"]]
for _, r in wg.iterrows():
    labels.append(f"Within\n{short(r.group)}")
    vals.append(r.bal_acc)
    chs.append(r.chance)

fig, ax = plt.subplots(figsize=(8.4, 3.9))
x = np.arange(len(labels))
cols = [viz.TEXT_MUTED] + [viz.SERIES[0]] * (len(labels) - 1)
ax.bar(x, vals, color=cols, width=0.56, zorder=3)
for xi, c in enumerate(chs):
    ax.plot([xi - 0.32, xi + 0.32], [c, c], color=viz.SERIES[1], lw=1.8, zorder=6)
for xi, v in enumerate(vals):
    ax.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=9.5,
            color=viz.TEXT_PRIMARY, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.2)
ax.set_ylabel("balanced accuracy")
ax.set_ylim(0, 0.92)
ax.grid(axis="x", visible=False)
ax.set_title("There is no modality shortcut, and tissue survives holding assay constant",
             fontsize=11.5, color=viz.TEXT_PRIMARY, pad=12)
ax.text(0.0, -0.34, "orange rule = chance   ·   grey bar = ceiling available from the assay label alone",
        transform=ax.transAxes, fontsize=8, color=viz.TEXT_SECONDARY)
fig.savefig("figures/f_probes_group.png")
plt.close(fig)

# ---------------------------------------------------------------- C. per-pair AUROC
fig, axes = plt.subplots(1, 2, figsize=(10.6, 2.9), sharex=True)
for ax, sub, ttl in [
    (axes[0], "NTv3_650M_post::all_biosamples", "All biosamples"),
    (axes[1], "NTv3_650M_post::tissues_only", "Primary tissues & cells only"),
]:
    pp = pd.DataFrame(F["delta"][sub]["per_pair"]).sort_values("auroc")
    yy = np.arange(len(pp))
    cols = [viz.SERIES[0] if v > 0.5 else viz.SERIES[7] for v in pp.auroc]
    ax.barh(yy, pp.auroc - 0.5, left=0.5, color=cols, height=0.55, zorder=3)
    ax.axvline(0.5, color=viz.TEXT_MUTED, lw=1.2, zorder=5)
    ax.set_yticks(yy)
    ax.set_yticklabels([short(x) for x in pp.pair], fontsize=8.4)
    for yi, v in zip(yy, pp.auroc):
        off = 0.005 if v > 0.5 else -0.005
        ax.text(v + off, yi, f"{v:.3f}", va="center",
                ha="left" if v > 0.5 else "right", fontsize=8.5,
                color=viz.TEXT_PRIMARY, fontweight="bold")
    ax.set_title(ttl, fontsize=10.5)
    ax.set_xlabel("AUROC: cosine ranks same-tissue pairs first")
    ax.set_xlim(0.40, 0.70)
    ax.grid(axis="y", visible=False)
fig.suptitle("Uneven across pairs: ChIP-seq × DNase-seq carries no tissue signal",
             y=1.06, fontsize=11.5, color=viz.TEXT_PRIMARY)
fig.tight_layout()
fig.savefig("figures/f_pair_auroc_group.png")
plt.close(fig)

# ---------------------------------------------------------------- D. transfer control
tr = pd.DataFrame(F["transfer"]["rows"])
tr["lbl"] = [f"{short(a)}\n→ {short(b)}" for a, b in zip(tr.train, tr.test)]
fig, ax = plt.subplots(figsize=(9.4, 3.7))
x = np.arange(len(tr))
w = 0.38
ax.bar(x - w / 2, tr.heldout, width=w, color=viz.SERIES[0], zorder=3, label="Held out within training assay")
ax.bar(x + w / 2, tr.transfer, width=w, color=viz.SERIES[1], zorder=3, label="Transferred to the other assay")
for xi, c in zip(x, tr.chance):
    ax.plot([xi - 0.52, xi + 0.52], [c, c], color=viz.TEXT_MUTED, lw=1.2, zorder=6)
ax.set_xticks(x)
ax.set_xticklabels(tr.lbl, fontsize=8.6)
ax.set_ylabel("balanced accuracy")
ax.set_ylim(0, 0.9)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("The probe is competent within an assay and collapses across assays",
             fontsize=11.5, color=viz.TEXT_PRIMARY, pad=12)
ax.text(0.0, -0.30, "grey rule = chance for that tissue set", transform=ax.transAxes,
        fontsize=8, color=viz.TEXT_SECONDARY)
fig.savefig("figures/f_transfer_control.png")
plt.close(fig)

# ---------------------------------------------------------------- E. centroid matching
cm = pd.DataFrame(F["centroid_matching"])
fig, ax = plt.subplots(figsize=(8.6, 2.9))
y = np.arange(len(cm))[::-1]
ax.barh(y, cm.top1, color=viz.SERIES[0], height=0.5, zorder=3)
ax.barh(y, cm.chance_top1, color=viz.SERIES[7], height=0.5, zorder=4)
for yi, r in zip(y, cm.itertuples()):
    ax.text(r.top1 + 0.008, yi,
            f"{r.top1:.3f}  vs chance {r.chance_top1:.3f}   (top-5 {r.top5:.2f}, z {r.z:.0f})",
            va="center", fontsize=8.5, color=viz.TEXT_PRIMARY)
ax.set_yticks(y)
ax.set_yticklabels([short(x) for x in cm.pair], fontsize=8.4)
ax.set_xlabel("top-1 accuracy matching a tissue's centroid across assays")
ax.set_xlim(0, 0.85)
ax.grid(axis="y", visible=False)
ax.set_title("With nothing trained, a tissue's assay-A centroid finds its own assay-B centroid",
             fontsize=11.5, color=viz.TEXT_PRIMARY, pad=12)
fig.savefig("figures/f_centroid_matching.png")
plt.close(fig)

# ---------------------------------------------------------------- F. grids at group level
head = load_head("data/head_NTv3_650M_post.npz")
tracks = add_assay_group(pd.read_parquet("data/tracks.parquet"), merge_rna=True)
V = effective_vectors(head)
d = tracks.copy()
d["alive"] = dead_track_mask(head)
d = d[d.alive & d.assay_group.notna() & d.term_id.notna()]
ng = d.groupby("term_id").assay_group.nunique()
d = d[d.term_id.isin(ng[ng >= 2].index)].reset_index(drop=True)
U = l2_normalize(V[d["index"].to_numpy()])
S = U @ U.T

for keys, ttl, sub, fn in [
    (["assay_group", "term_id"], "Ordered by assay group, then tissue",
     "Three blocks: assay is the dominant structure", "f_grid_group_first.png"),
    (["term_id", "assay_group"], "Same matrix, ordered by tissue, then assay group",
     "Tissue produces fine texture, not blocks", "f_grid_tissue_first.png"),
]:
    order = d.sort_values(keys).index.to_numpy()
    M = S[np.ix_(order, order)]
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(M, cmap=viz.diverging_cmap(), vmin=-0.6, vmax=0.6, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(ttl, fontsize=11.5, color=viz.TEXT_PRIMARY, pad=14)
    ax.text(0.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=viz.TEXT_SECONDARY)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("cosine similarity", fontsize=8.5, color=viz.TEXT_SECONDARY)
    cb.outline.set_visible(False)
    prim = d.loc[order, keys[0]].to_numpy()
    for b in np.flatnonzero(prim[1:] != prim[:-1]) + 1:
        ax.axhline(b, color=viz.SURFACE, lw=0.5)
        ax.axvline(b, color=viz.SURFACE, lw=0.5)
    fig.savefig(f"figures/{fn}")
    plt.close(fig)

print("saved figures/f_*.png")
