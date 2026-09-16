"""Figures for the confound audit: probe breakdown, kNN by neighbour distance, family pairs."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ntv3_interp import viz  # noqa: E402

viz.apply_style()
Path("figures").mkdir(exist_ok=True)

a6 = json.loads(Path("results/a6_probe_confound.json").read_text())
a7 = json.loads(Path("results/a7_family_level.json").read_text())

# ------------------------------------------------- 1. probe breakdown
lb = a6["label_baseline"]
wm = pd.DataFrame(a6["within_modality"]).sort_values("n", ascending=False)

labels, vals, chances = [], [], []
labels.append("Tissue from\nMODALITY LABEL only")
vals.append(lb["tissue_from_modality_label_only"])
chances.append(lb["chance"])
labels.append("Tissue from head vectors\n(all modalities pooled)")
vals.append(lb["tissue_from_head_vectors"])
chances.append(lb["chance"])
for _, r in wm.iterrows():
    labels.append(f"Within {r.modality}\n({r.n_tissues} tissues, n={r.n})")
    vals.append(r.bal_acc)
    chances.append(r.chance)

fig, ax = plt.subplots(figsize=(8.2, 3.9))
x = np.arange(len(labels))
colors = [viz.TEXT_MUTED] + [viz.SERIES[0]] * (len(labels) - 1)
ax.bar(x, vals, color=colors, width=0.56, zorder=3)
for xi, c in enumerate(chances):
    ax.plot([xi - 0.32, xi + 0.32], [c, c], color=viz.SERIES[1], lw=1.8, zorder=6)
for xi, v in enumerate(vals):
    ax.text(xi, v + 0.022, f"{v:.3f}", ha="center", fontsize=9.5,
            color=viz.TEXT_PRIMARY, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.8)
ax.set_ylabel("balanced accuracy")
ax.set_ylim(0, 1.0)
ax.grid(axis="x", visible=False)
ax.set_title(
    "The modality shortcut does not exist — and tissue survives holding modality constant",
    fontsize=11.5, color=viz.TEXT_PRIMARY, pad=12,
)
ax.text(
    0.0, -0.40, "orange rule = chance for that comparison",
    transform=ax.transAxes, fontsize=8, color=viz.TEXT_SECONDARY,
)
fig.savefig("figures/a8_probe_breakdown.png")
plt.close(fig)

# ------------------------------------------------- 2. kNN by neighbour distance
kn = a6["knn_by_distance"]
order = ["same modality", "different modality, same assay family", "different assay family"]
order = [o for o in order if o in kn]
pur = [kn[o]["purity"] for o in order]
enr = [kn[o]["enrichment"] for o in order]
ns = [kn[o]["n_pairs"] for o in order]
base = a6["knn_chance"]

fig, ax = plt.subplots(figsize=(7.6, 3.4))
y = np.arange(len(order))
ax.barh(y, pur, color=viz.SERIES[0], height=0.58, zorder=3)
ax.axvline(base, color=viz.SERIES[1], lw=1.8, zorder=6)
for yi, (p, e, n) in enumerate(zip(pur, enr, ns)):
    ax.text(p + 0.012, yi, f"{p:.3f}   {e:.1f}× chance   n={n:,}",
            va="center", fontsize=8.8, color=viz.TEXT_PRIMARY)
ax.set_yticks(y)
ax.set_yticklabels(
    ["Same modality", "Different modality,\nsame assay family", "Different assay family"],
    fontsize=8.8,
)
ax.set_xlabel("fraction of neighbours sharing the query's biosample")
ax.set_xlim(0, 1.12)
ax.grid(axis="y", visible=False)
ax.invert_yaxis()
ax.set_title(
    "Most of the cross-modality enrichment comes from comparing ChIP marks to each other",
    fontsize=11.5, color=viz.TEXT_PRIMARY, pad=12,
)
ax.text(base + 0.008, len(order) - 0.42, f"chance {base:.3f}",
        fontsize=8, color=viz.SERIES[1])
fig.savefig("figures/a8_knn_by_distance.png")
plt.close(fig)

# ------------------------------------------------- 3. per family-pair AUROC
fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), sharex=True)
for ax, sub, ttl in [
    (axes[0], "NTv3_650M_post::all_biosamples", "All biosamples"),
    (axes[1], "NTv3_650M_post::tissues_only", "Primary tissues & cells only"),
]:
    p = pd.DataFrame(a7[sub]["per_family_pair"]).sort_values("auroc")
    yy = np.arange(len(p))
    cols = [viz.SERIES[0] if v > 0.5 else viz.SERIES[7] for v in p.auroc]
    ax.barh(yy, p.auroc - 0.5, left=0.5, color=cols, height=0.6, zorder=3)
    ax.axvline(0.5, color=viz.TEXT_MUTED, lw=1.2, zorder=5)
    ax.set_yticks(yy)
    ax.set_yticklabels([s.replace(" x ", " × ") for s in p.pair], fontsize=8)
    for yi, (v, n) in enumerate(zip(p.auroc, p.n_pos)):
        off = 0.005 if v > 0.5 else -0.005
        ax.text(v + off, yi, f"{v:.3f}", va="center",
                ha="left" if v > 0.5 else "right",
                fontsize=8.5, color=viz.TEXT_PRIMARY, fontweight="bold")
    ax.set_title(ttl, fontsize=10.5)
    ax.set_xlabel("AUROC: cosine ranks same-tissue pairs first")
    ax.set_xlim(0.40, 0.75)
    ax.grid(axis="y", visible=False)
fig.suptitle(
    "Across genuinely different assay families the effect is real but uneven",
    y=1.04, fontsize=11.5, color=viz.TEXT_PRIMARY,
)
fig.tight_layout()
fig.savefig("figures/a8_family_pair_auroc.png")
plt.close(fig)

print("saved figures/a8_*.png")
