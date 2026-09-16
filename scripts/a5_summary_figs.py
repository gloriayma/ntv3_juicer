"""Headline figures: the null vs observed, the per-modality-pair breakdown, and the probes."""

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

TAG = "NTv3_650M_post"
SUB = "all_biosamples"

a3 = json.loads(Path("results/a3_summary.json").read_text())
a3c = json.loads(Path("results/a3c_decisive.json").read_text())
a1 = json.loads(Path("results/a1_geometry.json").read_text())
null = np.load(f"results/a3_null_{TAG}_{SUB}.npy")
obs = a3[f"{TAG}::{SUB}"]["observed_delta"]
lab_null_mean = a3c[f"{TAG}::permute within modality x lab"]["null_mean"]

# ---------------------------------------------------------------- null vs observed
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.hist(null, bins=40, color=viz.RECESSIVE, zorder=2)
ax.axvline(lab_null_mean, color=viz.SERIES[1], lw=2, zorder=4)
ax.axvline(obs, color=viz.SERIES[0], lw=2.5, zorder=5)
ax.set_xlabel("mean Δ  (same-tissue − different-tissue cosine, modality pair held fixed)")
ax.set_ylabel("permutations")
ax.set_xlim(-0.016, obs * 1.15)
ymax = ax.get_ylim()[1]
ax.annotate(
    f"observed\n{obs:+.4f}",
    xy=(obs, ymax * 0.80),
    xytext=(-10, 0),
    textcoords="offset points",
    ha="right",
    fontsize=9.5,
    color=viz.TEXT_PRIMARY,
    fontweight="bold",
)
ax.annotate(
    f"null holding lab fixed\nmean {lab_null_mean:+.4f}",
    xy=(lab_null_mean, ymax * 0.50),
    xytext=(14, 0),
    textcoords="offset points",
    ha="left",
    fontsize=8.5,
    color=viz.TEXT_SECONDARY,
)
ax.annotate(
    "null: tissue shuffled\nwithin modality",
    xy=(0.0002, ymax * 0.68),
    xytext=(-14, 0),
    textcoords="offset points",
    ha="right",
    fontsize=8.5,
    color=viz.TEXT_SECONDARY,
)
ax.set_title(
    "Same-tissue tracks are more aligned than chance, and batch does not explain it",
    fontsize=11.5,
    color=viz.TEXT_PRIMARY,
    pad=12,
)
fig.savefig("figures/a5_null_vs_observed.png")
plt.close(fig)

# ---------------------------------------------------------------- per modality pair
per = pd.read_csv(f"results/a3_strata_{TAG}_{SUB}.csv")
per["pair"] = [" × ".join(sorted([a, b])) for a, b in zip(per.m1, per.m2)]
bp = (
    per.groupby("pair")
    .agg(mean_delta=("delta", "mean"), n=("delta", "size"))
    .sort_values("n", ascending=False)
    .head(18)
    .sort_values("mean_delta")
)
fig, ax = plt.subplots(figsize=(7.6, 5.4))
colors = [viz.SERIES[0] if v > 0 else viz.SERIES[7] for v in bp.mean_delta]
ax.barh(range(len(bp)), bp.mean_delta, color=colors, height=0.68, zorder=3)
ax.set_yticks(range(len(bp)))
ax.set_yticklabels(bp.index, fontsize=7.5)
ax.axvline(0, color=viz.TEXT_MUTED, lw=1)
ax.set_xlabel("mean Δ within this modality pair")
ax.set_title(
    "The effect holds across genuinely different assay pairs",
    fontsize=11.5,
    color=viz.TEXT_PRIMARY,
    pad=12,
)
ax.grid(axis="y", visible=False)
fig.savefig("figures/a5_delta_by_modality_pair.png")
plt.close(fig)

# ---------------------------------------------------------------- probes
labels = [
    "Modality\n20 classes",
    "Assay family\n5 classes",
    "Tissue\n71 classes",
    "Modality from\namplitude only",
]
keys = ["probe_modality", "probe_assay_family", "probe_tissue", "probe_modality_amplitude"]
acc = [a1[k]["balanced_acc"] for k in keys]
chance = [a1[k]["chance"] for k in keys]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(len(labels))
ax.bar(x, acc, color=viz.SERIES[0], width=0.55, zorder=3)
# Chance shown as a short rule spanning the bar, in recessive ink above the fill.
for xi, c in enumerate(chance):
    ax.plot([xi - 0.32, xi + 0.32], [c, c], color=viz.TEXT_SECONDARY, lw=1.4, zorder=6)
for xi, a in enumerate(acc):
    ax.text(
        xi, a + 0.022, f"{a:.3f}", ha="center", fontsize=9.5,
        color=viz.TEXT_PRIMARY, fontweight="bold",
    )
ax.set_xticks(x)
ax.set_xticklabels(
    [f"{lab}\nchance {c:.3f}" for lab, c in zip(labels, chance)], fontsize=8.5
)
ax.set_ylabel("balanced accuracy (5-fold CV)")
ax.set_ylim(0, 1.08)
ax.grid(axis="x", visible=False)
ax.set_title(
    "Modality is almost perfectly decodable; tissue is decodable but far weaker",
    fontsize=11.5,
    color=viz.TEXT_PRIMARY,
    pad=12,
)
fig.savefig("figures/a5_probes.png")
plt.close(fig)

print("saved figures/a5_*.png")
