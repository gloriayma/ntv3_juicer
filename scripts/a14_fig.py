"""Figure: pairwise cosine is underpowered; centroid matching finds the organ code."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntv3_interp import viz  # noqa: E402

viz.apply_style()
A13 = json.loads(Path("results/a13_decompose.json").read_text())
A14 = json.loads(Path("results/a14_organ_centroid.json").read_text())

SHORT = {"ChIP-seq": "ChIP", "RNA-seq (total + polyA)": "RNA", "DNase-seq": "DNase"}
def sh(s):
    for k, v in SHORT.items():
        s = s.replace(k, v)
    return s.replace("->", "\u2192")

fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0),
                         gridspec_kw={"width_ratios": [1, 1.25]})

# ---- left: the underpowered instrument
keys = [("650M | tissues_only", "Tissues"),
        ("650M | tissues_only | no blood bucket", "Tissues,\nno blood")]
x = np.arange(len(keys))
w = 0.36
b1 = [A13[k]["auroc_same_biosample_vs_diff_organ"] for k, _ in keys]
b2 = [A13[k]["auroc_same_organ_vs_diff_organ"] for k, _ in keys]
ax = axes[0]
ax.bar(x - w / 2, [v - 0.5 for v in b1], bottom=0.5, width=w, color=viz.SERIES[0],
       zorder=3, label="Same biosample")
ax.bar(x + w / 2, [v - 0.5 for v in b2], bottom=0.5, width=w, color=viz.SERIES[1],
       zorder=3, label="Same organ,\ndifferent biosample")
ax.axhline(0.5, color=viz.TEXT_MUTED, lw=1.3, zorder=5)
for xi, (v1, v2) in enumerate(zip(b1, b2)):
    ax.text(xi - w / 2, v1 + (0.0012 if v1 >= 0.5 else -0.004), f"{v1:.3f}", ha="center",
            fontsize=8.6, color=viz.TEXT_PRIMARY, fontweight="bold")
    ax.text(xi + w / 2, v2 + (0.0012 if v2 >= 0.5 else -0.004), f"{v2:.3f}", ha="center",
            fontsize=8.6, color=viz.TEXT_PRIMARY, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([v for _, v in keys], fontsize=8.6)
ax.set_ylabel("AUROC vs different-organ pairs")
ax.set_ylim(0.462, 0.532)
ax.grid(axis="x", visible=False)
ax.legend(loc="lower left", fontsize=7.8)
ax.set_title("Per-track pairwise cosine:\nshows almost nothing", fontsize=10.5)

# ---- right: the powerful instrument
key = "650M | tissues_only | no blood"
rec = A14[key]
dirs = list(rec.keys())
ax = axes[1]
y = np.arange(len(dirs))[::-1]
top1 = [rec[k]["top1"] for k in dirs]
ch = [rec[k]["chance"] for k in dirs]
zs = [rec[k]["z"] for k in dirs]
ax.barh(y, top1, color=[viz.SERIES[0] if z > 3 else viz.RECESSIVE for z in zs],
        height=0.52, zorder=3)
ax.barh(y, ch, color=viz.SERIES[7], height=0.52, zorder=4)
for yi, (t, c, z) in zip(y, zip(top1, ch, zs)):
    ax.text(t + 0.012, yi, f"{t:.3f}   vs chance {c:.3f}   z {z:+.1f}", va="center",
            fontsize=8.6, color=viz.TEXT_PRIMARY)
ax.set_yticks(y)
ax.set_yticklabels([sh(k) for k in dirs], fontsize=8.6)
ax.set_xlabel("top-1 organ match, querying biosample excluded")
ax.set_xlim(0, 0.78)
ax.grid(axis="y", visible=False)
ax.set_title("Centroid matching, biosample held out:\norgan code is clearly present",
             fontsize=10.5)

fig.suptitle(
    "Same question, two instruments — the pairwise null was a power problem",
    y=1.04, fontsize=11.5, color=viz.TEXT_PRIMARY,
)
fig.tight_layout()
fig.savefig("figures/f_organ_code.png")
plt.close(fig)
print("saved figures/f_organ_code.png")
