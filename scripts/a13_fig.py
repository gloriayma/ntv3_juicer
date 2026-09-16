"""Figure: is the cross-assay tissue signal organ-level or biosample-specific?"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ntv3_interp import viz  # noqa: E402

viz.apply_style()
D = json.loads(Path("results/a13_decompose.json").read_text())

order = [
    ("650M | all_biosamples", "All biosamples"),
    ("650M | all_biosamples | no blood bucket", "All, blood bucket dropped"),
    ("650M | tissues_only", "Primary tissues & cells"),
    ("650M | tissues_only | no blood bucket", "Tissues, blood dropped"),
]

fig, ax = plt.subplots(figsize=(9.2, 3.9))
x = np.arange(len(order))
w = 0.36
b1 = [D[k]["auroc_same_biosample_vs_diff_organ"] for k, _ in order]
b2 = [D[k]["auroc_same_organ_vs_diff_organ"] for k, _ in order]

ax.bar(x - w / 2, [v - 0.5 for v in b1], bottom=0.5, width=w, color=viz.SERIES[0],
       zorder=3, label="Same biosample  (e.g. liver × liver)")
ax.bar(x + w / 2, [v - 0.5 for v in b2], bottom=0.5, width=w, color=viz.SERIES[1],
       zorder=3, label="Same organ, different biosample\n(e.g. liver × right lobe of liver)")
ax.axhline(0.5, color=viz.TEXT_MUTED, lw=1.3, zorder=5)

for xi, (v1, v2) in enumerate(zip(b1, b2)):
    ax.text(xi - w / 2, v1 + (0.0015 if v1 >= 0.5 else -0.0045), f"{v1:.3f}",
            ha="center", fontsize=8.8, color=viz.TEXT_PRIMARY, fontweight="bold")
    ax.text(xi + w / 2, v2 + (0.0015 if v2 >= 0.5 else -0.0045), f"{v2:.3f}",
            ha="center", fontsize=8.8, color=viz.TEXT_PRIMARY, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([lbl for _, lbl in order], fontsize=8.6)
ax.set_ylabel("AUROC vs different-organ pairs")
ax.set_ylim(0.465, 0.542)
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right", fontsize=8.2)
ax.set_title(
    "No organ-level generalization: anatomically related samples are not more alike",
    fontsize=11.5, color=viz.TEXT_PRIMARY, pad=12,
)
ax.text(0.0, -0.26,
        "cross-assay pairs only  ·  0.500 = no signal  ·  orange at or below the line in every configuration",
        transform=ax.transAxes, fontsize=8, color=viz.TEXT_SECONDARY)
fig.savefig("figures/f_organ_decomposition.png")
plt.close(fig)
print("saved figures/f_organ_decomposition.png")
