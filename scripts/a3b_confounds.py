"""A3b: is the tissue effect actually a batch (lab) effect?

ENCODE experiments from one lab share protocols, reagents and often donors. If same-tissue
tracks tend to come from the same lab, the A3 result could be batch structure wearing a
tissue label. Two checks, both at the pair level using the cross-modality AUROC framing:

  1. Substitute lab for tissue. If cosine ranks same-LAB pairs as strongly as same-tissue
     pairs, the tissue claim is in trouble.
  2. Restrict to CROSS-LAB pairs and re-ask the tissue question. Surviving this is the
     strong result: same tissue, different modality, different lab.

Run: uv run python scripts/a3b_confounds.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)

head = load_head("data/head_NTv3_650M_post.npz")
tracks = pd.read_parquet("data/tracks.parquet")
V = effective_vectors(head)
alive = dead_track_mask(head)

df = tracks.copy()
df["alive"] = alive
df = df[df.alive & df.modality.notna() & df.term_id.notna() & df.lab.notna()]
vc = df.modality.value_counts()
df = df[df.modality.isin(vc[vc >= 10].index)]
npm = df.groupby("term_id").modality.nunique()
df = df[df.term_id.isin(npm[npm >= 2].index)].reset_index(drop=True)

U = l2_normalize(V[df["index"].to_numpy()]).astype(np.float32)
print(
    f"{len(df)} tracks with lab metadata | "
    f"{df.term_id.nunique()} tissues | {df.lab.nunique()} labs"
)

S = (U @ U.T).astype(np.float32)
iu = np.triu_indices(len(df), k=1)
i, j = iu[0].astype(np.int32), iu[1].astype(np.int32)

mod = pd.factorize(df.modality)[0]
tis = pd.factorize(df.term_id)[0]
lab = pd.factorize(df.lab)[0]

cross_mod = mod[i] != mod[j]
i, j = i[cross_mod], j[cross_mod]
sims = S[i, j]
del S

same_tissue = tis[i] == tis[j]
same_lab = lab[i] == lab[j]

out = {}

# How entangled are tissue and lab to begin with?
out["pct_same_tissue_pairs_same_lab"] = float(same_lab[same_tissue].mean())
out["pct_diff_tissue_pairs_same_lab"] = float(same_lab[~same_tissue].mean())
print(
    f"\nsame-tissue pairs that share a lab : {out['pct_same_tissue_pairs_same_lab']:.1%}"
    f"\ndiff-tissue pairs that share a lab : {out['pct_diff_tissue_pairs_same_lab']:.1%}"
)

out["auroc_tissue_all_pairs"] = float(roc_auc_score(same_tissue, sims))
out["auroc_lab_all_pairs"] = float(roc_auc_score(same_lab, sims))
print(f"\nAUROC, cosine predicts same TISSUE : {out['auroc_tissue_all_pairs']:.4f}")
print(f"AUROC, cosine predicts same LAB    : {out['auroc_lab_all_pairs']:.4f}")

# The decisive test: cross-lab pairs only.
xl = ~same_lab
out["n_cross_lab_pairs"] = int(xl.sum())
out["n_cross_lab_same_tissue"] = int(same_tissue[xl].sum())
if same_tissue[xl].sum() >= 20:
    out["auroc_tissue_cross_lab_only"] = float(
        roc_auc_score(same_tissue[xl], sims[xl])
    )
    print(
        f"\nAUROC, same TISSUE among CROSS-LAB pairs only : "
        f"{out['auroc_tissue_cross_lab_only']:.4f}"
        f"\n   ({out['n_cross_lab_same_tissue']:,} same-tissue of "
        f"{out['n_cross_lab_pairs']:,} cross-lab cross-modality pairs)"
    )

# And the converse: same-lab pairs that are DIFFERENT tissue -- does cosine still rank them
# up? If lab alone did the work, this would look like the tissue effect.
sl = same_lab
if (~same_tissue[sl]).sum() >= 20 and same_tissue[sl].sum() >= 20:
    out["auroc_tissue_same_lab_only"] = float(roc_auc_score(same_tissue[sl], sims[sl]))
    print(
        f"AUROC, same TISSUE among SAME-LAB pairs only  : "
        f"{out['auroc_tissue_same_lab_only']:.4f}"
    )

Path("results").mkdir(exist_ok=True)
Path("results/a3b_confounds.json").write_text(json.dumps(out, indent=2))
print("\nsaved results/a3b_confounds.json")
