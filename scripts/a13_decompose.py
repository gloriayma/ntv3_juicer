"""A13: is there ANY organ-level generalization, or is the signal biosample-specific?

A12 produced a surprise: regrouping by organ LOWERED the cross-assay AUROC (0.556 -> 0.520),
the opposite of what "the fine ontology splits hairs and costs us" predicts. Merging must be
adding positives the model does not consider similar.

The direct test splits every cross-assay pair three ways:

  1. same biosample            e.g. liver ChIP  x  liver DNase
  2. same organ, different biosample   e.g. liver ChIP  x  right-lobe-of-liver DNase
  3. different organ           e.g. liver ChIP  x  lung DNase

If bucket 2 sits near bucket 1, the head has learned organ-level biology.
If bucket 2 sits near bucket 3, the head has learned something closer to sample identity,
with no generalization across donors of the same organ -- a much weaker claim, and one that
matters for the inductive-bias argument.

Run: uv run python scripts/a13_decompose.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from ntv3_interp.grouping import add_assay_group  # noqa: E402
from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)

BLOOD = "blood|bodily fluid"
out: dict = {}
tracks = add_assay_group(pd.read_parquet("data/tracks.parquet"), merge_rna=True)
head = load_head("data/head_NTv3_650M_post.npz")
V = effective_vectors(head)

for subset in ("all_biosamples", "tissues_only"):
    for drop_blood in (False, True):
        d = tracks.copy()
        d["alive"] = dead_track_mask(head)
        d = d[d.alive & d.assay_group.notna() & d.term_id.notna()
              & d.organ_slims.notna() & d.lab.notna()]
        if subset == "tissues_only":
            d = d[d.classification.isin(["tissue", "primary cell"])]
        if drop_blood:
            d = d[d.organ_slims != BLOOD]
        d = d.reset_index(drop=True)

        U = l2_normalize(V[d["index"].to_numpy()]).astype(np.float32)
        S = (U @ U.T).astype(np.float32)
        iu = np.triu_indices(len(d), k=1)
        i, j = iu[0].astype(np.int32), iu[1].astype(np.int32)

        ai = pd.factorize(d.assay_group)[0]
        ti = pd.factorize(d.term_id)[0]
        oi = pd.factorize(d.organ_slims)[0]
        li = pd.factorize(d.lab)[0]

        xa = ai[i] != ai[j]          # cross-assay only
        i, j = i[xa], j[xa]
        sims = S[i, j]
        del S

        same_t = ti[i] == ti[j]
        same_o = oi[i] == oi[j]
        same_l = li[i] == li[j]

        b1 = same_t                       # same biosample
        b2 = same_o & ~same_t             # same organ, different biosample
        b3 = ~same_o                      # different organ

        name = f"650M | {subset}{' | no blood bucket' if drop_blood else ''}"
        rec = {"n_cross_assay_pairs": int(len(sims))}
        print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
        for lbl, m in [("1. same biosample", b1),
                       ("2. same organ, diff biosample", b2),
                       ("3. different organ", b3)]:
            rec[lbl] = {"mean_cos": float(sims[m].mean()), "n": int(m.sum()),
                        "pct_same_lab": float(same_l[m].mean())}
            print(f"  {lbl:<32s} mean cos {sims[m].mean():+.4f}   "
                  f"n={int(m.sum()):>9,}   same-lab {same_l[m].mean():5.1%}")

        # How well does cosine separate each bucket from "different organ"?
        a13 = roc_auc_score(np.r_[np.ones(b1.sum()), np.zeros(b3.sum())],
                            np.r_[sims[b1], sims[b3]])
        a23 = roc_auc_score(np.r_[np.ones(b2.sum()), np.zeros(b3.sum())],
                            np.r_[sims[b2], sims[b3]])
        a12 = roc_auc_score(np.r_[np.ones(b1.sum()), np.zeros(b2.sum())],
                            np.r_[sims[b1], sims[b2]])
        rec["auroc_same_biosample_vs_diff_organ"] = float(a13)
        rec["auroc_same_organ_vs_diff_organ"] = float(a23)
        rec["auroc_same_biosample_vs_same_organ"] = float(a12)
        print(f"\n  AUROC  same biosample   vs different organ : {a13:.4f}")
        print(f"  AUROC  same organ (diff biosample) vs diff organ : {a23:.4f}"
              f"   <- is there organ-level generalization?")
        print(f"  AUROC  same biosample   vs same organ      : {a12:.4f}")

        # Same-biosample bucket split by lab, to see how much is batch.
        if (b1 & ~same_l).sum() > 200:
            rec["same_biosample_cross_lab_mean_cos"] = float(sims[b1 & ~same_l].mean())
            rec["same_biosample_same_lab_mean_cos"] = float(sims[b1 & same_l].mean())
            a13x = roc_auc_score(
                np.r_[np.ones((b1 & ~same_l).sum()), np.zeros((b3 & ~same_l).sum())],
                np.r_[sims[b1 & ~same_l], sims[b3 & ~same_l]])
            rec["auroc_same_biosample_vs_diff_organ_cross_lab"] = float(a13x)
            print(f"\n  same biosample, cross-lab only  mean cos "
                  f"{sims[b1 & ~same_l].mean():+.4f}  (n={int((b1 & ~same_l).sum()):,})")
            print(f"  AUROC same biosample vs diff organ, cross-lab only : {a13x:.4f}")

        out[name] = rec

Path("results").mkdir(exist_ok=True)
Path("results/a13_decompose.json").write_text(json.dumps(out, indent=2, default=float))
print("\nsaved results/a13_decompose.json")
