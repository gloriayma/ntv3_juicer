"""A10: is the weak cross-family transfer real, or just a badly-fit classifier?

A9/A6 found that a tissue classifier trained on ChIP-seq barely beats chance on RNA-seq.
That has two possible causes and they must be separated:

  (a) there is no shared cross-family tissue code, or
  (b) the probe is simply undertrained / the task is hard at this sample size.

POSITIVE CONTROL. For every family pair, fit the SAME classifier on the SAME tissue classes
with the SAME feature space, and evaluate two ways: held-out within the training family, and
transferred to the other family. If within-family is strong while transfer is at chance, the
classifier is competent and the transfer failure is real. If both are weak, (b) is the
explanation and the transfer result says nothing.

CENTROID MATCHING. A near-parameter-free version that cannot be blamed on fitting: take each
tissue's mean vector in family A and in family B, and ask whether tissue t's A-centroid is
closest to its own B-centroid among all candidates. Reports top-1, top-5 and mean reciprocal
rank against a shuffled null. Nothing is trained at all.

Run: uv run python scripts/a10_transfer_control.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)

N_PC = 50
out: dict = {}

head = load_head("data/head_NTv3_650M_post.npz")
tracks = pd.read_parquet("data/tracks.parquet")
V = effective_vectors(head)
alive = dead_track_mask(head)

df = tracks.copy()
df["alive"] = alive
df = df[df.alive & df.assay_family.notna() & df.term_id.notna()].reset_index(drop=True)
U = l2_normalize(V[df["index"].to_numpy()])
Z = PCA(n_components=N_PC, random_state=0).fit_transform(U)

fam = df.assay_family.to_numpy()
fams = [f for f, c in df.assay_family.value_counts().items() if c >= 150]
print(f"{len(df)} tracks | families: {fams}")


def topk_acc(proba, ytrue, classes, k):
    order = np.argsort(-proba, axis=1)[:, :k]
    hits = [ytrue[i] in classes[order[i]] for i in range(len(ytrue))]
    return float(np.mean(hits))


# ------------------------------------------------------------------ positive control
print("\n[POSITIVE CONTROL] same classifier, same classes -- held-out vs transferred\n")
rows = []
for ftr in fams:
    for fte in fams:
        if ftr == fte:
            continue
        tr_all = fam == ftr
        te_all = fam == fte
        shared = {
            t for t in set(df.term_id[tr_all]) & set(df.term_id[te_all])
            if (df.term_id[tr_all] == t).sum() >= 4 and (df.term_id[te_all] == t).sum() >= 3
        }
        if len(shared) < 5:
            continue
        trm = tr_all & df.term_id.isin(shared).to_numpy()
        tem = te_all & df.term_id.isin(shared).to_numpy()
        classes = np.array(sorted(shared))
        cmap = {c: i for i, c in enumerate(classes)}
        ytr = df.term_id[trm].map(cmap).to_numpy()
        yte = df.term_id[tem].map(cmap).to_numpy()
        nC = len(classes)

        # Held-out within the TRAINING family: identical classes and features.
        skf = StratifiedKFold(n_splits=min(4, int(np.bincount(ytr).min())), shuffle=True,
                              random_state=0)
        heldout = []
        for tr_i, te_i in skf.split(Z[trm], ytr):
            sc = StandardScaler().fit(Z[trm][tr_i])
            clf = LogisticRegression(max_iter=3000).fit(sc.transform(Z[trm][tr_i]), ytr[tr_i])
            heldout.append(balanced_accuracy_score(ytr[te_i], clf.predict(sc.transform(Z[trm][te_i]))))
        ho = float(np.mean(heldout))

        # Transfer to the other family, trained on ALL of the training family.
        sc = StandardScaler().fit(Z[trm])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(Z[trm]), ytr)
        pr = clf.predict_proba(sc.transform(Z[tem]))
        tf = balanced_accuracy_score(yte, clf.predict(sc.transform(Z[tem])))
        t5 = topk_acc(pr, yte, np.array(sorted(set(ytr))), 5)

        rows.append({
            "train": ftr, "test": fte, "n_tissues": nC,
            "chance": 1.0 / nC,
            "heldout_same_family": ho,
            "transfer": float(tf),
            "transfer_top5": t5,
            "chance_top5": min(1.0, 5.0 / nC),
            "ho_ratio": ho * nC,
            "tf_ratio": float(tf) * nC,
        })

pc = pd.DataFrame(rows)
print(pc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(
    f"\n  mean held-out within family : {pc.heldout_same_family.mean():.4f} "
    f"({pc.ho_ratio.mean():.1f}x chance)"
)
print(
    f"  mean transferred across fam : {pc.transfer.mean():.4f} "
    f"({pc.tf_ratio.mean():.1f}x chance)"
)
print(
    f"  transfer retains {pc.transfer.mean() / pc.heldout_same_family.mean():.1%} "
    f"of within-family accuracy"
)
out["positive_control"] = pc.to_dict("records")
out["positive_control_summary"] = {
    "mean_heldout": float(pc.heldout_same_family.mean()),
    "mean_transfer": float(pc.transfer.mean()),
    "mean_heldout_ratio": float(pc.ho_ratio.mean()),
    "mean_transfer_ratio": float(pc.tf_ratio.mean()),
    "retention": float(pc.transfer.mean() / pc.heldout_same_family.mean()),
}

# ------------------------------------------------------------------ centroid matching
print("\n[CENTROID MATCHING] nothing is trained; does tissue t in family A point at "
      "tissue t in family B?\n")
rng = np.random.default_rng(0)
rows = []
for i, fa in enumerate(fams):
    for fb in fams:
        if fa >= fb:
            continue
        a_all = fam == fa
        b_all = fam == fb
        shared = sorted({
            t for t in set(df.term_id[a_all]) & set(df.term_id[b_all])
            if (df.term_id[a_all] == t).sum() >= 3 and (df.term_id[b_all] == t).sum() >= 3
        })
        if len(shared) < 6:
            continue
        CA = np.stack([U[a_all & (df.term_id == t).to_numpy()].mean(0) for t in shared])
        CB = np.stack([U[b_all & (df.term_id == t).to_numpy()].mean(0) for t in shared])
        # Centre each family's centroid cloud so the shared family offset cannot dominate.
        CA -= CA.mean(0, keepdims=True)
        CB -= CB.mean(0, keepdims=True)
        CA /= np.maximum(np.linalg.norm(CA, axis=1, keepdims=True), 1e-12)
        CB /= np.maximum(np.linalg.norm(CB, axis=1, keepdims=True), 1e-12)

        S = CA @ CB.T
        n = len(shared)
        ranks = (-S).argsort(axis=1).argsort(axis=1)[np.arange(n), np.arange(n)] + 1
        top1 = float((ranks == 1).mean())
        top5 = float((ranks <= 5).mean())
        mrr = float((1.0 / ranks).mean())

        null_mrr = []
        for _ in range(2000):
            p = rng.permutation(n)
            r = (-S).argsort(axis=1).argsort(axis=1)[np.arange(n), p] + 1
            null_mrr.append((1.0 / r).mean())
        null_mrr = np.array(null_mrr)
        z = (mrr - null_mrr.mean()) / null_mrr.std()
        p_emp = float(((null_mrr >= mrr).sum() + 1) / (len(null_mrr) + 1))

        rows.append({
            "pair": f"{fa} x {fb}", "n_tissues": n,
            "top1": top1, "chance_top1": 1.0 / n,
            "top5": top5, "mrr": mrr,
            "null_mrr": float(null_mrr.mean()), "z": float(z), "p": p_emp,
        })

cm = pd.DataFrame(rows)
print(cm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
out["centroid_matching"] = cm.to_dict("records")

Path("results").mkdir(exist_ok=True)
Path("results/a10_transfer_control.json").write_text(json.dumps(out, indent=2, default=float))
print("\nsaved results/a10_transfer_control.json")
