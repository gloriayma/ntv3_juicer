"""A6: is the 0.311 tissue probe just decoding modality?

The A1 tissue probe was trained on all tracks pooled. If tissues have skewed assay
composition, a classifier could decode modality (which is near-trivial, 0.939) and read
tissue off it. Four checks, in increasing order of strictness:

  1. LABEL BASELINE. Predict tissue from the modality label alone -- nothing else. This is
     the exact quantity "how much of 0.311 is obtainable from modality". If this is near
     0.311 the original number is worthless; if near chance, there was no shortcut to take.
  2. WITHIN-MODALITY. Train and test tissue classifiers separately inside each modality, so
     modality is constant and cannot carry information.
  3. CROSS-MODALITY TRANSFER. Train on one assay family, test on a different one. This is
     the real test of a shared tissue code: a modality shortcut cannot survive it.
  4. MODALITY REMOVED. Subtract each modality centroid, re-probe.

Also breaks the kNN enrichment down by whether the neighbour differs in assay FAMILY, not
just in target -- ChIP:H3K4me3 vs ChIP:H3K27ac are different "modalities" under the fine
label but are still both ChIP-seq, which is a weaker notion of crossing modality.

Run: uv run python scripts/a6_probe_confound.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

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
df = df[df.alive & df.modality.notna() & df.term_id.notna()].reset_index(drop=True)
U = l2_normalize(V[df["index"].to_numpy()])

# Keep tissues with enough support, matching the A1 probe setup (>=20 tracks).
vc = df.term_id.value_counts()
df = df[df.term_id.isin(vc[vc >= 20].index)].reset_index(drop=True)
U = l2_normalize(V[df["index"].to_numpy()])
Z = PCA(n_components=N_PC, random_state=0).fit_transform(U)

y_tis = pd.factorize(df.term_id)[0]
y_mod = pd.factorize(df.modality)[0]
fam = df.assay_family.to_numpy()
n_cls = len(np.unique(y_tis))
chance = 1.0 / n_cls

print(f"{len(df)} tracks | {n_cls} tissues | {df.modality.nunique()} modalities "
      f"| {df.assay_family.nunique()} assay families | chance {chance:.4f}")


def cv_bal_acc(X, y, cv=5):
    clf = LogisticRegression(max_iter=3000, C=1.0)
    return cross_val_score(
        clf, StandardScaler().fit_transform(X), y,
        cv=StratifiedKFold(cv, shuffle=True, random_state=0),
        scoring="balanced_accuracy", n_jobs=-1,
    )


# ---------------------------------------------------------------- 1. label baseline
print("\n[1] How much tissue is obtainable from the MODALITY LABEL alone?")
Xmod = OneHotEncoder(sparse_output=False).fit_transform(y_mod.reshape(-1, 1))
acc_modlabel = cv_bal_acc(Xmod, y_tis)
acc_full = cv_bal_acc(Z, y_tis)
dummy = cross_val_score(
    DummyClassifier(strategy="most_frequent"), Z, y_tis,
    cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="balanced_accuracy",
)
print(f"    tissue from modality one-hot ONLY : {acc_modlabel.mean():.4f}")
print(f"    tissue from head vectors (50 PCs) : {acc_full.mean():.4f}")
print(f"    majority-class dummy              : {dummy.mean():.4f}   chance {chance:.4f}")
out["label_baseline"] = {
    "tissue_from_modality_label_only": float(acc_modlabel.mean()),
    "tissue_from_head_vectors": float(acc_full.mean()),
    "dummy": float(dummy.mean()),
    "chance": chance,
    "n_classes": int(n_cls),
}

# How entangled are the two labels at all?
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

nmi = normalized_mutual_info_score(y_mod, y_tis)
print(f"    normalized MI(modality; tissue)   : {nmi:.4f}")
out["nmi_modality_tissue"] = float(nmi)

# ---------------------------------------------------------------- 2. within-modality
print("\n[2] Tissue probes trained WITHIN each modality (modality held constant)")
rows = []
for m in df.modality.unique():
    sel = (df.modality == m).to_numpy()
    yy = df.term_id[sel]
    cnt = yy.value_counts()
    ok = yy.isin(cnt[cnt >= 5].index).to_numpy()
    if ok.sum() < 40 or yy[ok].nunique() < 3:
        continue
    y = pd.factorize(yy[ok])[0]
    acc = cv_bal_acc(Z[sel][ok], y, cv=min(5, int(np.bincount(y).min())))
    rows.append(
        {
            "modality": m,
            "n": int(ok.sum()),
            "n_tissues": int(len(np.unique(y))),
            "bal_acc": float(acc.mean()),
            "chance": 1.0 / len(np.unique(y)),
        }
    )
wm = pd.DataFrame(rows).sort_values("n", ascending=False)
wm["ratio_vs_chance"] = wm.bal_acc / wm.chance
print(wm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
out["within_modality"] = wm.to_dict("records")
out["within_modality_median_ratio"] = float(wm.ratio_vs_chance.median())
print(f"\n    median accuracy / chance across modalities: {wm.ratio_vs_chance.median():.1f}x")

# ---------------------------------------------------------------- 3. cross-family transfer
print("\n[3] CROSS-FAMILY TRANSFER: train tissue classifier on one assay family, test on another")


def modality_centered(Zmat: np.ndarray) -> np.ndarray:
    Zc = Zmat.copy()
    for m in np.unique(y_mod):
        s = y_mod == m
        Zc[s] -= Zc[s].mean(axis=0, keepdims=True)
    return Zc


Zc = modality_centered(Z)
fams = [f for f, c in df.assay_family.value_counts().items() if c >= 150]

for space_name, Zspace in [("as learned", Z), ("modality centroids removed", Zc)]:
    print(f"\n  -- feature space: {space_name} --")
    grid = []
    for ftr in fams:
        for fte in fams:
            if ftr == fte:
                continue
            tr = fam == ftr
            te = fam == fte
            shared = set(df.term_id[tr]) & set(df.term_id[te])
            # Need enough shared tissues with real support on both sides.
            shared = {
                t for t in shared
                if (df.term_id[tr] == t).sum() >= 3 and (df.term_id[te] == t).sum() >= 3
            }
            if len(shared) < 5:
                continue
            trm = tr & df.term_id.isin(shared).to_numpy()
            tem = te & df.term_id.isin(shared).to_numpy()
            classes = sorted(shared)
            cmap = {c: i for i, c in enumerate(classes)}
            ytr = df.term_id[trm].map(cmap).to_numpy()
            yte = df.term_id[tem].map(cmap).to_numpy()

            sc = StandardScaler().fit(Zspace[trm])
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Zspace[trm]), ytr)
            pred = clf.predict(sc.transform(Zspace[tem]))
            acc = balanced_accuracy_score(yte, pred)
            grid.append(
                {
                    "train": ftr,
                    "test": fte,
                    "n_train": int(trm.sum()),
                    "n_test": int(tem.sum()),
                    "n_tissues": len(classes),
                    "bal_acc": float(acc),
                    "chance": 1.0 / len(classes),
                    "ratio": float(acc * len(classes)),
                }
            )
    g = pd.DataFrame(grid)
    if g.empty:
        print("    no family pair with enough shared tissues")
        continue
    print(g.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(
        f"    mean bal_acc {g.bal_acc.mean():.4f} | mean chance {g.chance.mean():.4f} "
        f"| median ratio {g.ratio.median():.1f}x | pairs above chance: "
        f"{(g.bal_acc > g.chance).sum()}/{len(g)}"
    )
    key = "transfer_raw" if space_name == "as learned" else "transfer_modality_removed"
    out[key] = g.to_dict("records")
    out[key + "_summary"] = {
        "mean_bal_acc": float(g.bal_acc.mean()),
        "mean_chance": float(g.chance.mean()),
        "median_ratio": float(g.ratio.median()),
        "n_above_chance": int((g.bal_acc > g.chance).sum()),
        "n_pairs": int(len(g)),
    }

# ---------------------------------------------------------------- 4. kNN by family
print("\n[4] kNN tissue enrichment, split by how far the neighbour is from the query")
Ufull = U
S = Ufull @ Ufull.T
np.fill_diagonal(S, -np.inf)
K = 10
nn = np.argpartition(-S, K, axis=1)[:, :K]
del S

mod_a = df.modality.to_numpy()
fam_a = df.assay_family.to_numpy()
tis_a = df.term_id.to_numpy()

same_tis = tis_a[nn] == tis_a[:, None]
diff_mod = mod_a[nn] != mod_a[:, None]
diff_fam = fam_a[nn] != fam_a[:, None]

tv = pd.Series(tis_a).value_counts(normalize=True).to_numpy()
base = float((tv**2).sum())

buckets = {
    "same modality": ~diff_mod,
    "different modality, same assay family": diff_mod & ~diff_fam,
    "different assay family": diff_fam,
}
knn = {}
for name, mask in buckets.items():
    if mask.sum() == 0:
        continue
    p = float(same_tis[mask].mean())
    knn[name] = {"purity": p, "enrichment": p / base, "n_pairs": int(mask.sum())}
    print(f"    {name:42s} purity {p:.3f}  ({p / base:5.1f}x chance)  n={int(mask.sum()):,}")
print(f"    {'chance rate of sharing a tissue':42s} {base:.3f}")
out["knn_by_distance"] = knn
out["knn_chance"] = base

Path("results").mkdir(exist_ok=True)
Path("results/a6_probe_confound.json").write_text(json.dumps(out, indent=2, default=float))
print("\nsaved results/a6_probe_confound.json")
