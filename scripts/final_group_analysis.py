"""Everything, at assay-GROUP level, from one place.

Reruns the whole analysis with cross-group (genuinely different assay types) in place of
cross-modality (which was overwhelmingly ChIP-target vs ChIP-target).

    uv run python scripts/final_group_analysis.py            # RNA assays merged (default)
    uv run python scripts/final_group_analysis.py --split    # RNA assays kept apart
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from ntv3_interp.grouping import add_assay_group, label  # noqa: E402
from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)
from ntv3_interp.tissue_test import cross_modality_auroc, run_delta_test  # noqa: E402

MERGE = "--split" not in sys.argv
N_PERM = 200
N_PC = 50
TAGL = label(MERGE)
out: dict = {"grouping": TAGL}

tracks_raw = pd.read_parquet("data/tracks.parquet")
tracks = add_assay_group(tracks_raw, merge_rna=MERGE)

print(f"{'=' * 76}\nASSAY GROUPING: {TAGL}\n{'=' * 76}")
grp = (
    tracks[tracks.assay_group.notna()]
    .groupby("assay_group")
    .agg(tracks=("track_id", "size"), modalities=("modality", "nunique"),
         biosamples=("term_id", "nunique"))
    .sort_values("tracks", ascending=False)
)
print(grp.to_string())
out["groups"] = grp.reset_index().to_dict("records")


def prep(head, subset, need_lab=False):
    V = effective_vectors(head)
    d = tracks.copy()
    d["alive"] = dead_track_mask(head)
    d = d[d.alive & d.assay_group.notna() & d.term_id.notna()]
    if need_lab:
        d = d[d.lab.notna()]
    if subset == "tissues_only":
        d = d[d.classification.isin(["tissue", "primary cell"])]
    ng = d.groupby("term_id").assay_group.nunique()
    d = d[d.term_id.isin(ng[ng >= 2].index)].reset_index(drop=True)
    return d, l2_normalize(V[d["index"].to_numpy()])


# ============================================================ 1. probes
print(f"\n{'=' * 76}\n[1] PROBES (650M, all biosamples)\n{'=' * 76}")
head650 = load_head("data/head_NTv3_650M_post.npz")
d, U = prep(head650, "all_biosamples")
vc = d.term_id.value_counts()
dp = d[d.term_id.isin(vc[vc >= 20].index)].reset_index(drop=True)
Up = l2_normalize(effective_vectors(head650)[dp["index"].to_numpy()])
Z = PCA(n_components=N_PC, random_state=0).fit_transform(Up)
y_tis = pd.factorize(dp.term_id)[0]
y_grp = pd.factorize(dp.assay_group)[0]
nC = len(np.unique(y_tis))


def cv(X, y, k=5):
    return cross_val_score(
        LogisticRegression(max_iter=3000), StandardScaler().fit_transform(X), y,
        cv=StratifiedKFold(k, shuffle=True, random_state=0),
        scoring="balanced_accuracy", n_jobs=-1,
    ).mean()


probes = {
    "assay_group_from_vectors": float(cv(Z, y_grp)),
    "assay_group_chance": 1.0 / len(np.unique(y_grp)),
    "tissue_from_vectors": float(cv(Z, y_tis)),
    "tissue_from_group_label_only": float(
        cv(OneHotEncoder(sparse_output=False).fit_transform(y_grp.reshape(-1, 1)), y_tis)
    ),
    "tissue_chance": 1.0 / nC,
    "n_tissues": int(nC),
    "n_tracks": int(len(dp)),
}
print(f"  assay group from vectors     {probes['assay_group_from_vectors']:.4f} "
      f"(chance {probes['assay_group_chance']:.4f})")
print(f"  tissue  from vectors         {probes['tissue_from_vectors']:.4f} "
      f"(chance {probes['tissue_chance']:.4f}, {nC} classes)")
print(f"  tissue  from GROUP LABEL only {probes['tissue_from_group_label_only']:.4f}"
      f"   <- the confound ceiling")

wrows = []
for g in dp.assay_group.unique():
    sel = (dp.assay_group == g).to_numpy()
    yy = dp.term_id[sel]
    cnt = yy.value_counts()
    ok = yy.isin(cnt[cnt >= 5].index).to_numpy()
    if ok.sum() < 40 or yy[ok].nunique() < 3:
        continue
    y = pd.factorize(yy[ok])[0]
    a = cv(Z[sel][ok], y, k=min(5, int(np.bincount(y).min())))
    wrows.append({"group": g, "n": int(ok.sum()), "n_tissues": int(len(np.unique(y))),
                  "bal_acc": float(a), "chance": 1.0 / len(np.unique(y)),
                  "ratio": float(a * len(np.unique(y)))})
wm = pd.DataFrame(wrows).sort_values("n", ascending=False)
print("\n  tissue probes WITHIN each group (group held constant):")
print(wm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
probes["within_group"] = wm.to_dict("records")
out["probes"] = probes

# ============================================================ 2. kNN by distance
print(f"\n{'=' * 76}\n[2] kNN TISSUE PURITY BY NEIGHBOUR DISTANCE\n{'=' * 76}")
S = U @ U.T
np.fill_diagonal(S, -np.inf)
nn = np.argpartition(-S, 10, axis=1)[:, :10]
del S
tis_a, grp_a, mod_a = d.term_id.to_numpy(), d.assay_group.to_numpy(), d.modality.to_numpy()
same_t = tis_a[nn] == tis_a[:, None]
diff_g = grp_a[nn] != grp_a[:, None]
diff_m = mod_a[nn] != mod_a[:, None]
tv = pd.Series(tis_a).value_counts(normalize=True).to_numpy()
base = float((tv**2).sum())
knn = {}
for name, mask in {
    "same modality": ~diff_m,
    "different modality, same assay group": diff_m & ~diff_g,
    "different assay group": diff_g,
}.items():
    if mask.sum() == 0:
        continue
    p = float(same_t[mask].mean())
    knn[name] = {"purity": p, "enrichment": p / base, "n_pairs": int(mask.sum())}
    print(f"  {name:40s} {p:.3f}  ({p / base:5.1f}x)  n={int(mask.sum()):,}")
print(f"  {'chance':40s} {base:.3f}")
out["knn"] = {"buckets": knn, "chance": base}

# ============================================================ 3. delta + AUROC
print(f"\n{'=' * 76}\n[3] CROSS-GROUP DELTA TEST\n{'=' * 76}")
delta = {}
for hp in sorted(Path("data").glob("head_*.npz")):
    h = load_head(hp)
    tg = h.repo.split("/")[-1]
    for subset in ("all_biosamples", "tissues_only"):
        dd, UU = prep(h, subset, need_lab=True)
        UU = UU.astype(np.float32)
        tis, tis_u = pd.factorize(dd.term_id)
        gi, g_u = pd.factorize(dd.assay_group)
        glab = pd.factorize(dd.assay_group.astype(str) + "||" + dd.lab.astype(str))[0]
        au = cross_modality_auroc(UU, tis, gi)
        pp = au["per_stratum"].copy()
        pp["pair"] = [" × ".join(sorted([g_u[k // (gi.max() + 1)], g_u[k % (gi.max() + 1)]]))
                      for k in pp.modality_pair_key]
        print(f"\n  {tg} | {subset} | {len(dd)} tracks, {len(tis_u)} tissues")
        print(f"    cross-group AUROC (stratified) {au['stratified_auroc']:.4f}")
        print(pp[["pair", "auroc", "n_pos", "n_neg"]].to_string(index=False))
        rec = {"n_tracks": int(len(dd)), "n_tissues": int(len(tis_u)),
               "stratified_auroc": au["stratified_auroc"],
               "pooled_auroc": au["pooled_auroc"],
               "per_pair": pp[["pair", "auroc", "n_pos", "n_neg"]].to_dict("records")}
        for lb, gp in [("within_group", None), ("within_group_x_lab", glab.astype(np.int64))]:
            r = run_delta_test(UU, tis.astype(np.int64), gi.astype(np.int64),
                               n_tissue=len(tis_u), n_mod=len(g_u),
                               n_perm=N_PERM, perm_group=gp)
            if "error" in r:
                continue
            print(f"    permute {lb:20s} delta {r['observed_delta']:+.5f}  "
                  f"null {r['null_mean']:+.6f}  z {r['z']:+.1f}  p {r['p_perm']:.5f}")
            rec[lb] = {k: r[k] for k in
                       ("observed_delta", "null_mean", "null_sd", "z", "p_perm")}
            rec[lb]["frac_positive"] = float(np.nanmean(r["per_stratum"] > 0))
        delta[f"{tg}::{subset}"] = rec
out["delta"] = delta

# ============================================================ 4. lab confound
print(f"\n{'=' * 76}\n[4] LAB / BATCH CONFOUND (cross-group pairs only)\n{'=' * 76}")
dd, UU = prep(head650, "all_biosamples", need_lab=True)
UU = UU.astype(np.float32)
S2 = (UU @ UU.T).astype(np.float32)
iu = np.triu_indices(len(dd), k=1)
i, j = iu[0].astype(np.int32), iu[1].astype(np.int32)
gi = pd.factorize(dd.assay_group)[0]
ti = pd.factorize(dd.term_id)[0]
li = pd.factorize(dd.lab)[0]
xg = gi[i] != gi[j]
i, j = i[xg], j[xg]
sims = S2[i, j]
del S2
st, sl = ti[i] == ti[j], li[i] == li[j]
lab = {
    "auroc_tissue": float(roc_auc_score(st, sims)),
    "auroc_lab": float(roc_auc_score(sl, sims)),
    "auroc_tissue_cross_lab_only": float(roc_auc_score(st[~sl], sims[~sl])),
    "pct_same_tissue_share_lab": float(sl[st].mean()),
    "pct_diff_tissue_share_lab": float(sl[~st].mean()),
    "n_pairs": int(i.size),
    "n_cross_lab_same_tissue": int(st[~sl].sum()),
}
for k, v in lab.items():
    print(f"  {k:34s} {v}")
out["lab_confound"] = lab

# ============================================================ 5. transfer + centroids
print(f"\n{'=' * 76}\n[5] TRANSFER POSITIVE CONTROL & CENTROID MATCHING\n{'=' * 76}")
dA = tracks.copy()
dA["alive"] = dead_track_mask(head650)
dA = dA[dA.alive & dA.assay_group.notna() & dA.term_id.notna()].reset_index(drop=True)
UA = l2_normalize(effective_vectors(head650)[dA["index"].to_numpy()])
ZA = PCA(n_components=N_PC, random_state=0).fit_transform(UA)
gA = dA.assay_group.to_numpy()
groups = [g for g, c in dA.assay_group.value_counts().items() if c >= 150]

rows = []
for a in groups:
    for b in groups:
        if a == b:
            continue
        ta, tb = gA == a, gA == b
        shared = {t for t in set(dA.term_id[ta]) & set(dA.term_id[tb])
                  if (dA.term_id[ta] == t).sum() >= 4 and (dA.term_id[tb] == t).sum() >= 3}
        if len(shared) < 5:
            continue
        trm = ta & dA.term_id.isin(shared).to_numpy()
        tem = tb & dA.term_id.isin(shared).to_numpy()
        cls = np.array(sorted(shared))
        cm = {c: k for k, c in enumerate(cls)}
        ytr, yte = dA.term_id[trm].map(cm).to_numpy(), dA.term_id[tem].map(cm).to_numpy()
        skf = StratifiedKFold(min(4, int(np.bincount(ytr).min())), shuffle=True, random_state=0)
        ho = np.mean([
            balanced_accuracy_score(
                ytr[te2],
                LogisticRegression(max_iter=3000)
                .fit(StandardScaler().fit_transform(ZA[trm][tr2]), ytr[tr2])
                .predict(StandardScaler().fit(ZA[trm][tr2]).transform(ZA[trm][te2])))
            for tr2, te2 in skf.split(ZA[trm], ytr)
        ])
        sc = StandardScaler().fit(ZA[trm])
        clf = LogisticRegression(max_iter=3000).fit(sc.transform(ZA[trm]), ytr)
        tf = balanced_accuracy_score(yte, clf.predict(sc.transform(ZA[tem])))
        rows.append({"train": a, "test": b, "n_tissues": len(cls), "chance": 1 / len(cls),
                     "heldout": float(ho), "transfer": float(tf),
                     "ho_ratio": float(ho * len(cls)), "tf_ratio": float(tf * len(cls))})
pc = pd.DataFrame(rows)
print(pc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n  mean held-out {pc.heldout.mean():.4f} ({pc.ho_ratio.mean():.1f}x) | "
      f"mean transfer {pc.transfer.mean():.4f} ({pc.tf_ratio.mean():.1f}x) | "
      f"retention {pc.transfer.mean() / pc.heldout.mean():.1%}")
out["transfer"] = {"rows": pc.to_dict("records"),
                   "mean_heldout": float(pc.heldout.mean()),
                   "mean_transfer": float(pc.transfer.mean()),
                   "mean_heldout_ratio": float(pc.ho_ratio.mean()),
                   "mean_transfer_ratio": float(pc.tf_ratio.mean()),
                   "retention": float(pc.transfer.mean() / pc.heldout.mean())}

rng = np.random.default_rng(0)
crows = []
for a in groups:
    for b in groups:
        if a >= b:
            continue
        ta, tb = gA == a, gA == b
        shared = sorted({t for t in set(dA.term_id[ta]) & set(dA.term_id[tb])
                         if (dA.term_id[ta] == t).sum() >= 3
                         and (dA.term_id[tb] == t).sum() >= 3})
        if len(shared) < 6:
            continue
        CA = np.stack([UA[ta & (dA.term_id == t).to_numpy()].mean(0) for t in shared])
        CB = np.stack([UA[tb & (dA.term_id == t).to_numpy()].mean(0) for t in shared])
        CA -= CA.mean(0, keepdims=True)
        CB -= CB.mean(0, keepdims=True)
        CA /= np.maximum(np.linalg.norm(CA, axis=1, keepdims=True), 1e-12)
        CB /= np.maximum(np.linalg.norm(CB, axis=1, keepdims=True), 1e-12)
        Sm = CA @ CB.T
        n = len(shared)
        rk = (-Sm).argsort(1).argsort(1)[np.arange(n), np.arange(n)] + 1
        mrr = float((1 / rk).mean())
        nulls = np.array([
            (1 / ((-Sm).argsort(1).argsort(1)[np.arange(n), rng.permutation(n)] + 1)).mean()
            for _ in range(2000)
        ])
        crows.append({"pair": f"{a} × {b}", "n_tissues": n,
                      "top1": float((rk == 1).mean()), "chance_top1": 1 / n,
                      "top5": float((rk <= 5).mean()), "mrr": mrr,
                      "z": float((mrr - nulls.mean()) / nulls.std()),
                      "p": float(((nulls >= mrr).sum() + 1) / (len(nulls) + 1))})
cm_df = pd.DataFrame(crows)
print("\n  centroid matching (nothing trained):")
print(cm_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
out["centroid_matching"] = cm_df.to_dict("records")

Path("results").mkdir(exist_ok=True)
Path(f"results/final_{TAGL}.json").write_text(json.dumps(out, indent=2, default=float))
print(f"\nsaved results/final_{TAGL}.json")
