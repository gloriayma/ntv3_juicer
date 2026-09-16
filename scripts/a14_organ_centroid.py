"""A14: organ-level generalization, tested at centroid level with biosample held out.

A13's pairwise test said there is no organ-level generalization, but pairwise cosines are
noisy -- and the biosample-level centroid test was strong, so the same question deserves the
powerful version before the negative is believed.

Design. Build one centroid per (organ, biosample, assay). For each ChIP centroid, rank all
RNA (or DNase) centroids from *different biosamples* and ask whether the nearest one comes
from the same organ. Because the biosample is excluded on the other side, a hit cannot come
from sample identity -- only from a donor/site-independent organ code.

Chance is computed per query as the fraction of eligible targets sharing its organ, so the
baseline accounts for organ frequency rather than assuming uniformity.

Run: uv run python scripts/a14_organ_centroid.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ntv3_interp.grouping import add_assay_group  # noqa: E402
from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)

BLOOD = "blood|bodily fluid"
RNA = "RNA-seq (total + polyA)"
out: dict = {}

tracks = add_assay_group(pd.read_parquet("data/tracks.parquet"), merge_rna=True)
head = load_head("data/head_NTv3_650M_post.npz")
V = effective_vectors(head)


def centroids(d, U, assay):
    """One centroid per (organ, biosample) for the given assay."""
    m = (d.assay_group == assay).to_numpy()
    sub = d[m]
    idx = np.flatnonzero(m)
    rows, meta = [], []
    for (org, bio), g in sub.groupby(["organ_slims", "term_id"]):
        sel = idx[np.isin(sub.index.to_numpy(), g.index.to_numpy())]
        if len(sel) < 2:
            continue
        rows.append(U[sel].mean(0))
        meta.append({"organ": org, "biosample": bio, "n": len(sel)})
    if not rows:
        return None, None
    C = np.stack(rows)
    return C, pd.DataFrame(meta)


for subset in ("all_biosamples", "tissues_only"):
    for drop_blood in (False, True):
        d = tracks.copy()
        d["alive"] = dead_track_mask(head)
        d = d[d.alive & d.assay_group.notna() & d.term_id.notna() & d.organ_slims.notna()]
        if subset == "tissues_only":
            d = d[d.classification.isin(["tissue", "primary cell"])]
        if drop_blood:
            d = d[d.organ_slims != BLOOD]
        d = d.reset_index(drop=True)
        U = l2_normalize(V[d["index"].to_numpy()])

        name = f"650M | {subset}{' | no blood' if drop_blood else ''}"
        print(f"\n{'=' * 76}\n{name}\n{'=' * 76}")
        rec = {}

        for qa, ta in [("ChIP-seq", RNA), (RNA, "ChIP-seq"), ("ChIP-seq", "DNase-seq"),
                       ("DNase-seq", RNA)]:
            CQ, MQ = centroids(d, U, qa)
            CT, MT = centroids(d, U, ta)
            if CQ is None or CT is None or len(MT) < 8:
                continue
            # Centre each assay's centroid cloud so the shared assay offset cannot dominate.
            CQc = CQ - CQ.mean(0, keepdims=True)
            CTc = CT - CT.mean(0, keepdims=True)
            CQc /= np.maximum(np.linalg.norm(CQc, axis=1, keepdims=True), 1e-12)
            CTc /= np.maximum(np.linalg.norm(CTc, axis=1, keepdims=True), 1e-12)
            S = CQc @ CTc.T

            hits, chances, n_used = [], [], 0
            for i in range(len(MQ)):
                # Exclude the same biosample on the target side -- this is the whole point.
                elig = (MT.biosample != MQ.biosample.iloc[i]).to_numpy()
                same_org = (MT.organ == MQ.organ.iloc[i]).to_numpy() & elig
                if same_org.sum() == 0 or elig.sum() < 5:
                    continue
                best = np.argmax(np.where(elig, S[i], -np.inf))
                hits.append(bool(same_org[best]))
                chances.append(same_org.sum() / elig.sum())
                n_used += 1
            if n_used < 10:
                continue
            top1 = float(np.mean(hits))
            ch = float(np.mean(chances))
            # Binomial z against the per-query chance baseline.
            var = float(np.sum([c * (1 - c) for c in chances]))
            z = (np.sum(hits) - np.sum(chances)) / np.sqrt(max(var, 1e-12))
            key = f"{qa} -> {ta}"
            rec[key] = {"top1": top1, "chance": ch, "z": float(z), "n_queries": n_used,
                        "n_targets": int(len(MT))}
            print(f"  {key:<46s} top-1 {top1:.3f}  chance {ch:.3f}  "
                  f"z {z:+.2f}  (n={n_used} queries, {len(MT)} targets)")

        out[name] = rec

Path("results").mkdir(exist_ok=True)
Path("results/a14_organ_centroid.json").write_text(json.dumps(out, indent=2, default=float))
print("\nsaved results/a14_organ_centroid.json")
