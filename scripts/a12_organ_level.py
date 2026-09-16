"""A12: how much does the fine biosample ontology cost us?

ENCODE's biosample terms split hairs: `liver` and `right lobe of liver` are different terms,
as are seven CD4/CD8 T-cell subsets and three aortas. The cross-assay test scores those pairs
as NEGATIVES (different tissue) even though they are biologically near-identical, which
deflates Delta and pulls AUROC toward 0.5. So the headline number is a lower bound.

This regroups by ENCODE `organ_slims` (exact set match), which collapses exactly those cases:
  liver / left lobe of liver / right lobe of liver  -> endocrine gland|exocrine gland|liver
  lung / left lung / upper lobe of left lung / ...  -> lung
  aorta / ascending aorta / thoracic aorta         -> arterial blood vessel|blood vessel|vasculature
  every CD4/CD8 T-cell subset                      -> blood|bodily fluid

The last line is also the catch: `blood|bodily fluid` becomes one 1,188-track bucket holding
K562, GM12878, T cells, B cells and monocytes, which is over-merging in the other direction.
So the run is repeated with that bucket dropped, and the honest answer is a bracket rather than
a single number.

Both groupings are computed on the IDENTICAL track subset (those with organ_slims resolved),
so the comparison isolates the grouping change and nothing else.

Run: uv run python scripts/a12_organ_level.py [n_perm]
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
from ntv3_interp.tissue_test import cross_modality_auroc, run_delta_test  # noqa: E402

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BLOOD = "blood|bodily fluid"
out: dict = {}

tracks = add_assay_group(pd.read_parquet("data/tracks.parquet"), merge_rna=True)


def run(head, subset, key, drop_blood):
    """key is 'term_id' (fine) or 'organ_slims' (coarse)."""
    V = effective_vectors(head)
    d = tracks.copy()
    d["alive"] = dead_track_mask(head)
    # Identical base subset for both groupings: needs an organ label either way.
    d = d[d.alive & d.assay_group.notna() & d.term_id.notna() & d.organ_slims.notna()
          & d.lab.notna()]
    if subset == "tissues_only":
        d = d[d.classification.isin(["tissue", "primary cell"])]
    if drop_blood:
        d = d[d.organ_slims != BLOOD]
    ng = d.groupby(key).assay_group.nunique()
    d = d[d[key].isin(ng[ng >= 2].index)].reset_index(drop=True)
    if d.empty or d[key].nunique() < 3:
        return None

    U = l2_normalize(V[d["index"].to_numpy()]).astype(np.float32)
    gi, g_u = pd.factorize(d[key])
    ai, a_u = pd.factorize(d.assay_group)
    alab = pd.factorize(d.assay_group.astype(str) + "||" + d.lab.astype(str))[0]

    au = cross_modality_auroc(U, gi, ai)
    pp = au["per_stratum"].copy()
    pp["pair"] = [" × ".join(sorted([a_u[k // (ai.max() + 1)], a_u[k % (ai.max() + 1)]]))
                  for k in pp.modality_pair_key]
    rec = {
        "n_tracks": int(len(d)),
        "n_groups": int(len(g_u)),
        "stratified_auroc": au["stratified_auroc"],
        "pooled_auroc": au["pooled_auroc"],
        "per_pair": pp[["pair", "auroc", "n_pos", "n_neg"]].to_dict("records"),
    }
    for lb, gp in [("within_assay", None), ("within_assay_x_lab", alab.astype(np.int64))]:
        r = run_delta_test(U, gi.astype(np.int64), ai.astype(np.int64),
                           n_tissue=len(g_u), n_mod=len(a_u),
                           n_perm=N_PERM, perm_group=gp)
        if "error" in r:
            continue
        rec[lb] = {k: r[k] for k in
                   ("observed_delta", "null_mean", "null_sd", "z", "p_perm")}
    return rec


head = load_head("data/head_NTv3_650M_post.npz")
head100 = load_head("data/head_NTv3_100M_post.npz")

for tag, h in [("650M", head), ("100M", head100)]:
    for subset in ("all_biosamples", "tissues_only"):
        for drop_blood in (False, True):
            for key, kname in [("term_id", "biosample (fine)"), ("organ_slims", "organ (coarse)")]:
                r = run(h, subset, key, drop_blood)
                if r is None:
                    continue
                name = (f"{tag} | {subset} | {kname}"
                        f"{' | blood bucket dropped' if drop_blood else ''}")
                out[name] = r
                print(f"\n{name}")
                print(f"   tracks {r['n_tracks']}  groups {r['n_groups']}  "
                      f"AUROC {r['stratified_auroc']:.4f}  (pooled {r['pooled_auroc']:.4f})")
                if "within_assay" in r:
                    w = r["within_assay"]
                    wl = r.get("within_assay_x_lab", {})
                    print(f"   delta {w['observed_delta']:+.5f}  z {w['z']:+.1f}"
                          + (f"   |  lab-fixed z {wl['z']:+.1f}" if wl else ""))
                for p in r["per_pair"]:
                    print(f"      {p['pair']:<46s} {p['auroc']:.4f}  "
                          f"(n_pos {p['n_pos']:,})")

Path("results").mkdir(exist_ok=True)
Path("results/a12_organ_level.json").write_text(json.dumps(out, indent=2, default=float))
print("\nsaved results/a12_organ_level.json")
