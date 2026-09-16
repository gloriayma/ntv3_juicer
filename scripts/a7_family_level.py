"""A7: re-run the headline test with "cross-modality" tightened to cross-assay-FAMILY.

A6 showed the kNN enrichment was largely carried by pairs that differ in ChIP target but
are both still ChIP-seq. The A3 test has the same exposure: its modality label is
assay:target, so "ChIP-seq:H3K4me3 x ChIP-seq:H3K27ac" counted as a cross-modality
comparison. That is a much weaker notion of crossing modality than ChIP-seq x RNA-seq.

Here the modality variable is the assay FAMILY (ChIP-seq / RNA-seq / polyA plus RNA-seq /
DNase-seq / CAGE), so every comparison the statistic makes is between genuinely different
assay types. If the effect survives this, the tissue signal is not an artifact of comparing
histone marks to each other.

Run: uv run python scripts/a7_family_level.py [n_perm]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)
from ntv3_interp.tissue_test import cross_modality_auroc, run_delta_test  # noqa: E402

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
results = {}

for head_path in sorted(Path("data").glob("head_*.npz")):
    head = load_head(head_path)
    tag = head.repo.split("/")[-1]
    tracks = pd.read_parquet("data/tracks.parquet")
    V = effective_vectors(head)
    alive = dead_track_mask(head)

    for subset, fn in [
        ("all_biosamples", lambda d: d),
        ("tissues_only", lambda d: d[d.classification.isin(["tissue", "primary cell"])]),
    ]:
        df = tracks.copy()
        df["alive"] = alive
        df = df[df.alive & df.assay_family.notna() & df.term_id.notna()]
        df = fn(df)
        vc = df.assay_family.value_counts()
        df = df[df.assay_family.isin(vc[vc >= 20].index)]
        # A tissue is usable only if it appears in at least two different FAMILIES.
        npf = df.groupby("term_id").assay_family.nunique()
        df = df[df.term_id.isin(npf[npf >= 2].index)].reset_index(drop=True)
        if df.empty:
            continue

        U = l2_normalize(V[df["index"].to_numpy()]).astype(np.float32)
        tis, tis_u = pd.factorize(df.term_id)
        fam, fam_u = pd.factorize(df.assay_family)
        modlab = pd.factorize(
            df.assay_family.astype(str) + "||" + df.lab.astype(str)
        )[0]

        print(f"\n{'=' * 72}\n{tag} | {subset}\n{'=' * 72}")
        print(
            f"tracks={len(df)} tissues={len(tis_u)} families={len(fam_u)} "
            f"({', '.join(fam_u)})"
        )

        auroc = cross_modality_auroc(U, tis, fam)
        print(
            f"cross-FAMILY AUROC (stratified) : {auroc['stratified_auroc']:.4f}"
            f"  [pooled {auroc['pooled_auroc']:.4f}]"
        )
        print(
            f"   {auroc['n_strata']} family-pair strata, "
            f"{auroc['n_same_tissue_pairs']:,} same-tissue of {auroc['n_pairs']:,} pairs"
        )
        print("   per family pair:")
        pp = auroc["per_stratum"].copy()
        pp["pair"] = [
            " x ".join(sorted([fam_u[k // (fam.max() + 1)], fam_u[k % (fam.max() + 1)]]))
            for k in pp.modality_pair_key
        ]
        print(pp[["pair", "auroc", "n_pos", "n_neg"]].to_string(index=False))

        res = {}
        for label, grp in [
            ("within family", None),
            ("within family x lab", modlab.astype(np.int64)),
        ]:
            r = run_delta_test(
                U, tis.astype(np.int64), fam.astype(np.int64),
                n_tissue=len(tis_u), n_mod=len(fam_u),
                n_perm=N_PERM, perm_group=grp,
            )
            if "error" in r:
                print(f"  {label}: {r['error']}")
                continue
            print(
                f"\n  permute {label}:"
                f"\n    observed delta {r['observed_delta']:+.5f} | "
                f"null {r['null_mean']:+.6f} ± {r['null_sd']:.6f} | "
                f"z {r['z']:+.1f} | p {r['p_perm']:.5f}"
            )
            res[label] = {
                k: r[k] for k in
                ("observed_delta", "null_mean", "null_sd", "z", "p_perm", "n_strata")
            }
            res[label]["frac_positive"] = float(np.nanmean(r["per_stratum"] > 0))

        results[f"{tag}::{subset}"] = {
            "n_tracks": int(len(df)),
            "n_tissues": int(len(tis_u)),
            "families": list(fam_u),
            "stratified_auroc": auroc["stratified_auroc"],
            "pooled_auroc": auroc["pooled_auroc"],
            "per_family_pair": pp[["pair", "auroc", "n_pos", "n_neg"]].to_dict("records"),
            **res,
        }

Path("results").mkdir(exist_ok=True)
Path("results/a7_family_level.json").write_text(json.dumps(results, indent=2, default=float))
print("\nsaved results/a7_family_level.json")
