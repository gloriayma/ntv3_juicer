"""A3: does the head factorize tissue from assay modality?

Run: uv run python scripts/a3_tissue.py
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

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MIN_MOD_TRACKS = 10  # drop ultra-rare modalities: they make unstable strata

tracks = pd.read_parquet("data/tracks.parquet")

SUBSETS = {
    "all_biosamples": lambda d: d,
    "tissues_only": lambda d: d[d.classification.isin(["tissue", "primary cell"])],
}

results = {}

for head_path in sorted(Path("data").glob("head_*.npz")):
    head = load_head(head_path)
    tag = head.repo.split("/")[-1]
    V = effective_vectors(head)  # gamma-scaled, mean-centered (identifiable part)
    alive = dead_track_mask(head)

    for subset_name, subset_fn in SUBSETS.items():
        df = tracks.copy()
        df["alive"] = alive
        df = df[df.alive & df.modality.notna() & df.term_id.notna()]
        df = subset_fn(df)

        # Drop rare modalities.
        vc = df.modality.value_counts()
        df = df[df.modality.isin(vc[vc >= MIN_MOD_TRACKS].index)]
        # A tissue is only usable if it carries at least two different modalities.
        n_mod_per_tissue = df.groupby("term_id").modality.nunique()
        df = df[df.term_id.isin(n_mod_per_tissue[n_mod_per_tissue >= 2].index)]

        if df.empty:
            print(f"\n[{tag} / {subset_name}] no usable tracks")
            continue

        idx = df["index"].to_numpy()
        U = l2_normalize(V[idx]).astype(np.float32)

        tissue_codes, tissue_uniques = pd.factorize(df.term_id)
        mod_codes, mod_uniques = pd.factorize(df.modality)

        print(f"\n{'=' * 74}")
        print(f"{tag}  |  {subset_name}")
        print(f"{'=' * 74}")
        print(
            f"tracks={len(df)}  tissues={len(tissue_uniques)}  modalities={len(mod_uniques)}"
        )

        auroc = cross_modality_auroc(U, tissue_codes, mod_codes)
        if "error" not in auroc:
            print(
                f"cross-modality AUROC (stratified) : {auroc['stratified_auroc']:.4f}"
                f"   [pooled {auroc['pooled_auroc']:.4f}]"
            )
            print(
                f"   over {auroc['n_strata']} modality-pair strata, "
                f"{auroc['n_same_tissue_pairs']:,} same-tissue pairs "
                f"of {auroc['n_pairs']:,}"
            )

        print(f"running delta test with {N_PERM} within-modality permutations ...")
        res = run_delta_test(
            U,
            tissue_codes.astype(np.int64),
            mod_codes.astype(np.int64),
            n_tissue=len(tissue_uniques),
            n_mod=len(mod_uniques),
            n_perm=N_PERM,
        )
        if "error" in res:
            print(f"  {res['error']}")
            continue

        print(
            f"\n  observed delta : {res['observed_delta']:+.5f}"
            f"\n  null mean/sd   : {res['null_mean']:+.5f} / {res['null_sd']:.5f}"
            f"\n  z              : {res['z']:+.2f}"
            f"\n  p (permutation): {res['p_perm']:.5f}"
            f"\n  strata         : {res['n_strata']}"
        )

        # Per-modality-pair breakdown, so a single dominant pair can't carry the result.
        per = pd.DataFrame(
            {
                "tissue": [tissue_uniques[c] for c in res["strata_tissue"]],
                "m1": [mod_uniques[m] for m in res["strata_m1"]],
                "m2": [mod_uniques[m] for m in res["strata_m2"]],
                "delta": res["per_stratum"],
                "n_within": res["n_within"],
            }
        ).dropna(subset=["delta"])
        per["pair"] = [
            " x ".join(sorted([a, b])) for a, b in zip(per.m1, per.m2)
        ]
        bypair = (
            per.groupby("pair")
            .agg(mean_delta=("delta", "mean"), n=("delta", "size"))
            .sort_values("n", ascending=False)
        )
        print(f"\n  positive strata: {(per.delta > 0).mean():.1%} of {len(per)}")
        print("\n  top modality pairs by support:")
        print(bypair.head(12).to_string())

        bytissue = (
            per.groupby("tissue")
            .agg(mean_delta=("delta", "mean"), n=("delta", "size"))
            .sort_values("mean_delta", ascending=False)
        )
        print("\n  strongest tissues:")
        print(bytissue.head(10).to_string())
        print("\n  weakest tissues:")
        print(bytissue.tail(5).to_string())

        key = f"{tag}::{subset_name}"
        results[key] = {
            "n_tracks": int(len(df)),
            "n_tissues": int(len(tissue_uniques)),
            "n_modalities": int(len(mod_uniques)),
            "observed_delta": res["observed_delta"],
            "null_mean": res["null_mean"],
            "null_sd": res["null_sd"],
            "z": res["z"],
            "p_perm": res["p_perm"],
            "n_strata": res["n_strata"],
            "frac_positive_strata": float((per.delta > 0).mean()),
            "stratified_auroc": auroc.get("stratified_auroc"),
            "pooled_auroc": auroc.get("pooled_auroc"),
        }

        Path("results").mkdir(exist_ok=True)
        per.to_csv(f"results/a3_strata_{tag}_{subset_name}.csv", index=False)
        np.save(f"results/a3_null_{tag}_{subset_name}.npy", res["null"])

Path("results/a3_summary.json").write_text(json.dumps(results, indent=2))
print("\n\nsaved results/a3_summary.json")
