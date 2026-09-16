"""Extra controls for confounds that the main permutation null does not remove.

**The ChIP target confound.** Which TFs and histone marks were assayed in which
biosample is not random -- ENCODE profiled far more targets in K562 and HepG2 than
anywhere else. So for any modality pair involving ChIP, the set of targets carrying
the label "same tissue" differs systematically from the set carrying "different
tissue", and some of Δ could reflect assay design rather than tissue biology.

Two ways to address it, both run here:

1. **A stronger null.** Permute tissue labels within ``modality × experiment_target``
   instead of within modality alone. Each target's tissue distribution is then
   preserved exactly, so a target-composition effect cannot show up as signal.

2. **A target-free subset.** Restrict to modality pairs where neither side has an
   ``experiment_target`` at all -- Accessibility (ATAC/DNase) versus RNA. These are
   genuinely different molecular readouts with no shared target concept, so a
   positive Δ there cannot be a target artifact.
"""
from __future__ import annotations

import json

import numpy as np

from .config import RESULTS
from .head import cosine_matrix, load_head
from .metadata import load_metadata
from .organ_groups import add_tissue_group
from .tissue_test import run_test

TARGET_FREE = ("Accessibility", "RNA", "CAGE")


def main(n_perm: int = 1000) -> None:
    head = load_head("650M")
    md = add_tissue_group(load_metadata(head.track_ids))
    S = cosine_matrix(head.W).astype(np.float32)

    out = {}

    # 1. null that preserves each ChIP target's tissue distribution
    res = run_test(S, md, "tissue_strict", "modality_coarse", n_perm=n_perm,
                   strata=("modality", "experiment_target"), seed=0)
    out["null_stratified_by_target"] = res.aggregate
    print(f"target-stratified null: Δ={res.aggregate['observed_delta']:+.5f} "
          f"z={res.aggregate['z_vs_null']:.1f} p={res.aggregate['p_permutation']:.4g}")

    # 2. target-free modalities only (no experiment_target on either side)
    mask = md["modality_coarse"].isin(TARGET_FREE).to_numpy()
    res2 = run_test(S, md, "tissue_strict", "modality_coarse", n_perm=n_perm,
                    seed=0, subset=mask)
    out["target_free_modalities"] = res2.aggregate
    if res2.by_modality_pair is not None:
        res2.by_modality_pair.to_csv(
            RESULTS / "delta_by_modality_pair_target_free.csv", index=False)
    print(f"target-free subset:     Δ={res2.aggregate['observed_delta']:+.5f} "
          f"z={res2.aggregate['z_vs_null']:.1f} p={res2.aggregate['p_permutation']:.4g} "
          f"(n_tracks={int(mask.sum())}, n_cells={res2.aggregate['n_cells']})")

    with open(RESULTS / "extra_controls.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
