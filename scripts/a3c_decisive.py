"""A3c: the decisive test -- tissue effect with BOTH modality and lab held fixed.

A3b showed cosine predicts same-lab (0.618) somewhat better than same-tissue (0.566), so
batch structure is clearly present in the head. The question that remains is whether tissue
adds anything on top of it.

This permutes tissue labels within (modality x lab) strata. Modality composition and lab
composition are both preserved exactly; only the tissue assignment is destroyed. An effect
that survives cannot be explained by either confound.

Run: uv run python scripts/a3c_decisive.py [n_perm]
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
from ntv3_interp.tissue_test import run_delta_test  # noqa: E402

N_PERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
out = {}

for head_path in sorted(Path("data").glob("head_*.npz")):
    head = load_head(head_path)
    tag = head.repo.split("/")[-1]
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
    tissue_codes, tissue_uniques = pd.factorize(df.term_id)
    mod_codes, mod_uniques = pd.factorize(df.modality)
    modlab = pd.factorize(df.modality.astype(str) + "||" + df.lab.astype(str))[0]

    print(f"\n{'=' * 72}\n{tag}\n{'=' * 72}")
    print(
        f"tracks={len(df)} tissues={len(tissue_uniques)} "
        f"modalities={len(mod_uniques)} modality-x-lab strata={modlab.max() + 1}"
    )

    for label, grp in [
        ("permute within modality", None),
        ("permute within modality x lab", modlab.astype(np.int64)),
    ]:
        res = run_delta_test(
            U,
            tissue_codes.astype(np.int64),
            mod_codes.astype(np.int64),
            n_tissue=len(tissue_uniques),
            n_mod=len(mod_uniques),
            n_perm=N_PERM,
            perm_group=grp,
        )
        print(
            f"\n  {label}"
            f"\n    observed delta : {res['observed_delta']:+.5f}"
            f"\n    null mean/sd   : {res['null_mean']:+.6f} / {res['null_sd']:.6f}"
            f"\n    z              : {res['z']:+.1f}"
            f"\n    p (permutation): {res['p_perm']:.5f}"
        )
        out[f"{tag}::{label}"] = {
            k: res[k]
            for k in ("observed_delta", "null_mean", "null_sd", "z", "p_perm", "n_strata")
        }

Path("results").mkdir(exist_ok=True)
Path("results/a3c_decisive.json").write_text(json.dumps(out, indent=2))
print("\nsaved results/a3c_decisive.json")
