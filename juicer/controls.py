"""Alignment and sanity controls that must pass before anything is interpreted.

A note on a control that was *dropped*. The original plan proposed using the 1,207
same-experiment plus/minus strand mates (``ENCSR580GSX_P``/``_M``) as a positive
control, expecting near-identical head rows. That reasoning was wrong: the two
strand tracks of one stranded RNA-seq experiment measure *complementary* signals --
a plus-strand gene appears in the plus track and is absent from the minus track --
so the head must predict different things for them. Measured cosine is ~0.06 raw and
slightly negative once the LayerNorm gain is folded in, i.e. mates are no more
similar than random and are faintly anti-aligned. That is consistent with correct
alignment, not with a broken join, so it is reported as a diagnostic instead.

The controls that actually discriminate are biological orderings that a scrambled
row mapping could not reproduce.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .head import Head, l2_normalize


def assay_block_matrix(S: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Mean off-diagonal cosine for every pair of label blocks."""
    cats = sorted(set(labels))
    out = pd.DataFrame(index=cats, columns=cats, dtype=float)
    idx = {c: np.where(labels == c)[0] for c in cats}
    for a in cats:
        for b in cats:
            block = S[np.ix_(idx[a], idx[b])]
            if a == b:
                n = len(idx[a])
                if n < 2:
                    out.loc[a, b] = np.nan
                    continue
                tri = block[np.triu_indices(n, k=1)]
                out.loc[a, b] = tri.mean()
            else:
                out.loc[a, b] = block.mean()
    return out


def strand_mate_diagnostic(S: np.ndarray, md: pd.DataFrame, rng) -> dict:
    """Cosine between same-experiment plus/minus mates vs a random-pair background."""
    stranded = md[md["strand"] != ""]
    groups = stranded.groupby("expt_base")["row"].apply(list)
    pairs = [(v[0], v[1]) for v in groups if len(v) == 2]
    i = np.array([p[0] for p in pairs])
    j = np.array([p[1] for p in pairs])
    same = S[i, j]
    a = rng.integers(0, S.shape[0], 200_000)
    b = rng.integers(0, S.shape[0], 200_000)
    keep = a != b
    bg = S[a[keep], b[keep]]
    return {
        "n_pairs": len(pairs),
        "mate_median": float(np.median(same)),
        "mate_mean": float(same.mean()),
        "background_median": float(np.median(bg)),
        "background_mean": float(bg.mean()),
    }


def run_gates(S: np.ndarray, md: pd.DataFrame, head: Head, variant: str, rng) -> dict:
    """Structural gates. Each returns a pass/fail plus the numbers behind it."""
    labels = md["assay"].values
    M = assay_block_matrix(S, labels)

    cats = list(M.index)
    within = np.nanmean([M.loc[c, c] for c in cats])
    between = np.nanmean([M.loc[a, b] for a in cats for b in cats if a != b])

    gates = {}

    # Gate A: modality blocks must be internally more similar than across blocks.
    gates["A_within_gt_between"] = {
        "pass": bool(within - between > 0.05),
        "within_assay_mean": float(within),
        "between_assay_mean": float(between),
        "delta": float(within - between),
    }

    # Gate B: biologically ordered off-diagonals. ATAC and DNase both assay open
    # chromatin, so they must be closer to each other than either is to RNA-seq.
    atac_dnase = float(M.loc["ATAC-seq", "DNase-seq"])
    atac_rna = float(M.loc["ATAC-seq", "total RNA-seq"])
    rna_polya = float(M.loc["RNA-seq", "polyA plus RNA-seq"])
    rna_atac = float(M.loc["RNA-seq", "ATAC-seq"])
    gates["B_biological_ordering"] = {
        "pass": bool(atac_dnase > atac_rna and rna_polya > rna_atac),
        "cos_ATAC_DNase": atac_dnase,
        "cos_ATAC_totalRNA": atac_rna,
        "cos_RNAseq_polyA": rna_polya,
        "cos_RNAseq_ATAC": rna_atac,
    }

    gates["C_strand_mate_diagnostic"] = strand_mate_diagnostic(S, md, rng)
    gates["_assay_block_matrix"] = M
    gates["variant"] = variant
    gates["checkpoint"] = head.name
    return gates


def cross_model_agreement(heads: dict[str, Head], md: pd.DataFrame, variant: str) -> dict:
    """Do two independently trained checkpoints agree on assay-block geometry?

    The 650M and 100M heads have different widths (1536 vs 768) and were trained
    separately, so agreement here cannot come from a shared bug in the row mapping.
    """
    mats = {}
    for name, h in heads.items():
        U = l2_normalize(h.variant(variant))
        S = (U @ U.T).astype(np.float32)
        mats[name] = assay_block_matrix(S, md["assay"].values)
    names = list(mats)
    a = mats[names[0]].values.astype(float).ravel()
    b = mats[names[1]].values.astype(float).ravel()
    ok = ~(np.isnan(a) | np.isnan(b))
    r = float(np.corrcoef(a[ok], b[ok])[0, 1])
    return {"pair": names, "pearson_r": r, "pass": bool(r > 0.8), "variant": variant}
