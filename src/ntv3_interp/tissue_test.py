"""A3 -- the actual hypothesis test.

Question: holding the pair of modalities fixed, are same-tissue track vectors more aligned
than different-tissue ones?

    Delta_c^(m1,m2) = E[ cos(v~_i, v~_j) | c_i=c_j=c, m_i=m1, m_j=m2 ]
                    - E[ cos(v~_i, v~_j) | c_i=c, c_j!=c, m_i=m1, m_j=m2 ]

Fixing the modality pair is what stops "RNA-seq and DNase are intrinsically more alike than
RNA-seq and ChIP" from masquerading as tissue structure.

Two things make this cheap and exact. Writing U for the L2-normalised effective vectors, so
that S = U U^T, the sum of cosines over a block of tracks A x B is just

    sum_{i in A, j in B} S_ij = (sum_{i in A} u_i) . (sum_{j in B} u_j)

so every block statistic reduces to a dot product of per-cell sum vectors -- the full
7362 x 7362 similarity matrix is never materialised. And because the test only ever compares
*different* modalities, near-duplicate pairs (the _M/_P strand splits of one experiment, and
replicate runs of one assay) can never land inside a within-tissue block: they always share a
modality. The duplicate-inflation worry is structurally eliminated, not merely filtered.

The null permutes tissue labels *within* modality strata, which preserves modality
composition and per-modality tissue counts and destroys only tissue alignment. Reordering the
similarity matrix would not be a null at all -- S itself is unchanged by reordering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Strata:
    """Pre-computed index structure for the block statistics."""

    tissue_idx: np.ndarray  # (n_tracks,) tissue id per track
    mod_idx: np.ndarray  # (n_tracks,) modality id per track
    n_tissue: int
    n_mod: int
    s_tissue: np.ndarray = field(default_factory=lambda: np.array([]))  # (S,) stratum tissue
    s_m1: np.ndarray = field(default_factory=lambda: np.array([]))
    s_m2: np.ndarray = field(default_factory=lambda: np.array([]))


def build_strata(
    tissue_idx: np.ndarray,
    mod_idx: np.ndarray,
    n_tissue: int,
    n_mod: int,
    min_between: int = 10,
) -> Strata:
    """Enumerate (tissue, m1, m2) strata that have support, from the OBSERVED labels.

    The stratum list is held fixed across permutations so that the null and the observed
    statistic average over exactly the same set of comparisons.
    """
    counts = np.zeros((n_tissue, n_mod), dtype=np.int64)
    np.add.at(counts, (tissue_idx, mod_idx), 1)
    mod_totals = counts.sum(axis=0)

    ts, m1s, m2s = [], [], []
    for c in range(n_tissue):
        present = np.flatnonzero(counts[c] > 0)
        for m1 in present:
            for m2 in present:
                if m1 == m2:
                    continue
                if mod_totals[m2] - counts[c, m2] < min_between:
                    continue
                ts.append(c)
                m1s.append(m1)
                m2s.append(m2)

    return Strata(
        tissue_idx=tissue_idx,
        mod_idx=mod_idx,
        n_tissue=n_tissue,
        n_mod=n_mod,
        s_tissue=np.asarray(ts, dtype=np.int64),
        s_m1=np.asarray(m1s, dtype=np.int64),
        s_m2=np.asarray(m2s, dtype=np.int64),
    )


def _cell_sums(U: np.ndarray, tissue_idx: np.ndarray, mod_idx: np.ndarray, st: Strata):
    """Per-(tissue, modality) sum of unit vectors, and per-modality totals."""
    n_cells = st.n_tissue * st.n_mod
    cell = tissue_idx * st.n_mod + mod_idx
    M = np.zeros((n_cells, U.shape[1]), dtype=np.float32)
    np.add.at(M, cell, U)
    counts = np.bincount(cell, minlength=n_cells).astype(np.float64)

    mod_tot = np.zeros((st.n_mod, U.shape[1]), dtype=np.float32)
    np.add.at(mod_tot, mod_idx, U)
    mod_counts = np.bincount(mod_idx, minlength=st.n_mod).astype(np.float64)
    return M, counts, mod_tot, mod_counts


def delta_statistic(
    U: np.ndarray,
    tissue_idx: np.ndarray,
    st: Strata,
    return_per_stratum: bool = False,
):
    """Mean Delta over strata. ``tissue_idx`` is an argument so permutations can vary it."""
    M, counts, mod_tot, _ = _cell_sums(U, tissue_idx, st.mod_idx, st)

    i1 = st.s_tissue * st.n_mod + st.s_m1
    i2 = st.s_tissue * st.n_mod + st.s_m2

    n1 = counts[i1]
    n2 = counts[i2]

    A = M[i1]  # (S, D) sum of unit vectors in (c, m1)
    B = M[i2]  # (S, D) sum of unit vectors in (c, m2)
    T2 = mod_tot[st.s_m2]  # (S, D) sum over ALL tracks of modality m2

    within_sum = np.einsum("sd,sd->s", A, B)
    between_sum = np.einsum("sd,sd->s", A, T2) - within_sum

    n_within = n1 * n2
    n_between = n1 * (np.bincount(st.mod_idx, minlength=st.n_mod).astype(np.float64)[st.s_m2] - n2)

    valid = (n_within > 0) & (n_between > 0)
    delta = np.full(st.s_tissue.shape, np.nan)
    delta[valid] = (
        within_sum[valid] / n_within[valid] - between_sum[valid] / n_between[valid]
    )

    stat = float(np.nanmean(delta))
    if return_per_stratum:
        return stat, delta, n_within, n_between
    return stat


def permute_within_modality(
    tissue_idx: np.ndarray, mod_idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Shuffle tissue labels inside each modality group.

    Preserves how many tracks each modality has and the per-modality tissue distribution;
    destroys only the correspondence between a track's modality and its tissue.

    ``mod_idx`` may encode a composite stratum -- e.g. (modality, lab) -- in which case the
    null additionally holds lab structure fixed, so a surviving effect cannot be batch.
    """
    out = tissue_idx.copy()
    for m in np.unique(mod_idx):
        sel = np.flatnonzero(mod_idx == m)
        out[sel] = rng.permutation(tissue_idx[sel])
    return out


def run_delta_test(
    U: np.ndarray,
    tissue_idx: np.ndarray,
    mod_idx: np.ndarray,
    n_tissue: int,
    n_mod: int,
    n_perm: int = 500,
    min_between: int = 10,
    seed: int = 0,
    perm_group: np.ndarray | None = None,
) -> dict:
    """``perm_group`` defaults to modality. Pass a composite (modality, lab) code to hold
    batch structure fixed under the null as well."""
    st = build_strata(tissue_idx, mod_idx, n_tissue, n_mod, min_between)
    if st.s_tissue.size == 0:
        return {"error": "no strata with support"}

    obs, per_stratum, n_within, n_between = delta_statistic(
        U, tissue_idx, st, return_per_stratum=True
    )

    groups = mod_idx if perm_group is None else perm_group
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = permute_within_modality(tissue_idx, groups, rng)
        null[k] = delta_statistic(U, perm, st)
        if (k + 1) % 50 == 0:
            print(f"    perm {k + 1}/{n_perm}", end="\r")

    mu, sd = float(null.mean()), float(null.std(ddof=1))
    # +1 in numerator and denominator: the observed value is itself one draw under H0.
    p = float(((null >= obs).sum() + 1) / (n_perm + 1))

    return {
        "observed_delta": obs,
        "null_mean": mu,
        "null_sd": sd,
        "z": float((obs - mu) / sd) if sd > 0 else float("nan"),
        "p_perm": p,
        "n_strata": int(st.s_tissue.size),
        "n_perm": n_perm,
        "per_stratum": per_stratum,
        "strata_tissue": st.s_tissue,
        "strata_m1": st.s_m1,
        "strata_m2": st.s_m2,
        "n_within": n_within,
        "n_between": n_between,
        "null": null,
    }


# --------------------------------------------------------------------------------------
# Cross-modality AUROC -- one interpretable number
# --------------------------------------------------------------------------------------


def cross_modality_auroc(
    U: np.ndarray,
    tissue_idx: np.ndarray,
    mod_idx: np.ndarray,
    seed: int = 0,
) -> dict:
    """Among cross-modality track pairs only, how well does cosine rank same-tissue first?

    Stratified by modality pair, so the headline number cannot be produced by one modality
    pair that happens to be both common and self-similar.

    Every cross-modality pair is enumerated exactly -- with a few thousand tracks the full
    similarity matrix is only ~100 MB, so there is no need to subsample.
    """
    n = U.shape[0]
    S = (U @ U.T).astype(np.float32)
    iu = np.triu_indices(n, k=1)
    i, j = iu[0].astype(np.int32), iu[1].astype(np.int32)

    keep = mod_idx[i] != mod_idx[j]
    i, j = i[keep], j[keep]
    if i.size == 0:
        return {"error": "no cross-modality pairs"}

    sims = S[i, j]
    same = tissue_idx[i] == tissue_idx[j]
    del S

    # Unordered modality-pair key.
    a = np.minimum(mod_idx[i], mod_idx[j])
    b = np.maximum(mod_idx[i], mod_idx[j])
    key = a * (mod_idx.max() + 1) + b

    from sklearn.metrics import roc_auc_score

    rows = []
    for k in np.unique(key):
        sel = key == k
        y, s = same[sel], sims[sel]
        if y.sum() < 10 or (~y).sum() < 10:
            continue
        rows.append(
            {
                "modality_pair_key": int(k),
                "auroc": float(roc_auc_score(y, s)),
                "n_pos": int(y.sum()),
                "n_neg": int((~y).sum()),
            }
        )

    if not rows:
        return {"error": "no modality-pair stratum with enough same-tissue pairs"}

    df = pd.DataFrame(rows)
    w = df["n_pos"].to_numpy(float)
    return {
        "stratified_auroc": float(np.average(df["auroc"], weights=w)),
        "pooled_auroc": float(roc_auc_score(same, sims)),
        "n_strata": len(df),
        "n_pairs": int(i.size),
        "n_same_tissue_pairs": int(same.sum()),
        "per_stratum": df,
    }
