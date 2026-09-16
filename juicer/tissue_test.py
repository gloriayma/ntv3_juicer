"""The actual hypothesis test: is tissue structure present *after* fixing modality?

For each unordered modality pair (m, m') with m != m', and each tissue c having at
least one track in each::

    within_c  = mean cos over { (i,j) : i in (c,m),  j in (c,m')  }
    between_c = mean cos over { (i,j) : i in (c,m),  j in (~c,m') }
                         plus { (i,j) : i in (c,m'), j in (~c,m)  }
    delta_c   = within_c - between_c

Holding the modality pair fixed is the whole point. It prevents "RNA-RNA replicates
are intrinsically similar" and "RNA-ATAC is intrinsically closer than RNA-ChIP" from
masquerading as tissue structure, and because some modalities come wholly from one
source (CAGE is all FANTOM5, DNase all ENCODE), fixing the modality pair fixes the
dataset pair too.

Aggregation weights every (tissue, modality-pair) cell equally so that K562 -- 517
tracks -- cannot dominate.

The null is a permutation of tissue labels *within modality strata*. Permuting inside
a stratum preserves every (tissue, modality) group size exactly, so the null differs
from the observed statistic only in whether a tissue's identity corresponds across
modalities. Reordering a heatmap is *not* a null; this is.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PairBlock:
    """Precomputed cosine submatrix for one ordered modality pair."""

    m_a: str
    m_b: str
    rows: np.ndarray          # track indices with modality m_a
    cols: np.ndarray          # track indices with modality m_b
    S: np.ndarray             # (len(rows), len(cols)) float32 cosine block
    same_expt: np.ndarray | None = None  # bool mask of same-experiment entries


@dataclass
class DeltaResult:
    cells: pd.DataFrame
    aggregate: dict = field(default_factory=dict)
    by_modality_pair: pd.DataFrame | None = None
    nulls: np.ndarray | None = None


def build_pair_blocks(S: np.ndarray, modality: np.ndarray, expt_base: np.ndarray,
                      modalities: list[str]) -> list[PairBlock]:
    """Cache one cosine submatrix per unordered modality pair."""
    blocks = []
    for m_a, m_b in itertools.combinations(modalities, 2):
        rows = np.where(modality == m_a)[0]
        cols = np.where(modality == m_b)[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        sub = np.ascontiguousarray(S[np.ix_(rows, cols)], dtype=np.float32)
        # Same-experiment pairs must never contribute. Across two different
        # modalities this should be empty (one experiment has one assay), but we
        # check rather than assume, and mask if it ever is not.
        eq = expt_base[rows][:, None] == expt_base[cols][None, :]
        mask = eq if eq.any() else None
        if mask is not None:
            sub = sub.copy()
            sub[mask] = 0.0
        blocks.append(PairBlock(m_a, m_b, rows, cols, sub, mask))
    return blocks


def _block_stats(block: PairBlock, codes: np.ndarray, n_tissue: int):
    """Sum, sum-of-squares and count of cosines per (row-tissue, col-tissue) block."""
    rc = codes[block.rows]
    cc = codes[block.cols]
    flat = (rc[:, None].astype(np.int64) * n_tissue + cc[None, :]).ravel()
    vals = block.S.ravel().astype(np.float64)
    size = n_tissue * n_tissue
    g_sum = np.bincount(flat, weights=vals, minlength=size).reshape(n_tissue, n_tissue)
    g_sq = np.bincount(flat, weights=vals * vals, minlength=size).reshape(n_tissue, n_tissue)
    if block.same_expt is None:
        n_a = np.bincount(rc, minlength=n_tissue).astype(np.float64)
        n_b = np.bincount(cc, minlength=n_tissue).astype(np.float64)
        g_cnt = np.outer(n_a, n_b)
    else:
        w = (~block.same_expt).ravel().astype(np.float64)
        g_cnt = np.bincount(flat, weights=w, minlength=size).reshape(n_tissue, n_tissue)
    return g_sum, g_sq, g_cnt


def _cells_for_block(block: PairBlock, codes: np.ndarray, n_tissue: int):
    """Per-tissue within/between sums for one modality pair."""
    g_sum, g_sq, g_cnt = _block_stats(block, codes, n_tissue)

    w_sum = np.diag(g_sum).copy()
    w_sq = np.diag(g_sq).copy()
    w_cnt = np.diag(g_cnt).copy()

    row_sum, col_sum = g_sum.sum(axis=1), g_sum.sum(axis=0)
    row_sq, col_sq = g_sq.sum(axis=1), g_sq.sum(axis=0)
    row_cnt, col_cnt = g_cnt.sum(axis=1), g_cnt.sum(axis=0)

    # symmetrised "different tissue, same modality pair"
    b_sum = (row_sum - w_sum) + (col_sum - w_sum)
    b_sq = (row_sq - w_sq) + (col_sq - w_sq)
    b_cnt = (row_cnt - w_cnt) + (col_cnt - w_cnt)

    valid = (w_cnt > 0) & (b_cnt > 0)
    return valid, (w_sum, w_sq, w_cnt), (b_sum, b_sq, b_cnt)


def aggregate_delta(blocks: list[PairBlock], codes: np.ndarray, n_tissue: int) -> float:
    """Mean delta over all (tissue, modality-pair) cells, equally weighted."""
    total, n = 0.0, 0
    for block in blocks:
        valid, (w_sum, _, w_cnt), (b_sum, _, b_cnt) = _cells_for_block(block, codes, n_tissue)
        if not valid.any():
            continue
        d = w_sum[valid] / w_cnt[valid] - b_sum[valid] / b_cnt[valid]
        total += float(d.sum())
        n += int(valid.sum())
    return total / n if n else np.nan


def run_test(S: np.ndarray, md: pd.DataFrame, tissue_col: str, modality_col: str,
             n_perm: int = 1000, strata: tuple[str, ...] = ("modality",),
             seed: int = 0, subset: np.ndarray | None = None) -> DeltaResult:
    """Observed delta, per-cell table, per-modality-pair table and permutation null.

    ``strata`` names the columns whose combination defines a permutation block.
    ``"modality"`` is always included implicitly via ``modality_col``.
    """
    work = md if subset is None else md.loc[subset].copy()
    work = work.reset_index(drop=True)

    # Re-index the cosine matrix to the working subset.
    if subset is not None:
        keep = md.loc[subset, "row"].to_numpy()
        S = np.ascontiguousarray(S[np.ix_(keep, keep)])

    tissues = pd.Categorical(work[tissue_col])
    codes = tissues.codes.astype(np.int64)
    n_tissue = len(tissues.categories)
    modality = work[modality_col].to_numpy()
    expt = work["expt_base"].to_numpy()
    modalities = sorted(set(modality))

    blocks = build_pair_blocks(S, modality, expt, modalities)

    # ---- observed, with a full per-cell table ----
    rows = []
    for block in blocks:
        valid, (w_sum, w_sq, w_cnt), (b_sum, b_sq, b_cnt) = _cells_for_block(
            block, codes, n_tissue)
        for c in np.where(valid)[0]:
            wm = w_sum[c] / w_cnt[c]
            bm = b_sum[c] / b_cnt[c]
            wv = max(w_sq[c] / w_cnt[c] - wm * wm, 0.0)
            bv = max(b_sq[c] / b_cnt[c] - bm * bm, 0.0)
            rows.append({
                "modality_pair": f"{block.m_a} | {block.m_b}",
                "tissue": tissues.categories[c],
                "within_mean": wm, "between_mean": bm, "delta": wm - bm,
                "n_within": int(w_cnt[c]), "n_between": int(b_cnt[c]),
                "within_var": wv, "between_var": bv,
            })
    cells = pd.DataFrame(rows)
    if cells.empty:
        return DeltaResult(cells=cells, aggregate={"n_cells": 0})

    observed = float(cells["delta"].mean())

    # ---- permutation null: shuffle tissue within modality (x optional strata) ----
    rng = np.random.default_rng(seed)
    strat_cols = [modality_col] + [c for c in strata if c != "modality"]
    strat_key = work[strat_cols].astype(str).agg("\x1f".join, axis=1).to_numpy()
    groups = [np.where(strat_key == k)[0] for k in pd.unique(strat_key)]

    null = np.empty(n_perm, dtype=float)
    for p in range(n_perm):
        perm = codes.copy()
        for g in groups:
            perm[g] = rng.permutation(perm[g])
        null[p] = aggregate_delta(blocks, perm, n_tissue)

    null_mean = float(np.nanmean(null))
    null_sd = float(np.nanstd(null))
    n_ge = int(np.sum(null >= observed))
    p_val = (n_ge + 1) / (n_perm + 1)

    # pooled effect size from accumulated sums (no need to store every pair)
    w_n = cells["n_within"].to_numpy()
    b_n = cells["n_between"].to_numpy()
    pooled_wv = float(np.average(cells["within_var"], weights=w_n))
    pooled_bv = float(np.average(cells["between_var"], weights=b_n))
    denom = np.sqrt((pooled_wv + pooled_bv) / 2) or np.nan
    cohen_d = float(observed / denom) if denom and not np.isnan(denom) else np.nan

    by_pair = (cells.groupby("modality_pair")
                    .agg(delta=("delta", "mean"), n_cells=("delta", "size"),
                         n_within_pairs=("n_within", "sum"),
                         frac_positive=("delta", lambda s: float((s > 0).mean())))
                    .sort_values("delta", ascending=False)
                    .reset_index())

    aggregate = {
        "observed_delta": observed,
        "n_cells": int(len(cells)),
        "n_tissues": int(cells["tissue"].nunique()),
        "n_modality_pairs": int(cells["modality_pair"].nunique()),
        "frac_cells_positive": float((cells["delta"] > 0).mean()),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z_vs_null": float((observed - null_mean) / null_sd) if null_sd > 0 else np.nan,
        "p_permutation": float(p_val),
        "n_perm": n_perm,
        "cohen_d_pooled": cohen_d,
        # AUC implied by the pooled effect size under a normal approximation:
        # P(within-pair cosine > between-pair cosine) = Phi(d / sqrt(2)).
        "auc_approx": float(0.5 * (1 + math.erf(cohen_d / 2))) if not np.isnan(cohen_d) else np.nan,
        "tissue_col": tissue_col,
        "modality_col": modality_col,
        "strata": ",".join(strat_cols),
    }
    return DeltaResult(cells=cells, aggregate=aggregate, by_modality_pair=by_pair,
                       nulls=null)
