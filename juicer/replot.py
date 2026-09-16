"""Regenerate every figure from cached results, without redoing the analysis.

``run_all`` writes its tables to ``results/`` and the UMAP embedding to
``results/umap_650M_raw.npy``, so figures can be rebuilt cheaply after a styling
change. The only thing not cached is the permutation null distribution, which is
recomputed for the headline configuration alone.

Usage::

    .venv/bin/python -m juicer.replot            # reuse cached null if present
    .venv/bin/python -m juicer.replot --perm 1000
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import plots, simgrid, viz
from .config import FIGURES, RESULTS
from .head import cosine_matrix, load_head
from .metadata import load_metadata
from .organ_groups import add_tissue_group
from .tissue_test import run_test

NULL_CACHE = RESULTS / "permutation_nulls_650M_raw_strict.npy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=1000)
    args = ap.parse_args()

    head = load_head("650M")
    md = add_tissue_group(load_metadata(head.track_ids))

    # ---- block matrices from cached CSVs ----
    for var in ("raw", "gain"):
        p = RESULTS / f"assay_block_matrix_650M_{var}.csv"
        if p.exists():
            M = pd.read_csv(p, index_col=0)
            simgrid.plot_block_matrix(
                M, f"Mean cosine between track-head rows, by assay ({var})",
                "off-diagonal = cross-assay; ATAC/DNase and the RNA family stand out",
                FIGURES / f"assay_block_matrix_650M_{var}.png")

    p = RESULTS / "tissue_block_matrix_650M_raw.csv"
    if p.exists():
        Mt = pd.read_csv(p, index_col=0)
        simgrid.plot_block_matrix(
            Mt, "Mean cosine by tissue (14 most-sampled, 650M raw)",
            "diagonal is inflated by within-modality replicates — hence Analysis 3",
            FIGURES / "tissue_block_matrix_650M_raw.png")

    # ---- per-track scalars ----
    p = RESULTS / "per_track_scalars_650M.csv"
    if p.exists():
        tbl = pd.read_csv(p)
        plots.plot_norms(tbl, "w_norm", "Weight-row norm by assay (650M)",
                         "norm is not direction: reported separately, never z-scored away",
                         FIGURES / "norm_by_assay_650M.png")
        plots.plot_norms(tbl, "bias_effective",
                         "Effective per-track offset by assay (650M)",
                         "w·β + b, the constant term of the head's linear function",
                         FIGURES / "bias_by_assay_650M.png")

    # ---- predictability ----
    p = RESULTS / "label_predictability.csv"
    if p.exists():
        pred = pd.read_csv(p)
        sub = pred[(pred["variant"] == "raw") & (pred["space"] == "full")]
        if not sub.empty:
            plots.plot_predictability(
                sub,
                "How predictable are modality and tissue from the head's PCs? (650M, raw)",
                "grouped CV prevents replicate/strand-mate leakage",
                FIGURES / "predictability_650M_raw.png")
        sub2 = pred[(pred["variant"] == "raw") & (pred["space"] == "modality_removed")]
        if not sub2.empty:
            plots.plot_predictability(
                sub2, "After projecting out the modality subspace (650M, raw)",
                "modality removed by least squares, not by deleting PC1",
                FIGURES / "predictability_modality_removed_650M_raw.png")

    # ---- UMAP from the cached embedding ----
    p = RESULTS / "umap_650M_raw.npy"
    if p.exists():
        emb = np.load(p)
        plots.plot_umap_facets(
            emb, md["modality_coarse"].to_numpy(),
            "Head rows (650M, raw) — UMAP, cosine metric, by modality",
            "small multiples: one panel per modality against the full cloud",
            FIGURES / "umap_by_modality_650M.png", colors=viz.MODALITY_COLORS)
        top = list(md["tissue_strict"].value_counts().index[:9])
        plots.plot_umap_facets(
            emb, md["tissue_strict"].to_numpy(),
            "Same embedding, by tissue (nine most-sampled)",
            "tissue does not form the dominant clusters; modality does",
            FIGURES / "umap_by_tissue_650M.png", order=top)
        plots.plot_umap_facets(
            emb, md["biosample_type"].to_numpy(),
            "Same embedding, by biosample type",
            "cell line vs primary tissue vs single cell",
            FIGURES / "umap_by_biosample_650M.png")

    # ---- similarity grids (cheap to recompute) ----
    S = cosine_matrix(head.W).astype(np.float32)
    for by, tag in [(["modality_coarse", "tissue_strict"], "modality_then_tissue"),
                    (["tissue_strict", "modality_coarse"], "tissue_then_modality"),
                    (["dataset", "modality_coarse"], "dataset_then_modality")]:
        simgrid.plot_full_matrix(
            S, md, by, f"Cosine similarity, ordered {' → '.join(by)}",
            FIGURES / f"simgrid_{tag}_650M_raw.png")

    # ---- delta figures ----
    p = RESULTS / "delta_by_modality_pair_650M_raw_strict.csv"
    if p.exists():
        plots.plot_delta_by_pair(
            pd.read_csv(p),
            "Same-tissue advantage, holding the modality pair fixed (650M, raw)",
            "Δ > 0 means two tracks of different modalities from the same tissue "
            "are more similar than from different tissues",
            FIGURES / "delta_by_modality_pair_650M_raw.png")
    p = RESULTS / "delta_by_assay_pair_650M_raw.csv"
    if p.exists():
        plots.plot_delta_by_pair(
            pd.read_csv(p),
            "Same-tissue advantage by assay pair (650M, raw, 8 assays)",
            "finer modality definition; each bar holds one assay pair fixed",
            FIGURES / "delta_by_assay_pair_650M_raw.png")
    p = RESULTS / "delta_cells_650M_raw_strict.csv"
    if p.exists():
        cells = pd.read_csv(p)
        plots.plot_delta_hist(
            cells, "Per-cell Δ distribution (650M, raw, strict tissue)",
            "one cell = one (tissue × modality-pair) comparison",
            FIGURES / "delta_cells_hist_650M_raw.png")

    # ---- permutation null (recompute or reuse cache) ----
    if NULL_CACHE.exists():
        res_nulls = np.load(NULL_CACHE)
        grid = pd.read_csv(RESULTS / "sensitivity_grid.csv")
        row = grid[(grid.checkpoint == "650M") & (grid.variant == "raw")
                   & (grid.tissue_col == "tissue_strict")].iloc[0]
        observed, nm, ns = row["observed_delta"], row["null_mean"], row["null_sd"]
        n_perm = int(row["n_perm"])
    else:
        res = run_test(S, md, "tissue_strict", "modality_coarse",
                       n_perm=args.perm, seed=0)
        res_nulls = res.nulls
        np.save(NULL_CACHE, res_nulls)
        observed = res.aggregate["observed_delta"]
        nm, ns = res.aggregate["null_mean"], res.aggregate["null_sd"]
        n_perm = args.perm
    plots.plot_null(
        observed, nm, ns, res_nulls,
        "Observed Δ against the permutation null (650M, raw)",
        f"tissue labels shuffled within modality, {n_perm} permutations",
        FIGURES / "permutation_null_650M_raw.png")
    print("figures regenerated")


if __name__ == "__main__":
    main()
