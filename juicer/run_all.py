"""End-to-end run: controls, geometry, similarity structure, hypothesis test.

Usage::

    .venv/bin/python -m juicer.run_all              # full run, both checkpoints
    .venv/bin/python -m juicer.run_all --quick      # fewer permutations, no UMAP
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import plots, simgrid, viz
from .config import FIGURES, RESULTS
from .controls import cross_model_agreement, run_gates
from .geometry import (label_predictability, norm_table, pca_fit,
                       project_out_modality, umap_embed)
from .head import cosine_matrix, load_head
from .metadata import load_metadata
from .organ_groups import add_tissue_group, coverage_report
from .tissue_test import run_test
from .validate import negative_control, synthetic_positive_control

VARIANTS = ("raw", "gain")
CHECKPOINT_NAMES = ("650M", "100M")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="fewer permutations and skip UMAP")
    args = ap.parse_args()

    n_perm = 200 if args.quick else 1000
    n_perm_side = 100 if args.quick else 300

    summary: dict = {"n_perm": n_perm}

    heads = {name: load_head(name) for name in CHECKPOINT_NAMES}
    md = load_metadata(heads["650M"].track_ids)
    md = add_tissue_group(md)
    _log(f"loaded heads {[f'{k}:{v.d}d' for k, v in heads.items()]}, "
         f"{len(md)} human tracks")

    cov = coverage_report(md)
    cov.to_csv(RESULTS / "organ_group_coverage.csv", index=False)
    summary["organ_groups_used"] = int(len(cov))

    md.drop(columns=["row"]).to_csv(RESULTS / "human_track_labels.csv", index=False)

    # ---------------- cosine matrices ----------------
    S = {}
    for name, head in heads.items():
        for var in VARIANTS:
            S[(name, var)] = cosine_matrix(head.variant(var)).astype(np.float32)
    _log("cosine matrices built")

    # ---------------- Analysis 0: controls ----------------
    rng = np.random.default_rng(0)
    gate_report = {}
    for var in VARIANTS:
        g = run_gates(S[("650M", var)], md, heads["650M"], var, rng)
        M = g.pop("_assay_block_matrix")
        M.to_csv(RESULTS / f"assay_block_matrix_650M_{var}.csv")
        gate_report[var] = g
        simgrid.plot_block_matrix(
            M, f"Mean cosine between track-head rows, by assay ({var})",
            "off-diagonal = cross-assay; ATAC/DNase and the RNA family should stand out",
            FIGURES / f"assay_block_matrix_650M_{var}.png")
    gate_report["cross_model"] = {v: cross_model_agreement(heads, md, v)
                                  for v in VARIANTS}
    summary["gates"] = gate_report
    _log("gates: " + json.dumps({k: (v.get("A_within_gt_between", {}).get("pass")
                                     if isinstance(v, dict) else None)
                                 for k, v in gate_report.items()}))

    # ---------------- estimator validation ----------------
    val = {
        "negative_control": negative_control(
            heads["650M"].W, md, "tissue_strict", "modality_coarse",
            n_perm=n_perm_side),
        "synthetic_positive_control": synthetic_positive_control(
            heads["650M"].W, md, "tissue_strict", "modality_coarse",
            strength=0.25, n_perm=n_perm_side),
    }
    summary["estimator_validation"] = val
    _log(f"validation: negative Δ={val['negative_control']['observed_delta']:.5f}, "
         f"positive Δ={val['synthetic_positive_control']['observed_delta']:.5f}")

    # ---------------- Analysis 1: geometry ----------------
    tbl = norm_table(heads["650M"], md)
    tbl.to_csv(RESULTS / "per_track_scalars_650M.csv", index=False)
    plots.plot_norms(tbl, "w_norm",
                     "Weight-row norm by assay (650M)",
                     "norm is not direction: reported separately, never z-scored away",
                     FIGURES / "norm_by_assay_650M.png")
    plots.plot_norms(tbl, "bias_effective",
                     "Effective per-track offset by assay (650M)",
                     "w·β + b, the constant term of the head's linear function",
                     FIGURES / "bias_by_assay_650M.png")

    pred_frames = []
    for var in VARIANTS:
        Z, pca = pca_fit(heads["650M"].variant(var), n_components=50)
        np.save(RESULTS / f"pca_scores_650M_{var}.npy", Z[:, :20])
        pd.DataFrame({
            "pc": np.arange(1, len(pca.explained_variance_ratio_) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }).to_csv(RESULTS / f"pca_scree_650M_{var}.csv", index=False)

        for label, group in [("modality_coarse", "tissue_strict"),
                             ("assay", "tissue_strict"),
                             ("tissue_strict", "expt_base")]:
            df = label_predictability(Z, md, label, group)
            if not df.empty:
                df["variant"] = var
                df["space"] = "full"
                pred_frames.append(df)

        # after removing everything modality can linearly explain
        resid = project_out_modality(heads["650M"].variant(var), md)
        Zr, _ = pca_fit(resid, n_components=50)
        for label, group in [("modality_coarse", "tissue_strict"),
                             ("tissue_strict", "expt_base")]:
            df = label_predictability(Zr, md, label, group)
            if not df.empty:
                df["variant"] = var
                df["space"] = "modality_removed"
                pred_frames.append(df)
        _log(f"predictability done ({var})")

    pred = pd.concat(pred_frames, ignore_index=True)
    pred.to_csv(RESULTS / "label_predictability.csv", index=False)
    sub = pred[(pred["variant"] == "raw") & (pred["space"] == "full")]
    if not sub.empty:
        plots.plot_predictability(
            sub, "How predictable are modality and tissue from the head's PCs? (650M, raw)",
            "grouped CV prevents replicate/strand-mate leakage; dotted = permuted labels",
            FIGURES / "predictability_650M_raw.png")
    sub2 = pred[(pred["variant"] == "raw") & (pred["space"] == "modality_removed")]
    if not sub2.empty:
        plots.plot_predictability(
            sub2, "After projecting out the modality subspace (650M, raw)",
            "modality removed by least squares, not by deleting PC1",
            FIGURES / "predictability_modality_removed_650M_raw.png")

    if not args.quick:
        emb = umap_embed(heads["650M"].W)
        np.save(RESULTS / "umap_650M_raw.npy", emb)
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
        _log("umap done")

    # ---------------- Analysis 2: similarity structure ----------------
    Sraw = S[("650M", "raw")]
    for by, tag in [(["modality_coarse", "tissue_strict"], "modality_then_tissue"),
                    (["tissue_strict", "modality_coarse"], "tissue_then_modality"),
                    (["dataset", "modality_coarse"], "dataset_then_modality")]:
        simgrid.plot_full_matrix(
            Sraw, md, by, f"Cosine similarity, ordered {' → '.join(by)}",
            FIGURES / f"simgrid_{tag}_650M_raw.png")

    top_tis = list(md["tissue_strict"].value_counts().index[:14])
    sel = md[md["tissue_strict"].isin(top_tis)]
    Mt = simgrid.block_means(Sraw[np.ix_(sel["row"], sel["row"])],
                             sel["tissue_strict"].to_numpy(), top_tis)
    Mt.to_csv(RESULTS / "tissue_block_matrix_650M_raw.csv")
    simgrid.plot_block_matrix(
        Mt, "Mean cosine by tissue (14 most-sampled, 650M raw)",
        "diagonal is inflated by within-modality replicates — hence Analysis 3",
        FIGURES / "tissue_block_matrix_650M_raw.png")

    # ---------------- Analysis 3: the hypothesis test ----------------
    grid_rows, headline = [], None
    for name in CHECKPOINT_NAMES:
        for var in VARIANTS:
            for tissue_col in ("tissue_strict", "tissue_group"):
                res = run_test(S[(name, var)], md, tissue_col, "modality_coarse",
                               n_perm=n_perm, seed=0)
                agg = dict(res.aggregate)
                agg.update({"checkpoint": name, "variant": var})
                grid_rows.append(agg)
                _log(f"Δ[{name}/{var}/{tissue_col}] = {agg['observed_delta']:+.5f} "
                     f"z={agg['z_vs_null']:.1f} p={agg['p_permutation']:.4g}")
                if (name, var, tissue_col) == ("650M", "raw", "tissue_strict"):
                    headline = res
                    # cached so juicer.replot can redraw without re-permuting
                    np.save(RESULTS / "permutation_nulls_650M_raw_strict.npy",
                            res.nulls)
                    res.cells.to_csv(RESULTS / "delta_cells_650M_raw_strict.csv",
                                     index=False)
                    res.by_modality_pair.to_csv(
                        RESULTS / "delta_by_modality_pair_650M_raw_strict.csv",
                        index=False)

    # stricter null and stratified subsets, on the headline configuration
    extra = {}
    strict_null = run_test(S[("650M", "raw")], md, "tissue_strict",
                           "modality_coarse", n_perm=n_perm,
                           strata=("modality", "biosample_type", "dataset"), seed=0)
    extra["strict_strata_null"] = strict_null.aggregate

    for label, mask in [("tissue_only", md["biosample_type"] == "tissue"),
                        ("cell_line_only", md["biosample_type"] == "cell line")]:
        if mask.sum() > 50:
            r = run_test(S[("650M", "raw")], md, "tissue_strict", "modality_coarse",
                         n_perm=n_perm_side, seed=0, subset=mask.to_numpy())
            extra[label] = r.aggregate
            _log(f"Δ[{label}] = {r.aggregate['observed_delta']:+.5f} "
                 f"p={r.aggregate['p_permutation']:.4g}")

    # fine-grained modality (8 assays) as an extra robustness view
    fine = run_test(S[("650M", "raw")], md, "tissue_strict", "assay",
                    n_perm=n_perm_side, seed=0)
    extra["fine_assay_modality"] = fine.aggregate
    fine.by_modality_pair.to_csv(RESULTS / "delta_by_assay_pair_650M_raw.csv",
                                 index=False)

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(RESULTS / "sensitivity_grid.csv", index=False)
    summary["sensitivity_grid"] = grid_rows
    summary["extra"] = extra

    if headline is not None:
        plots.plot_delta_by_pair(
            headline.by_modality_pair,
            "Same-tissue advantage, holding the modality pair fixed (650M, raw)",
            "Δ > 0 means two tracks of different modalities from the same tissue "
            "are more similar than from different tissues",
            FIGURES / "delta_by_modality_pair_650M_raw.png")
        plots.plot_delta_hist(
            headline.cells, "Per-cell Δ distribution (650M, raw, strict tissue)",
            "one cell = one (tissue × modality-pair) comparison",
            FIGURES / "delta_cells_hist_650M_raw.png")
        plots.plot_null(
            headline.aggregate["observed_delta"],
            headline.aggregate["null_mean"], headline.aggregate["null_sd"],
            headline.nulls,
            "Observed Δ against the permutation null (650M, raw)",
            f"tissue labels shuffled within modality, {n_perm} permutations",
            FIGURES / "permutation_null_650M_raw.png")
        plots.plot_delta_by_pair(
            fine.by_modality_pair,
            "Same-tissue advantage by assay pair (650M, raw, 8 assays)",
            "finer modality definition; each bar holds one assay pair fixed",
            FIGURES / "delta_by_assay_pair_650M_raw.png")

    with open(RESULTS / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _log(f"wrote {RESULTS/'summary.json'}")


if __name__ == "__main__":
    main()
