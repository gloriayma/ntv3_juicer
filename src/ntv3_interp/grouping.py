"""Assay grouping levels.

Three granularities, from finest to coarsest:

  modality      assay_term_name:target   -- ChIP-seq:H3K4me3, ChIP-seq:CTCF, RNA-seq, ...
                635 of the ~639 distinct values are ChIP-seq targets, so "cross-modality"
                at this level is overwhelmingly ChIP-vs-ChIP.

  assay_family  assay_term_name          -- ChIP-seq, RNA-seq, polyA plus RNA-seq,
                DNase-seq, CAGE.

  assay_group   assay_family, with the two RNA assays merged (default).
                ENCODE's "RNA-seq" (total RNA) and "polyA plus RNA-seq" measure the same
                thing with different library prep, and they are by far the most similar
                pair in every analysis. Treating them as distinct families lets an
                essentially-duplicate comparison inflate any cross-family aggregate.
                Merging is therefore the conservative choice: it removes the easiest pair
                and reclassifies those same-tissue RNA/polyA comparisons as within-group,
                excluding them from the cross-group test entirely.
"""

from __future__ import annotations

import pandas as pd

RNA_ASSAYS = {"RNA-seq", "polyA plus RNA-seq"}
RNA_MERGED = "RNA-seq (total + polyA)"


def add_assay_group(df: pd.DataFrame, merge_rna: bool = True) -> pd.DataFrame:
    """Add an ``assay_group`` column at the chosen granularity."""
    out = df.copy()
    if merge_rna:
        out["assay_group"] = out.assay_family.where(
            ~out.assay_family.isin(RNA_ASSAYS), RNA_MERGED
        )
    else:
        out["assay_group"] = out.assay_family
    return out


def label(merge_rna: bool) -> str:
    return "merged_rna" if merge_rna else "split_rna"
