"""Curated organ families over the strict tissue labels (sensitivity analysis only).

These are **explicit member lists, not keyword patterns**, deliberately. A keyword
pass over the real labels mis-merged badly: a ``cortex`` pattern for brain swallowed
``renal cortex interstitium``, a ``renal`` pattern for kidney swallowed ``adrenal
gland``, and a ``gastr`` pattern for stomach swallowed ``gastrocnemius medialis``.
Listing members makes every merge auditable.

Scope is intentionally narrow: only anatomical subregions of one organ, plus primary
cells unambiguously derived from that organ. Immune cell subsets are *not* grouped --
a CD4 and a CD8 T cell are genuinely different cell types, not two views of one
tissue. Cell lines are never merged into an organ family; they keep their own label.

The strict-label analysis remains primary. This grouping exists so cross-dataset
pairs (ENCODE ``hepatocyte`` vs FANTOM5 ``hepatocyte``) and cross-subregion pairs
can contribute, and so the headline result can be shown not to hinge on one choice.
"""
from __future__ import annotations

import pandas as pd

ORGAN_GROUPS: dict[str, list[str]] = {
    "liver_family": [
        "liver", "left lobe of liver", "right lobe of liver", "hepatocyte",
    ],
    "lung_family": [
        "lung", "upper lobe of left lung", "left lung", "right lung",
    ],
    "colon_family": [
        "colon", "colonic mucosa", "large intestine", "sigmoid colon",
        "transverse colon", "mucosa of descending colon",
    ],
    "heart_family": [
        "heart", "heart left ventricle", "heart right ventricle",
        "left cardiac atrium", "right cardiac atrium",
        "right atrium auricular region", "cardiac muscle cell",
        "cardiac fibroblast", "cardiac myoblast", "cardiac atrium",
    ],
    "brain_family": [
        "brain", "caudate nucleus", "cerebellum", "cerebellar cortex",
        "dorsolateral prefrontal cortex", "frontal cortex", "occipital lobe",
        "temporal lobe", "parietal lobe", "cingulate gyrus", "hippocampus",
        "putamen", "substantia nigra", "hypothalamus", "amygdala",
        "middle frontal area 46", "layer of hippocampus",
    ],
    "skeletal_muscle_family": [
        "muscle of arm", "muscle of back", "muscle of leg", "muscle of trunk",
        "gastrocnemius medialis", "psoas muscle", "skeletal muscle tissue",
        "myotube", "skeletal muscle myoblast", "skeletal muscle satellite cell",
        "muscle of thigh", "skeletal muscle cell",
    ],
    "esophagus_family": [
        "esophagus", "esophagus muscularis mucosa",
        "esophagus squamous epithelium", "gastroesophageal sphincter",
    ],
    "pancreas_family": [
        "pancreas", "body of pancreas", "endocrine pancreas",
        "pancreatic islet", "islet of langerhans",
    ],
    "kidney_family": [
        "kidney", "left kidney", "right kidney", "kidney epithelial cell",
        "renal cortex interstitium", "left renal cortex interstitium",
        "right renal cortex interstitium", "renal pelvis", "left renal pelvis",
        "right renal pelvis",
    ],
    "stomach_family": [
        "stomach", "mucosa of stomach", "gastric mucosa",
    ],
    "uterus_family": [
        "uterus", "endometrium", "myometrium",
    ],
    "breast_family": [
        "breast epithelium", "mammary epithelial cell", "luminal epithelial cell of mammary gland",
        "mammary gland",
    ],
}

# label -> family, built once; a label absent here keeps its strict form.
_LABEL_TO_GROUP = {
    label: group for group, labels in ORGAN_GROUPS.items() for label in labels
}


def add_tissue_group(md: pd.DataFrame, cell_line_types=("cell line",)) -> pd.DataFrame:
    """Add a ``tissue_group`` column: organ family where curated, else strict label.

    Cell lines are excluded from family assignment so that e.g. a hepatoma line is
    never folded into ``liver_family``.
    """
    md = md.copy()
    is_cell_line = md["biosample_type"].isin(cell_line_types)
    mapped = md["tissue_strict"].map(_LABEL_TO_GROUP)
    md["tissue_group"] = mapped.where(~is_cell_line & mapped.notna(),
                                      md["tissue_strict"])
    return md


def coverage_report(md: pd.DataFrame) -> pd.DataFrame:
    """Which families actually captured tracks, and how many labels each merged."""
    rows = []
    for group in ORGAN_GROUPS:
        sub = md[md["tissue_group"] == group]
        if sub.empty:
            continue
        rows.append({
            "family": group,
            "n_tracks": len(sub),
            "n_strict_labels_merged": sub["tissue_strict"].nunique(),
            "n_modalities": sub["modality_coarse"].nunique(),
            "strict_labels": ", ".join(sorted(sub["tissue_strict"].unique())),
        })
    return pd.DataFrame(rows).sort_values("n_tracks", ascending=False)
