"""Human track metadata, aligned to the head's row order.

Provenance: ``data/functional_tracks_metadata.csv`` comes from the *ungated* HF Space
``InstaDeepAI/ntv3_tracks`` (path ``data/functional_tracks_metadata.csv``, commit
``516378d``). 15,889 rows across 24 species; the 7,362 ``specie == "human"`` rows are
the ones that matter here.

The load path asserts that those human rows appear in exactly the same order as
``config.json["bigwigs_per_species"]["human"]``, so head weight row *i* corresponds to
metadata row *i*. Everything downstream depends on this, so it is a hard failure.
"""
from __future__ import annotations

import re

import pandas as pd

from .config import METADATA_CSV, N_HUMAN_TRACKS, SPECIES

# assay -> coarse modality. Kept explicit rather than inferred so the grouping is
# auditable: CAGE is held apart from RNA-seq because it measures 5' ends.
COARSE_MODALITY = {
    "total RNA-seq": "RNA",
    "polyA plus RNA-seq": "RNA",
    "RNA-seq": "RNA",
    "CAGE": "CAGE",
    "DNase-seq": "Accessibility",
    "ATAC-seq": "Accessibility",
    "Histone ChIP-seq": "HistoneChIP",
    "TF ChIP-seq": "TFChIP",
}

_WS = re.compile(r"\s+")
_STRAND_SUFFIX = re.compile(r"_(P|M)$")


def normalize_tissue(raw: str) -> str:
    """Normalise a free-text tissue label without merging distinct tissues.

    Casefolds, strips trailing punctuation/whitespace and collapses internal runs of
    whitespace. This alone reconciles FANTOM5's trailing commas (``"Hepatocyte,"`` ->
    ``hepatocyte``) and inconsistent capitalisation (``"Mesenchymal Stem Cells -
    hepatic,"`` vs ``"Mesenchymal stem cells - hepatic,"``). It deliberately does
    *not* merge ``liver`` with ``left lobe of liver`` -- that is the job of the
    curated organ groups in :mod:`juicer.organ_groups`.
    """
    s = str(raw).strip()
    s = s.strip(" ,;\t")
    s = _WS.sub(" ", s)
    return s.casefold()


def experiment_base(file_id: str) -> str:
    """Strip a trailing ``_P``/``_M`` strand suffix to get the experiment identity.

    ENCODE stranded RNA-seq ships as ``ENCSR580GSX_P`` / ``ENCSR580GSX_M`` and FANTOM5
    CAGE as ``CNhs12331_P`` / ``_M``: two bigwigs of the *same* experiment. Collapsing
    the suffix identifies those mates, which we use both as the alignment gate and as
    pairs to exclude from the tissue test.
    """
    return _STRAND_SUFFIX.sub("", str(file_id))


def load_metadata(track_ids: list[str]) -> pd.DataFrame:
    """Load human track metadata in head-row order, asserting alignment.

    ``track_ids`` is ``config.json["bigwigs_per_species"]["human"]`` from the
    checkpoint being analysed.
    """
    df = pd.read_csv(METADATA_CSV, dtype=str, keep_default_na=False)
    human = df[df["specie"] == SPECIES].reset_index(drop=True)

    if len(human) != N_HUMAN_TRACKS:
        raise AssertionError(f"metadata has {len(human)} human rows, expected {N_HUMAN_TRACKS}")
    if human["file_id"].duplicated().any():
        raise AssertionError("duplicate file_id among human rows")

    csv_ids = human["file_id"].tolist()
    if set(csv_ids) != set(track_ids):
        raise AssertionError("human file_id set differs from checkpoint track list")
    if csv_ids != track_ids:
        n_bad = sum(a != b for a, b in zip(csv_ids, track_ids))
        raise AssertionError(
            f"human metadata row order differs from checkpoint track order "
            f"at {n_bad} positions; refusing to proceed on a misaligned join"
        )

    human = human.rename(columns={"specie": "species"})
    human["assay"] = human["assay"].str.strip()
    human["modality_fine"] = human["assay"]
    human["modality_coarse"] = human["assay"].map(COARSE_MODALITY)
    if human["modality_coarse"].isna().any():
        missing = sorted(human.loc[human["modality_coarse"].isna(), "assay"].unique())
        raise AssertionError(f"assay(s) with no coarse modality mapping: {missing}")

    human["tissue_strict"] = human["tissue"].map(normalize_tissue)
    human["expt_base"] = human["file_id"].map(experiment_base)
    human["row"] = range(len(human))
    return human
