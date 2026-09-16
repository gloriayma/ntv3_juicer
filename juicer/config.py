"""Paths and constants for the NTv3 track-head geometry analysis.

The NTv3 repos on HuggingFace are gated (401 unauthenticated) and no HF token
exists on this machine, but the post-trained checkpoints are already present on a
group-readable shared mount, so nothing needs to be downloaded.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
FIGURES = REPO / "figures"
RESULTS = REPO / "results"
CACHE = DATA / "cache"

HUB = Path("/mnt/weka/shared_datasets/evo3_checkpoint_import/hub")

# Human is species head 21. Verified two ways: the 9 non-empty species-head sizes
# match the per-species track counts in the metadata exactly, and head 21 is the
# only one with 7362 outputs (== the human track count).
HUMAN_HEAD_IDX = 21

_PREFIX = f"core.bigwig_head.species_heads.{HUMAN_HEAD_IDX}"
TENSORS = {
    "W": f"{_PREFIX}.head.weight",            # (7362, d) one row per track
    "b": f"{_PREFIX}.head.bias",              # (7362,)   per-track scalar offset
    "gamma": f"{_PREFIX}.layer_norm.weight",  # (d,) shared LayerNorm gain
    "beta": f"{_PREFIX}.layer_norm.bias",     # (d,) shared LayerNorm shift
}

CHECKPOINTS = {
    "650M": HUB / "models--InstaDeepAI--NTv3_650M_post/snapshots"
                  "/ad622051abbe9376bdea3f2727863fa559957e9f",
    "100M": HUB / "models--InstaDeepAI--NTv3_100M_post/snapshots"
                  "/b7292f0bd1b5b004d28561783a1890e1ec6f94fd",
}

METADATA_CSV = DATA / "functional_tracks_metadata.csv"
SPECIES = "human"
N_HUMAN_TRACKS = 7362

for _d in (FIGURES, RESULTS, CACHE):
    _d.mkdir(parents=True, exist_ok=True)
