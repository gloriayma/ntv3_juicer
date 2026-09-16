"""A9: exactly which fine-grained modalities roll up into which assay family.

"Modality" in the first pass was `assay_term_name:target` -- so ChIP-seq:H3K4me3 and
ChIP-seq:CTCF were different modalities. "Assay family" is just the assay_term_name, which
is the level at which ChIP-seq x RNA-seq is a genuinely different measurement. This script
writes the full mapping so the grouping is auditable rather than implicit.

Run: uv run python scripts/a9_family_mapping.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

tracks = pd.read_parquet("data/tracks.parquet")
Path("results").mkdir(exist_ok=True)

res = tracks[tracks.modality.notna()].copy()

print(f"{len(tracks)} human tracks | {len(res)} with a resolved assay "
      f"({len(res) / len(tracks):.1%})\n")

fam = (
    res.groupby("assay_family")
    .agg(
        tracks=("track_id", "size"),
        distinct_modalities=("modality", "nunique"),
        biosamples=("term_id", "nunique"),
    )
    .sort_values("tracks", ascending=False)
)
print("ASSAY FAMILIES")
print(fam.to_string())

full = (
    res.groupby(["assay_family", "modality"])
    .agg(tracks=("track_id", "size"), biosamples=("term_id", "nunique"))
    .reset_index()
    .sort_values(["assay_family", "tracks"], ascending=[True, False])
)
full.to_csv("results/a9_modality_to_family.csv", index=False)
fam.to_csv("results/a9_family_summary.csv")

print("\n\nMODALITIES WITHIN EACH FAMILY")
for f in fam.index:
    sub = full[full.assay_family == f]
    print(f"\n{'=' * 72}\n{f}  —  {fam.loc[f, 'tracks']} tracks, "
          f"{len(sub)} distinct modalities\n{'=' * 72}")
    show = sub.head(30)
    for _, r in show.iterrows():
        lbl = r.modality.split(":", 1)[1] if ":" in r.modality else "(no target)"
        print(f"  {lbl:<34s} {r.tracks:>5d} tracks   {r.biosamples:>4d} biosamples")
    if len(sub) > 30:
        tail = sub.iloc[30:]
        print(f"  ... {len(tail)} further targets, {tail.tracks.sum()} tracks "
              f"(full list in results/a9_modality_to_family.csv)")

print("\n\nsaved results/a9_modality_to_family.csv and results/a9_family_summary.csv")
