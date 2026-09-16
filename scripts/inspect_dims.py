import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd

tracks = pd.read_parquet("data/tracks.parquet")
for name, d in [
    ("all_biosamples", tracks),
    (
        "tissues_only",
        tracks[tracks.classification.isin(["tissue", "primary cell"])],
    ),
]:
    d = d[d.modality.notna() & d.term_id.notna()]
    vc = d.modality.value_counts()
    d = d[d.modality.isin(vc[vc >= 10].index)]
    npm = d.groupby("term_id").modality.nunique()
    d = d[d.term_id.isin(npm[npm >= 2].index)]
    print(f"\n=== {name} ===")
    print(f"tracks {len(d)}  tissues {d.term_id.nunique()}  modalities {d.modality.nunique()}")
    print("modality counts head:")
    print(d.modality.value_counts().head(12).to_string())
    print(f"tissues with >=2 modalities: {d.term_id.nunique()}")
    print("tracks per tissue (top):")
    print(d.term_id.value_counts().head(8).to_string())
