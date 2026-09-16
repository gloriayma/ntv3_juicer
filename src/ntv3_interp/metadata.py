"""Resolve NTv3 human track IDs to modality and biosample (tissue/cell type).

Track-ID composition of the 7,362 human tracks:
    5,609  ENCODE experiment accessions (ENCSR*, some with _M/_P strand suffix)
    1,276  FANTOM5 CAGE (CNhs*, always _M/_P strand-split)
      222  ``kai*``   -- unknown provenance, left unresolved
      169  GEO samples (GSM*)
       ~30 GTEx (GTEX-donor-tissue-SM-*)

ENCODE is the workhorse and its REST API is fully public (no auth). FANTOM5 contributes a
cleanly-labelled extra modality (CAGE), which is useful precisely because the headline test
needs *cross-modality* pairs.

The ``_M``/``_P`` suffixes are minus/plus strand of the SAME underlying experiment. They are
near-duplicate tracks, so downstream code must group by ``base_id`` and never count a strand
pair as independent evidence that two tracks from one tissue agree.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ENCODE_SEARCH = "https://www.encodeproject.org/search/"
CACHE = Path("data/encode_cache.json")

STRAND_RE = re.compile(r"_(M|P)$")


def split_strand(track_id: str) -> tuple[str, str | None]:
    """``ENCSR561FEE_P`` -> ``("ENCSR561FEE", "P")``."""
    m = STRAND_RE.search(track_id)
    if m:
        return track_id[: m.start()], m.group(1)
    return track_id, None


def source_of(base_id: str) -> str:
    if base_id.startswith("ENCSR"):
        return "ENCODE"
    if base_id.startswith("CNhs"):
        return "FANTOM5"
    if base_id.startswith("GSM"):
        return "GEO"
    if base_id.startswith("GTEX"):
        return "GTEx"
    if base_id.startswith("kai"):
        return "kai"
    return "other"


# --------------------------------------------------------------------------------------
# ENCODE
# --------------------------------------------------------------------------------------

_FIELDS = [
    "accession",
    "assay_term_name",
    "assay_title",
    "target.label",
    "biosample_ontology.term_id",
    "biosample_ontology.term_name",
    "biosample_ontology.classification",
    "biosample_ontology.organ_slims",
    "biosample_ontology.cell_slims",
    "biosample_ontology.system_slims",
    "biosample_summary",
    "lab.title",
    "status",
]


def _get_json(url: str, retries: int = 4) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_encode(accessions: list[str], batch: int = 60) -> dict[str, dict]:
    """Batch-query the ENCODE portal, caching to disk so reruns are free."""
    cache: dict[str, dict] = {}
    if CACHE.exists():
        cache = json.loads(CACHE.read_text())

    todo = sorted(set(accessions) - set(cache))
    for i in range(0, len(todo), batch):
        chunk = todo[i : i + batch]
        params = [("type", "Experiment"), ("limit", "all"), ("format", "json")]
        params += [("field", f) for f in _FIELDS]
        params += [("accession", a) for a in chunk]
        url = ENCODE_SEARCH + "?" + urllib.parse.urlencode(params)
        try:
            res = _get_json(url)
            for g in res.get("@graph", []):
                cache[g["accession"]] = g
        except Exception as e:  # noqa: BLE001 - one bad batch shouldn't kill the run
            print(f"  ! batch {i // batch} failed: {e}")
        # Record misses so we don't re-query them every run.
        for a in chunk:
            cache.setdefault(a, {})
        print(f"  ENCODE {min(i + batch, len(todo))}/{len(todo)}", end="\r")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    return cache


def _modality_from_encode(rec: dict) -> str | None:
    """Assay name, with the ChIP/eCLIP target folded in.

    H3K4me3 ChIP and CTCF ChIP are wildly different biology; collapsing both to "ChIP-seq"
    would make the modality-matched test compare things that aren't comparable.
    """
    assay = rec.get("assay_term_name") or rec.get("assay_title")
    if not assay:
        return None
    target = (rec.get("target") or {}).get("label")
    if target:
        return f"{assay}:{target}"
    return assay


# --------------------------------------------------------------------------------------
# FANTOM5
# --------------------------------------------------------------------------------------

FANTOM_SDRF = (
    "https://fantom.gsc.riken.jp/5/datafiles/reprocessed/hg38_latest/basic/"
    "HumanSamples2.0.sdrf.tsv"
)
FANTOM_CACHE = Path("data/fantom5_samples.tsv")


def fetch_fantom5() -> dict[str, dict]:
    """Map CNhs id -> sample annotation. Returns {} if the file is unreachable.

    FANTOM5 is a bonus modality, not load-bearing, so a fetch failure degrades to
    "CAGE with unknown tissue" rather than aborting the run.
    """
    if not FANTOM_CACHE.exists():
        try:
            req = urllib.request.Request(FANTOM_SDRF, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                FANTOM_CACHE.parent.mkdir(parents=True, exist_ok=True)
                FANTOM_CACHE.write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"  ! FANTOM5 annotation unavailable ({e}); CAGE tissue will be unknown")
            return {}

    try:
        df = pd.read_csv(FANTOM_CACHE, sep="\t", dtype=str, on_bad_lines="skip")
    except Exception as e:  # noqa: BLE001
        print(f"  ! FANTOM5 parse failed ({e})")
        return {}

    # Column naming varies across FANTOM releases; find the CNhs id column empirically.
    id_col = None
    for c in df.columns:
        if df[c].astype(str).str.contains(r"CNhs\d+", na=False).any():
            id_col = c
            break
    if id_col is None:
        return {}

    name_cols = [c for c in df.columns if re.search(r"name|tissue|organ|source", c, re.I)]
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        m = re.search(r"(CNhs\d+)", str(row[id_col]))
        if not m:
            continue
        label = next((str(row[c]) for c in name_cols if pd.notna(row.get(c))), None)
        out[m.group(1)] = {"term_name": label}
    return out


# --------------------------------------------------------------------------------------
# Build the table
# --------------------------------------------------------------------------------------


def build_track_table(track_ids: list[str]) -> pd.DataFrame:
    bases = [split_strand(t) for t in track_ids]
    enc = [b for b, _ in bases if b.startswith("ENCSR")]

    print(f"resolving {len(set(enc))} unique ENCODE accessions ...")
    encode = fetch_encode(enc)
    print()
    print("resolving FANTOM5 CAGE samples ...")
    fantom = fetch_fantom5()

    rows = []
    for idx, (tid, (base, strand)) in enumerate(zip(track_ids, bases)):
        src = source_of(base)
        modality = biosample = term_id = classification = organ = lab = None

        if src == "ENCODE":
            rec = encode.get(base) or {}
            if rec:
                modality = _modality_from_encode(rec)
                bo = rec.get("biosample_ontology") or {}
                biosample = bo.get("term_name")
                term_id = bo.get("term_id")
                classification = bo.get("classification")
                slims = bo.get("organ_slims") or []
                organ = "|".join(sorted(slims)) if slims else None
                lab = (rec.get("lab") or {}).get("title")
        elif src == "FANTOM5":
            modality = "CAGE"
            rec = fantom.get(base) or {}
            biosample = rec.get("term_name")
            term_id = f"FANTOM:{biosample}" if biosample else None
            classification = "fantom5_sample"
        elif src == "GTEx":
            modality = "RNA-seq"
            classification = "tissue"

        rows.append(
            {
                "index": idx,
                "track_id": tid,
                "base_id": base,
                "strand": strand,
                "source": src,
                "modality": modality,
                "biosample": biosample,
                "term_id": term_id,
                "classification": classification,
                "organ_slims": organ,
                "lab": lab,
            }
        )

    df = pd.DataFrame(rows)
    # Coarse assay family (ChIP-seq:CTCF -> ChIP-seq), useful for plots where the
    # target-level split is too granular to read.
    df["assay_family"] = df["modality"].str.split(":").str[0]
    return df


def main() -> None:
    import argparse

    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="data/head_NTv3_650M_post.npz")
    ap.add_argument("--out", default="data/tracks.parquet")
    args = ap.parse_args()

    d = np.load(args.head, allow_pickle=True)
    track_ids = [str(x) for x in d["track_ids"]]
    df = build_track_table(track_ids)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    n = len(df)
    print(f"\ntracks                 : {n}")
    print(f"modality resolved      : {df.modality.notna().sum()} ({df.modality.notna().mean():.1%})")
    print(
        f"biosample resolved     : {df.biosample.notna().sum()} "
        f"({df.biosample.notna().mean():.1%})"
    )
    print(f"strand-paired tracks   : {df.strand.notna().sum()}")
    print(f"\nsources:\n{df.source.value_counts().to_string()}")
    print(f"\ntop assay families:\n{df.assay_family.value_counts().head(15).to_string()}")
    print(f"\ntop biosamples:\n{df.biosample.value_counts().head(15).to_string()}")
    print(f"\nclassification:\n{df.classification.value_counts().to_string()}")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
