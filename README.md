# ntv3_juicer

Does NTv3's per-track output head encode biological context (tissue / cell type), or
only assay modality?

NTv3's functional-track head has no inductive bias coupling tracks: it is literally
one independent linear projection per track. This repo tests whether tissue structure
nevertheless emerges in the learned projections, **holding the modality pair fixed**.

See [`FINDINGS.md`](FINDINGS.md) for results.

## Setup

```bash
uv venv --python 3.12
uv pip install -e .
```

No HuggingFace token or download is needed. The post-trained checkpoints are read
from a local shared mount (see `juicer/config.py`); only the four human-head tensors
are read out of each `model.safetensors`, not the whole checkpoint.

## Run

```bash
.venv/bin/python -m juicer.run_all           # full: 1000 permutations + UMAP
.venv/bin/python -m juicer.run_all --quick   # fewer permutations, no UMAP
```

Outputs land in `figures/` (PNG) and `results/` (CSV + `summary.json`).

## Layout

| Module | Role |
|---|---|
| `config.py` | Checkpoint paths, human head index (21), tensor names |
| `head.py` | Reads `W`, `b`, `gamma`, `beta`; exposes raw and gain-folded weights |
| `metadata.py` | Track metadata, **asserts** row order matches the checkpoint |
| `organ_groups.py` | Curated organ families (explicit member lists) for sensitivity |
| `controls.py` | Analysis 0 — alignment/structural gates |
| `geometry.py` | Analysis 1 — norms, PCA, label predictability, UMAP |
| `simgrid.py` | Analysis 2 — cosine matrix under different orderings |
| `tissue_test.py` | Analysis 3 — modality-matched Δ + permutation nulls |
| `validate.py` | Negative and synthetic-positive controls for the estimator |
| `viz.py`, `plots.py` | Validated palette, figure style, figures |

## Data provenance

`data/functional_tracks_metadata.csv` — from the ungated HF Space
`InstaDeepAI/ntv3_tracks`, path `data/functional_tracks_metadata.csv`, commit
`516378d`. 15,889 rows across 24 species; 7,362 are human.

The NTv3 weights are under a non-commercial license and are read read-only from the
shared mount; nothing from them is redistributed here.

### Checkpoint authenticity (verified)

The local checkpoints came from a third-party import, and the whole analysis rests on
them, so they were checked against the official gated HF release
(`InstaDeepAI/NTv3_650M_post`, revision `ad62205`):

| Artifact | Check | Result |
|---|---|---|
| `config.json` | sha256 vs official | identical (`a770f4b2…`) |
| `model.safetensors` header (64,264 B) | byte compare | identical |
| human head weight (`species_heads.21.head.weight`, 45,232,128 B) | byte compare via HTTP range | **identical** |

So the tensors analysed here are the released ones, unmodified.

## Two methodological notes

1. **Rows are L2-normalised, never per-row z-scored.** Subtracting a row's own mean
   rotates the vector, destroying the direction under study.
2. **The LayerNorm gain matters.** The head is
   `softplus(w_t·(γ⊙x̂ + β) + b_t)`, so the functionally faithful comparison vector is
   `w_t ⊙ γ`. γ spans 0.23–5.72 in the 650M head, so this is not cosmetic. Every
   analysis runs on both `raw` and `gain`.
