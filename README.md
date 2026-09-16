# Does the NTv3 track head factorize tissue from assay modality?

NTv3's functional-track head is the thinnest thing it could be. From
[`heads.py`](https://github.com/instadeepai/nucleotide-transformer/blob/main/nucleotide_transformer_v3/heads.py):

```python
class LinearHead(nnx.Module):
    def __init__(self, embed_dim, num_labels, *, rngs):
        self.layer_norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.head = nnx.Linear(in_features=embed_dim, out_features=num_labels, rngs=rngs)
    def __call__(self, x):
        return jax.nn.softplus(self.head(self.layer_norm(x)))
```

Track `t`'s prediction is `softplus(w_t · LN(x) + b_t)`, where `w_t` is one column of a single
weight matrix. **Nothing ties RNA-seq tracks to each other, and nothing tells the model that a
liver ATAC track and a liver RNA track share a tissue.** Any such structure had to be learned
from the prediction objective alone.

**Answer: it emerges anyway — but modality dominates, and tissue is a second-order effect.**

## Headline results

| Question | Answer |
|---|---|
| Does the head cluster by modality? | Overwhelmingly. Linear probe **0.939** balanced accuracy over 20 assay types (chance 0.05). |
| Does it also encode tissue? | Yes. Probe **0.311** over 71 tissues (chance 0.014, 22× chance). |
| Same tissue ⇒ more aligned, modality held fixed? | **Δ = +0.066, z = +199, p < 0.005.** Replicates on both checkpoints. |
| Is it just batch/lab? | No. Holding lab fixed too, the null moves to +0.0047 — **7% of the effect** — leaving z = +148. |
| Is it just track amplitude? | No. Amplitude-only probe gets 0.213 vs 0.939 for the full vector. |

Among the 10 nearest neighbours of a track, **73.8% of the different-assay neighbours share its
biosample**, against a 1.7% chance rate — a **43× enrichment**.

## The preprocessing correction

The linear sits directly on a `LayerNorm`, whose output is **exactly zero-mean across
`embed_dim`**. Expanding, with `v_t := γ ⊙ w_t`:

```
w_t · LN(x) + b_t = (γ ⊙ w_t) · n(x) + (w_t · β + b_t),    ⟨1, n(x)⟩ = 0
```

so `(v_t + c·1) · n(x) == v_t · n(x)` for every input and every `c`. **The all-ones component of
`v_t` is a gauge freedom** — it provably cannot change a prediction, and the induced shift in the
constant term is absorbed by the free bias. The identifiable object is therefore

```
ṽ_t = center(γ ⊙ w_t)
```

and `cos(ṽ_i, ṽ_j)` is the Pearson correlation of `γ⊙w_i` and `γ⊙w_j`.

Consequences: mean-centering is **mandatory** (gauge removal), not a stylistic choice; the `γ`
rescaling matters because LayerNorm reweights every coordinate before the dot product; and `‖ṽ_t‖`
(the track's effective gain) is real information but is reported separately rather than allowed to
contaminate a cosine.

`scripts/a0_gauge.py` verifies this numerically rather than asserting it — relative prediction
drift under the gauge shift is **3.4e-12**. **Honest caveat:** the correction is provably right but
empirically modest. Raw and corrected similarity matrices correlate at **r = 0.969**, so it does not
overturn any conclusion here; it reaches 48% of a vector's variance only for individual outlier
tracks.

## Data

No model inference is needed — only four tensors, pulled from `model.safetensors` by HTTP range
request using the safetensors header offsets (tens of MB rather than 2.6 GB, and no torch/JAX).

- Human is a **separate head module** (`bigwig_head.species_heads.{i}`), not rows of a shared
  matrix. Human is index 21; **7,362 tracks**; 1,536-d (650M) / 768-d (100M).
- Track IDs come from `config.json → bigwigs_per_species["human"]`: 5,609 ENCODE accessions,
  1,276 FANTOM5 CAGE, 169 GEO, 84 GTEx, 222 unresolved `kai*`.
- Modality and biosample resolved from the **public ENCODE API** (no auth): 94.7% modality,
  76.2% biosample coverage.
- `_M`/`_P` suffixes are strand splits of one experiment. They cannot contaminate the headline
  test: the test only compares *different* modalities, and a strand pair always shares one.

## Reproducing

NTv3 repos are gated. Accept the licence on
[NTv3_650M_post](https://huggingface.co/InstaDeepAI/NTv3_650M_post) and
[NTv3_100M_post](https://huggingface.co/InstaDeepAI/NTv3_100M_post), then:

```bash
echo -n "hf_..." > ~/.cache/huggingface/token
uv sync

uv run python src/ntv3_interp/fetch_head.py --repo InstaDeepAI/NTv3_650M_post
uv run python src/ntv3_interp/fetch_head.py --repo InstaDeepAI/NTv3_100M_post
uv run python src/ntv3_interp/metadata.py        # ENCODE resolution, cached

uv run python scripts/a0_gauge.py                # gauge check + diagnostics
uv run python scripts/a1_geometry.py             # PCA, UMAP, probes, kNN purity
uv run python scripts/a2_grids.py 200            # similarity grids + modality removal
uv run python scripts/a3_tissue.py 200           # the headline test
uv run python scripts/a3b_confounds.py           # lab/batch confound
uv run python scripts/a3c_decisive.py 200        # permute within modality x lab
uv run python scripts/a5_summary_figs.py
```

## Method notes

- **Δ statistic.** `Δ_c^(m1,m2)` = mean same-tissue cosine minus mean different-tissue cosine, with
  the ordered modality pair held fixed. Fixing the pair is what stops "RNA-seq and DNase are
  intrinsically more alike than RNA-seq and ChIP" from masquerading as tissue structure.
- **Computed without materializing S.** Since `S = U Uᵀ`, any block sum is
  `(Σ_{i∈A} u_i)·(Σ_{j∈B} u_j)`, so every statistic reduces to dot products of per-cell sum vectors.
- **Null.** Tissue labels are permuted *within* modality strata (and, in `a3c`, within
  modality × lab), preserving composition and destroying only tissue alignment. Reordering the
  similarity matrix would not be a null at all — `S` is unchanged by permuting its rows.
- **Removing modality** is done by subtracting each modality's centroid, not by deleting PC1.
  Modality is categorical and occupies many dimensions — **PC1 explains only 9.9%** of variance.

## Limitations

- Lab is a coarse batch proxy; ENCODE donor was not resolved, so residual donor confounding is not
  excluded. Lab-only accounts for ~7% of the effect.
- FANTOM5's sample annotation URL 404s, so the 1,276 CAGE tracks carry modality but no tissue and
  are excluded from the tissue test.
- 222 `kai*` tracks are of unknown provenance and are unresolved throughout.
- Absolute effect sizes are modest (cross-modality AUROC 0.57–0.61). The claim is that tissue
  structure is *present and robust*, not that it is strong.
- `RNA-seq` and `polyA plus RNA-seq` are counted as distinct modalities though they are near
  neighbours biologically; the per-pair breakdown shows the effect holds across genuinely
  different assays (distinct histone marks, ChIP × RNA-seq, ChIP × DNase), so this does not drive
  the result.
