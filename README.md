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

**Answer: it learns a shared organ code — one that holds across donors and assays — but it is
faint, uneven, and not linearly transferable.**

## Headline results

All numbers at **assay-group** level (ChIP-seq / RNA-seq / DNase-seq / CAGE), 650M checkpoint.

| Question | Answer |
|---|---|
| Does the head cluster by assay? | Essentially perfectly. Probe **0.999** (chance 0.333). |
| Is the tissue signal just a modality shortcut? | No. Tissue from the **assay label alone** = 0.019 vs 0.015 chance. |
| Tissue with assay held constant? | **0.45–0.69** balanced accuracy within a single assay (14–34× chance). |
| Same tissue ⇒ more aligned across assays? | **Δ = +0.0121, z = +28**, p < 0.005. Replicates on 100M. |
| Is it batch? | No. Same-lab AUROC 0.523 vs same-tissue 0.527; cross-lab-only tissue 0.531; z = 24.9 holding lab fixed. |
| Does a tissue classifier transfer across assays? | **Barely — 7.5% retention**, despite being competent within assay (24× chance). |
| Do tissue centroids align across assays? | **Yes, strongly.** ChIP→RNA top-1 **0.392** over 97 tissues (chance 0.010), nothing trained. |

### Granularity matters — this corrects the first version

"Different modality" was originally `assay:target`, but **635 of the ~639 modality labels are
ChIP-seq targets**, so that aggregate was mostly ChIP-vs-ChIP. Recomputed properly:

| Comparison level | Δ | z | AUROC |
|---|---|---|---|
| Cross fine-modality (mostly ChIP × ChIP) | +0.0660 | 199 | 0.613 |
| Cross assay family (RNA-seq, polyA kept apart) | +0.0246 | 24.7 | 0.564 |
| **Cross assay group (RNA assays merged — strictest)** | **+0.0121** | **28.2** | **0.556** |

RNA-seq and polyA-plus-RNA-seq are merged deliberately: they measure the same thing with
different library prep and were the most similar pair in every analysis, so merging removes the
easiest comparison rather than flattering the result.

The effect is also **uneven** — DNase × RNA 0.611, ChIP × RNA 0.558, and **ChIP × DNase 0.512
(0.447, below chance, on primary tissues only)**.

### Organ biology, not sample identity

ENCODE biosample terms split hairs (`liver` vs `right lobe of liver`, seven T-cell subsets,
three aortas), so "same tissue" could mean "same specific sample" rather than "same organ".

1. Regrouping by `organ_slims` **lowered** cross-assay AUROC (0.556 → 0.520), and a pairwise
   three-way split showed same-organ/different-biosample pairs were *not* more similar than
   different-organ pairs (AUROC 0.473–0.507). That reads as "no organ-level generalization".
2. **That conclusion was wrong — it was underpowered.** Per-track pairwise cosine is a weak
   instrument here; even the same-biosample cross-assay effect is only AUROC 0.50–0.53.

The properly powered test is **leave-biosample-out organ matching**: one centroid per
(organ, biosample, assay); for each query, rank target centroids *from different biosamples*
and ask whether the nearest shares the organ. A hit cannot come from sample identity.
Primary tissues only, blood bucket dropped:

| Direction | top-1 | chance | z |
|---|---|---|---|
| RNA-seq → ChIP-seq | **0.404** | 0.028 | +21.9 |
| ChIP-seq → RNA-seq | **0.295** | 0.025 | +13.6 |
| ChIP-seq → DNase-seq | 0.136 | 0.031 | +4.0 |
| DNase-seq → RNA-seq | 0.050 | 0.027 | +0.9 (n.s.) |

So a donor- and site-independent organ code is present and shared across assays — concentrated
in ChIP ↔ RNA, weak for ChIP ↔ DNase, absent for DNase ↔ RNA.

### Retracted from the first version

"73.8% of cross-modality neighbours share tissue, 43× chance" rested on ChIP-target-vs-ChIP-target
pairs. At group level it is 0.750 (31×) but on **only 128 neighbour pairs** — too thin to headline.
Superseded by centroid matching, which uses every tissue.

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

uv run python scripts/a9_family_mapping.py       # modality -> family/group mapping
uv run python scripts/final_group_analysis.py    # everything at assay-GROUP level (default)
uv run python scripts/final_group_analysis.py --split   # RNA assays kept apart
uv run python scripts/final_figs.py

uv run python scripts/a12_organ_level.py 200      # regroup by organ_slims
uv run python scripts/a13_decompose.py            # biosample / same-organ / diff-organ split
uv run python scripts/a14_organ_centroid.py       # leave-biosample-out organ matching
uv run python scripts/a13_fig.py && uv run python scripts/a14_fig.py
```

Granularity is defined in `src/ntv3_interp/grouping.py`; the full modality→group table is written
to `results/a9_modality_to_family.csv`.

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

- **Only three assay groups enter the tissue test.** CAGE has no resolvable biosample labels
  (FANTOM5's annotation URL 404s), so cross-group evidence rests on ChIP-seq, RNA-seq and
  DNase-seq — three pairs, one of which is null.
- **ChIP-seq is internally heterogeneous.** Treating 635 targets as one group means "within-ChIP"
  spans H3K9me3 and CTCF. Probably makes the cross-group contrast conservative, but it is not a
  clean control.
- **Donor was not resolved.** Lab is a coarse batch proxy, though lab is no longer a rival
  explanation at group level.
- **Coverage.** Assay resolved for 94.7% of 7,362 human tracks, biosample for 76.2%. 222 `kai*`
  tracks are of unknown provenance.
- **Per-track SNR is low enough to mislead.** Two analyses here pointed the wrong way before
  being redone at centroid level. Treat any pairwise cosine result on this head as underpowered
  until checked against an averaged version.
- **Organ matching rests on modest query counts** — 40–89 queries per direction in the strictest
  configuration, since it needs organs with >=2 independently-assayed biosamples.
- **Not validated against real data.** These are weight-space statistics; correlating head
  geometry against empirical track covariation would test whether it reflects biology.
