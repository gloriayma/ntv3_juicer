# Does NTv3's track head learn biology, or just modality?

**Both — but very unequally.** Modality dominates the head's geometry almost
completely. Tissue identity is also there, it is statistically unambiguous, and it
survives every control we could throw at it — but it is a second-order effect
riding on top of a geometry organised by assay.

Headline number: holding the pair of modalities fixed, two tracks from the **same**
tissue are more similar than two tracks from **different** tissues by

> **Δ = +0.0218 cosine**, z = 30.8 against a permutation null
> (null mean 5.5 × 10⁻⁶, SD 7.1 × 10⁻⁴), p = 0.000999 (1/1001, the floor for 1000
> permutations), across 618 (tissue × modality-pair) cells spanning 224 tissues.
> 75.9% of cells are positive. Pooled Cohen's d = 0.50, implying ≈64% chance that a
> same-tissue cross-modality pair beats a different-tissue one.

So NTv3 has spontaneously learned *some* shared biological-context representation
despite being given no tissue × modality factorization. The honest framing is that
it learned a strong assay representation and a weak tissue representation.

---

## 1. The premise is correct, and confirmed from source

Not inferred — read from the vendored modeling code
(`dev/arc/src/scnona/ntv3_base/modeling_ntv3_posttrained.py:483`):

```python
class LinearHead(nn.Module):
    """A linear head that predicts one scalar value per track."""
    def __init__(self, embed_dim, num_labels):
        self.layer_norm = LayerNormFP32(embed_dim)      # shared across all tracks
        self.head = nn.Linear(embed_dim, num_labels)    # one row per track
    def forward(self, x):
        x = self.layer_norm(x.float()); x = self.head(x); return F.softplus(x)
```

`MultiSpeciesHead` (line 506) is a plain `ModuleList` of one independent `LinearHead`
per species. Each track is exactly one learned linear projection plus a scalar bias,
with **zero cross-track coupling** — nothing ties two RNA-seq projections together,
and nothing tells the model that an RNA-seq and an ATAC track share a tissue.

The paper agrees: *"species-specific functional-track heads that output the set of
tracks available for each species"*, with species conditioning via adaptive
LayerNorm — so species conditioning never enters the head itself.

Because the `layer_norm` is shared by every track in a species head, all 7,362 human
weight rows live in one common basis, which is what makes cosine comparison
well-posed.

## 2. What was verified before anything was interpreted

| Check | Result |
|---|---|
| Head tensor | `core.bigwig_head.species_heads.21.head.weight` = `[7362, 1536]` (650M), `[7362, 768]` (100M) |
| Species identification | the 9 non-empty head sizes match per-species track counts exactly (7362 human, 2450 mouse, 1899 arabidopsis, …) |
| **Row alignment** | `config.json["bigwigs_per_species"]["human"]` is set-**and** order-identical to the metadata's 7,362 human rows, for **both** checkpoints. Asserted in code; hard failure otherwise |
| **Checkpoint authenticity** | `config.json` sha256, the safetensors header, and the 45,232,128-byte human head weight are all **byte-identical** to the official gated HF release (`ad62205`) |
| Modality blocks (Gate A) | within-assay mean cosine 0.321 vs between-assay 0.088 (Δ 0.232) |
| Biological ordering (Gate B) | cos(ATAC, DNase) = 0.302 ≫ cos(ATAC, total RNA) = 0.036; cos(RNA-seq, polyA+) = 0.318 ≫ cos(RNA-seq, ATAC) = 0.060 |
| **Cross-model agreement** | assay-block geometry correlates r = **0.990** (raw) / **0.997** (gain) between the independently trained 650M (d=1536) and 100M (d=768) heads |

Gate B matters because a scrambled row mapping could not put ATAC next to DNase and
the RNA protocols next to each other; and cross-model agreement cannot come from a
shared bug in our own join.

### A control we designed, ran, and had to discard

The plan proposed the 1,207 same-experiment ±strand mates (`ENCSR580GSX_P`/`_M`,
`CNhs12331_P`/`_M`) as a positive control, expecting near-identical head rows. **That
reasoning was wrong.** The two strand tracks of one stranded RNA-seq experiment
measure *complementary* signals — a plus-strand gene appears in the plus track and is
absent from the minus track — so the head must predict different things for them.

Measured: median cosine **0.064** raw (background 0.058) and **−0.059** gain-folded
(background +0.038). Mates are no more similar than random pairs, and are faintly
*anti*-aligned once the LayerNorm gain is folded in.

This is a finding, not a failure: **the head encodes what a track measures, not which
experiment it came from.** It also shows up geometrically — in the UMAP, CAGE splits
into two clusters and RNA into three, which is the strand split.

### The estimator itself was validated

| Control | Δ | p |
|---|---|---|
| Negative (tissue labels shuffled once, within modality) | +0.00022 | 0.42 |
| Synthetic positive (small per-tissue vector injected) | +0.1006 | 0.0033 (floor) |

The estimator is unbiased under the null and has ample power, so a null result would
have meant absence of signal rather than lack of sensitivity.

## 3. Modality dominates — and it is *not* PC1

| Predicted from top-k PCs (grouped CV) | k=2 | k=5 | k=10 | k=20 | k=50 | chance |
|---|---|---|---|---|---|---|
| modality (5 classes) | 0.748 | 0.925 | **0.985** | 0.990 | 0.989 | 0.199 |
| assay (8 classes) | 0.438 | 0.761 | 0.889 | 0.961 | 0.970 | 0.123 |
| tissue (152 classes) | 0.013 | 0.019 | 0.053 | 0.160 | **0.356** | 0.0064 |

Two things fall out of this:

- **"Is PC1 the modality?" No.** Modality needs ~10 PCs to saturate (74.8% at 2 PCs →
  98.5% at 10). PC1 explains only **7.0%** of variance, the top 10 only 29.7%. The
  geometry is high-dimensional, so deleting PC1 would have removed very little and
  would have been the wrong move.
- **Tissue is genuinely encoded**, at 0.356 balanced accuracy over 152 classes against
  a 0.0064 permuted baseline — ~55× chance — but far more weakly than modality.

Grouped CV matters here: folds are grouped by experiment/tissue so that replicates
and ±strand mates cannot leak across folds and inflate accuracy.

## 4. Tissue information is *orthogonal* to modality

Rather than deleting a PC, we regressed the one-hot modality design matrix out of the
weight rows (removing everything modality can linearly explain) and re-measured:

| After projecting out the modality subspace | k=10 | k=20 | k=50 |
|---|---|---|---|
| modality | 0.242 | 0.208 | 0.225 (≈ chance 0.199) |
| tissue | 0.060 | 0.176 | **0.355** (was 0.356) |

Modality collapses to chance — the removal worked — while **tissue predictability is
completely unchanged**. Tissue information is not a by-product of modality; it lives
in a modality-orthogonal part of the head geometry.

## 5. Similarity structure

Mean cosine by assay (650M, raw) is biologically coherent rather than arbitrary:

- ATAC ↔ DNase **0.302** (both assay open chromatin) — the largest cross-assay value
- RNA-seq ↔ polyA+ **0.318**, RNA-seq ↔ total RNA **0.243**, polyA+ ↔ total **0.197**
- CAGE sits nearer accessibility (0.11 / 0.10) than RNA (0.05–0.07), consistent with
  CAGE marking active promoters
- Within-assay self-similarity is *low* for Histone ChIP (**0.090**) and TF ChIP
  (**0.133**) — different marks and different TFs are genuinely different targets,
  so "same assay" is not "same thing"

Ordering the same matrix by modality→tissue shows crisp modality blocks; ordering by
tissue→modality does not produce comparable tissue blocks. Per the plan, this is
recorded as a **visual cue only** — reordering does not change the matrix and is not a
statistical null. The null is in §6.

Per-track scalars also track assay: weight-row norms run from ~5.6 (CAGE, sparse and
peaky) down to ~1.2 (TF ChIP, low amplitude), i.e. the norm largely encodes dynamic
range. This is why rows are L2-normalised for direction work and the norm is reported
separately — and never per-row z-scored, which would have rotated the vectors and
destroyed the very directions under study.

## 6. The hypothesis test

For each unordered modality pair (m ≠ m′) and tissue c with ≥1 track in each:

```
within_c   = mean cos over { i in (c,m),  j in (c,m')  }
between_c  = mean cos over { i in (c,m),  j in (~c,m') } ∪ { i in (c,m'), j in (~c,m) }
Δ_c        = within_c - between_c
```

Unique unordered pairs only (the C(n,2) correction — no self-pairs, no double
counting); same-experiment pairs dropped outright; every (tissue × modality-pair) cell
weighted equally so K562's 517 tracks cannot dominate. The null shuffles tissue labels
**within modality strata**, which preserves every (tissue, modality) group size
exactly and changes only whether a tissue's identity corresponds across modalities.

### By modality pair (650M, raw, strict tissue)

| modality pair | Δ | tissues | % positive |
|---|---|---|---|
| Accessibility ↔ TF ChIP | **+0.064** | 71 | 92% |
| Histone ChIP ↔ TF ChIP | **+0.043** | 105 | 88% |
| Histone ChIP ↔ RNA | +0.015 | 128 | 84% |
| Accessibility ↔ RNA | +0.012 | 124 | 77% |
| RNA ↔ TF ChIP | +0.007 | 78 | 67% |
| Accessibility ↔ Histone ChIP | +0.004 | 99 | 52% |
| CAGE ↔ anything | −0.017 … +0.007 | **2–4** | — |

The CAGE rows are **uninterpretable, not negative results**: CAGE is entirely FANTOM5,
so under strict tissue matching only 2–4 tissues overlap another modality. Ignore them.

Under the finer 8-assay definition the largest effects are within-family protocol
pairs (RNA-seq ↔ polyA+ **+0.175**, polyA+ ↔ total RNA **+0.142**, ATAC ↔ DNase
**+0.103**, all 100% of cells positive). Those are the easiest case — same underlying
biology, different protocol — which is why the coarse grouping that merges the RNA
protocols is used for the headline.

### Sensitivity: the result does not depend on any analysis choice

| checkpoint | variant | tissue def | Δ | z | Cohen's d |
|---|---|---|---|---|---|
| 650M | raw | strict | +0.0218 | 30.8 | 0.50 |
| 650M | raw | organ groups | +0.0225 | 28.3 | 0.53 |
| 650M | gain | strict | +0.0175 | 24.2 | 0.40 |
| 650M | gain | organ groups | +0.0180 | 22.2 | 0.42 |
| 100M | raw | strict | +0.0178 | 20.8 | 0.35 |
| 100M | raw | organ groups | +0.0183 | 18.9 | 0.37 |
| 100M | gain | strict | +0.0157 | 18.6 | 0.32 |
| 100M | gain | organ groups | +0.0161 | 16.9 | 0.33 |

All eight cells: p = 0.000999 (permutation floor). No disagreement to flag. Folding in
the LayerNorm gain shrinks the effect by ~20% but never changes its sign or
significance. **The effect is consistently stronger in the 650M head than the 100M
head** (d = 0.50 vs 0.35), i.e. it strengthens with scale.

### Stratified and confound controls

| Analysis | Δ | z | note |
|---|---|---|---|
| Stricter null: permute within modality × biosample_type × dataset | +0.0218 | 31.1 | pipeline/cell-line confounds removed |
| **Null stratified by ChIP target** | +0.0218 | **41.8** | each target's tissue distribution preserved |
| **Target-free modalities only** (Accessibility/RNA/CAGE) | **+0.0118** | 17.0 | no `experiment_target` on either side; 3,570 tracks, 131 cells |
| Primary tissues only (`biosample_type == "tissue"`) | +0.0141 | 16.0 | 83 tissues, d = 0.32 |
| Cell lines only | +0.0369 | 22.4 | 78 lines, d = 0.85 |

The two target controls address a confound the main null does not: which TFs and
histone marks were assayed in which biosample is far from random (ENCODE profiled many
more targets in K562 and HepG2 than anywhere else), so part of Δ on ChIP-involving
pairs could have reflected assay design. Stratifying the null by `experiment_target`
leaves Δ unchanged, and restricting to modality pairs with no target concept at all
still gives Δ = +0.0118 at z = 17. **The effect is not a target artifact.**

The cell-line/primary-tissue split is the most biologically interesting contrast: the
effect is **2.6× larger in cell lines** (d = 0.85) than in primary tissues (d = 0.32).
Per-tissue, the strongest effects are distinctive cancer lines (MCF 10A +0.128,
MM.1S +0.105, PC-3 +0.073, MCF-7 +0.064) plus `uterus` (+0.096); the weakest are
stem/progenitor states (H9, mesendoderm, neuronal stem cell, all slightly negative).
84% of the 118 tissues with ≥3 modality pairs have a positive mean Δ. A plausible
reading is that the head is picking up idiosyncratic, strongly-deviating epigenomes
more than a graded notion of tissue identity.

## 7. Direct answers to the questions asked

1. **Does it cluster by modality or by cell type?** Overwhelmingly by modality. The
   UMAP shows one well-separated island per modality; tissues are scattered *across*
   those islands as tight sub-clusters within each, never forming clusters of their own.
2. **Is PC1 the modality?** No. PC1 is 7.0% of variance and modality needs ~10 PCs.
   Removing PC1 would have been the wrong operation; regressing out the modality
   subspace is the right one, and it leaves tissue untouched.
3. **Should the vectors be z-scored?** No — your instinct was right. Per-row z-scoring
   rotates `w_t`. L2-normalise and report the norm separately (it mostly encodes
   dynamic range).
4. **Conditioned on same tissue, are cross-modality projections more similar?** Yes:
   Δ = +0.0218, z = 30.8, p < 0.001, robust to eight analysis variants and to target,
   dataset and biosample-type confounds.
5. **Can tissues be matched across modalities at all?** Yes — 228 human tissues have
   ≥2 modalities, giving ~102k cross-modality same-tissue pairs. Labels needed
   normalisation (1,391 raw values, 472 singletons, FANTOM5 trailing commas) plus
   curated organ families for cross-dataset pairs.

## 8. Caveats

- **Effect size is small in absolute terms.** Δ ≈ 0.02 cosine against within-assay
  similarities of ~0.3. AUC ≈ 0.64 — real and highly significant, but not a
  representation you could read tissue off reliably per-pair.
- **Driven more by cell lines than primary tissue** (d 0.85 vs 0.32), so "learned cell
  type" overstates it; "learned idiosyncratic epigenome" is closer.
- **CAGE is untestable here** under strict tissue matching (2–4 overlapping tissues).
- **Tissue grouping involves judgment.** Organ families are explicit audited member
  lists, not keyword patterns — a keyword pass mis-merged `renal cortex interstitium`
  into brain (via `cortex`), `adrenal gland` into kidney (via `renal`), and
  `gastrocnemius medialis` into stomach (via `gastr`). Immune subsets were left
  ungrouped on purpose.
- **This is correlational.** It shows tissue information is present in and recoverable
  from the head's geometry. It does not show the model *uses* it, nor that adding an
  explicit tissue × modality factorization would improve predictions — that needs a
  training intervention.

## Reproduce

```bash
uv venv --python 3.12 && uv pip install -e .
.venv/bin/python -m juicer.run_all          # ~35 min, 1000 permutations
.venv/bin/python -m juicer.extra_controls    # target confound controls
.venv/bin/python -m juicer.replot            # figures from cached results
```

Tables in `results/`, figures in `figures/`. No HF token or download required.
