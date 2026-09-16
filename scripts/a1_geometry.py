"""A1: what does the head-vector geometry actually organize by?

Replaces "is PC1 the modality?" with measurements: variance structure, linear probes for
modality vs tissue, and kNN label purity -- including purity computed over *cross-modality*
neighbours only, which is the version that speaks to the tissue question.

Run: uv run python scripts/a1_geometry.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from ntv3_interp import viz  # noqa: E402
from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    effective_vectors,
    l2_normalize,
    load_head,
)

viz.apply_style()
Path("figures").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

HEAD = "data/head_NTv3_650M_post.npz"
N_PC = 50
results: dict = {}

head = load_head(HEAD)
tag = head.repo.split("/")[-1]
tracks = pd.read_parquet("data/tracks.parquet")

V = effective_vectors(head)
alive = dead_track_mask(head)

df = tracks.copy()
df["alive"] = alive
df = df[df.alive & df.modality.notna()]
idx = df["index"].to_numpy()
U = l2_normalize(V[idx])

print(f"{tag}: {U.shape[0]} tracks with resolved modality, dim {U.shape[1]}")

# ---------------------------------------------------------------- PCA
pca = PCA(n_components=N_PC, random_state=0)
Z = pca.fit_transform(U)
evr = pca.explained_variance_ratio_
results["pca"] = {
    "evr_top10": [float(x) for x in evr[:10]],
    "cumulative_10": float(evr[:10].sum()),
    "cumulative_50": float(evr.sum()),
}
print(f"PC1 explains {evr[0]:.1%}; top-10 {evr[:10].sum():.1%}; top-50 {evr.sum():.1%}")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
axes[0].bar(
    np.arange(1, 21), evr[:20] * 100, color=viz.HIGHLIGHT, width=0.65, zorder=3
)
axes[0].set_xlabel("principal component")
axes[0].set_ylabel("variance explained (%)")
axes[0].set_title("Variance is spread across many PCs")
axes[1].plot(np.arange(1, N_PC + 1), np.cumsum(evr) * 100, color=viz.HIGHLIGHT, zorder=3)
axes[1].set_xlabel("number of components")
axes[1].set_ylabel("cumulative variance (%)")
axes[1].set_title("Cumulative variance")
axes[1].set_ylim(0, 100)
fig.suptitle(
    "No single dominant axis — modality cannot be 'PC1'", y=1.04, fontsize=11.5,
    color=viz.TEXT_PRIMARY,
)
fig.savefig("figures/a1_pca_scree.png")
plt.close(fig)

# ---------------------------------------------------------------- probes
def probe(X: np.ndarray, labels: pd.Series, min_count: int, name: str) -> dict:
    vc = labels.value_counts()
    keep = labels.isin(vc[vc >= min_count].index).to_numpy()
    y = pd.factorize(labels[keep])[0]
    Xs = StandardScaler().fit_transform(X[keep])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    acc = cross_val_score(clf, Xs, y, cv=5, scoring="balanced_accuracy", n_jobs=-1)

    rng = np.random.default_rng(0)
    y_perm = rng.permutation(y)
    acc_perm = cross_val_score(clf, Xs, y_perm, cv=5, scoring="balanced_accuracy", n_jobs=-1)

    out = {
        "n": int(keep.sum()),
        "n_classes": int(len(np.unique(y))),
        "balanced_acc": float(acc.mean()),
        "balanced_acc_sd": float(acc.std()),
        "permuted_baseline": float(acc_perm.mean()),
        "chance": float(1.0 / len(np.unique(y))),
    }
    print(
        f"  {name:28s} n={out['n']:5d} classes={out['n_classes']:4d} "
        f"bal.acc={out['balanced_acc']:.3f} (perm {out['permuted_baseline']:.3f}, "
        f"chance {out['chance']:.3f})"
    )
    return out


print("\nlinear probes on top-50 PCs:")
results["probe_modality"] = probe(Z, df.modality.reset_index(drop=True), 20, "modality")
results["probe_assay_family"] = probe(
    Z, df.assay_family.reset_index(drop=True), 20, "assay family"
)
bio = df.term_id.reset_index(drop=True)
results["probe_tissue"] = probe(Z, bio.fillna("__na__"), 20, "tissue (term_id)")

# Amplitude-only baseline: is apparent structure just track gain?
amp = np.column_stack(
    [np.linalg.norm(V[idx], axis=1), head.b[idx]]
)
print("\namplitude-only baseline (||v~||, bias):")
results["probe_modality_amplitude"] = probe(
    amp, df.modality.reset_index(drop=True), 20, "modality from amplitude"
)

# ---------------------------------------------------------------- kNN purity
print("\nkNN label purity (k=10, cosine):")
S = U @ U.T
np.fill_diagonal(S, -np.inf)
K = 10
nn = np.argpartition(-S, K, axis=1)[:, :K]

mod_arr = df.modality.to_numpy()
tis_arr = df.term_id.to_numpy()

mod_purity = float(np.mean(mod_arr[nn] == mod_arr[:, None]))
has_tissue = pd.notna(tis_arr)
tis_purity = float(
    np.mean((tis_arr[nn] == tis_arr[:, None])[has_tissue])
)

# The version that matters: among neighbours of a DIFFERENT modality, how often is the
# neighbour the same tissue? This cannot be explained by modality clustering.
cross_mask = mod_arr[nn] != mod_arr[:, None]
same_tissue = tis_arr[nn] == tis_arr[:, None]
valid = cross_mask & has_tissue[:, None] & pd.notna(tis_arr[nn])
cross_tissue_purity = float(same_tissue[valid].mean()) if valid.any() else float("nan")

# Baseline rate: probability two random tissue-resolved tracks share a tissue.
tv = pd.Series(tis_arr[has_tissue]).value_counts(normalize=True).to_numpy()
baseline_same_tissue = float((tv**2).sum())

print(f"  modality purity              : {mod_purity:.3f}")
print(f"  tissue purity (all neighbours): {tis_purity:.3f}")
print(f"  tissue purity (cross-modality neighbours only): {cross_tissue_purity:.3f}")
print(f"  chance rate of sharing tissue : {baseline_same_tissue:.3f}")
results["knn"] = {
    "k": K,
    "modality_purity": mod_purity,
    "tissue_purity_all": tis_purity,
    "tissue_purity_cross_modality": cross_tissue_purity,
    "chance_same_tissue": baseline_same_tissue,
    "enrichment": cross_tissue_purity / baseline_same_tissue,
}
del S

# ---------------------------------------------------------------- UMAP small multiples
cache = Path("results/a1_umap.npy")
if cache.exists() and np.load(cache).shape[0] == U.shape[0]:
    print("\nloading cached UMAP ...")
    emb = np.load(cache)
else:
    print("\ncomputing UMAP ...")
    import umap  # noqa: E402

    emb = umap.UMAP(
        n_neighbors=25, min_dist=0.15, metric="cosine", random_state=0
    ).fit_transform(U)
    np.save(cache, emb)

def facet(
    labels: pd.Series,
    groups: list[str],
    title: str,
    fname: str,
    display: dict[str, str] | None = None,
) -> None:
    n = len(groups)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.5 * nrow))
    axes = np.atleast_1d(axes).ravel()
    lab = labels.to_numpy()
    for ax, g in zip(axes, groups):
        m = lab == g
        ax.scatter(emb[~m, 0], emb[~m, 1], s=1.5, c=viz.RECESSIVE, linewidths=0, rasterized=True)
        ax.scatter(emb[m, 0], emb[m, 1], s=3.5, c=viz.HIGHLIGHT, linewidths=0, rasterized=True)
        name = (display or {}).get(g, g)
        ax.set_title(f"{name}  (n={int(m.sum())})", fontsize=8.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    for ax in axes[len(groups) :]:
        ax.axis("off")
    fig.suptitle(title, y=1.0, fontsize=11.5, color=viz.TEXT_PRIMARY)
    fig.tight_layout()
    fig.savefig(f"figures/{fname}")
    plt.close(fig)


fams = df.assay_family.value_counts().head(10).index.tolist()
facet(
    df.assay_family.reset_index(drop=True),
    fams,
    "UMAP of head vectors, one panel per assay — modality dominates the geometry",
    "a1_umap_by_modality.png",
)

top_tis = df[df.term_id.notna()].term_id.value_counts().head(10).index.tolist()
name_map = df.dropna(subset=["term_id"]).drop_duplicates("term_id").set_index("term_id").biosample
facet(
    df.term_id.reset_index(drop=True).fillna("__na__"),
    top_tis,
    "Same UMAP, one panel per biosample — tissue is sub-structure *within* each modality island",
    "a1_umap_by_tissue.png",
    display={t: str(name_map.get(t, t)) for t in top_tis},
)

results["top_tissue_names"] = {t: str(name_map.get(t, t)) for t in top_tis}
Path("results/a1_geometry.json").write_text(json.dumps(results, indent=2))
print("\nsaved results/a1_geometry.json and figures/a1_*.png")
