"""Analysis 1: geometry of the head -- norms, PCA, label predictability, UMAP.

Two deliberate choices, both from the plan:

* Rows are L2-normalised, never per-row z-scored. Subtracting a row's own mean
  rotates the vector and would destroy the direction under study.
* We do not assert "PC1 = modality" and we do not delete PC1. Modality is categorical
  and can occupy several dimensions, so instead we *measure* how predictable modality
  and tissue are from the top-k PCs, sweeping k. If modality is to be removed, we
  project out a fitted modality subspace rather than deleting a principal component.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .head import Head, l2_normalize


def norm_table(head: Head, md: pd.DataFrame) -> pd.DataFrame:
    """Per-track weight norm and effective offset, with labels attached."""
    out = md[["file_id", "assay", "modality_coarse", "tissue_strict",
              "biosample_type", "dataset"]].copy()
    out["w_norm"] = np.linalg.norm(head.W, axis=1)
    out["w_gain_norm"] = np.linalg.norm(head.W_gain, axis=1)
    out["bias"] = head.b
    out["bias_effective"] = head.b_eff
    return out


def pca_fit(X: np.ndarray, n_components: int = 50) -> tuple[np.ndarray, PCA]:
    """PCA on L2-normalised rows (centred, not per-row standardised)."""
    U = l2_normalize(X)
    pca = PCA(n_components=min(n_components, min(U.shape) - 1), svd_solver="randomized",
              random_state=0)
    Z = pca.fit_transform(U)
    return Z, pca


def label_predictability(Z: np.ndarray, md: pd.DataFrame, label_col: str,
                         group_col: str, ks=(2, 5, 10, 20, 50),
                         min_class: int = 10, n_splits: int = 5,
                         seed: int = 0) -> pd.DataFrame:
    """Balanced accuracy for predicting ``label_col`` from the top-k PCs.

    Grouped CV is essential here. Grouping by ``group_col`` (experiment, or tissue)
    keeps replicates and plus/minus strand mates of one experiment out of the training
    fold when their sibling is being predicted; without it, accuracy is inflated by
    near-duplicate rows rather than reflecting genuine structure.

    Each row is paired with a label-permuted baseline so the number is interpretable.
    """
    y = md[label_col].to_numpy()
    groups = md[group_col].to_numpy()

    counts = pd.Series(y).value_counts()
    keep_classes = set(counts[counts >= min_class].index)
    mask = np.isin(y, list(keep_classes))
    if mask.sum() < n_splits * 2 or len(keep_classes) < 2:
        return pd.DataFrame()

    Zm, ym, gm = Z[mask], y[mask], groups[mask]
    rng = np.random.default_rng(seed)
    y_perm = rng.permutation(ym)

    rows = []
    for k in ks:
        if k > Zm.shape[1]:
            continue
        # multinomial is sklearn's default for multiclass since 1.5; the explicit
        # multi_class kwarg was removed in 1.7
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0),
        )
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        try:
            real = cross_val_score(clf, Zm[:, :k], ym, groups=gm, cv=cv,
                                   scoring="balanced_accuracy", n_jobs=-1)
            perm = cross_val_score(clf, Zm[:, :k], y_perm, groups=gm, cv=cv,
                                   scoring="balanced_accuracy", n_jobs=-1)
        except ValueError:
            continue
        rows.append({
            "label": label_col, "group": group_col, "k_pcs": k,
            "balanced_acc": float(np.mean(real)),
            "balanced_acc_sd": float(np.std(real)),
            "permuted_baseline": float(np.mean(perm)),
            "n_classes": int(len(set(ym))), "n_tracks": int(len(ym)),
        })
    return pd.DataFrame(rows)


def project_out_modality(X: np.ndarray, md: pd.DataFrame,
                         modality_col: str = "modality_coarse") -> np.ndarray:
    """Remove the learned modality subspace by least-squares, not by deleting a PC.

    Regresses the one-hot modality design matrix out of the (normalised) weight rows
    and returns the residual. This removes everything modality can linearly explain,
    which is the principled version of "subtract PC1".
    """
    U = l2_normalize(X)
    D = pd.get_dummies(md[modality_col].to_numpy()).to_numpy(dtype=float)
    # include an intercept
    D = np.hstack([np.ones((len(D), 1)), D])
    coef, *_ = np.linalg.lstsq(D, U, rcond=None)
    return U - D @ coef


def umap_embed(X: np.ndarray, seed: int = 0, n_neighbors: int = 30,
               min_dist: float = 0.3) -> np.ndarray:
    """UMAP with cosine metric on the head rows."""
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine",
                        random_state=seed, n_components=2)
    return reducer.fit_transform(l2_normalize(X))
