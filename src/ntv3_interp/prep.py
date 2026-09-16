"""Turn raw head weights into the identifiable per-track vectors, and prove the correction.

The head computes, for track t:

    y_t(x) = softplus( w_t . LN(x) + b_t ),    LN(x) = gamma * n(x) + beta

where n(x) = (x - mu) / sigma is *exactly zero-mean across embed_dim* by construction.
Expanding:

    w_t . LN(x) + b_t = (gamma * w_t) . n(x) + (w_t . beta + b_t)

Write v_t = gamma * w_t. Because <1, n(x)> = 0 for every input x:

    (v_t + c*1) . n(x) == v_t . n(x)      for all x, all c

So the all-ones component of v_t is a GAUGE FREEDOM -- it cannot change any prediction, and
the induced shift in the constant term is absorbed by the free bias b_t. Two tracks can
differ in raw cosine purely because of a coordinate the model is provably indifferent to.

The identifiable object is therefore

    v~_t = center(gamma * w_t)

and cos(v~_i, v~_j) is the Pearson correlation of (gamma*w_i) and (gamma*w_j).

Two corrections relative to the analysis as originally scoped:
  * applying gamma at all -- LayerNorm reweights every input coordinate before the dot
    product, so comparing raw w_t measures similarity in the wrong metric;
  * mean-centering is mandatory (gauge removal), not a stylistic choice.
The track's *norm* ||v~_t|| is kept separately: it is the effective gain / dynamic range,
which is real information, but it must not contaminate a cosine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Head:
    repo: str
    track_ids: list[str]
    W: np.ndarray  # (T, D) raw rows w_t
    b: np.ndarray  # (T,)
    gamma: np.ndarray  # (D,)
    beta: np.ndarray  # (D,)

    @property
    def n_tracks(self) -> int:
        return self.W.shape[0]

    @property
    def embed_dim(self) -> int:
        return self.W.shape[1]


def load_head(path: str | Path) -> Head:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    return Head(
        repo=meta["repo"],
        track_ids=[str(x) for x in d["track_ids"]],
        W=d["W"].astype(np.float64),
        b=d["b"].astype(np.float64),
        gamma=d["gamma"].astype(np.float64),
        beta=d["beta"].astype(np.float64),
    )


def effective_vectors(head: Head, apply_gamma: bool = True, center: bool = True) -> np.ndarray:
    """v~ = center(gamma * W). Flags exist so A0 can compare against the un-corrected forms."""
    V = head.W * head.gamma[None, :] if apply_gamma else head.W.copy()
    if center:
        V = V - V.mean(axis=1, keepdims=True)
    return V


def l2_normalize(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), eps)


def cosine_matrix(V: np.ndarray) -> np.ndarray:
    U = l2_normalize(V)
    S = U @ U.T
    return np.clip(S, -1.0, 1.0)


# --------------------------------------------------------------------------------------
# A0: verify the gauge claim numerically rather than trusting the algebra
# --------------------------------------------------------------------------------------


def verify_gauge_invariance(head: Head, n_samples: int = 256, seed: int = 0) -> dict:
    """Assert that adding c*1 to (gamma*w) changes predictions by exactly zero.

    The whole preprocessing decision rests on this, so it is checked against actual
    LayerNorm outputs rather than asserted.
    """
    rng = np.random.default_rng(seed)
    D = head.embed_dim
    x = rng.normal(size=(n_samples, D)) * rng.uniform(0.5, 3.0, size=(n_samples, 1))
    x += rng.normal(size=(n_samples, 1)) * 5.0  # arbitrary per-sample mean offset

    # LayerNorm's normalized part, exactly as flax/torch compute it.
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    n = (x - mu) / np.sqrt(var + 1e-5)

    max_abs_mean = float(np.abs(n.mean(axis=1)).max())

    v = head.W[:64] * head.gamma[None, :]  # a sample of tracks
    c = rng.normal(size=(64, 1)) * 3.0
    base = v @ n.T
    shifted = (v + c) @ n.T
    max_drift = float(np.abs(base - shifted).max())
    scale = float(np.abs(base).mean())

    return {
        "layernorm_output_max_abs_mean": max_abs_mean,
        "max_prediction_drift": max_drift,
        "mean_abs_logit": scale,
        "relative_drift": max_drift / max(scale, 1e-12),
        "passes": max_drift / max(scale, 1e-12) < 1e-9,
    }


def gauge_report(head: Head) -> dict:
    """How much does the unidentifiable component actually matter, empirically?"""
    V_gamma = head.W * head.gamma[None, :]
    ones = np.ones(head.embed_dim) / np.sqrt(head.embed_dim)

    proj = V_gamma @ ones  # component along the all-ones direction
    total_sq = (V_gamma**2).sum(axis=1)
    frac = (proj**2) / np.maximum(total_sq, 1e-300)

    S_raw = cosine_matrix(head.W)  # no gamma, no centering
    S_gamma = cosine_matrix(V_gamma)  # gamma, no centering
    S_corr = cosine_matrix(effective_vectors(head))  # gamma + centering

    iu = np.triu_indices(head.n_tracks, k=1)
    # Subsample: 7362^2/2 pairs is fine in memory but correlating all of them is wasteful.
    rng = np.random.default_rng(0)
    sel = rng.choice(iu[0].size, size=min(2_000_000, iu[0].size), replace=False)
    a, bq = iu[0][sel], iu[1][sel]

    return {
        "ones_variance_fraction_mean": float(frac.mean()),
        "ones_variance_fraction_median": float(np.median(frac)),
        "ones_variance_fraction_max": float(frac.max()),
        "corr_raw_vs_corrected": float(np.corrcoef(S_raw[a, bq], S_corr[a, bq])[0, 1]),
        "corr_gamma_vs_corrected": float(np.corrcoef(S_gamma[a, bq], S_corr[a, bq])[0, 1]),
        "mean_cos_raw": float(S_raw[a, bq].mean()),
        "mean_cos_gamma": float(S_gamma[a, bq].mean()),
        "mean_cos_corrected": float(S_corr[a, bq].mean()),
    }


def track_diagnostics(head: Head) -> dict:
    """Norms and biases -- used to identify gradient-starved 'dead' tracks."""
    V = effective_vectors(head)
    norms = np.linalg.norm(V, axis=1)
    return {
        "norms": norms,
        "bias": head.b,
        "norm_percentiles": {
            str(p): float(np.percentile(norms, p)) for p in (0, 1, 5, 25, 50, 75, 95, 99, 100)
        },
        "bias_percentiles": {
            str(p): float(np.percentile(head.b, p)) for p in (0, 1, 5, 25, 50, 75, 95, 99, 100)
        },
    }


def dead_track_mask(head: Head, norm_pct: float = 1.0) -> np.ndarray:
    """True for tracks to KEEP.

    A track whose effective vector has near-zero norm contributes an essentially random
    direction: softplus has driven it flat, so its weights carry little gradient signal.
    Averaging such directions into a similarity statistic adds variance and biases cosines
    toward zero. Cut the bottom percentile and report how many were removed.
    """
    norms = np.linalg.norm(effective_vectors(head), axis=1)
    thresh = np.percentile(norms, norm_pct)
    return norms > thresh
