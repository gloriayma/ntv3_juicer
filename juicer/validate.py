"""Validation of the Analysis-3 estimator itself, independent of NTv3.

Two checks, both necessary before any delta is believed:

* **Negative control** -- run the test on a weight matrix with tissue labels shuffled
  once. The estimator must return ~0. A non-zero answer here would mean the statistic
  is biased by group-size artifacts rather than measuring tissue correspondence.
* **Synthetic positive control** -- add a small per-tissue vector to a copy of the real
  weights and confirm the test recovers a significant positive delta. This proves the
  test has the power to detect what it claims to detect, so a null result on the real
  head can be read as evidence of absence rather than lack of sensitivity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .head import cosine_matrix
from .tissue_test import run_test


def negative_control(W: np.ndarray, md: pd.DataFrame, tissue_col: str,
                     modality_col: str, n_perm: int = 200, seed: int = 0) -> dict:
    """Shuffle tissue labels within modality once, then run the full test."""
    rng = np.random.default_rng(seed)
    shuffled = md.copy()
    codes = shuffled[tissue_col].to_numpy().copy()
    for m in shuffled[modality_col].unique():
        idx = np.where(shuffled[modality_col].to_numpy() == m)[0]
        codes[idx] = rng.permutation(codes[idx])
    shuffled["_tissue_shuffled"] = codes
    S = cosine_matrix(W).astype(np.float32)
    res = run_test(S, shuffled, "_tissue_shuffled", modality_col,
                   n_perm=n_perm, seed=seed)
    return res.aggregate


def synthetic_positive_control(W: np.ndarray, md: pd.DataFrame, tissue_col: str,
                               modality_col: str, strength: float = 0.25,
                               n_perm: int = 200, seed: int = 0) -> dict:
    """Inject a per-tissue direction into the weights and confirm detection.

    ``strength`` is the injected vector's norm relative to the median weight-row
    norm, so it is a deliberately modest perturbation.
    """
    rng = np.random.default_rng(seed)
    W2 = W.copy()
    scale = strength * float(np.median(np.linalg.norm(W, axis=1)))
    for tis in md[tissue_col].unique():
        idx = np.where(md[tissue_col].to_numpy() == tis)[0]
        v = rng.normal(size=W.shape[1])
        v /= np.linalg.norm(v)
        W2[idx] += scale * v
    S = cosine_matrix(W2).astype(np.float32)
    res = run_test(S, md, tissue_col, modality_col, n_perm=n_perm, seed=seed)
    return res.aggregate
