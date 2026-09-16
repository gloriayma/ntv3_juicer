"""Read the human track head out of an NTv3 post-trained checkpoint.

We read the four tensors directly with ``safetensors.safe_open`` rather than going
through ``AutoModel.from_pretrained``. Two reasons:

1. The cached snapshots contain only ``config.json`` + ``model.safetensors`` -- no
   tokenizer, and ``.no_exist`` markers for the ``trust_remote_code`` modules -- so
   ``from_pretrained(local_files_only=True)`` would not resolve.
2. The human head is ~43 MiB inside a 2.5 GiB file; ``get_tensor`` reads only that
   slice instead of materialising the whole checkpoint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from safetensors import safe_open

from .config import CHECKPOINTS, N_HUMAN_TRACKS, SPECIES, TENSORS


@dataclass
class Head:
    """The human functional-track head of one checkpoint.

    The effective per-track function is

        softplus( w_t . (gamma * x_hat + beta) + b_t )
          = softplus( (w_t * gamma) . x_hat + (w_t . beta + b_t) )

    so ``W`` is the literal learned projection and ``W_gain`` is the functionally
    faithful comparison basis. Both are exposed; analyses run on both.
    """

    name: str
    W: np.ndarray       # (7362, d) raw weight rows
    b: np.ndarray       # (7362,)
    gamma: np.ndarray   # (d,)
    beta: np.ndarray    # (d,)
    track_ids: list[str]

    @property
    def d(self) -> int:
        return self.W.shape[1]

    @property
    def W_gain(self) -> np.ndarray:
        """Weight rows with the shared LayerNorm gain folded in (``w_t * gamma``)."""
        return self.W * self.gamma[None, :]

    @property
    def b_eff(self) -> np.ndarray:
        """Effective per-track offset, ``w_t . beta + b_t``."""
        return self.W @ self.beta + self.b

    def variant(self, which: str) -> np.ndarray:
        if which == "raw":
            return self.W
        if which == "gain":
            return self.W_gain
        raise ValueError(f"unknown variant {which!r}; expected 'raw' or 'gain'")


def load_head(name: str) -> Head:
    """Load the human head and its ordered track ids from checkpoint ``name``."""
    snap = CHECKPOINTS[name]
    with open(snap / "config.json") as fh:
        cfg = json.load(fh)
    track_ids = list(cfg["bigwigs_per_species"][SPECIES])

    arrays = {}
    with safe_open(snap / "model.safetensors", framework="np") as fh:
        available = set(fh.keys())
        for key, tensor_name in TENSORS.items():
            if tensor_name not in available:
                raise KeyError(f"{tensor_name} missing from {snap.name}")
            arrays[key] = np.asarray(fh.get_tensor(tensor_name), dtype=np.float64)

    W = arrays["W"]
    if W.shape[0] != N_HUMAN_TRACKS:
        raise AssertionError(f"{name}: head has {W.shape[0]} rows, expected {N_HUMAN_TRACKS}")
    if len(track_ids) != N_HUMAN_TRACKS:
        raise AssertionError(f"{name}: config lists {len(track_ids)} human tracks")
    if arrays["gamma"].shape[0] != W.shape[1]:
        raise AssertionError(f"{name}: gamma/weight dim mismatch")

    return Head(name=name, W=W, b=arrays["b"], gamma=arrays["gamma"],
                beta=arrays["beta"], track_ids=track_ids)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation.

    Deliberately *not* per-row z-scoring: subtracting a row's own mean rotates the
    vector, which would destroy the very direction we want to study.
    """
    n = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.all(n > 0):
        raise AssertionError("zero-norm weight row(s)")
    return X / n


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    """Full pairwise cosine similarity of the rows of ``X``."""
    U = l2_normalize(X)
    S = U @ U.T
    return np.clip(S, -1.0, 1.0)
