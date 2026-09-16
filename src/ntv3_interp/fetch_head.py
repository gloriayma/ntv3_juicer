"""Fetch NTv3 human functional-track head tensors without downloading the whole model.

We only need four tensors out of ``model.safetensors``. The safetensors format puts a JSON
header at the front listing every tensor's byte range, so we can read the header and then
issue HTTP range requests for just the bytes we want. For the 650M checkpoint this pulls
tens of MB instead of ~2.6 GB.

Stdlib-only (urllib) apart from numpy, so it runs before project deps are installed.
"""

from __future__ import annotations

import json
import os
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HF = "https://huggingface.co"
REPO_650M = "InstaDeepAI/NTv3_650M_post"
REPO_100M = "InstaDeepAI/NTv3_100M_post"

# The tensors that define the per-track linear head for one species.
#   head.weight       (num_tracks, embed_dim)  -> row t is w_t
#   head.bias         (num_tracks,)            -> b_t
#   layer_norm.weight (embed_dim,)             -> gamma   (essential: the metric on w)
#   layer_norm.bias   (embed_dim,)             -> beta
HEAD_SUFFIXES = ("head.weight", "head.bias", "layer_norm.weight", "layer_norm.bias")


def get_token() -> str:
    """Read the HF token from the environment or the standard cache location."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if p.exists():
            return p.read_text().strip()
    raise RuntimeError(
        "No HF token found. NTv3 repos are gated: accept the license on the model page, "
        "then write the token to ~/.cache/huggingface/token or set $HF_TOKEN."
    )


def _request(url: str, token: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {token}"}
    if byte_range is not None:
        # HTTP Range is inclusive on both ends; safetensors offsets are half-open.
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1] - 1}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                f"401 for {url}. The token lacks access -- accept the license on the "
                f"model page (gated, non-commercial) with the same account."
            ) from e
        raise


def _resolve_url(repo: str, filename: str) -> str:
    return f"{HF}/{repo}/resolve/main/{filename}"


def fetch_config(repo: str, token: str) -> dict:
    raw = _request(_resolve_url(repo, "config.json"), token)
    return json.loads(raw)


def read_safetensors_header(url: str, token: str) -> tuple[dict, int]:
    """Return (header dict, offset where the tensor data block begins)."""
    # First 8 bytes: little-endian uint64 giving the JSON header length.
    n = struct.unpack("<Q", _request(url, token, (0, 8)))[0]
    header = json.loads(_request(url, token, (8, 8 + n)))
    return header, 8 + n


def _to_f32(buf: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    """Decode a safetensors buffer to float32, handling bf16 which numpy lacks."""
    if dtype == "F32":
        arr = np.frombuffer(buf, dtype="<f4")
    elif dtype == "F16":
        arr = np.frombuffer(buf, dtype="<f2").astype(np.float32)
    elif dtype == "BF16":
        # bf16 is the top 16 bits of an f32: widen by shifting left 16.
        u16 = np.frombuffer(buf, dtype="<u2").astype(np.uint32)
        arr = (u16 << 16).view(np.float32)
    elif dtype == "F64":
        arr = np.frombuffer(buf, dtype="<f8").astype(np.float32)
    else:
        raise ValueError(f"unhandled safetensors dtype {dtype!r}")
    return arr.reshape(shape).astype(np.float32)


def human_head_index(config: dict) -> tuple[int, str, list[str]]:
    """Resolve which species_heads slot is human.

    Mirrors model.py:995-999 -- heads are ordered by species token id, and only species
    that actually have bigwig tracks get a head.
    """
    bigwigs = config["bigwigs_per_species"]
    tok = config["species_to_token_id"]
    ordered = sorted(bigwigs.keys(), key=lambda s: tok[s])
    human_keys = [s for s in ordered if s.lower() in ("human", "homo_sapiens", "hg38")]
    if not human_keys:
        raise RuntimeError(f"no human species key among {ordered}")
    key = human_keys[0]
    return ordered.index(key), key, list(bigwigs[key])


@dataclass
class HumanHead:
    repo: str
    species_key: str
    head_index: int
    track_ids: list[str]
    W: np.ndarray  # (num_tracks, embed_dim) -- row t is w_t
    b: np.ndarray  # (num_tracks,)
    gamma: np.ndarray  # (embed_dim,)
    beta: np.ndarray  # (embed_dim,)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W=self.W,
            b=self.b,
            gamma=self.gamma,
            beta=self.beta,
            track_ids=np.array(self.track_ids, dtype=object),
            meta=np.array(
                json.dumps(
                    {
                        "repo": self.repo,
                        "species_key": self.species_key,
                        "head_index": self.head_index,
                    }
                )
            ),
        )


def fetch_human_head(repo: str, token: str | None = None) -> HumanHead:
    token = token or get_token()
    config = fetch_config(repo, token)
    idx, key, track_ids = human_head_index(config)

    url = _resolve_url(repo, "model.safetensors")
    header, data_start = read_safetensors_header(url, token)

    # Tensor names carry an optional prefix; match on the suffix we care about.
    want = {s: f"bigwig_head.species_heads.{idx}.{s}" for s in HEAD_SUFFIXES}
    resolved: dict[str, str] = {}
    for short, target in want.items():
        hits = [k for k in header if k != "__metadata__" and k.endswith(target)]
        if len(hits) != 1:
            raise RuntimeError(f"expected exactly one tensor ending {target!r}, got {hits}")
        resolved[short] = hits[0]

    out: dict[str, np.ndarray] = {}
    for short, name in resolved.items():
        entry = header[name]
        s, e = entry["data_offsets"]
        buf = _request(url, token, (data_start + s, data_start + e))
        out[short] = _to_f32(buf, entry["dtype"], entry["shape"])

    W, b = out["head.weight"], out["head.bias"]
    gamma, beta = out["layer_norm.weight"], out["layer_norm.bias"]

    # Torch stores Linear as (out_features, in_features); pretrained.py:386-387 transposes
    # on load into JAX. Assert rather than trust, since a silent transpose would make every
    # downstream number wrong but still plausible-looking.
    n_tracks = len(track_ids)
    if W.shape[0] != n_tracks:
        raise RuntimeError(
            f"head.weight rows {W.shape[0]} != {n_tracks} track ids -- layout assumption broken"
        )
    if W.shape[1] != gamma.shape[0]:
        raise RuntimeError(f"head.weight cols {W.shape[1]} != embed_dim {gamma.shape[0]}")
    if b.shape[0] != n_tracks:
        raise RuntimeError(f"head.bias {b.shape} != ({n_tracks},)")

    return HumanHead(repo, key, idx, track_ids, W, b, gamma, beta)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_650M)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    head = fetch_human_head(args.repo)
    out = Path(args.out or f"data/head_{args.repo.split('/')[-1]}.npz")
    head.save(out)
    print(f"repo         : {head.repo}")
    print(f"species key  : {head.species_key!r} (head index {head.head_index})")
    print(f"W            : {head.W.shape}   b: {head.b.shape}")
    print(f"gamma/beta   : {head.gamma.shape} / {head.beta.shape}")
    print(f"tracks       : {len(head.track_ids)}  e.g. {head.track_ids[:5]}")
    print(f"saved        : {out}")


if __name__ == "__main__":
    main()
