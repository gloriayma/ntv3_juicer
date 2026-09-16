"""A0: verify the gauge argument and quantify how much the correction actually changes."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from ntv3_interp.prep import (  # noqa: E402
    dead_track_mask,
    gauge_report,
    load_head,
    track_diagnostics,
    verify_gauge_invariance,
)

out = {}
for path in sorted(Path("data").glob("head_*.npz")):
    head = load_head(path)
    tag = head.repo.split("/")[-1]
    print(f"\n{'=' * 70}\n{tag}   W={head.W.shape}\n{'=' * 70}")

    inv = verify_gauge_invariance(head)
    print("-- gauge invariance check --")
    for k, v in inv.items():
        print(f"   {k:38s} {v}")
    assert inv["passes"], "gauge invariance FAILED -- the preprocessing argument is wrong"

    rep = gauge_report(head)
    print("-- how much does the correction matter? --")
    for k, v in rep.items():
        print(f"   {k:38s} {v:.6f}")

    diag = track_diagnostics(head)
    print("-- track diagnostics --")
    print(f"   ||v~|| percentiles  {diag['norm_percentiles']}")
    print(f"   bias  percentiles  {diag['bias_percentiles']}")

    keep = dead_track_mask(head)
    print(f"   dead tracks removed: {(~keep).sum()} of {head.n_tracks}")

    out[tag] = {
        "shape": list(head.W.shape),
        "gauge_check": inv,
        "gauge_report": rep,
        "norm_percentiles": diag["norm_percentiles"],
        "bias_percentiles": diag["bias_percentiles"],
        "n_dead": int((~keep).sum()),
    }

Path("results").mkdir(exist_ok=True)
Path("results/a0_gauge.json").write_text(json.dumps(out, indent=2))
print("\nsaved results/a0_gauge.json")
