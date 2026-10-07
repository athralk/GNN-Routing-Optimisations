#!/usr/bin/env python3
"""Score trained congestion GNNs on the official benchmark tiers.

For every case in a tier, the GNN predicts per tile-layer usage from the instance
alone; the "truth" is the usage of the tier's verified reference solution. We
report R^2 (0 = no better than predicting the average, 1 = perfect) for lateral
and via usage. None of these cases is ever used for training, and most tiers are
far larger than the training grids, so this measures generalization.

Usage (from this repo's root)::

    python -m gnn.evaluate --ckpt models/v1_gat8.pt models/v2_gat8.pt
    python -m gnn.evaluate --ckpt models/v2_gat8.pt --tiers benchmarks_hard benchmarks_designs
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gnn._paths  # noqa: E402,F401  (finds the challenge repo's m3d toolkit)
from gnn._paths import repo_path  # noqa: E402
from gnn.features import build_targets  # noqa: E402
from gnn.predict import load_model, predict  # noqa: E402
from m3d.model import Instance, Submission  # noqa: E402

TIERS = ["benchmarks_hard", "benchmarks", "benchmarks_congested", "benchmarks_designs",
         "benchmarks_scale", "benchmarks_stress"]


def r2(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    m = mask > 0
    return float(1 - ((pred[m] - y[m]) ** 2).mean() / y[m].var())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--tiers", nargs="+", default=TIERS)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {os.path.splitext(os.path.basename(c))[0]: load_model(c, device)
              for c in args.ckpt}
    header = f"{'tier':22s} {'max side':>8} | " + " | ".join(
        f"{k:^21s}" for k in models)
    print(header)
    print(f"{'':22s} {'':>8} | " + " | ".join(f"{'lat R2':>10} {'via R2':>10}"
                                              for _ in models))
    for tier in args.tiers:
        tdir = repo_path(tier)
        with open(os.path.join(tdir, "suite.json")) as fh:
            cases = json.load(fh)["cases"]
        ys, masks = [], []
        preds = {k: [] for k in models}
        for c in cases:
            inst = Instance.load(os.path.join(tdir, c["instance_file"]))
            sol = Submission.load(os.path.join(tdir, c["reference_file"]))
            for k, (model, ck) in models.items():
                preds[k].append(predict(model, ck, inst, device).reshape(-1, 2))
            tile = next(iter(models.values()))[1]["tile"]
            y, m = build_targets(inst, sol, tile)
            ys.append(y); masks.append(m)
        Y, M = np.concatenate(ys), np.concatenate(masks)
        cells = []
        for k in models:
            P = np.concatenate(preds[k])
            cells.append(f"{r2(P[:, 0], Y[:, 0], M[:, 0]):>+10.3f} "
                         f"{r2(P[:, 1], Y[:, 1], M[:, 1]):>+10.3f}")
        side = max(c["width"] for c in cases)
        print(f"{tier:22s} {side:>8d} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
