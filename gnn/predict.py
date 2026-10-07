#!/usr/bin/env python3
"""Run a trained congestion GNN on one instance and plot prediction vs truth.

Works on any instance JSON -- a generated sample *or* an official benchmark case
(that is the real generalization test: official cases are bigger than anything
in training, and the ``designs`` tier comes from real circuits).

Usage (from the repo root)::

    python -m gnn.predict --ckpt models/v2_gat8.pt --sample val/sample_000003
    python -m gnn.predict --ckpt models/v2_gat8.pt \
        --case benchmarks_hard/case_05.json --sol benchmarks_hard/reference/case_05.sol.json
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gnn._paths  # noqa: E402,F401  (finds the challenge repo's m3d toolkit)

from gnn.features import FEATURE_NAMES, RAW_ROOT, build_graph, build_targets  # noqa: E402
from gnn.model import CongestionGNN  # noqa: E402
from m3d.model import Instance, Submission  # noqa: E402


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CongestionGNN.from_config(ck["model_config"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def predict(model, ck, inst: Instance, device) -> np.ndarray:
    """Return predicted usage, shape [layers, tiles_y, tiles_x, 2]."""
    g = build_graph(inst, ck["tile"])
    # take exactly the feature columns this checkpoint was trained on (features are
    # append-only, so a v1 model just uses the first 15)
    cols = [FEATURE_NAMES.index(n) for n in ck["features"]]
    x = (torch.from_numpy(g["x"][:, cols]) - ck["feat_mean"]) / ck["feat_std"]
    out = model(x.to(device), torch.from_numpy(g["edge_index"]).to(device),
                torch.from_numpy(g["edge_attr"]).to(device))
    L, TH, TW = g["shape"]
    return out.cpu().numpy().reshape(L, TH, TW, -1)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sample", help="SPLIT/NAME of a generated sample")
    ap.add_argument("--case", help="any instance JSON (e.g. an official benchmark)")
    ap.add_argument("--sol", help="its solution JSON (for the 'truth' row)")
    ap.add_argument("--out", help="output PNG (default: gnn/data/viz/...)")
    args = ap.parse_args(argv)
    if args.sample:
        stem = os.path.join(RAW_ROOT, args.sample)
        args.case, args.sol = stem + ".inst.json", stem + ".sol.json"
    if not args.case:
        raise SystemExit("give --sample or --case")
    from gnn._paths import repo_path
    args.case = repo_path(args.case)
    if args.sol:
        args.sol = repo_path(args.sol)
    from gnn._paths import repo_path
    args.case = repo_path(args.case)
    if args.sol:
        args.sol = repo_path(args.sol)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.ckpt, device)
    inst = Instance.load(args.case)
    pred = predict(model, ck, inst, device)
    L = inst.layers
    pred[L - 1, :, :, 1] = 0.0   # no vias go up from the top layer; output there is meaningless
    truth = None
    if args.sol:
        y, _ = build_targets(inst, Submission.load(args.sol), ck["tile"])
        truth = y.reshape(pred.shape)

    rows = [("pred lateral", pred[..., 0])]
    if truth is not None:
        rows.append(("true lateral", truth[..., 0]))
    rows.append(("pred via", pred[..., 1]))
    if truth is not None:
        rows.append(("true via", truth[..., 1]))
    vmax = max(1e-6, max(float(r[1].max()) for r in rows))

    fig, axes = plt.subplots(len(rows), L, figsize=(2.4 * L, 2.4 * len(rows) + 0.6),
                             squeeze=False)
    for r, (label, arr) in enumerate(rows):
        for z in range(L):
            ax = axes[r][z]
            ax.imshow(arr[z], origin="lower", cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={z}  delay={inst.layer_delay[z]}", fontsize=9)
            if z == 0:
                ax.set_ylabel(label, fontsize=9)
    title = f"{inst.name}: {inst.width}x{inst.height}x{L}, {len(inst.nets)} nets"
    if truth is not None:
        mae = np.abs(pred - truth)[:L - 1].mean()   # skip top layer (no vias there)
        title += f"   |   MAE {mae:.4f}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    case_name = os.path.basename(args.case).replace(".inst.json", "").replace(".json", "")
    folder = os.path.basename(os.path.dirname(os.path.abspath(args.case)))
    out = args.out or os.path.join(_ROOT, "gnn", "data", "viz",
                                   f"pred_{folder}_{case_name}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"[predict] wrote {out}")


if __name__ == "__main__":
    main()
