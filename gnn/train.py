#!/usr/bin/env python3
"""Train the congestion GNN on the processed tile graphs.

Before training it prints two non-GNN reference scores on the validation set, so
the GNN's number means something:

* ``mean``   -- always predict the training-set average usage
* ``linear`` -- least-squares linear model on the node's own features (no graph)

A GNN that can't beat ``linear`` isn't using its neighbours. ``--layers 0`` trains
the same network with no message passing (a per-node MLP) as a second ablation.

Metrics (masked; via usage is undefined on the top layer):

* ``mse`` -- the training loss
* ``mae`` -- mean absolute error, in "fraction of edges used"
* ``r2``  -- 1 - mse / variance: 0 = no better than the mean, 1 = perfect

Usage (from the repo root)::

    python -m gnn.train --epochs 3 --run smoke            # quick check that it works
    python -m gnn.train --epochs 60 --run v2_gat8         # real run, v2 features
    python -m gnn.train --epochs 60 --layers 0 --run v2_mlp   # no-graph ablation
    python -m gnn.train --max-side 40 --run v2_small      # ignore the large samples
    python -m gnn.train --features 1 --dropout 0 --patience 0 --run v1_again  # old recipe

Outputs: ``gnn/checkpoints/<run>/best.pt`` (lowest val loss) and
``gnn/runs/<run>/metrics.csv`` (one row per epoch).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gnn._paths  # noqa: E402,F401  (finds the challenge repo's m3d toolkit)

from gnn.features import FEATURE_VERSION, TARGET_NAMES, proc_dir  # noqa: E402
from gnn.model import CongestionGNN  # noqa: E402

CKPT_ROOT = os.path.join(_ROOT, "gnn", "checkpoints")
RUNS_ROOT = os.path.join(_ROOT, "gnn", "runs")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def split_feature_names(split: str, tile: int, version: int) -> List[str]:
    with open(os.path.join(proc_dir(tile, split, version), "features.json")) as fh:
        return json.load(fh)["features"]


def load_split(split: str, tile: int, version: int, limit: int = 0,
               min_side: int = 0, max_side: int = 0) -> List[Data]:
    files = sorted(glob.glob(os.path.join(proc_dir(tile, split, version), "*.pt")))
    if not files:
        raise SystemExit(f"no processed v{version} data for split={split!r}; "
                         f"run: python -m gnn.features --split {split}")
    out = []
    for f in files:
        d = torch.load(f, weights_only=False)
        # v1 files have no "grid"; tiles * tile size is the same up to rounding
        side = int(d["grid"][0]) if "grid" in d else int(d["shape"][2]) * tile
        if (min_side and side < min_side) or (max_side and side > max_side):
            continue
        out.append(Data(x=d["x"], edge_index=d["edge_index"], edge_attr=d["edge_attr"],
                        y=d["y"], y_mask=d["y_mask"], name=d["name"]))
        if limit and len(out) >= limit:
            break
    return out


def feature_stats(graphs: List[Data]):
    x = torch.cat([g.x for g in graphs])
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-6] = 1.0      # constant features (e.g. via_norm on one profile)
    return mean, std


def normalize(graphs: List[Data], mean, std) -> None:
    for g in graphs:
        g.x = (g.x - mean) / std


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class Meter:
    """Accumulates masked squared/absolute error per target over a whole split."""

    def __init__(self, n_targets: int):
        self.se = torch.zeros(n_targets, dtype=torch.float64)
        self.ae = torch.zeros(n_targets, dtype=torch.float64)
        self.n = torch.zeros(n_targets, dtype=torch.float64)

    def add(self, pred, y, mask):
        pred, y, mask = pred.double().cpu(), y.double().cpu(), mask.double().cpu()
        self.se += (((pred - y) ** 2) * mask).sum(0)
        self.ae += ((pred - y).abs() * mask).sum(0)
        self.n += mask.sum(0)

    def result(self, var) -> Dict[str, float]:
        mse = self.se / self.n
        out = {"mse": float(mse.mean())}
        for i, name in enumerate(TARGET_NAMES):
            out[f"{name}_mae"] = float(self.ae[i] / self.n[i])
            out[f"{name}_r2"] = float(1 - mse[i] / var[i])
        return out


def target_var(graphs: List[Data]):
    y = torch.cat([g.y for g in graphs]).double()
    m = torch.cat([g.y_mask for g in graphs]).double()
    mu = (y * m).sum(0) / m.sum(0)
    return mu, ((((y - mu) ** 2) * m).sum(0) / m.sum(0))


def reference_scores(train: List[Data], val: List[Data], var_val) -> Dict[str, Dict]:
    mu, _ = target_var(train)
    res = {}
    m = Meter(len(TARGET_NAMES))
    for g in val:
        m.add(mu.expand_as(g.y), g.y, g.y_mask)
    res["mean"] = m.result(var_val)

    # linear least squares on the node's own (normalized) features + bias
    X = torch.cat([torch.cat([g.x, torch.ones(len(g.x), 1)], 1) for g in train]).double()
    Y = torch.cat([g.y for g in train]).double()
    W = torch.linalg.lstsq(X, Y).solution
    m = Meter(len(TARGET_NAMES))
    for g in val:
        p = (torch.cat([g.x, torch.ones(len(g.x), 1)], 1).double() @ W).clamp(0, 1)
        m.add(p, g.y, g.y_mask)
    res["linear"] = m.result(var_val)
    return res


def fmt(r: Dict[str, float]) -> str:
    return (f"mse {r['mse']:.5f} | lat mae {r['lateral_usage_mae']:.4f} "
            f"r2 {r['lateral_usage_r2']:+.3f} | via mae {r['via_usage_mae']:.4f} "
            f"r2 {r['via_usage_r2']:+.3f}")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def masked_mse(pred, y, mask):
    return (((pred - y) ** 2) * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, loader, device, var) -> Dict[str, float]:
    model.eval()
    m = Meter(len(TARGET_NAMES))
    for b in loader:
        b = b.to(device)
        m.add(model(b.x, b.edge_index, b.edge_attr), b.y, b.y_mask)
    return m.result(var)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="v1", help="run name (checkpoint/log folder)")
    ap.add_argument("--tile", type=int, default=4)
    ap.add_argument("--features", type=int, default=FEATURE_VERSION,
                    help="processed feature-set version (1 = original 15 features)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32, help="graphs per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=8, help="message-passing layers (0 = MLP)")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=15,
                    help="stop after N epochs without a new best val loss (0 = never)")
    ap.add_argument("--min-side", type=int, default=0, help="only train on grids >= this")
    ap.add_argument("--max-side", type=int, default=0, help="only train on grids <= this")
    ap.add_argument("--extra-val", default="val_large",
                    help="second val split, reported every epoch but not used to pick "
                         "the checkpoint (skipped if it has no processed data)")
    ap.add_argument("--limit", type=int, default=0, help="use only N train graphs (debug)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)

    t0 = time.time()
    feat_names = split_feature_names("train", args.tile, args.features)
    train = load_split("train", args.tile, args.features, args.limit,
                       args.min_side, args.max_side)
    val = load_split("val", args.tile, args.features)
    extra: List[Data] = []
    if args.extra_val and glob.glob(os.path.join(
            proc_dir(args.tile, args.extra_val, args.features), "*.pt")):
        extra = load_split(args.extra_val, args.tile, args.features)
    mean, std = feature_stats(train)
    for split in (train, val, extra):
        normalize(split, mean, std)
    _, var_val = target_var(val)
    var_extra = target_var(extra)[1] if extra else None
    print(f"[train] features v{args.features} ({len(feat_names)}), "
          f"{len(train)} train / {len(val)} val"
          + (f" / {len(extra)} {args.extra_val}" if extra else "")
          + f" graphs, {sum(g.num_nodes for g in train):,} train nodes, "
          f"loaded in {time.time() - t0:.1f}s")
    print(f"[train] device: {device}"
          + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))

    refs = reference_scores(train, val, var_val)
    for k, r in refs.items():
        print(f"[ref]   {k:<6} {fmt(r)}")

    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val, batch_size=64)
    extra_loader = DataLoader(extra, batch_size=8) if extra else None

    model = CongestionGNN(in_dim=len(feat_names), out_dim=len(TARGET_NAMES),
                          hidden=args.hidden, n_layers=args.layers, heads=args.heads,
                          dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model: {args.layers} GATv2 layers, hidden {args.hidden}, "
          f"dropout {args.dropout}, {n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(train) / args.batch)
    warm = min(200, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(        # linear warm-up, cosine decay
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    ckpt_dir = os.path.join(CKPT_ROOT, args.run)
    run_dir = os.path.join(RUNS_ROOT, args.run)
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({**vars(args), "n_params": n_params, "n_train": len(train),
                   "features_list": feat_names, "reference": refs}, fh, indent=1)
    log_path = os.path.join(run_dir, "metrics.csv")
    log_fh = open(log_path, "w", newline="")
    log = None

    best, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        tot, n = 0.0, 0
        for b in train_loader:
            b = b.to(device)
            loss = masked_mse(model(b.x, b.edge_index, b.edge_attr), b.y, b.y_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * b.num_graphs; n += b.num_graphs
        r = evaluate(model, val_loader, device, var_val)
        row = {"epoch": epoch, "train_mse": tot / n, **{f"val_{k}": v for k, v in r.items()}}
        rx = evaluate(model, extra_loader, device, var_extra) if extra else None
        if rx:
            row.update({f"{args.extra_val}_{k}": v for k, v in rx.items()})
        row.update({"lr": sched.get_last_lr()[0], "seconds": time.time() - te})
        if log is None:
            log = csv.DictWriter(log_fh, fieldnames=list(row)); log.writeheader()
        log.writerow(row); log_fh.flush()

        mark = ""
        if r["mse"] < best:
            best, best_epoch = r["mse"], epoch
            torch.save({"model_config": model.config, "state_dict": model.state_dict(),
                        "feat_mean": mean, "feat_std": std, "tile": args.tile,
                        "features": feat_names, "targets": TARGET_NAMES,
                        "epoch": epoch, "val": r, "extra_val": rx, "args": vars(args)},
                       os.path.join(ckpt_dir, "best.pt"))
            mark = "  * best"
        extra_txt = (f" || large: lat r2 {rx['lateral_usage_r2']:+.3f} "
                     f"via r2 {rx['via_usage_r2']:+.3f}") if rx else ""
        print(f"epoch {epoch:3d}  train {tot / n:.5f} | val {fmt(r)}{extra_txt}  "
              f"{row['seconds']:.1f}s{mark}", flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            print(f"[train] early stop: no improvement for {args.patience} epochs")
            break

    log_fh.close()
    print(f"[train] done in {(time.time() - t0) / 60:.1f} min; "
          f"best val mse {best:.5f} at epoch {best_epoch}")
    print(f"[train] checkpoint: {os.path.join(ckpt_dir, 'best.pt')}")
    print(f"[train] log:        {log_path}")


if __name__ == "__main__":
    main()
