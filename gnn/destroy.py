#!/usr/bin/env python3
"""Learned destroy policy: a GNN that decides which nets LNS should rip up.

LNS spends almost all its time on useless moves (on hard, ~0.05% of moves improve
anything after a few minutes). This GNN looks at the current routing and predicts,
for every net, "if I rip up this net and some of its neighbours, will delay drop?".
LNS then picks seed nets in proportion to that prediction (plus some exploration).

Graph: one node per net; nets whose bounding boxes overlap are connected (they
compete for the same space). Node features (``FastRouter.NET_FEATURES``): how far
the net is from its ideal delay, fanout, length, cross-die, crowding around it,
detour, ...

Labels come for free from the engine. On *generated* instances (never the
benchmarks) we take snapshots during LNS and, at each, try a few moves seeded at
every net without keeping them: label 1 if any of them improved delay.

Usage (from this repo's root)::

    python -m gnn.destroy data  --n 300 --workers 8      # ~20 min, CPU
    python -m gnn.destroy train                          # ~2 min, GPU
    python -m engine.run --suite benchmarks_hard --lns-seconds 300 --destroy models/destroy_v1.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gnn._paths  # noqa: E402,F401  (finds the challenge repo's m3d toolkit)

DATA_DIR = os.path.join(_ROOT, "gnn", "data", "destroy")
RAW_DIR = os.path.join(_ROOT, "gnn", "data", "raw_engine", "train")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def _collect(job: Dict) -> Dict:
    import torch
    from engine.fast import build_router
    from m3d.model import Instance
    inst = Instance.load(job["inst"])
    out_path = os.path.join(job["data_dir"], os.path.basename(job["inst"]).replace(".inst.json", ".pt"))
    t0 = time.time()
    r = build_router(inst)
    if r is None:
        return {"name": inst.name, "ok": False}
    rng = np.random.default_rng(job["seed"])
    snaps = []
    # snapshots at increasing LNS depth, so labels cover early and late search states
    for wait in [0.0] + job["waits"]:
        if wait:
            r.lns(wait, seed=int(rng.integers(1 << 30)))
        x, ei = r.net_features()
        y, best, rate = r.snapshot_labels(job["tries"], rng)
        snaps.append({"x": torch.from_numpy(x), "edge_index": torch.from_numpy(ei),
                      "y": torch.from_numpy(y), "best": torch.from_numpy(best),
                      "rate": torch.from_numpy(rate)})
    torch.save(snaps, out_path)
    return {"name": inst.name, "ok": True, "nets": len(r.driver),
            "pos": float(np.mean([s["y"].mean().item() for s in snaps])),
            "seconds": round(time.time() - t0, 1)}


def make_data(n: int, workers: int, tries: int, waits: List[float], data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    rows = [json.loads(l) for l in open(os.path.join(RAW_DIR, "index.jsonl")) if l.strip()]
    # contended instances are where LNS matters: skip the sparse profile
    rows = [r for r in rows if r.get("ok") and r["profile"] != "sparse"]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    todo = [r for r in rows if not os.path.exists(
        os.path.join(data_dir, r["instance_file"].replace(".inst.json", ".pt")))][:n]
    jobs = [{"inst": os.path.join(RAW_DIR, r["instance_file"]), "seed": i, "tries": tries,
             "waits": waits, "data_dir": data_dir} for i, r in enumerate(todo)]
    print(f"[destroy] collecting {len(jobs)} instances, {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(_collect, jobs), 1):
            if r["ok"]:
                print(f"  [{k}/{len(jobs)}] {r['name']}: {r['nets']} nets, "
                      f"{100 * r['pos']:.1f}% improvable, {r['seconds']}s", flush=True)
    print(f"[destroy] done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def make_model(in_dim: int, hidden: int = 64, layers: int = 4):
    import torch.nn as nn
    from torch_geometric.nn import GATv2Conv

    class DestroyGNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"in_dim": in_dim, "hidden": hidden, "layers": layers}
            self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden))
            self.convs = nn.ModuleList(GATv2Conv(hidden, hidden // 4, heads=4)
                                       for _ in range(layers))
            self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                      nn.GELU(), nn.Linear(hidden, 1))

        def forward(self, x, edge_index):
            h = self.inp(x)
            for conv, norm in zip(self.convs, self.norms):
                h = h + conv(norm(h), edge_index)
            return self.head(h).squeeze(-1)      # logit: "worth ripping up"
    return DestroyGNN()


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y > 0.5], score[y < 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _hit_at(score: np.ndarray, y: np.ndarray, frac: float = 0.1) -> float:
    k = max(1, int(len(score) * frac))
    return float(y[np.argsort(-score)[:k]].mean())


def train(epochs: int, out: str, data_dir: str) -> None:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    n_val = max(1, len(files) // 7)
    graphs = {"train": [], "val": []}
    for i, f in enumerate(files):
        for s in torch.load(f, weights_only=False):
            graphs["val" if i < n_val else "train"].append(
                # train on the improvement *rate* when available (v2 data), evaluate on y
                Data(x=s["x"], edge_index=s["edge_index"], y=s["y"],
                     t=s.get("rate", s["y"])))
    X = torch.cat([g.x for g in graphs["train"]])
    mean, std = X.mean(0), X.std(0).clamp(min=1e-6)
    for split in graphs.values():
        for g in split:
            g.x = (g.x - mean) / std
    ytr = torch.cat([g.y for g in graphs["train"]])
    pos_w = float((1 - ytr.mean()) / ytr.mean().clamp(min=1e-6))
    print(f"[destroy] {len(graphs['train'])} train / {len(graphs['val'])} val snapshots, "
          f"{100 * ytr.mean():.1f}% positive")

    # what LNS uses today: rank nets by regret (feature 0 = regret ratio, raw)
    def evaluate(model=None):
        scores, ys, aucs, hits = [], [], [], []
        for g in graphs["val"]:
            if model is None:
                sc = (g.x[:, 0] * std[0] + mean[0]).numpy()
            else:
                with torch.no_grad():
                    sc = model(g.x.to(dev), g.edge_index.to(dev)).cpu().numpy()
            yy = g.y.numpy()
            if 0 < yy.sum() < len(yy):
                aucs.append(_auc(sc, yy)); hits.append(_hit_at(sc, yy))
        return float(np.nanmean(aucs)), float(np.mean(hits))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a, h = evaluate()
    print(f"[ref]   regret ranking: AUC {a:.3f}  hit-rate of top 10% {h:.3f}  "
          f"(random = {ytr.mean():.3f})")
    model = make_model(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))
    loader = DataLoader(graphs["train"], batch_size=32, shuffle=True)
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for b in loader:
            b = b.to(dev)
            loss = lossf(model(b.x, b.edge_index), b.t)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        a, h = evaluate(model)
        mark = ""
        if a > best:
            best = a
            torch.save({"config": model.config, "state_dict": model.state_dict(),
                        "mean": mean, "std": std}, out)
            mark = "  * best"
        print(f"epoch {ep:3d}  val AUC {a:.3f}  hit-rate top 10% {h:.3f}{mark}", flush=True)
    print(f"[destroy] saved {out}")


# ---------------------------------------------------------------------------
# policy used inside LNS (runs on CPU inside each routing worker)
# ---------------------------------------------------------------------------
_CACHE: Dict[str, object] = {}


def load_policy(path: str, temperature: float = 1.0):
    import torch
    torch.set_num_threads(1)       # many routing workers in parallel
    if path not in _CACHE:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = make_model(**ck["config"])
        m.load_state_dict(ck["state_dict"]); m.eval()
        _CACHE[path] = (m, ck["mean"], ck["std"])
    model, mean, std = _CACHE[path]

    def policy(router) -> np.ndarray:
        x, ei = router.net_features()
        with torch.no_grad():
            logit = model((torch.from_numpy(x) - mean) / std, torch.from_numpy(ei))
        p = torch.softmax(logit / temperature, 0).numpy().astype(np.float64)
        return p / p.sum()
    return policy


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--n", type=int, default=300)
    d.add_argument("--data-dir", default=DATA_DIR)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--tries", type=int, default=4)
    d.add_argument("--waits", type=float, nargs="+", default=[2, 4, 8, 16],
                   help="LNS seconds between snapshots")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--data-dir", default=DATA_DIR)
    t.add_argument("--out", default=os.path.join(_ROOT, "models", "destroy_v1.pt"))
    args = ap.parse_args(argv)
    if args.cmd == "data":
        make_data(args.n, args.workers, args.tries, args.waits, args.data_dir)
    else:
        train(args.epochs, args.out, args.data_dir)


if __name__ == "__main__":
    main()
