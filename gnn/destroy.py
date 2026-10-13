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


_CONG: Dict[str, object] = {}


def _congestion(inst):
    """Congestion-GNN (v3_engine) prediction on CPU, cached model per worker."""
    import torch
    from gnn.predict import load_model, predict
    torch.set_num_threads(1)
    path = os.path.join(_ROOT, "models", "v3_engine.pt")
    if path not in _CONG:
        _CONG[path] = load_model(path, torch.device("cpu"))
    m, ck = _CONG[path]
    return predict(m, ck, inst, torch.device("cpu"))


def _collect_v3(job: Dict) -> Dict:
    """v3 labels: blocking graph + history features, node AND partner (edge) labels,
    snapshots taken deeper into the search than v2 (late states are where LNS stalls)."""
    import torch
    from engine.fast import build_router
    from m3d.model import Instance
    inst = Instance.load(job["inst"])
    out_path = os.path.join(job["data_dir"], os.path.basename(job["inst"]).replace(".inst.json", ".pt"))
    t0 = time.time()
    r = build_router(inst)
    if r is None:
        return {"name": inst.name, "ok": False}
    r.cong = _congestion(inst)
    rng = np.random.default_rng(job["seed"])
    snaps = []
    for wait in [0.0] + job["waits"]:
        if wait:
            r.lns(wait, seed=int(rng.integers(1 << 30)))
        x, ei, ea, y, rate, e_win, e_cnt = r.snapshot_labels_v3(job["tries"], rng)
        snaps.append({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in
                      dict(x=x, edge_index=ei, edge_attr=ea, y=y, rate=rate,
                           e_win=e_win, e_cnt=e_cnt).items()})
    torch.save(snaps, out_path)
    return {"name": inst.name, "ok": True, "nets": len(r.driver),
            "pos": float(np.mean([s_["y"].mean().item() for s_ in snaps])),
            "seconds": round(time.time() - t0, 1)}


def _collect_v4(job: Dict) -> Dict:
    """Phase 3 labels: v3 graph + improvement *sizes*, deeper search states."""
    import torch
    from engine.fast import build_router
    from m3d.model import Instance
    inst = Instance.load(job["inst"])
    out_path = os.path.join(job["data_dir"], os.path.basename(job["inst"]).replace(".inst.json", ".pt"))
    t0 = time.time()
    r = build_router(inst)
    if r is None:
        return {"name": inst.name, "ok": False}
    r.cong = _congestion(inst)
    rng = np.random.default_rng(job["seed"])
    snaps = []
    for wait in job["waits"]:                 # v4: no t=0 snapshot, late states only
        r.lns(wait, seed=int(rng.integers(1 << 30)))
        x, ei, ea, y, rate, gain, e_win, e_gain, e_cnt = r.snapshot_labels_v4(job["tries"], rng)
        snaps.append({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in
                      dict(x=x, edge_index=ei, edge_attr=ea, y=y, rate=rate, gain=gain,
                           e_win=e_win, e_gain=e_gain, e_cnt=e_cnt).items()})
        torch.save(snaps, out_path)           # save as we go: survives a hard stop
    return {"name": inst.name, "ok": True, "nets": len(r.driver),
            "pos": float(np.mean([s_["y"].mean().item() for s_ in snaps])),
            "seconds": round(time.time() - t0, 1)}


def make_data(n: int, workers: int, tries: int, waits: List[float], data_dir: str,
              version: int = 2) -> None:
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
        fn = {2: _collect, 3: _collect_v3}.get(version, _collect_v4)
        for k, r in enumerate(pool.imap_unordered(fn, jobs), 1):
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


def make_model_v3(in_dim: int, edge_dim: int, hidden: int = 128, layers: int = 6,
                  dropout: float = 0.0):
    """Destroy GNN v3: GATv2 over the blocking graph (edge features in the attention),
    a *seed* head per net and a *partner* head per edge (src = seed, dst = candidate)."""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import GATv2Conv

    class DestroyGNN3(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = {"version": 3, "in_dim": in_dim, "edge_dim": edge_dim,
                           "hidden": hidden, "layers": layers, "dropout": dropout}
            self.drop = nn.Dropout(dropout)
            self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden))
            self.eemb = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU())
            self.convs = nn.ModuleList(GATv2Conv(hidden, hidden // 4, heads=4,
                                                 edge_dim=hidden) for _ in range(layers))
            self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
            self.ffs = nn.ModuleList(nn.Sequential(nn.LayerNorm(hidden),
                                                   nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                                   nn.Linear(2 * hidden, hidden))
                                     for _ in range(layers))
            self.seed = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                      nn.GELU(), nn.Linear(hidden, 1))
            self.partner = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(),
                                         nn.Linear(hidden, 1))

        def forward(self, x, edge_index, edge_attr):
            e = self.eemb(edge_attr)
            h = self.inp(x)
            for conv, norm, ff in zip(self.convs, self.norms, self.ffs):
                h = h + self.drop(conv(norm(h), edge_index, e))
                h = h + self.drop(ff(h))
            node = self.seed(h).squeeze(-1)
            edge = self.partner(torch.cat([h[edge_index[0]], h[edge_index[1]], e], -1)).squeeze(-1)
            return node, edge
    return DestroyGNN3()


def train_v3(epochs: int, out: str, data_dir: str, hidden: int = 128, layers: int = 6,
             lr: float = 1e-3) -> None:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    n_val = max(1, len(files) // 7)            # split by *instance*, not by snapshot
    graphs = {"train": [], "val": []}
    for i, f in enumerate(files):
        for s_ in torch.load(f, weights_only=False):
            cnt = s_["e_cnt"]
            graphs["val" if i < n_val else "train"].append(Data(
                x=s_["x"], edge_index=s_["edge_index"], edge_attr=s_["edge_attr"],
                y=s_["y"], t=s_["rate"], e_t=torch.where(cnt > 0, s_["e_win"] / cnt.clamp(min=1),
                                                         torch.zeros_like(cnt)), e_cnt=cnt))
    X = torch.cat([g.x for g in graphs["train"]])
    EA = torch.cat([g.edge_attr for g in graphs["train"]])
    mean, std = X.mean(0), X.std(0).clamp(min=1e-6)
    emean, estd = EA.mean(0), EA.std(0).clamp(min=1e-6)
    raw_block = {id(g): g.edge_attr[:, 3].clone() for g in graphs["val"]}
    raw_regret = {id(g): g.x[:, 0].clone() for g in graphs["val"]}
    for split in graphs.values():
        for g in split:
            g.x = (g.x - mean) / std
            g.edge_attr = (g.edge_attr - emean) / estd
    ytr = torch.cat([g.t for g in graphs["train"]])
    etr = torch.cat([g.e_t[g.e_cnt > 0] for g in graphs["train"]])
    pos_n = float((1 - ytr.mean()) / ytr.mean().clamp(min=1e-6))
    pos_e = float((1 - etr.mean()) / etr.mean().clamp(min=1e-6))
    print(f"[destroy3] {len(graphs['train'])} train / {len(graphs['val'])} val snapshots, "
          f"node positive rate {ytr.mean():.3f}, partner positive rate {etr.mean():.3f}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(model=None):
        na, ea_, hits = [], [], []
        for g in graphs["val"]:
            if model is None:
                ns, es = raw_regret[id(g)].numpy(), raw_block[id(g)].numpy()
            else:
                with torch.no_grad():
                    n_, e_ = model(g.x.to(dev), g.edge_index.to(dev), g.edge_attr.to(dev))
                ns, es = n_.cpu().numpy(), e_.cpu().numpy()
            yy = g.y.numpy()
            if 0 < yy.sum() < len(yy):
                na.append(_auc(ns, yy)); hits.append(_hit_at(ns, yy))
            m = g.e_cnt.numpy() > 0
            ey = (g.e_t.numpy()[m] > 0).astype(np.float32)
            if 0 < ey.sum() < len(ey):
                ea_.append(_auc(es[m], ey))
        return float(np.nanmean(na)), float(np.mean(hits)), float(np.nanmean(ea_))

    a, h, e = evaluate()
    print(f"[ref]   seed: regret AUC {a:.3f} hit@10% {h:.3f} | partner: block-frac AUC {e:.3f}")
    model = make_model_v3(X.shape[1], EA.shape[1], hidden, layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    loader = DataLoader(graphs["train"], batch_size=16, shuffle=True)
    best = -1.0
    for ep in range(1, epochs + 1):
        model.train()
        for b in loader:
            b = b.to(dev)
            n_, e_ = model(b.x, b.edge_index, b.edge_attr)
            ln = bce(n_, b.t, pos_weight=torch.tensor(pos_n, device=dev))
            m = b.e_cnt > 0
            le = (bce(e_[m], b.e_t[m], pos_weight=torch.tensor(pos_e, device=dev),
                      reduction="none") * b.e_cnt[m]).sum() / b.e_cnt[m].sum()
            loss = ln + le
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        a, h, e = evaluate(model)
        mark = ""
        if a + e > best:
            best = a + e
            torch.save({"config": model.config, "state_dict": model.state_dict(),
                        "mean": mean, "std": std, "emean": emean, "estd": estd}, out)
            mark = "  * best"
        print(f"epoch {ep:3d}  seed AUC {a:.3f} hit@10% {h:.3f} | partner AUC {e:.3f}{mark}",
              flush=True)
    print(f"[destroy3] saved {out}")


def train_v4(out: str, data_dir: str, epochs: int = 40,
             sweep=((64, 4, 0.1), (128, 4, 0.1), (64, 6, 0.1)), ensemble: int = 3,
             lr: float = 1e-3, wd: float = 1e-3) -> None:
    """Phase 3 training. Loss = listwise ranking: the GNN's softmax over nets (and over
    each seed's partners) should match where the *improvement size* is. Since LNS
    samples seeds from that same softmax, this directly trains "spend moves where the
    big gains are". Model selection by *gain lift*: expected gain when sampling from
    the policy / expected gain when sampling uniformly (val, split by instance)."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.utils import softmax as seg_softmax, scatter
    files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    n_val = max(1, len(files) // 7)
    graphs = {"train": [], "val": []}
    for i, f in enumerate(files):
        for s_ in torch.load(f, weights_only=False):
            cnt = s_["e_cnt"]
            graphs["val" if i < n_val else "train"].append(Data(
                x=s_["x"], edge_index=s_["edge_index"], edge_attr=s_["edge_attr"],
                y=s_["y"], t=s_["rate"], g=s_["gain"],
                e_t=torch.where(cnt > 0, s_["e_win"] / cnt.clamp(min=1), torch.zeros_like(cnt)),
                e_g=torch.where(cnt > 0, s_["e_gain"] / cnt.clamp(min=1), torch.zeros_like(cnt)),
                e_cnt=cnt))
    X = torch.cat([g.x for g in graphs["train"]])
    EA = torch.cat([g.edge_attr for g in graphs["train"]])
    mean, std = X.mean(0), X.std(0).clamp(min=1e-6)
    emean, estd = EA.mean(0), EA.std(0).clamp(min=1e-6)
    raw_regret = {id(g): g.x[:, 0].clone() for g in graphs["val"]}
    for split in graphs.values():
        for g in split:
            g.x = (g.x - mean) / std
            g.edge_attr = (g.edge_attr - emean) / estd
    print(f"[destroy4] {len(graphs['train'])} train / {len(graphs['val'])} val snapshots "
          f"from {len(files)} instances", flush=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def lift(scores_n, scores_e, g):
        """(expected gain under softmax policy) / (under uniform), nodes and partners."""
        gn = g.g.numpy()
        pn = np.exp(scores_n - scores_n.max()); pn /= pn.sum()
        ln = (pn * gn).sum() / max(gn.mean(), 1e-9)
        src = g.edge_index[0].numpy(); m = g.e_cnt.numpy() > 0
        eg = g.e_g.numpy()
        les = []
        for i in np.unique(src[m]):
            sel = m & (src == i)
            if sel.sum() < 2 or eg[sel].sum() <= 0:
                continue
            z = scores_e[sel]; q = np.exp(z - z.max()); q /= q.sum()
            les.append((q * eg[sel]).sum() / eg[sel].mean())
        return ln, (np.mean(les) if les else 1.0)

    def evaluate(model=None):
        L_n, L_e = [], []
        for g in graphs["val"]:
            if g.g.sum() <= 0:
                continue
            if model is None:
                sn = raw_regret[id(g)].numpy() * 5.0
                se = np.zeros(g.edge_index.shape[1])
            else:
                with torch.no_grad():
                    n_, e_ = model(g.x.to(dev), g.edge_index.to(dev), g.edge_attr.to(dev))
                sn, se = n_.cpu().numpy(), e_.cpu().numpy()
            a, b = lift(sn, se, g)
            L_n.append(a); L_e.append(b)
        return float(np.mean(L_n)), float(np.mean(L_e))

    a, b = evaluate()
    print(f"[ref]   lift (1.0 = uniform): regret seed {a:.3f} | random partners {b:.3f}", flush=True)

    def fit(hidden, layers, dropout, seed):
        torch.manual_seed(seed)
        model = make_model_v3(X.shape[1], EA.shape[1], hidden, layers, dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        loader = DataLoader(graphs["train"], batch_size=16, shuffle=True)
        best, best_sd = -1.0, None
        for ep in range(1, epochs + 1):
            model.train()
            for bt in loader:
                bt = bt.to(dev)
                n_, e_ = model(bt.x, bt.edge_index, bt.edge_attr)
                # listwise ranking over each snapshot's nets: target ∝ mean gain
                tn = bt.g / scatter(bt.g, bt.batch, reduce="sum").clamp(min=1e-9)[bt.batch]
                pn = seg_softmax(n_, bt.batch)
                ln = -(tn * torch.log(pn + 1e-12)).sum() / bt.num_graphs
                # listwise over each seed's tried partners: target ∝ mean gain
                m = bt.e_cnt > 0
                src = bt.edge_index[0][m]
                ge = bt.e_g[m]
                te = ge / scatter(ge, src, dim_size=bt.num_nodes, reduce="sum").clamp(min=1e-9)[src]
                pe = seg_softmax(e_[m], src, num_nodes=bt.num_nodes)
                has = scatter(ge, src, dim_size=bt.num_nodes, reduce="sum")[src] > 0
                le = -(te[has] * torch.log(pe[has] + 1e-12)).sum() / max(1, int(has.sum()))
                loss = ln + 2.0 * le
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            model.eval()
            a, b = evaluate(model)
            if a * b > best:
                best = a * b
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_ab = (a, b, ep)
        return best, best_sd, model.config, best_ab

    results = []
    for (hdn, lay, dr) in sweep:
        t0 = time.time()
        sc, sd, cfg, (a, b, ep) = fit(hdn, lay, dr, 0)
        results.append((sc, sd, cfg))
        print(f"[sweep] hidden {hdn} layers {lay} dropout {dr}: seed lift {a:.3f} "
              f"partner lift {b:.3f} (epoch {ep})  {time.time() - t0:.0f}s", flush=True)
    sc, sd, cfg = max(results, key=lambda t: t[0])
    members = [sd]
    for k in range(1, ensemble):
        _, sdk, _, (a, b, ep) = fit(cfg["hidden"], cfg["layers"], cfg["dropout"], k)
        members.append(sdk)
        print(f"[ensemble] member {k}: seed lift {a:.3f} partner lift {b:.3f}", flush=True)
    torch.save({"config": cfg, "state_dict": members[0], "members": members,
                "mean": mean, "std": std, "emean": emean, "estd": estd}, out)
    print(f"[destroy4] saved {out}: config {cfg}, ensemble of {len(members)}", flush=True)


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
        cfg = dict(ck["config"])
        if cfg.pop("version", 2) >= 3:
            ms = []
            for sd in ck.get("members", [ck["state_dict"]]):
                mm = make_model_v3(**cfg); mm.load_state_dict(sd); mm.eval(); ms.append(mm)

            def m(*a):                         # ensemble: average the members' logits
                outs = [mm(*a) for mm in ms]
                return (sum(o[0] for o in outs) / len(outs), sum(o[1] for o in outs) / len(outs))
        else:
            m = make_model(**cfg)
            m.load_state_dict(ck["state_dict"]); m.eval()
        _CACHE[path] = (m, ck)
    model, ck = _CACHE[path]
    mean, std = ck["mean"], ck["std"]

    if ck["config"].get("version", 2) >= 3:
        emean, estd = ck["emean"], ck["estd"]

        def policy_v3(router):
            x, ei, ea = router.net_graph_v3()
            with torch.no_grad():
                n_, e_ = model((torch.from_numpy(x) - mean) / std, torch.from_numpy(ei),
                               (torch.from_numpy(ea) - emean) / estd)
            p = torch.softmax(n_ / temperature, 0).numpy().astype(np.float64)
            el = e_.numpy().astype(np.float64)
            src, dst = ei
            order = np.argsort(src, kind="stable")
            bounds = np.searchsorted(src[order], np.arange(len(p) + 1))
            partners = []
            for i in range(len(p)):
                idx = order[bounds[i]:bounds[i + 1]]
                if len(idx) == 0:
                    partners.append((np.zeros(0, dtype=np.int64), np.zeros(0)))
                    continue
                z = el[idx] / temperature
                q = np.exp(z - z.max()); q /= q.sum()
                partners.append((dst[idx], q))
            return {"seed": p / p.sum(), "partners": partners}
        return policy_v3

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
    d.add_argument("--version", type=int, default=2, help="3 = blocking graph + partner labels")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--data-dir", default=DATA_DIR)
    t.add_argument("--out", default=os.path.join(_ROOT, "models", "destroy_v1.pt"))
    t.add_argument("--version", type=int, default=2)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args(argv)
    if args.cmd == "data":
        make_data(args.n, args.workers, args.tries, args.waits, args.data_dir, args.version)
    elif args.version >= 4:
        train_v4(args.out, args.data_dir, epochs=args.epochs, lr=args.lr)
    elif args.version >= 3:
        train_v3(args.epochs, args.out, args.data_dir, args.hidden, args.layers, args.lr)
    else:
        train(args.epochs, args.out, args.data_dir)


if __name__ == "__main__":
    main()
