#!/usr/bin/env python3
"""Route with GNN guidance: the congestion prediction becomes a prior cost inside
the repo's negotiated-congestion (PathFinder) router.

PathFinder normally *discovers* congestion: it routes, sees overlaps, raises a
history cost on overused edges, and repeats. Here the GNN *predicts* where usage
will be high before routing starts, and every edge gets an extra prior cost

    edge_cost = delay * (1 + alpha * predicted_usage ** gamma + history + present)

so nets steer around predicted hot spots from the very first pass. ``alpha = 0``
is exactly the stock negotiated router (a built-in sanity check).

The decoder is still a real router, so the output is legal or it reports failure
-- the GNN can only change *which* legal routes we find, never break legality.

Usage (from the repo root)::

    # compare alphas on the hard tier (alpha 0 = plain negotiated = the baseline)
    python -m gnn.route --ckpt models/v2_gat8.pt --suite benchmarks_hard \
        --alpha 0 0.5 1 2

    # write a submission directory with one alpha
    python -m gnn.route --ckpt ... --suite benchmarks_hard --alpha 1 \
        --out-dir submissions/hard/athralk-gnn
"""
from __future__ import annotations

import argparse
import heapq
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gnn._paths  # noqa: E402,F401  (finds the challenge repo's m3d toolkit)

from m3d.checker import check  # noqa: E402
from m3d.grid import edge_key  # noqa: E402
from m3d.model import Instance  # noqa: E402
from m3d.negotiated import VCONG, Negotiated  # noqa: E402


class GuidedNegotiated(Negotiated):
    """``Negotiated`` with a per-edge prior cost from the GNN's usage prediction."""

    def __init__(self, inst: Instance, pred: Optional[np.ndarray], tile: int,
                 alpha: float, gamma: float = 2.0, decay: float = 1.0, **kw):
        super().__init__(inst, **kw)
        # prior is scaled by (pres_fac0 / pres_fac) ** decay. pres_fac grows 1.7x per
        # PathFinder iteration, so decay=1 leaves ~7% of the prior after 5 rounds:
        # the GNN steers the first passes, then measured congestion takes over.
        # decay=0 keeps it constant (which can stop the negotiation converging).
        self.decay = decay
        W, H, L = inst.width, inst.height, inst.layers
        n = W * H * L
        # per-vertex prior for the lateral edge / the upward via that starts there
        self.p_lat = [0.0] * n
        self.p_via = [0.0] * n
        if pred is not None and alpha > 0:
            prior = alpha * np.clip(pred, 0.0, 1.0) ** gamma        # [L, TH, TW, 2]
            z, y, x = np.meshgrid(np.arange(L), np.arange(H), np.arange(W), indexing="ij")
            tiles = prior[z, y // tile, x // tile]                   # [L, H, W, 2]
            self.p_lat = tiles[..., 0].ravel().tolist()               # vid order (z,y,x)
            self.p_via = tiles[..., 1].ravel().tolist()

    def _route_net(self, nid, pres_fac):
        # identical to Negotiated._route_net except for the `prior` term
        g = self.g
        wh = g.wh
        pins = self.net_pins[nid]
        forbidden_pins = {v for v, o in self.pin_owner.items() if o != nid}
        tree_v: Set[int] = {pins[0]}
        tree_e: Set[Tuple[int, int]] = set()
        remaining = set(pins[1:])
        h_v, h_e, ov, oe = self.h_v, self.h_e, self.owners_v, self.owners_e
        p_lat, p_via = self.p_lat, self.p_via
        scale = (self.pres_fac0 / pres_fac) ** self.decay
        while remaining:
            dist: Dict[int, float] = {s: 0.0 for s in tree_v}
            prev: Dict[int, int] = {}
            heap: List[Tuple[float, int]] = [(0.0, s) for s in tree_v]
            heapq.heapify(heap)
            found = None
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist.get(u, d):
                    continue
                if u in remaining:
                    found = u
                    break
                for v, w in g.neighbors(u):
                    if v in forbidden_pins or v in tree_v:
                        continue
                    ek = edge_key(u, v)
                    lo = ek[0]
                    prior = scale * (p_via[lo] if ek[1] - lo == wh else p_lat[lo])
                    ec = w * (1.0 + prior + h_e.get(ek, 0.0) + pres_fac * len(oe.get(ek, ())))
                    vc = VCONG * (h_v.get(v, 0.0) + pres_fac * len(ov.get(v, ())))
                    nd = d + ec + vc
                    if nd < dist.get(v, 1e30):
                        dist[v] = nd
                        prev[v] = u
                        heapq.heappush(heap, (nd, v))
            if found is None:
                return None
            path = [found]
            cur = found
            while cur in prev:
                cur = prev[cur]
                path.append(cur)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                tree_v.add(a); tree_v.add(b)
                tree_e.add(edge_key(a, b))
            remaining.discard(found)
        return tree_v, tree_e


def route_case(job: Dict) -> Dict:
    inst = Instance.load(job["case"])
    t0 = time.time()
    router = GuidedNegotiated(inst, job["pred"], job["tile"], job["alpha"], job["gamma"],
                              job["decay"], max_iters=job["max_iters"])
    sub, stats = router.run()
    secs = time.time() - t0
    out = {"name": job["name"], "alpha": job["alpha"], "iterations": stats.iterations,
           "seconds": round(secs + job["infer_seconds"], 2), "legal": False,
           "total_delay": None, "used": "gnn"}
    if sub is not None:
        res = check(inst, sub)
        out["legal"], out["total_delay"] = res.legal, res.total_delay
        if not res.legal:
            sub = None
    # portfolio: also run the plain negotiated router and keep the better legal
    # result ("best"), or only run it when the guided route failed ("fail").
    # Runtime counts both runs, so the comparison stays honest.
    fb = job.get("fallback", "none")
    if job["alpha"] > 0 and (fb == "best" or (fb == "fail" and sub is None)):
        t1 = time.time()
        sub2, _ = Negotiated(inst, max_iters=job["max_iters"]).run()
        out["seconds"] = round(out["seconds"] + time.time() - t1, 2)
        if sub2 is not None:
            res2 = check(inst, sub2)
            if res2.legal and (sub is None or res2.total_delay < out["total_delay"]):
                sub, out["legal"], out["total_delay"] = sub2, True, res2.total_delay
                out["used"] = "plain"
    if job.get("out_path") and sub is not None:
        sub.save(job["out_path"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--cases", nargs="*", help="only these case names (default: all)")
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 1.0])
    ap.add_argument("--gamma", type=float, default=2.0,
                    help="prior = alpha * usage**gamma (>1 focuses on hot spots)")
    ap.add_argument("--decay", type=float, default=1.0,
                    help="prior fades as (pres_fac0/pres_fac)**decay; 0 = constant")
    ap.add_argument("--max-iters", type=int, default=50,
                    help="PathFinder iteration cap (the reference uses 50)")
    ap.add_argument("--control", choices=["none", "shuffle"], default="none",
                    help="shuffle = scramble each layer's predictions across tile "
                         "positions: same strength and layer profile, no spatial "
                         "information. If the GNN beats this, its *locations* matter.")
    ap.add_argument("--seed", type=int, default=0, help="seed for --control shuffle")
    ap.add_argument("--fallback", choices=["none", "fail", "best"], default="none",
                    help="also run the plain negotiated router: 'fail' = only when the "
                         "guided route is illegal, 'best' = always, keep the better one")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out-dir", help="write <case>.sol.json + runtime.json here "
                                      "(requires exactly one --alpha)")
    args = ap.parse_args(argv)
    if args.out_dir and len(args.alpha) != 1:
        raise SystemExit("--out-dir needs exactly one --alpha")

    import torch
    from gnn.predict import load_model, predict

    from gnn._paths import repo_path
    args.suite = repo_path(args.suite)
    from gnn._paths import repo_path
    args.suite = repo_path(args.suite)
    with open(os.path.join(args.suite, "suite.json")) as fh:
        manifest = json.load(fh)
    cases = [c for c in manifest["cases"] if not args.cases or c["name"] in args.cases]
    baseline = {c["name"]: c["baseline_total"] for c in cases}

    # GNN inference in the parent (GPU); routing in worker processes (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.ckpt, device)
    preds: Dict[str, Tuple[np.ndarray, float]] = {}
    for c in cases:
        inst = Instance.load(os.path.join(args.suite, c["instance_file"]))
        t0 = time.time()
        p = predict(model, ck, inst, device)
        dt = time.time() - t0
        if args.control == "shuffle":
            rng = np.random.default_rng(args.seed)
            L, TH, TW, C = p.shape
            flat = p.reshape(L, TH * TW, C).copy()
            for z in range(L):
                flat[z] = flat[z][rng.permutation(TH * TW)]
            p = flat.reshape(L, TH, TW, C)
        preds[c["name"]] = (p, dt)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    jobs = []
    for a in args.alpha:
        for c in cases:
            pred, infer = preds[c["name"]]
            jobs.append({"case": os.path.join(args.suite, c["instance_file"]),
                         "name": c["name"], "alpha": a, "gamma": args.gamma,
                         "decay": args.decay, "max_iters": args.max_iters,
                         "fallback": args.fallback,
                         "pred": pred if a > 0 else None, "tile": ck["tile"],
                         "infer_seconds": infer if a > 0 else 0.0,
                         "out_path": os.path.join(args.out_dir, f"{c['name']}.sol.json")
                         if args.out_dir else None})
    print(f"[route] {len(cases)} cases x {len(args.alpha)} alphas on {args.suite}, "
          f"control={args.control}, decay={args.decay}, max_iters={args.max_iters}, "
          f"{args.workers} workers", flush=True)
    results: List[Dict] = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(route_case, jobs):
            results.append(r)
            ratio = (f"{baseline[r['name']] / r['total_delay']:.4f}"
                     if r["legal"] else "ILLEGAL")
            print(f"  alpha {r['alpha']:<4} {r['name']}: iters {r['iterations']:3d}  "
                  f"delay {r['total_delay']}  vs baseline {ratio}  {r['seconds']:.1f}s"
                  f"  [{r['used']}]",
                  flush=True)

    print(f"\n{'alpha':>6} | {'legal':>5} | {'aggregate':>9} | {'agg (legal only)':>16} "
          f"| {'iters':>5} | {'seconds':>7}")
    for a in args.alpha:
        rs = [r for r in results if r["alpha"] == a]
        ok = [r for r in rs if r["legal"]]
        geo = (float(np.exp(np.mean([np.log(baseline[r["name"]] / r["total_delay"])
                                     for r in ok]))) if ok else 0.0)
        agg = geo if len(ok) == len(rs) else 0.0
        print(f"{a:>6} | {len(ok):>2}/{len(rs):<2} | {agg:>9.4f} | {geo:>16.4f} | "
              f"{sum(r['iterations'] for r in rs):>5} | {sum(r['seconds'] for r in rs):>7.1f}")
    print("(aggregate = geometric mean of baseline/your delay, like the leaderboard: 0 if "
          "any case is illegal; alpha 0 should be exactly 1.0000)")

    if args.out_dir:
        with open(os.path.join(args.out_dir, "runtime.json"), "w") as fh:
            json.dump({r["name"]: r["seconds"] for r in results}, fh, indent=1)
        print(f"[route] wrote solutions + runtime.json to {args.out_dir}")


if __name__ == "__main__":
    main()
