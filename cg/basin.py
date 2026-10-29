#!/usr/bin/env python3
"""Basin hopping: big regional kicks + short LNS + Metropolis acceptance.

Part F showed that our best solutions are locally optimal for re-arrangements of
~15-20 nets, so LNS can't leave its basin. A full restart throws away the 80 % of
nets that are already well placed. A basin hop sits in between:

  repeat until the deadline:
    1. kick:   pick a region (centre: the destroy GNN's seed net half of the time,
               uniform otherwise) and rip up every net in it: 15-35 % of all nets
    2. rebuild the region with negotiated routing (mini-PathFinder), others fixed
    3. settle: short LNS (destroy GNN v2, the final runs' settings)
    4. accept the new state if better, or if worse with probability
       exp(-(new - cur) / (T * cur))  (Metropolis); otherwise restore
  the best state ever seen is saved (only if it beats the starting solution)

Usage (from this repo's root)::

    python -m cg.basin --case case_09 --until 19:30
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import engine  # noqa: E402,F401
from engine.fast import FastRouter, routes_from_submission  # noqa: E402
from gnn._paths import repo_path  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.model import Instance, Submission  # noqa: E402


def deadline(hhmm: str) -> float:
    h, m = map(int, hhmm.split(":"))
    now = time.localtime()
    t = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, h, m, 0, 0, 0, -1))
    return t if t > time.time() else t + 86400


def region(r: FastRouter, rng, s: int, frac: float):
    """Nets whose bounding box meets a square around a random point of net s's box,
    grown until it holds ~frac of all nets."""
    bb = r._bb
    x0, x1, y0, y1 = bb[s]
    cx = int(rng.integers(x0, x1 + 1)); cy = int(rng.integers(y0, y1 + 1))
    target = max(4, int(frac * len(r._nets)))
    for rad in range(2, max(r.W, r.H)):
        m = np.nonzero((bb[:, 0] <= cx + rad) & (bb[:, 1] >= cx - rad) &
                       (bb[:, 2] <= cy + rad) & (bb[:, 3] >= cy - rad))[0]
        if len(m) >= target:
            break
    # keep the nets nearest the centre (box distance), exactly `target` of them
    dx = np.maximum(0, np.maximum(bb[m, 0] - cx, cx - bb[m, 1]))
    dy = np.maximum(0, np.maximum(bb[m, 2] - cy, cy - bb[m, 3]))
    m = m[np.argsort(dx + dy + rng.random(len(m)) * 0.5)[:target]]
    return [int(r._nets[i]) for i in m]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--case", required=True)
    ap.add_argument("--start", default="results/hard_final3")
    ap.add_argument("--until", default="19:30")
    ap.add_argument("--settle", type=float, default=120, help="LNS seconds after each kick")
    ap.add_argument("--frac", type=float, nargs=2, default=[0.10, 0.25])
    ap.add_argument("--temp", type=float, default=0.002, help="Metropolis T (fraction of delay)")
    ap.add_argument("--gnn-centre", type=float, default=0.5)
    ap.add_argument("--destroy", default="models/destroy_v2.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/basin")
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    end = deadline(args.until)
    logf = open(args.log or f"results/logs/basin_{args.case}.log", "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} [{args.case}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    from gnn.destroy import load_policy
    policy = load_policy(args.destroy)
    suite = repo_path(args.suite)
    meta = {c["name"]: c for c in json.load(open(os.path.join(suite, "suite.json")))["cases"]}[args.case]
    inst = Instance.load(os.path.join(suite, meta["instance_file"]))
    base = meta["baseline_total"]
    sub0 = Submission.load(os.path.join(args.start, f"{args.case}.sol.json"))
    start_d = check(inst, sub0).total_delay
    r = FastRouter(inst)
    r.load_routes(routes_from_submission(inst, sub0))
    r.prepare_lns()
    rng = np.random.default_rng(args.seed)
    cur = best = r.total_delay()
    log(f"start {base / cur:.4f}; hops until {args.until}, settle {args.settle:.0f}s, "
        f"kick {args.frac[0]:.0%}-{args.frac[1]:.0%} of nets, T {args.temp}")
    hop = acc = fails = 0
    while time.time() + args.settle + 10 < end:
        hop += 1
        snap = (dict(r.routes), dict(r.delay))
        if rng.random() < args.gnn_centre:
            res = policy(r)
            p = res["seed"] if isinstance(res, dict) else res
            s = int(rng.choice(len(r._nets), p=p))
        else:
            s = int(rng.integers(len(r._nets)))
        group = region(r, rng, s, rng.uniform(*args.frac))
        for n in group:
            r._remove(n)
        if r._negotiate_group(group, rng, iters=30) is None:
            r._restore(*snap)
            fails += 1
            log(f"hop {hop}: kick of {len(group)} nets could not be legalized")
            continue
        kicked = r.total_delay()
        r.lns(args.settle, k=8, seed=int(rng.integers(1 << 30)), policy=policy, anneal=0.05)
        new = r.total_delay()
        ok = new < cur - 1e-9 or rng.random() < math.exp(-(new - cur) / (args.temp * cur))
        verdict = "accept" if ok else "reject"
        if ok:
            cur = new
            acc += 1
        else:
            r._restore(*snap)
        if cur < best - 1e-9:
            best = cur
            sub = r.to_submission()
            if check(inst, sub).legal and best < start_d:
                os.makedirs(args.out, exist_ok=True)
                sub.save(os.path.join(args.out, f"{args.case}.sol.json"))
                verdict += "  NEW BEST (saved)"
        log(f"hop {hop}: kicked {len(group)} nets -> {base / kicked:.4f} -> settled "
            f"{base / new:.4f}  {verdict} | current {base / cur:.4f} best {base / best:.4f}")
    log(f"DONE {hop} hops, {acc} accepted, {fails} failed kicks | start {base / start_d:.4f} "
        f"-> best {base / best:.4f}")


if __name__ == "__main__":
    main()
