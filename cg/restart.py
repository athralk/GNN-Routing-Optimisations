#!/usr/bin/env python3
"""LP-guided restart (basin hop): start PathFinder from the LP solution, not from
our current best, to land in a *different* basin.

  1. solve the LP over the saved column pool -> fractions x per candidate tree
  2. each net takes a tree drawn ~ x (sample 0 = the argmax); overlaps allowed
  3. negotiated repair (PathFinder from these routes) -> legal
  4. exact refine + LNS (destroy GNN v2, same settings as our final runs)
  5. compare with the old best; save if better

Usage (from this repo's root)::

    python -m cg.restart --case case_01 --sample 0 --lns-seconds 600
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import engine  # noqa: E402,F401
from cg.colgen import Pool, solve_lp  # noqa: E402
from engine.fast import FastRouter  # noqa: E402
from gnn._paths import repo_path  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.model import Instance, Submission  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--case", required=True)
    ap.add_argument("--pools", default="results/cg_hard/pools")
    ap.add_argument("--sample", type=int, default=0, help="0 = argmax, >0 = random draw seed")
    ap.add_argument("--lns-seconds", type=float, default=600)
    ap.add_argument("--destroy", default="models/destroy_v2.pt")
    ap.add_argument("--best", default="results/hard_final3")
    ap.add_argument("--out", default="results/cg_restart")
    args = ap.parse_args(argv)
    t0 = time.time()
    tag = f"[restart {args.case} s{args.sample}]"
    suite = repo_path(args.suite)
    meta = {c["name"]: c for c in json.load(open(os.path.join(suite, "suite.json")))["cases"]}[args.case]
    inst = Instance.load(os.path.join(suite, meta["instance_file"]))
    base = meta["baseline_total"]
    old = check(inst, Submission.load(os.path.join(args.best, f"{args.case}.sol.json"))).total_delay
    d = pickle.load(open(os.path.join(args.pools, f"{args.case}.pkl"), "rb"))
    pool = Pool(d["nets"])
    for n in pool.nets:
        for tv, te, dl in d["cols"][n]:
            pool.add(n, tv, te, dl)
    lp, _mu, _pi, x, _ = solve_lp(pool)

    rng = np.random.default_rng(args.sample)
    start, j = {}, 0
    for n in pool.nets:
        m = len(pool.cols[n])
        xn = np.maximum(x[j:j + m], 0) + 1e-9
        k = int(np.argmax(xn)) if args.sample == 0 else int(rng.choice(m, p=xn / xn.sum()))
        start[n] = pool.cols[n][k]
        j += m
    over = None
    for beta in (0.5, 0.25, 0.0):
        r = FastRouter(inst)
        r.beta = beta
        r.load_routes(start)
        over = int((r.occ_v > 1).sum())
        if r.repair():
            break
    else:
        print(f"{tag} repair failed", flush=True)
        return
    rep_d = r.total_delay()
    r.refine(20)
    ref_d = r.total_delay()
    from gnn.destroy import load_policy
    r.lns(args.lns_seconds, k=8, seed=args.sample, policy=load_policy(args.destroy),
          anneal=0.05)
    sub = r.to_submission()
    chk = check(inst, sub)
    fin = chk.total_delay if chk.legal else None
    print(f"{tag} LP {base / lp:.4f} | start clashes {over} cells | repaired {base / rep_d:.4f}"
          f" -> refined {base / ref_d:.4f} -> LNS {args.lns_seconds:.0f}s "
          f"{base / fin if fin else 0:.4f}  (old best {base / old:.4f})  "
          f"{'NEW BEST' if fin and fin < old else ''}  {time.time() - t0:.0f}s", flush=True)
    if fin and fin < old:
        os.makedirs(args.out, exist_ok=True)
        sub.save(os.path.join(args.out, f"{args.case}.sol.json"))


if __name__ == "__main__":
    main()
