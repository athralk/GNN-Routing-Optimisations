#!/usr/bin/env python3
"""Step 4 of column generation, done exactly on small pieces (an LP-guided sub-MIP).

The full 0/1 problem over a 10-16k column pool is too big to solve, and rounding the
LP directly gives clashes (the LP lets two nets each take half a cell). Instead:

  repeat until time is up:
    1. seed = a net whose LP solution disagrees with the current solution
    2. free the seed + the nets currently sitting on cells its LP trees want (10-20)
    3. every other net stays fixed; drop free-net columns that hit a fixed net
    4. solve that small 0/1 problem *exactly* (HiGHS, seconds), warm-started from
       the current choice; accept if the total delay drops

Each step is optimal for its piece and can never make the solution worse.

Usage (from this repo's root, after cg.colgen saved the pools)::

    python -m cg.select --case case_01 --seconds 900
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
from cg.colgen import Pool, solve_ilp, solve_lp  # noqa: E402
from engine.fast import FastRouter, _grow, _spt, routes_from_submission  # noqa: E402
from gnn._paths import repo_path  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.model import Instance, Submission  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--case", required=True)
    ap.add_argument("--pools", default="results/cg_hard/pools")
    ap.add_argument("--start", nargs="+", default=["results/cg_hard", "results/hard_final3"])
    ap.add_argument("--seconds", type=float, default=900)
    ap.add_argument("--k", type=int, default=16, help="max nets freed per sub-MIP")
    ap.add_argument("--sub-seconds", type=float, default=10)
    ap.add_argument("--out", default="results/cg_hard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.0, 2.0, 6.0, 20.0])
    args = ap.parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    suite = repo_path(args.suite)
    meta = {c["name"]: c for c in json.load(open(os.path.join(suite, "suite.json")))["cases"]}[args.case]
    inst = Instance.load(os.path.join(suite, meta["instance_file"]))
    base = meta["baseline_total"]
    d = pickle.load(open(os.path.join(args.pools, f"{args.case}.pkl"), "rb"))
    pool = Pool(d["nets"])
    for n in pool.nets:
        for tv, te, dl in d["cols"][n]:
            pool.add(n, tv, te, dl)

    # current solution = best legal saved one; make sure its trees are columns
    cur, cur_d = None, None
    for folder in args.start:
        f = os.path.join(folder, f"{args.case}.sol.json")
        if os.path.exists(f):
            res = check(inst, Submission.load(f))
            if res.legal and (cur_d is None or res.total_delay < cur_d):
                cur, cur_d = routes_from_submission(inst, Submission.load(f)), res.total_delay
    for n, (tv, te, dl) in cur.items():
        pool.add(n, tv, te, dl)
    keys = {n: np.unique(cur[n][0]).tobytes() for n in pool.nets}
    pick = {n: next(k for k, c in enumerate(pool.cols[n]) if c[0].tobytes() == keys[n])
            for n in pool.nets}
    start_d = cur_d

    # LP guidance: how much each net's LP solution disagrees with the current pick
    _lp, _mu, _pi, x, _ = solve_lp(pool)
    xs, j = {}, 0
    for n in pool.nets:
        m = len(pool.cols[n]); xs[n] = x[j:j + m]; j += m
    print(f"[select {args.case}] start {cur_d} (ratio {base / cur_d:.4f})  LP {_lp:.0f} "
          f"(ratio {base / _lp:.4f})  pool {pool.size()}", flush=True)

    R = FastRouter(inst)
    Z_v = np.zeros(R.n, dtype=np.int64); Z_ei = np.zeros(3 * R.n, dtype=np.int64)
    Z_e = np.zeros(3 * R.n)
    owner = np.full(R.n, -1, dtype=np.int64)

    def rebuild_owner():
        owner[:] = -1
        for n in pool.nets:
            owner[pool.cols[n][pick[n]][0]] = n

    rebuild_owner()
    it = acc = 0
    while time.time() - t0 < args.seconds:
        it += 1
        dis = np.array([1.0 - xs[n][pick[n]] for n in pool.nets]) + 0.02
        seed = pool.nets[rng.choice(len(dis), p=dis / dis.sum())]
        # nets sitting on cells that the seed's LP trees want
        want = [pool.cols[seed][k][0] for k in np.flatnonzero(xs[seed] > 0.01)] or \
               [pool.cols[seed][pick[seed]][0]]
        nb = np.unique(owner[np.concatenate(want)])
        nb = [int(v) for v in nb if v >= 0 and v != seed]
        rng.shuffle(nb)
        free = [seed] + nb[:args.k - 1]
        fs = set(free)
        fixed_cells = owner.copy()
        fixed_cells[np.isin(owner, free)] = -1
        sub = Pool(free)
        sub_idx = {n: [] for n in free}
        for n in free:
            for k, (tv, te, dl) in enumerate(pool.cols[n]):
                if k == pick[n] or not (fixed_cells[tv] >= 0).any():
                    if sub.add(n, tv, te, dl):
                        sub_idx[n].append(k)
        # context-aware columns: fixed nets are walls, the other free nets' current
        # cells cost s extra (s = 0 ignores them ... large s avoids them)
        wall = (fixed_cells >= 0).astype(np.float64) * 1e6
        freecell = np.zeros(R.n)
        for n in free:
            freecell[pool.cols[n][pick[n]][0]] += 1
        for n in free:
            mine = np.zeros(R.n); mine[pool.cols[n][pick[n]][0]] = 1
            for sc in args.scales:
                hv = np.ascontiguousarray(wall + sc * (freecell - mine))
                R.cur += 1
                cands = [_spt(R.driver[n], R.sinks[n], R.W, R.H, R.L, R.ldelay, R.via,
                              R.pin_owner, n, Z_v, Z_ei, hv, Z_e, 0.0, 1.0, False,
                              R.prior_lat, R.prior_via, 0.0, R.dist, R.tdist, R.pred,
                              R.stamp, R.cur)]
                for b in (1.0, 0.5):
                    R.tcur += 1
                    tv, te, dl, R.cur = _grow(R.driver[n], R.sinks[n], R.W, R.H, R.L,
                                              R.ldelay, R.via, R.pin_owner, n, Z_v, Z_ei,
                                              hv, Z_e, 0.0, 1.0, R.prior_lat, R.prior_via,
                                              0.0, b, R.dist, R.tdist, R.pred, R.stamp,
                                              R.cur, R.tmark, R.tcur, R.ttree)
                    cands.append((tv, te, dl))
                for tv, te, dl in cands:
                    if dl < 0 or (fixed_cells[tv] >= 0).any():
                        continue
                    pool.add(n, tv, te, dl)
                    k = len(pool.cols[n]) - 1
                    if pool.cols[n][k][0].tobytes() == np.unique(tv).tobytes() and                             sub.add(n, tv, te, dl):
                        sub_idx[n].append(k)
                        xs[n] = np.append(xs[n], 0.0)
        start = {n: pool.cols[n][pick[n]] for n in free}
        old = sum(pool.cols[n][pick[n]][2] for n in free)
        try:
            sp_, val, _b = solve_ilp(sub, args.sub_seconds, start)
        except Exception as e:                       # noqa: BLE001
            print(f"  sub-MIP error: {e}", flush=True)
            continue
        if val < old - 1e-6:
            for n in free:
                pick[n] = sub_idx[n][sp_[n]]
            cur_d -= old - val
            acc += 1
            rebuild_owner()
            print(f"  [{time.time() - t0:5.0f}s] it {it}: freed {len(free)} nets, "
                  f"-{old - val:.0f} -> {cur_d:.0f} (ratio {base / cur_d:.4f})", flush=True)

    # polish with the exact single-net refine, check, save if better
    r = FastRouter(inst)
    for n in pool.nets:
        tv, te, dl = pool.cols[n][pick[n]]
        r._add(n, tv, te, dl)
    r.refine(20)
    sub = r.to_submission()
    chk = check(inst, sub)
    print(f"[select {args.case}] {it} sub-MIPs, {acc} accepted | legal {chk.legal} "
          f"total {chk.total_delay} ratio {base / chk.total_delay if chk.legal else 0:.4f} "
          f"(start {base / start_d:.4f})  {time.time() - t0:.0f}s", flush=True)
    if chk.legal and chk.total_delay < start_d:
        os.makedirs(args.out, exist_ok=True)
        sub.save(os.path.join(args.out, f"{args.case}.sol.json"))


if __name__ == "__main__":
    main()
