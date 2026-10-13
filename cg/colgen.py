#!/usr/bin/env python3
"""Column generation for M3D routing (steps 2-3 of the plan).

Problem as an integer program over a pool of candidate trees ("columns"):

    min  sum delay[n,k] * x[n,k]
    s.t. sum_k x[n,k] = 1                         every net picks one tree
         sum_{(n,k) uses vertex v} x[n,k] <= 1    a vertex belongs to at most one net
         x binary

Loop:
  1. pool = every net's tree from every saved legal solution (so a legal answer always
     exists and the ILP can *recombine* nets from different runs);
  2. solve the LP relaxation -> duals: mu_n per net, pi_v >= 0 per vertex (price);
  3. pricing: for every net, shortest-path trees on cost = delay + s * pi_v; keep a
     tree if its reduced cost  delay + sum_v pi_v - mu_n  is negative;
  4. repeat until no net finds a negative column; then solve the ILP on the pool.

Usage (from this repo's root)::

    python -m cg.colgen --suite benchmarks_hard --case case_09 --rounds 40 --out results/cg
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, linprog, milp  # noqa: E402

import engine  # noqa: E402,F401
from engine.fast import FastRouter, _grow, _spt, routes_from_submission  # noqa: E402
from gnn._paths import repo_path  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.model import Instance, Submission  # noqa: E402


class Pool:
    """Candidate trees per net, deduplicated by vertex set."""

    def __init__(self, nets):
        self.nets = list(nets)
        self.cols = {n: [] for n in self.nets}      # n -> [(tv, te, delay)]
        self.keys = {n: set() for n in self.nets}

    def add(self, n, tv, te, d) -> bool:
        tv = np.unique(np.asarray(tv, dtype=np.int64))
        key = tv.tobytes()
        if key in self.keys[n]:
            return False
        self.keys[n].add(key)
        self.cols[n].append((tv, np.asarray(te, dtype=np.int64), float(d)))
        return True

    def size(self) -> int:
        return sum(len(c) for c in self.cols.values())

    def matrices(self, all_rows: bool = False):
        """Column list, costs, net-equality matrix, vertex-capacity matrix, and the
        vertex ids of those rows. ``all_rows``: a row for every vertex any column uses
        (needed for prices: a cell held by one net must still have a price); otherwise
        only vertices wanted by >= 2 nets (enough for the ILP)."""
        owner, cost, rows_v, cols_v = [], [], [], []
        j = 0
        for i, n in enumerate(self.nets):
            for tv, _te, d in self.cols[n]:
                owner.append(i); cost.append(d)
                rows_v.append(tv); cols_v.append(np.full(len(tv), j))
                j += 1
        owner = np.array(owner); cost = np.array(cost)
        rv = np.concatenate(rows_v); cv = np.concatenate(cols_v)
        # keep only vertices contested by at least two different nets
        pair = np.unique(np.stack([rv, owner[cv]]), axis=1)
        verts, cnt = np.unique(pair[0], return_counts=True)
        contested = verts if all_rows else verts[cnt >= 2]
        m = np.isin(rv, contested)
        vrow = np.searchsorted(contested, rv[m])
        A_v = sp.csr_matrix((np.ones(m.sum()), (vrow, cv[m])), shape=(len(contested), j))
        A_n = sp.csr_matrix((np.ones(j), (owner, np.arange(j))), shape=(len(self.nets), j))
        return owner, cost, A_n, A_v, contested


def solve_lp(pool):
    """LP relaxation by interior point *without crossover*: the duals are central
    (every contested cell gets a meaningful price) instead of the arbitrary vertex
    duals a simplex returns at a degenerate, all-integer optimum."""
    import highspy
    owner, cost, A_n, A_v, rows = pool.matrices(all_rows=True)
    A = sp.vstack([A_n, A_v]).tocsc()
    nn, nv = A_n.shape[0], A_v.shape[0]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "off")
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(A.shape[1]); lp.col_upper_ = np.ones(A.shape[1])
    lp.row_lower_ = np.concatenate([np.ones(nn), np.full(nv, -highspy.kHighsInf)])
    lp.row_upper_ = np.ones(nn + nv)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    h.passModel(lp)
    h.run()
    sol = h.getSolution()
    y = np.array(sol.row_dual)
    x = np.array(sol.col_value)
    mu = y[:nn]                                     # per net
    pi = np.zeros(int(rows.max()) + 1)
    pi[rows] = np.maximum(0.0, -y[nn:])             # >= 0 vertex prices
    return float(cost @ x), mu, pi, x, owner


def solve_ilp(pool, time_limit, start_routes):
    """0/1 solve with HiGHS, warm-started from a legal solution (so it always has an
    answer). Returns picks, objective, and the proven lower bound for this pool."""
    import highspy
    owner, cost, A_n, A_v, _ = pool.matrices()
    A = sp.vstack([A_n, A_v]).tocsc()
    nn, nv = A_n.shape[0], A_v.shape[0]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("mip_rel_gap", 1e-6)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = cost
    lp.col_lower_ = np.zeros(A.shape[1]); lp.col_upper_ = np.ones(A.shape[1])
    lp.row_lower_ = np.concatenate([np.ones(nn), np.full(nv, -highspy.kHighsInf)])
    lp.row_upper_ = np.ones(nn + nv)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr; lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    lp.integrality_ = [highspy.HighsVarType.kInteger] * A.shape[1]
    h.passModel(lp)
    # warm start: the column of each net that matches the start solution
    x0 = np.zeros(A.shape[1])
    j = 0
    for n in pool.nets:
        key = np.unique(np.asarray(start_routes[n][0], dtype=np.int64)).tobytes()
        for tv, _te, _d in pool.cols[n]:
            if tv.tobytes() == key:
                x0[j] = 1.0
            j += 1
    sol = highspy.HighsSolution()
    sol.col_value = list(x0)
    h.setSolution(sol)
    h.run()
    x = np.round(np.array(h.getSolution().col_value)).astype(int)
    if x.sum() != nn:                      # solver returned nothing usable: keep start
        x = x0.astype(int)
    info = h.getInfo()
    picks, j = {}, 0
    for n in pool.nets:
        for k in range(len(pool.cols[n])):
            if x[j]:
                picks[n] = k
            j += 1
    return picks, float(cost @ x), float(info.mip_dual_bound)


def mc_round(inst, pool, x, rng, seconds, max_tries=24):
    """Monte Carlo rounding of the LP solution. Each sample: visit nets (confident
    ones first), draw a tree with probability ~ its LP fraction among the trees that
    fit; nets with no fitting tree are rerouted around the others (exact SPT); then
    exact refine. Returns (best total, routes, number of legal samples)."""
    t0 = time.time()
    xs, j = {}, 0
    for n in pool.nets:
        m = len(pool.cols[n]); xs[n] = np.maximum(x[j:j + m], 0) + 1e-4; j += m
    conf = {n: xs[n].max() for n in pool.nets}
    best, n_ok = None, 0
    while time.time() - t0 < seconds:
        r = FastRouter(inst)
        order = sorted(pool.nets, key=lambda n: -conf[n] * rng.random() ** 0.5)
        left = []
        for n in order:
            p = xs[n] / xs[n].sum()
            tries = rng.choice(len(p), size=min(max_tries, len(p)), replace=False, p=p)
            for k in tries:
                tv, te, d = pool.cols[n][k]
                if not r.occ_v[tv].any():
                    r._add(n, tv, te, d)
                    break
            else:
                left.append(n)
        ok = True
        for n in left:
            tv, te, d = r._route(n, 0.0, True)
            if d < 0:
                ok = False
                break
            r._add(n, tv, te, d)
        if not ok:
            continue
        r.refine(5)
        n_ok += 1
        tot = r.total_delay()
        if best is None or tot < best[0]:
            best = (tot, {n: (r.routes[n][0].copy(), r.routes[n][1].copy(), r.delay[n])
                          for n in pool.nets})
    return None if best is None else (best[0], best[1], n_ok)


def price(r: FastRouter, pool: Pool, mu, pi, scales, betas):
    """Shortest-path trees under vertex prices; add every negative-reduced-cost tree."""
    n = r.n
    price_v = np.zeros(n); price_v[:len(pi)] = pi
    zeros_e = np.zeros(3 * n); zeros_v = np.zeros(n, dtype=np.int64)
    zeros_ei = np.zeros(3 * n, dtype=np.int64)
    added, best_rc = 0, 0.0
    for i, nid in enumerate(pool.nets):
        cands = []
        for s in scales:
            hv = np.ascontiguousarray(price_v * s)
            r.cur += 1
            tv, te, d = _spt(r.driver[nid], r.sinks[nid], r.W, r.H, r.L, r.ldelay, r.via,
                             r.pin_owner, nid, zeros_v, zeros_ei, hv, zeros_e, 0.0, 1.0,
                             False, r.prior_lat, r.prior_via, 0.0, r.dist, r.tdist, r.pred,
                             r.stamp, r.cur)
            if d >= 0:
                cands.append((tv, te, d))
            for b in betas:
                r.tcur += 1
                tv, te, d, r.cur = _grow(r.driver[nid], r.sinks[nid], r.W, r.H, r.L,
                                         r.ldelay, r.via, r.pin_owner, nid, zeros_v,
                                         zeros_ei, hv, zeros_e, 0.0, 1.0, r.prior_lat,
                                         r.prior_via, 0.0, b, r.dist, r.tdist, r.pred,
                                         r.stamp, r.cur, r.tmark, r.tcur, r.ttree)
                if d >= 0:
                    cands.append((tv, te, d))
        for tv, te, d in cands:
            rc = d + price_v[np.unique(tv)].sum() - mu[i]
            if rc < -1e-6 and pool.add(nid, tv, te, d):
                added += 1
                best_rc = min(best_rc, rc)
    return added, best_rc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--case", default="case_09")
    ap.add_argument("--seed-glob", default="results/*/{case}.sol.json results/*/*/{case}.sol.json")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    ap.add_argument("--betas", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--ilp-seconds", type=float, default=300)
    ap.add_argument("--mc-seconds", type=float, default=120)
    ap.add_argument("--out", default="results/cg")
    args = ap.parse_args(argv)
    t0 = time.time()
    suite = repo_path(args.suite)
    meta = {c["name"]: c for c in json.load(open(os.path.join(suite, "suite.json")))["cases"]}[args.case]
    inst = Instance.load(os.path.join(suite, meta["instance_file"]))
    base = meta["baseline_total"]
    r = FastRouter(inst)
    pool = Pool(sorted(r.driver))

    # 1. seed pool with every saved legal solution
    best_seed = None
    for pat in args.seed_glob.split():
        for f in sorted(glob.glob(pat.format(case=args.case))):
            sub = Submission.load(f)
            res = check(inst, sub)
            if not res.legal:
                continue
            for nid, (tv, te, d) in routes_from_submission(inst, sub).items():
                pool.add(nid, tv, te, d)
            if best_seed is None or res.total_delay < best_seed[0]:
                best_seed = (res.total_delay, f)
                seed_routes = routes_from_submission(inst, sub)
    print(f"[seed] {pool.size()} columns from saved solutions; best seed "
          f"{best_seed[0]} (ratio {base / best_seed[0]:.4f}) {best_seed[1]}", flush=True)

    # 2-3. column generation
    for it in range(1, args.rounds + 1):
        lp, mu, pi, x, owner = solve_lp(pool)
        added, rc = price(r, pool, mu, pi, args.scales, args.betas)
        frac = int(((x > 1e-6) & (x < 1 - 1e-6)).sum())
        print(f"[cg {it:2d}] LP {lp:.1f} (ratio {base / lp:.4f})  fractional cols {frac}  "
              f"added {added} (min rc {rc:.1f})  pool {pool.size()}  {time.time() - t0:.0f}s",
              flush=True)
        if added == 0:
            break

    # 4a. Monte Carlo rounding: sample one tree per net from the LP fractions,
    #     reroute clashing nets around the rest, exact refine; keep the best legal one
    _lp, _mu, _pi, x, _owner = solve_lp(pool)
    rng = np.random.default_rng(0)
    best = mc_round(inst, pool, x, rng, args.mc_seconds)
    if best is not None:
        tot, routes, n_ok = best
        print(f"[mc] best {tot:.0f} (ratio {base / tot:.4f})  {n_ok} legal samples  "
              f"{time.time() - t0:.0f}s", flush=True)
        for nid, (tv, te, d) in routes.items():
            pool.add(nid, tv, te, d)            # refined trees become columns too
    os.makedirs(os.path.join(args.out, "pools"), exist_ok=True)
    with open(os.path.join(args.out, "pools", f"{args.case}.pkl"), "wb") as fh:
        pickle.dump({"nets": pool.nets, "cols": pool.cols}, fh)

    # 4b. exact 0/1 solve, warm-started from the best known legal solution
    start = routes if best is not None and best[0] < best_seed[0] else seed_routes
    picks, val, bound = solve_ilp(pool, args.ilp_seconds, start)
    print(f"[ilp] {val:.0f} (ratio {base / val:.4f})  proven bound {bound:.0f} "
          f"(ratio <= {base / max(bound, 1):.4f})  {time.time() - t0:.0f}s", flush=True)
    out = FastRouter(inst)
    for nid, k in picks.items():
        tv, te, d = pool.cols[nid][k]
        out._add(nid, tv, te, d)
    out.refine(20)
    sub = out.to_submission()
    chk = check(inst, sub)
    print(f"[check] legal {chk.legal}  total {chk.total_delay}  "
          f"ratio {base / chk.total_delay if chk.legal else 0:.4f}  "
          f"(best seed {base / best_seed[0]:.4f})", flush=True)
    if chk.legal and chk.total_delay < best_seed[0]:
        os.makedirs(args.out, exist_ok=True)
        sub.save(os.path.join(args.out, f"{args.case}.sol.json"))


if __name__ == "__main__":
    main()
