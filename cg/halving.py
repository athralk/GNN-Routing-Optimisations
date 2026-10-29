#!/usr/bin/env python3
"""Successive halving over fresh starts ("make the top scorers compete").

Stage 1: N fresh, diverse starts per case (noisy negotiation + refine + short LNS).
Stage 2: the best N/4 per case continue (warm start) with more LNS.
Stage 3: the best one per case continues with the longest LNS.
Then best-of with the previous final submission.

Diversity comes from random initial history costs in the negotiation, so each start
lands in a different layout before LNS. One parent process writes the log (no
interleaved writes).

Usage (from this repo's root)::

    python -m cg.halving --starts 12 --stages 120 360 1200 --keep 3 1 --workers 9
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import engine  # noqa: E402,F401
from engine.fast import FastRouter, routes_from_submission  # noqa: E402
from gnn._paths import repo_path  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.model import Instance, Submission  # noqa: E402

_POLICY = {}


def _job(j):
    from gnn.destroy import load_policy
    inst = Instance.load(j["inst"])
    rng = np.random.default_rng(j["seed"])
    if j["warm"]:
        r = FastRouter(inst)
        r.load_routes(routes_from_submission(inst, Submission.load(j["warm"])))
    else:
        r = None
        for beta in (0.5, 0.25, 0.0):
            r = FastRouter(inst)
            r.beta = beta
            r.hist_v[:] = rng.exponential(j["noise"], r.n)      # diversity
            if r.negotiate():
                break
            r = None
        if r is None:
            return {**j, "delay": None}
        r.refine(20)
    if j["destroy"] not in _POLICY:
        _POLICY[j["destroy"]] = load_policy(j["destroy"])
    r.lns(j["seconds"], k=8, seed=j["seed"], policy=_POLICY[j["destroy"]], anneal=0.05)
    sub = r.to_submission()
    res = check(inst, sub)
    if not res.legal:
        return {**j, "delay": None}
    os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
    sub.save(j["out"])
    return {**j, "delay": res.total_delay}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--stages", type=float, nargs="+", default=[120, 360, 1200])
    ap.add_argument("--keep", type=int, nargs="+", default=[3, 1])
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--destroy", default="models/destroy_v2.pt")
    ap.add_argument("--best", default="results/hard_final3")
    ap.add_argument("--out", default="results/halving")
    ap.add_argument("--log", default="results/logs/halving_live.log")
    args = ap.parse_args(argv)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logf = open(args.log, "a", encoding="utf-8")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    suite = repo_path(args.suite)
    cases = json.load(open(os.path.join(suite, "suite.json")))["cases"]
    base = {c["name"]: c["baseline_total"] for c in cases}
    old = {}
    for c in cases:
        inst = Instance.load(os.path.join(suite, c["instance_file"]))
        old[c["name"]] = check(inst, Submission.load(
            os.path.join(args.best, f"{c['name']}.sol.json"))).total_delay
    log(f"successive halving: {args.starts} starts/case, stages {args.stages}s, keep {args.keep}")
    alive = {c["name"]: [(i, None) for i in range(args.starts)] for c in cases}
    pool = Pool(args.workers)
    for st, secs in enumerate(args.stages, 1):
        jobs = []
        for c in cases:
            for i, warm in alive[c["name"]]:
                jobs.append({"case": c["name"], "id": i, "stage": st, "seconds": secs,
                             "inst": os.path.join(suite, c["instance_file"]),
                             "warm": warm, "seed": 1000 * st + i, "noise": args.noise,
                             "destroy": args.destroy,
                             "out": os.path.join(args.out, f"stage{st}",
                                                 f"{c['name']}_r{i}.sol.json")})
        log(f"--- stage {st}: {len(jobs)} runs x {secs:.0f}s")
        done = {c["name"]: [] for c in cases}
        for res in pool.imap_unordered(_job, jobs):
            n = res["case"]
            d = res["delay"]
            done[n].append(res)
            log(f"  s{st} {n} r{res['id']:2d}: " +
                (f"{base[n] / d:.4f}" if d else "illegal") + f"   (best so far {base[n] / old[n]:.4f})")
        summary = []
        for c in cases:
            n = c["name"]
            ok = sorted([r for r in done[n] if r["delay"]], key=lambda r: r["delay"])
            if not ok:
                continue
            ratios = [base[n] / r["delay"] for r in ok]
            summary.append(ratios[0])
            log(f"  [stage {st}] {n}: top {ratios[0]:.4f}  median {np.median(ratios):.4f}  "
                f"(best so far {base[n] / old[n]:.4f})")
            if st <= len(args.keep):
                alive[n] = [(r["id"], r["out"]) for r in ok[:args.keep[st - 1]]]
        if summary:
            g = math.exp(sum(map(math.log, summary)) / len(summary))
            log(f"=== stage {st} done: geomean of per-case top = {g:.4f}  (final submission 1.3138)")
    pool.close()
    last = os.path.join(args.out, f"stage{len(args.stages)}")
    log("=== best of (final submission, halving winners):")
    improved = []
    for c in cases:
        n = c["name"]
        for f in sorted(os.listdir(last)) if os.path.isdir(last) else []:
            if f.startswith(n + "_"):
                d = check(Instance.load(os.path.join(suite, c["instance_file"])),
                          Submission.load(os.path.join(last, f))).total_delay
                if d < old[n]:
                    improved.append((n, base[n] / old[n], base[n] / d))
    for n, a, b in improved:
        log(f"  NEW BEST {n}: {a:.4f} -> {b:.4f}")
    if not improved:
        log("  no case beat the final submission")
    log("ALL DONE")


if __name__ == "__main__":
    main()
