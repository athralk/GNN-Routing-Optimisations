#!/usr/bin/env python3
"""Route a whole tier with the fast engine (optionally with the GNN prior) and score it.

Usage (from this repo's root)::

    python -m engine.run --suite benchmarks_hard --workers 8
    python -m engine.run --suite benchmarks_hard --ckpt models/v2_gat8.pt --alpha 1
    python -m engine.run --suite benchmarks_hard --ckpt models/v2_gat8.pt --out-dir results/hard_engine
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import engine  # noqa: E402,F401  (sets up paths)
from engine.fast import solve  # noqa: E402
from gnn._paths import repo_path  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.model import Instance  # noqa: E402


def _job(job: Dict) -> Dict:
    inst = Instance.load(job["case"])
    policy = None
    if job.get("destroy"):
        from gnn.destroy import load_policy
        policy = load_policy(job["destroy"])
    warm = None
    if job.get("warm") and os.path.exists(job["warm"]):
        from engine.fast import routes_from_submission
        from m3d.model import Submission
        warm = routes_from_submission(inst, Submission.load(job["warm"]))
    t0 = time.time()
    sub, st = solve(inst, job["prior"], job["tile"], job["alpha"], job["max_iters"],
                    job["refine_passes"], lns_seconds=job["lns_seconds"], lns_k=job["lns_k"],
                    seed=job["seed"], policy=policy, anneal=job["anneal"], mix=job["mix"],
                    stall_kick=job["stall_kick"], warm=warm, cong=job.get("cong"))
    secs = time.time() - t0 + job["infer"]
    out = {"name": job["name"], "seconds": round(secs, 2), "legal": False, "delay": None, **st}
    if sub is not None:
        res = check(inst, sub)
        out["legal"], out["delay"] = res.legal, res.total_delay
        if res.legal and job["out"]:
            sub.save(job["out"])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--ckpt", help="GNN checkpoint for the congestion prior (optional)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-iters", type=int, default=100)
    ap.add_argument("--refine-passes", type=int, default=20)
    ap.add_argument("--lns-seconds", type=float, default=0.0, help="LNS time per case")
    ap.add_argument("--lns-k", type=int, default=8, help="nets ripped up per LNS move")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anneal", type=float, default=0.1,
                    help="LNS may worsen a move by this x mean net delay (decays to 0)")
    ap.add_argument("--mix", type=float, nargs=3, default=[1.0, 0.0, 0.0],
                    help="probabilities of greedy / negotiated / window moves")
    ap.add_argument("--stall-kick", type=float, default=2.0,
                    help="seconds without improvement before a negotiated move (0 = off)")
    ap.add_argument("--destroy", help="destroy-policy GNN checkpoint: picks what LNS rips up")
    ap.add_argument("--warm", help="folder of saved solutions to continue from (warm start)")
    ap.add_argument("--cong-ckpt", default=None,
                    help="congestion GNN whose predictions feed destroy-GNN v3 features "
                         "(default: models/v3_engine.pt when --destroy is a v3 model)")
    ap.add_argument("--json-out", help="write per-case results as JSON (for gnn/ab.py)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)

    suite = repo_path(args.suite)
    with open(os.path.join(suite, "suite.json")) as fh:
        cases = [c for c in json.load(fh)["cases"] if not args.cases or c["name"] in args.cases]
    preds, tile = {}, 4
    if args.ckpt:
        import torch
        from gnn.predict import load_model, predict
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, ck = load_model(args.ckpt, dev)
        tile = ck["tile"]
        for c in cases:
            inst = Instance.load(os.path.join(suite, c["instance_file"]))
            t0 = time.time()
            preds[c["name"]] = (predict(model, ck, inst, dev), time.time() - t0)
    # congestion predictions as destroy-GNN v3 features (no effect on routing costs)
    congs = {}
    cong_ckpt = args.cong_ckpt
    if cong_ckpt is None and args.destroy:
        import torch
        ckd = torch.load(args.destroy, map_location="cpu", weights_only=False)
        if ckd["config"].get("version", 2) >= 3:
            cong_ckpt = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "models", "v3_engine.pt")
    if cong_ckpt:
        import torch
        from gnn.predict import load_model, predict
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cm, cck = load_model(cong_ckpt, dev)
        for c in cases:
            inst = Instance.load(os.path.join(suite, c["instance_file"]))
            congs[c["name"]] = predict(cm, cck, inst, dev)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    jobs = [{"case": os.path.join(suite, c["instance_file"]), "name": c["name"],
             "prior": preds.get(c["name"], (None, 0))[0], "infer": preds.get(c["name"], (None, 0))[1],
             "tile": tile, "alpha": args.alpha if args.ckpt else 0.0,
             "max_iters": args.max_iters, "refine_passes": args.refine_passes,
             "lns_seconds": args.lns_seconds, "lns_k": args.lns_k, "seed": args.seed,
             "destroy": args.destroy, "anneal": args.anneal, "mix": tuple(args.mix),
             "stall_kick": args.stall_kick, "cong": congs.get(c["name"]),
             "warm": os.path.join(args.warm, f"{c['name']}.sol.json") if args.warm else None,
             "out": os.path.join(args.out_dir, f"{c['name']}.sol.json") if args.out_dir else None}
            for c in cases]
    base = {c["name"]: c["baseline_total"] for c in cases}
    print(f"[engine] {len(jobs)} cases from {args.suite}, prior={'yes' if args.ckpt else 'no'}",
          flush=True)
    results = []
    with Pool(min(args.workers, len(jobs))) as pool:
        for r in pool.imap_unordered(_job, jobs):
            results.append(r)
            ratio = f"{base[r['name']] / r['delay']:.4f}" if r["legal"] else "ILLEGAL"
            neg = r.get("delay_negotiated")
            print(f"  {r['name']}: ratio {ratio}  (after negotiation "
                  f"{base[r['name']] / neg:.4f})" if neg else f"  {r['name']}: {ratio}",
                  f" iters {r['iterations']}  refined {r.get('refined_nets', 0)} nets  "
                  + (f"lns {r['lns_moves']} moves ({r['lns_improving']} improving), "
                     f"ideal ratio {base[r['name']] / r['lns_ideal']:.4f}  "
                     if 'lns_moves' in r else "")
                  + f"{r['seconds']:.1f}s", flush=True)
    ok = [r for r in results if r["legal"]]
    agg = float(np.exp(np.mean([np.log(base[r["name"]] / r["delay"]) for r in ok]))) if ok else 0
    print(f"[engine] legal {len(ok)}/{len(results)}  aggregate "
          f"{agg if len(ok) == len(results) else 0:.4f}  (legal-only {agg:.4f})  "
          f"total {sum(r['seconds'] for r in results):.1f}s")
    if args.out_dir:
        with open(os.path.join(args.out_dir, "runtime.json"), "w") as fh:
            json.dump({r["name"]: r["seconds"] for r in results}, fh, indent=1)
    if args.json_out:
        keep = ("name", "legal", "delay", "seconds", "lns_moves", "lns_improving", "lns_start")
        with open(args.json_out, "w") as fh:
            json.dump({"aggregate": agg if len(ok) == len(results) else 0.0,
                       "baseline": base,
                       "cases": [{k_: r.get(k_) for k_ in keep} for r in results]}, fh, indent=1)


if __name__ == "__main__":
    main()
