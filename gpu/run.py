#!/usr/bin/env python3
"""Route a tier with the GPU Lagrangian router, finish on the CPU engine, and score.

Pipeline per case: GPU phase (all nets in parallel, ``gpu/lagrange.py``) -> CPU repair
of leftover overlaps -> exact refinement -> optional LNS (``gpu/handoff.py``). The GPU
phase runs one case at a time in this process; CPU finishing runs in a worker pool, so
case k is being finished while case k+1 is on the GPU.

Usage (from this repo's root)::

    python -m gpu.run --suite benchmarks_hard --cases case_01 --verbose
    python -m gpu.run --suite benchmarks_hard --gpu-seconds 60 --lns-seconds 300 --out-dir results/hard_gpu
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

import gpu  # noqa: E402,F401  (sets up paths)
from gnn._paths import repo_path  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.model import Instance  # noqa: E402


def _finish_job(job: Dict) -> Dict:
    from gpu.handoff import finish
    inst = Instance.load(job["case"])
    policy = None
    if job.get("destroy"):
        from gnn.destroy import load_policy
        policy = load_policy(job["destroy"])
    sub, st = finish(inst, job["routes"], job["lns_seconds"], job["seed"], policy=policy)
    out = {"name": job["name"], "legal": False, "delay": None, **job["gpu"], **st}
    if sub is not None:
        res = check(inst, sub)
        out["legal"], out["delay"] = res.legal, res.total_delay
        if res.legal and job["out"]:
            sub.save(job["out"])
    out["seconds"] = round(job["gpu"]["gpu_seconds"] + st.get("t_cpu", 0.0), 2)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--gpu-iters", type=int, default=300)
    ap.add_argument("--gpu-seconds", type=float, default=60.0, help="GPU phase cap per case")
    ap.add_argument("--step0", type=float, default=0.5, help="price step, x mean delay")
    ap.add_argument("--p-over", type=float, default=0.5)
    ap.add_argument("--p-rest", type=float, default=0.05)
    ap.add_argument("--lns-seconds", type=float, default=0.0)
    ap.add_argument("--destroy", help="destroy-policy GNN for the LNS (optional)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-dir")
    args = ap.parse_args(argv)

    import torch
    from gpu.lagrange import LagrangeRouter
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    suite = repo_path(args.suite)
    with open(os.path.join(suite, "suite.json")) as fh:
        cases = [c for c in json.load(fh)["cases"] if not args.cases or c["name"] in args.cases]
    base = {c["name"]: c["baseline_total"] for c in cases}
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    print(f"[gpu] {len(cases)} cases from {args.suite} on {dev}", flush=True)

    results, pending = [], []
    with Pool(min(args.workers, len(cases))) as pool:
        for c in cases:
            path = os.path.join(suite, c["instance_file"])
            inst = Instance.load(path)
            t0 = time.time()
            lr = LagrangeRouter(inst, dev, step0=args.step0, p_over=args.p_over,
                                p_rest=args.p_rest, seed=args.seed)
            st = lr.run(args.gpu_iters, args.gpu_seconds, verbose=args.verbose)
            gsec = time.time() - t0
            if not st["ok"]:
                print(f"  {c['name']}: GPU phase failed ({st['reason']})", flush=True)
                continue
            print(f"  {c['name']}: GPU {st['iterations']} iters in {gsec:.1f}s, best at "
                  f"{st['best_it']}: overuse {st['overuse']}, delay {st['delay']:.0f} "
                  f"(baseline {base[c['name']]})", flush=True)
            job = {"case": path, "name": c["name"], "routes": lr.best_routes(),
                   "gpu": {"gpu_seconds": gsec, "gpu_overuse": st["overuse"],
                           "gpu_delay": st["delay"]},
                   "lns_seconds": args.lns_seconds, "seed": args.seed,
                   "destroy": args.destroy,
                   "out": os.path.join(args.out_dir, f"{c['name']}.sol.json")
                   if args.out_dir else None}
            pending.append(pool.apply_async(_finish_job, (job,)))
        for p in pending:
            r = p.get()
            results.append(r)
            b = base[r["name"]]
            ratio = f"{b / r['delay']:.4f}" if r["legal"] else "ILLEGAL"
            print(f"  {r['name']}: final {ratio} | after repair {b / r['delay_repaired']:.4f} "
                  f"| after refine {b / r['delay_refined']:.4f}"
                  + (" | repair FAILED -> CPU fallback" if r.get("fallback") else "")
                  + f" | {r['seconds']:.1f}s", flush=True)

    ok = [r for r in results if r["legal"]]
    agg = float(np.exp(np.mean([np.log(base[r["name"]] / r["delay"]) for r in ok]))) if ok else 0
    print(f"[gpu] legal {len(ok)}/{len(cases)}  aggregate "
          f"{agg if len(ok) == len(cases) else 0:.4f}  (legal-only {agg:.4f})")
    if args.out_dir:
        with open(os.path.join(args.out_dir, "runtime.json"), "w") as fh:
            json.dump({r["name"]: r["seconds"] for r in results}, fh, indent=1)


if __name__ == "__main__":
    main()
