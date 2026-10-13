#!/usr/bin/env python3
"""Equal-time A/B test of LNS policies on a tier (Phase 0 measurement harness).

Each config runs ``engine.run`` with the same time budget, seeds and (optionally) the
same warm start; we compare the aggregate score and *improving moves per minute* (the
leading indicator: it moves long before the score does).

Usage (from this repo's root)::

    python -m gnn.ab --suite benchmarks_hard --seconds 600 --seeds 1 2 --warm results/hard_final \
        --config v2="--destroy models/destroy_v2.pt" --config v3="--destroy models/destroy_v3.pt"
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="benchmarks_hard")
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--warm", help="start every run from these saved solutions")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--config", action="append", required=True,
                    help='NAME="extra engine.run args" (repeatable)')
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "ab"))
    ap.add_argument("--report", default=None, help="write the summary table (markdown) here")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for cfg in args.config:
        name, extra = cfg.split("=", 1)
        aggs, rates = [], []
        for seed in args.seeds:
            tag = f"{name}_s{seed}"
            js = os.path.join(args.out, f"{tag}.json")
            cmd = [sys.executable, "-m", "engine.run", "--suite", args.suite,
                   "--workers", str(args.workers), "--lns-seconds", str(args.seconds),
                   "--seed", str(seed), "--out-dir", os.path.join(args.out, tag),
                   "--json-out", js] + (["--warm", args.warm] if args.warm else []) + \
                shlex.split(extra)
            print(f"[ab] {tag}: {' '.join(cmd[2:])}", flush=True)
            with open(os.path.join(args.out, f"{tag}.log"), "w") as log:
                subprocess.run(cmd, cwd=_ROOT, stdout=log, stderr=subprocess.STDOUT)
            res = json.load(open(js))
            imp = [c["lns_improving"] or 0 for c in res["cases"]]
            aggs.append(res["aggregate"])
            rates.append(float(np.mean(imp)) / (args.seconds / 60))
            print(f"[ab] {tag}: aggregate {res['aggregate']:.4f}, "
                  f"{rates[-1]:.1f} improving moves/min per case", flush=True)
        rows.append((name, float(np.mean(aggs)), aggs, float(np.mean(rates))))

    lines = [f"| config | mean aggregate | per seed | improving moves/min |", "|---|---:|---|---:|"]
    for name, m, aggs, r in sorted(rows, key=lambda t: -t[1]):
        lines.append(f"| {name} | {m:.4f} | {', '.join(f'{a:.4f}' for a in aggs)} | {r:.1f} |")
    table = "\n".join(lines)
    print(table)
    best = max(rows, key=lambda t: t[1])[0]
    print(f"[ab] winner: {best}")
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(table + f"\n\nwinner: {best}\n")


if __name__ == "__main__":
    main()
