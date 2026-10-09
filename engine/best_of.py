#!/usr/bin/env python3
"""Combine several result folders into one submission: the best legal route per case.

Every candidate is re-verified with the official checker. ``runtime.json`` sums the
runtime of all contributing runs, so the combined entry reports its true total cost.

Usage (from this repo's root)::

    python -m engine.best_of --suite benchmarks_hard --runs results/hard_engine results/hard_lns3h --out results/hard_best
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import engine  # noqa: E402,F401
from gnn._paths import repo_path  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.model import Instance, Submission  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    suite = repo_path(args.suite)
    cases = json.load(open(os.path.join(suite, "suite.json")))["cases"]
    os.makedirs(args.out, exist_ok=True)
    runtime, picks = {}, {}
    for c in cases:
        inst = Instance.load(os.path.join(suite, c["instance_file"]))
        best, best_d, total_t = None, None, 0.0
        for r in args.runs:
            f = os.path.join(r, f"{c['name']}.sol.json")
            if not os.path.exists(f):
                continue
            rt = os.path.join(r, "runtime.json")
            if os.path.exists(rt):
                total_t += json.load(open(rt)).get(c["name"], 0.0)
            res = check(inst, Submission.load(f))
            if res.legal and (best_d is None or res.total_delay < best_d):
                best, best_d = r, res.total_delay
        if best is None:
            print(f"  {c['name']}: no legal candidate")
            continue
        shutil.copy(os.path.join(best, f"{c['name']}.sol.json"), args.out)
        runtime[c["name"]] = round(total_t, 2)
        picks[c["name"]] = (os.path.basename(best), round(c["baseline_total"] / best_d, 4))
    json.dump(runtime, open(os.path.join(args.out, "runtime.json"), "w"), indent=1)
    for k, (r, ratio) in picks.items():
        print(f"  {k}: {ratio}  from {r}")


if __name__ == "__main__":
    main()
