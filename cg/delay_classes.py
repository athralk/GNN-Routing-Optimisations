#!/usr/bin/env python3
"""How many *genuinely different* routes does a sink have?

Delay depends only on how many steps run on each layer plus the number of vias, not
on where the bends are. For every (driver, sink) pair we enumerate the set of
distinct delay values over all xy-shortest paths (any layer changes allowed) and
compare it with the raw number of such paths.

Usage::

    python -m cg.delay_classes benchmarks_hard/case_09.json benchmarks_designs/router.json
"""
from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import engine  # noqa: E402,F401
from gnn._paths import repo_path  # noqa: E402
from m3d.model import Instance  # noqa: E402


def delay_values(n_steps, z0, z1, ld, via, slack):
    """Distinct delays of xy-shortest paths with n_steps lateral steps, within slack."""
    L = len(ld)

    def vias(sets):
        out = [set() for _ in range(L)]
        for z in range(L):
            for d in sets[z]:
                for z2 in range(L):
                    out[z2].add(d + via * abs(z - z2))
        return out

    sets = [set() for _ in range(L)]
    sets[z0].add(0)
    sets = vias(sets)
    for _ in range(n_steps):
        sets = vias([{d + ld[z] for d in sets[z]} for z in range(L)])
        lo = min(min(s) for s in sets if s)
        cap = lo + (n_steps * max(ld) + via * L)          # loose prune
        sets = [{d for d in s if d <= cap} for s in sets]
    final = {d + via * abs(z - z1) for z in range(L) for d in sets[z]}
    best = min(final)
    return best, sorted(d for d in final if d <= best * (1 + slack))


def main(paths):
    for p in paths:
        inst = Instance.load(repo_path(p))
        ld, via = list(inst.layer_delay), inst.via_delay
        pins = {q.id: q for q in inst.pins}
        n_opt, n_10, n_30, log_paths = [], [], [], []
        for net in inst.nets:
            d = pins[net.driver]
            for s in set(net.sinks):
                t = pins[s]
                dx, dy = abs(d.x - t.x), abs(d.y - t.y)
                best, vals = delay_values(dx + dy, d.z, t.z, ld, via, 0.30)
                n_30.append(len(vals))
                n_10.append(sum(v <= best * 1.10 for v in vals))
                log_paths.append(math.log10(math.comb(dx + dy, dx)) if dx + dy else 0.0)
        a = np.array
        print(f"{inst.name}: {len(n_30)} sinks, layer delays {ld}, via {via}")
        print(f"  raw xy-shortest paths per sink (one layer): median 10^{np.median(log_paths):.1f}, "
              f"max 10^{max(log_paths):.1f}")
        print(f"  distinct delay values within +10% of best: median {np.median(a(n_10)):.0f}, "
              f"max {max(n_10)}")
        print(f"  distinct delay values within +30% of best: median {np.median(a(n_30)):.0f}, "
              f"max {max(n_30)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["benchmarks_hard/case_09.json", "benchmarks_designs/router.json"])
