"""GPU solution -> CPU engine: repair leftover overlaps, refine, LNS.

The GPU phase produces near-legal, delay-aware trees for every net. The CPU engine
(``engine/fast.py``) takes them as its starting state, reroutes only the nets that
still overlap (``FastRouter.repair``), then runs the usual exact refinement and LNS.
If repair fails, it falls back to the engine's own negotiation from scratch, so the
pipeline never produces less than the CPU-only result.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from engine.fast import FastRouter, build_router
from m3d.model import Instance, Submission


def finish(inst: Instance, routes: Dict, lns_seconds: float = 0.0, seed: int = 0,
           max_iters: int = 100, refine_passes: int = 20,
           policy=None) -> Tuple[Optional[Submission], Dict]:
    t0 = time.time()
    r = FastRouter(inst)
    r.beta = 0.5
    r.load_routes(routes)
    stats = {"start_delay": r.total_delay()}
    ok = r.repair(max_iters=max_iters)
    stats["repaired"] = ok
    if not ok:                                  # safety net: CPU-only pipeline
        r = build_router(inst, refine_passes=0)
        if r is None:
            return None, {**stats, "ok": False}
        stats["fallback"] = True
    stats["delay_repaired"] = r.total_delay()
    stats["refined_nets"] = r.refine(refine_passes)
    stats["delay_refined"] = r.total_delay()
    if lns_seconds > 0:
        st = r.lns(lns_seconds, seed=seed, policy=policy)
        stats.update({f"lns_{k}": v for k, v in st.items()})
    stats["delay"] = r.total_delay()
    stats["t_cpu"] = time.time() - t0
    return r.to_submission(), stats
