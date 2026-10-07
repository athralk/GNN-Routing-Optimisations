#!/usr/bin/env python3
"""Generate a training dataset of (instance, solution) pairs for the GNN.

Each sample is a fresh random instance from ``m3d.generator`` plus a verified
legal solution from one of the repo's routers (the "label"). We save the raw JSON
pair, not tensors: feature extraction lives in ``gnn/features.py``, so changing
features never requires regenerating data (the slow part).

Diversity is the point. Every sample draws its own grid size, layer count, delay
profile, net density, fanout, cross-die share and locality from a *profile*:

* ``sparse`` -- intro/scale-like: few nets per area, mostly local
* ``medium`` -- in between
* ``dense``  -- hard/congested-like: crowded middle layers, long nets

Output layout (resumable: re-running skips samples that already exist)::

    gnn/data/raw/<split>/
        sample_000000.inst.json    the problem (m3d-instance format)
        sample_000000.sol.json     the label   (m3d-submission format)
        ...
        index.jsonl                one line of metadata per finished sample

Usage (from the repo root)::

    python -m gnn.datagen --split train --n 2000 --workers 8
    python -m gnn.datagen --split val   --n 200
    python -m gnn.datagen --split train --summary
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

# make the repo root importable when run as a script (python gnn/datagen.py)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gnn._paths  # noqa: E402,F401  (finds the challenge repo's m3d toolkit)

from m3d.checker import check  # noqa: E402
from m3d.generator import GenConfig, generate_feasible  # noqa: E402

DEFAULT_OUT = os.path.join(_ROOT, "gnn", "data", "raw")

# Different salts so train / val / test never share a seed (no leakage), and
# none of them reuse the official benchmark master seed (20260923).
_SPLIT_SALT = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000,
               "val_large": 4_000_000}   # held-out big grids: size generalization
DATA_MASTER_SEED = 777_000_000

PROFILES = ("sparse", "medium", "dense")
PROFILE_WEIGHTS = (0.3, 0.3, 0.4)   # lean toward dense: that's where routing is hard


# ---------------------------------------------------------------------------
# sampling one configuration
# ---------------------------------------------------------------------------
def sample_config(idx: int, split: str, min_side: int, max_side: int,
                  profile: Optional[str] = None) -> Tuple[GenConfig, str]:
    """Draw a random-but-reproducible ``(GenConfig, profile)`` for sample ``idx``."""
    seed = DATA_MASTER_SEED + _SPLIT_SALT[split] + idx
    rng = random.Random(seed)
    prof = profile or rng.choices(PROFILES, weights=PROFILE_WEIGHTS)[0]

    side = rng.randint(min_side, max_side)
    # mostly 6 layers (the official setting), but vary it so the model can't
    # memorize "layer index z" and must learn from the delay values instead
    layers = rng.choices([4, 6, 8], weights=[0.2, 0.6, 0.2])[0]
    # delay profile: layer_delay[z] = center + slope * |2z - (layers-1)|
    center_delay = rng.choice([1, 1, 2])
    layer_slope = rng.choice([1, 1, 2])
    via_delay = rng.choice([2, 3, 3, 4, 5])

    if prof == "sparse":           # ~ intro / scale
        density = rng.uniform(0.35, 1.0)        # nets per unit of side length
        frac_local = rng.uniform(0.4, 0.6)
        p_twopin = rng.uniform(0.55, 0.75)
        max_fanout = rng.randint(3, 6)
        cell_min, cell_max, pins_per_cell, cell_gap = 2, rng.choice([3, 4]), 3, 1
        router, max_attempts = "baseline", 16
    elif prof == "medium":
        density = rng.uniform(1.0, 1.8)
        frac_local = rng.uniform(0.2, 0.45)
        p_twopin = rng.uniform(0.55, 0.7)
        max_fanout = rng.randint(4, 6)
        cell_min, cell_max, pins_per_cell, cell_gap = 2, rng.choice([2, 3]), \
            rng.choice([2, 3]), rng.choice([0, 1])
        router, max_attempts = "negotiated", 10
    else:                          # "dense" ~ hard / congested
        density = rng.uniform(1.8, 2.7)
        frac_local = rng.uniform(0.08, 0.2)
        p_twopin = rng.uniform(0.55, 0.65)
        max_fanout = rng.randint(5, 7)
        cell_min, cell_max, pins_per_cell, cell_gap = 2, 2, 2, 0
        router, max_attempts = "negotiated", 10

    # fewer layers = less routing room; without this, dense 4-layer configs are
    # often unroutable and the generator burns many retries (seen: 13 attempts)
    density *= min(1.0, layers / 6)

    return GenConfig(
        name=f"sample_{idx:06d}", width=side, height=side, layers=layers,
        center_delay=center_delay, layer_slope=layer_slope, via_delay=via_delay,
        n_nets=max(4, round(density * side)), frac_cross=rng.uniform(0.3, 0.5),
        p_twopin=p_twopin, max_fanout=max_fanout, frac_local=frac_local,
        cell_min=cell_min, cell_max=cell_max, pins_per_cell=pins_per_cell,
        cell_gap=cell_gap, seed=rng.randrange(1, 2 ** 31 - 1),
        master_seed=DATA_MASTER_SEED, max_attempts=max_attempts, router=router,
    ), prof


# ---------------------------------------------------------------------------
# one sample (runs in a worker process)
# ---------------------------------------------------------------------------
def _paths(out_dir: str, idx: int):
    stem = os.path.join(out_dir, f"sample_{idx:06d}")
    return stem + ".inst.json", stem + ".sol.json"


def _atomic_save(obj, path: str) -> None:
    # write to a temp file then rename, so a killed run never leaves half a file
    tmp = path + ".tmp"
    obj.save(tmp)
    os.replace(tmp, path)


def make_sample(job: Dict) -> Dict:
    idx, split, out_dir = job["idx"], job["split"], job["out_dir"]
    cfg, prof = sample_config(idx, split, job["min_side"], job["max_side"],
                              job.get("profile"))
    t0 = time.time()
    try:
        res = generate_feasible(cfg)
    except Exception as e:  # generator gave up (rare); record and move on
        return {"idx": idx, "ok": False, "error": f"{type(e).__name__}: {e}",
                "profile": prof, "seconds": round(time.time() - t0, 2)}
    inst, sol = res.instance, res.reference

    # paranoia: re-check with the independent checker before trusting a label
    chk = check(inst, sol)
    if not chk.legal:
        return {"idx": idx, "ok": False, "error": "label failed checker",
                "profile": prof, "seconds": round(time.time() - t0, 2)}

    inst_path, sol_path = _paths(out_dir, idx)
    _atomic_save(inst, inst_path)
    _atomic_save(sol, sol_path)
    return {
        "idx": idx, "ok": True, "profile": prof, "router": cfg.router,
        "width": inst.width, "height": inst.height, "layers": inst.layers,
        "layer_delay": inst.layer_delay, "via_delay": inst.via_delay,
        "n_nets": len(inst.nets), "n_pins": len(inst.pins),
        "total_delay": chk.total_delay, "gen_attempts": res.attempts,
        "seconds": round(time.time() - t0, 2),
        "instance_file": os.path.basename(inst_path),
        "solution_file": os.path.basename(sol_path),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _done_indices(out_dir: str) -> set:
    done = set()
    idx_path = os.path.join(out_dir, "index.jsonl")
    if os.path.exists(idx_path):
        with open(idx_path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    e = json.loads(line)
                    if e.get("ok"):
                        done.add(e["idx"])
    # only trust index entries whose files actually exist
    return {i for i in done if all(os.path.exists(p) for p in _paths(out_dir, i))}


def generate(split: str, n: int, start: int, workers: int, out_root: str,
             min_side: int, max_side: int, profile: Optional[str]) -> None:
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)
    done = _done_indices(out_dir)
    todo = [i for i in range(start, start + n) if i not in done]
    print(f"[datagen] split={split} out={out_dir}")
    print(f"[datagen] {len(done)} already done, {len(todo)} to generate, "
          f"{workers} workers, side {min_side}..{max_side}", flush=True)
    if not todo:
        return

    jobs = [{"idx": i, "split": split, "out_dir": out_dir, "min_side": min_side,
             "max_side": max_side, "profile": profile} for i in todo]
    t0 = time.time()
    n_ok = n_fail = 0
    # only the parent process writes index.jsonl, so no write races
    with open(os.path.join(out_dir, "index.jsonl"), "a") as idx_fh, \
            Pool(processes=workers) as pool:
        for k, entry in enumerate(pool.imap_unordered(make_sample, jobs), 1):
            idx_fh.write(json.dumps(entry) + "\n")
            idx_fh.flush()
            if entry["ok"]:
                n_ok += 1
            else:
                n_fail += 1
            el = time.time() - t0
            eta = el / k * (len(jobs) - k)
            tag = (f"{entry['width']}x{entry['height']}x{entry['layers']} "
                   f"{entry['n_nets']} nets, {entry['router']}"
                   if entry["ok"] else f"FAILED ({entry['error']})")
            print(f"  [{k}/{len(jobs)}] sample_{entry['idx']:06d} {entry['profile']:<6} "
                  f"{tag}  {entry['seconds']:.1f}s  | eta {eta / 60:.1f} min", flush=True)
    print(f"[datagen] finished: {n_ok} ok, {n_fail} failed, "
          f"{(time.time() - t0) / 60:.1f} min")


def summary(split: str, out_root: str) -> None:
    idx_path = os.path.join(out_root, split, "index.jsonl")
    if not os.path.exists(idx_path):
        print(f"no index at {idx_path}")
        return
    rows: List[Dict] = []
    with open(idx_path) as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    ok = [r for r in rows if r.get("ok")]
    print(f"{split}: {len(ok)} ok, {len(rows) - len(ok)} failed")
    for prof in PROFILES:
        rs = [r for r in ok if r["profile"] == prof]
        if not rs:
            continue
        sides = [r["width"] for r in rs]
        nets = [r["n_nets"] for r in rs]
        secs = [r["seconds"] for r in rs]
        print(f"  {prof:<6} n={len(rs):5d}  side {min(sides)}-{max(sides)}  "
              f"nets {min(nets)}-{max(nets)}  avg {sum(secs) / len(secs):.1f}s/sample")
    layers: Dict[int, int] = {}
    for r in ok:
        layers[r["layers"]] = layers.get(r["layers"], 0) + 1
    print(f"  layers: {dict(sorted(layers.items()))}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=sorted(_SPLIT_SALT))
    ap.add_argument("--n", type=int, default=100, help="number of samples")
    ap.add_argument("--start", type=int, default=0, help="first sample index")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=DEFAULT_OUT, help="root output dir")
    ap.add_argument("--min-side", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=40,
                    help="keep small: pure-Python routers get slow on big grids")
    ap.add_argument("--profile", choices=PROFILES, default=None,
                    help="force one profile (default: random mix)")
    ap.add_argument("--summary", action="store_true", help="print dataset stats and exit")
    args = ap.parse_args(argv)

    if args.summary:
        summary(args.split, args.out)
        return
    generate(args.split, args.n, args.start, args.workers, args.out,
             args.min_side, args.max_side, args.profile)


if __name__ == "__main__":   # required on Windows for multiprocessing
    main()
