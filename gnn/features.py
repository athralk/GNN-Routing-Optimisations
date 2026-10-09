#!/usr/bin/env python3
"""Turn raw (instance, solution) JSON pairs into tile-graph tensors for the GNN.

Version 1 task: *congestion prediction*. The 3D routing grid is coarsened into
tiles of ``T x T`` grid vertices, one graph node per (tile, layer). The model
sees only the problem (pins, nets, delays) and must predict how heavily a good
router uses each tile on each layer.

Everything is size- and stack-agnostic on purpose: no feature depends on the
absolute grid size, the absolute layer index or the total number of nets, so the
same trained model can run on a 16x16x4 case or a 530x530x6 one.

Node features (one row per tile-layer node), in order -- see ``FEATURE_NAMES``:

* layer physics: normalized step delay, fastness rank, via cost, relative
  height in the stack, is-die-layer
* geometry: fraction of the tile that is inside the grid (edge tiles are partial)
* pins: pin density on this tile-layer, and on the bottom / top die below/above it
* RUDY (Rectangular Uniform wire DensitY, the classic quick congestion estimate:
  spread each net's half-perimeter wirelength evenly over its bounding box and
  sum over nets), split by net length (short / medium / long), cross-die nets,
  and a fanout-weighted version (the objective counts every sink separately)

Graph edges: tile-layer to its 4 lateral neighbours on the same layer and to the
tiles directly above/below. Edge features: ``[is_via, step_delay / max_delay]``.

Targets ``y`` (one row per node):

* ``y[:, 0]`` = fraction of the tile-layer's lateral grid edges used by the solution
* ``y[:, 1]`` = fraction of possible vias from this layer to the next one used
  (undefined on the top layer -> ``y_mask[:, 1] = 0`` there)

Only numpy + torch are needed (no PyTorch Geometric). Each saved ``.pt`` is a plain
dict; ``gnn/train.py`` wraps it in a PyG ``Data`` object.

Usage (from the repo root)::

    python -m gnn.features --split train            # process all raw samples
    python -m gnn.features --split val
    python -m gnn.features --viz train/sample_000004 # picture: features vs target
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, Tuple

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gnn._paths  # noqa: E402,F401  (finds the challenge repo's m3d toolkit)

from m3d.model import Instance, Submission  # noqa: E402

RAW_ROOT = os.path.join(_ROOT, "gnn", "data", "raw")
PROC_ROOT = os.path.join(_ROOT, "gnn", "data", "processed")
DEFAULT_TILE = 4
# bump when FEATURE_NAMES changes; processed data lives in processed/v<N>_tile<T>/
# so models trained on an older feature set keep working (features are append-only)
FEATURE_VERSION = 2
EDGE_DIST_CAP = 4   # tiles; "4+ tiles from the border" all look the same (= interior)


DATASET = ""   # "" = reference-router labels (raw/); "engine" = fast-engine labels (raw_engine/)


def set_dataset(name: str) -> None:
    """Switch every path in this module to another labelled dataset."""
    global DATASET, RAW_ROOT
    DATASET = name or ""
    RAW_ROOT = os.path.join(_ROOT, "gnn", "data", f"raw_{name}" if name else "raw")


def proc_dir(tile: int, split: str, version: int = FEATURE_VERSION) -> str:
    # v1 data predates versioning and lives in processed/tile<T>/
    name = f"tile{tile}" if version == 1 else f"v{version}_tile{tile}"
    if DATASET:
        name = f"{DATASET}_{name}"
    return os.path.join(PROC_ROOT, name, split)

# net-length buckets for the RUDY split, in grid steps (absolute on purpose:
# delay is per step, so "long" means the same thing on any grid size)
SHORT_MAX, MEDIUM_MAX = 4, 12

FEATURE_NAMES = [
    "layer_delay_norm",   # step delay / max step delay
    "layer_fastness",     # 1 = fastest layer of the stack, 0 = slowest
    "via_norm",           # via delay / max step delay
    "rel_z",              # 0 = bottom die layer, 1 = top die layer
    "is_die_layer",
    "area_frac",          # part of the tile inside the grid
    "pin_density_here",   # pins on this tile-layer per vertex
    "pin_density_bottom", # pins on the bottom die in this tile column
    "pin_density_top",    # pins on the top die in this tile column
    "rudy_all",           # log1p of RUDY, all nets
    "rudy_short",
    "rudy_medium",
    "rudy_long",
    "rudy_cross_die",
    "rudy_fanout",        # each net weighted by its number of sinks
    # --- v2 (append-only, so v1 models can still slice out the first 15) ---
    "cross_pin_here",     # pins of cross-die nets on this tile-layer per vertex
    "cross_pin_column",   # cross-die pins in this tile column (either die): via stacks
    "cross_sink_column",  # same, counting each cross-die net's sinks (delay weight)
    "edge_dist",          # tiles to the nearest grid border, capped, in [0, 1]
]
FEATURES_V1 = FEATURE_NAMES[:15]
N_FEATURES = len(FEATURE_NAMES)
TARGET_NAMES = ["lateral_usage", "via_usage"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def pool_tiles(a: np.ndarray, t: int) -> np.ndarray:
    """Sum the last two axes (H, W) of ``a`` into ceil(H/t) x ceil(W/t) tiles."""
    h, w = a.shape[-2:]
    th, tw = -(-h // t), -(-w // t)
    pad = [(0, 0)] * (a.ndim - 2) + [(0, th * t - h), (0, tw * t - w)]
    a = np.pad(a, pad)
    a = a.reshape(a.shape[:-2] + (th, t, tw, t))
    return a.sum(axis=(-3, -1))


def _rudy_map(boxes, h: int, w: int) -> np.ndarray:
    """RUDY over the H x W vertex grid from ``(x0, y0, x1, y1, weight)`` boxes."""
    diff = np.zeros((h + 1, w + 1), dtype=np.float64)
    for x0, y0, x1, y1, wt in boxes:
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        dens = wt * ((x1 - x0) + (y1 - y0)) / (bw * bh)   # wire per vertex
        diff[y0, x0] += dens
        diff[y0, x1 + 1] -= dens
        diff[y1 + 1, x0] -= dens
        diff[y1 + 1, x1 + 1] += dens
    # clip float round-off from the prefix sums (would show up as -0.000)
    return np.maximum(diff.cumsum(0).cumsum(1)[:h, :w], 0.0)


# ---------------------------------------------------------------------------
# instance -> graph
# ---------------------------------------------------------------------------
def build_graph(inst: Instance, t: int = DEFAULT_TILE) -> Dict[str, np.ndarray]:
    W, H, L = inst.width, inst.height, inst.layers
    TW, TH = -(-W // t), -(-H // t)
    n_nodes = L * TH * TW
    ld = np.asarray(inst.layer_delay, dtype=np.float64)
    dmax, dmin = ld.max(), ld.min()

    # --- per-tile area (edge tiles are partial)
    area = pool_tiles(np.ones((H, W)), t)                       # [TH, TW]

    # --- pins per vertex, per layer
    pins = np.zeros((L, H, W))
    for p in inst.pins:
        pins[p.z, p.y, p.x] += 1
    pins_t = pool_tiles(pins, t) / area                          # [L, TH, TW]

    # --- cross-die pins: a net spanning both dies must stack vias through every
    # layer somewhere near its pins, which shows up as vertical via "columns"
    pin_xy = {p.id: (p.x, p.y, p.die) for p in inst.pins}
    pin_obj = {p.id: p for p in inst.pins}
    cross = np.zeros((L, H, W))
    cross_sink = np.zeros((H, W))
    for n in inst.nets:
        if len({pin_xy[pid][2] for pid in n.pins()}) < 2:
            continue
        for pid in n.pins():
            p = pin_obj[pid]
            cross[p.z, p.y, p.x] += 1
            cross_sink[p.y, p.x] += len(n.sinks)
    cross_t = pool_tiles(cross, t) / area                        # [L, TH, TW]
    cross_col = cross_t.sum(axis=0)                              # [TH, TW]
    cross_sink_col = np.log1p(pool_tiles(cross_sink, t) / area)

    # --- distance (in tiles) to the nearest border, capped
    ty, tx = np.meshgrid(np.arange(TH), np.arange(TW), indexing="ij")
    edge_dist = np.minimum(np.minimum(tx, TW - 1 - tx), np.minimum(ty, TH - 1 - ty))
    edge_dist = np.minimum(edge_dist, EDGE_DIST_CAP) / EDGE_DIST_CAP

    # --- RUDY maps (2D, shared by every layer of a tile column)
    buckets = {"all": [], "short": [], "medium": [], "long": [], "cross": [], "fan": []}
    for n in inst.nets:
        pts = [pin_xy[pid] for pid in n.pins()]
        xs = [q[0] for q in pts]
        ys = [q[1] for q in pts]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        hpwl = (x1 - x0) + (y1 - y0)
        box = (x0, y0, x1, y1, 1.0)
        buckets["all"].append(box)
        key = "short" if hpwl <= SHORT_MAX else "medium" if hpwl <= MEDIUM_MAX else "long"
        buckets[key].append(box)
        if len({q[2] for q in pts}) > 1:
            buckets["cross"].append(box)
        buckets["fan"].append((x0, y0, x1, y1, float(len(n.sinks))))
    rudy = {k: np.log1p(pool_tiles(_rudy_map(v, H, W), t) / area)
            for k, v in buckets.items()}                         # each [TH, TW]

    # --- assemble node features [L, TH, TW, F]
    z = np.arange(L, dtype=np.float64)[:, None, None] * np.ones((1, TH, TW))
    feat = np.zeros((L, TH, TW, N_FEATURES), dtype=np.float32)
    feat[..., 0] = (ld / dmax)[:, None, None]
    feat[..., 1] = ((dmax - ld) / (dmax - dmin if dmax > dmin else 1.0))[:, None, None]
    feat[..., 2] = inst.via_delay / dmax
    feat[..., 3] = z / max(1, L - 1)
    feat[..., 4] = ((z == 0) | (z == L - 1)).astype(np.float32)
    feat[..., 5] = area / (t * t)
    feat[..., 6] = pins_t
    feat[..., 7] = pins_t[0]
    feat[..., 8] = pins_t[L - 1]
    for i, k in enumerate(["all", "short", "medium", "long", "cross", "fan"]):
        feat[..., 9 + i] = rudy[k]
    feat[..., 15] = cross_t
    feat[..., 16] = cross_col
    feat[..., 17] = cross_sink_col
    feat[..., 18] = edge_dist
    x = feat.reshape(n_nodes, N_FEATURES)

    # --- edges (both directions), node id = (z*TH + ty)*TW + tx
    nid = np.arange(n_nodes).reshape(L, TH, TW)
    src, dst, attr = [], [], []

    def add(a: np.ndarray, b: np.ndarray, is_via: float, delay: np.ndarray):
        a, b, delay = a.ravel(), b.ravel(), delay.ravel()
        src.extend([a, b]); dst.extend([b, a])
        e = np.stack([np.full_like(delay, is_via), delay], axis=1)
        attr.extend([e, e])

    lat_delay = (ld / dmax)[:, None, None]
    add(nid[:, :, :-1], nid[:, :, 1:], 0.0, np.broadcast_to(lat_delay, (L, TH, TW - 1)))
    add(nid[:, :-1, :], nid[:, 1:, :], 0.0, np.broadcast_to(lat_delay, (L, TH - 1, TW)))
    add(nid[:-1], nid[1:], 1.0, np.full((L - 1, TH, TW), inst.via_delay / dmax))
    edge_index = np.stack([np.concatenate(src), np.concatenate(dst)]).astype(np.int64)
    edge_attr = np.concatenate(attr).astype(np.float32)

    pos = np.stack(np.meshgrid(np.arange(L), np.arange(TH), np.arange(TW),
                               indexing="ij"), axis=-1).reshape(n_nodes, 3)
    return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr,
            "pos": pos.astype(np.int64),  # (z, ty, tx), for plotting only -- NOT a feature
            "shape": np.array([L, TH, TW], dtype=np.int64),
            "grid": np.array([W, H, L], dtype=np.int64)}   # for filtering by size


# ---------------------------------------------------------------------------
# solution -> targets
# ---------------------------------------------------------------------------
def build_targets(inst: Instance, sol: Submission, t: int = DEFAULT_TILE
                  ) -> Tuple[np.ndarray, np.ndarray]:
    W, H, L = inst.width, inst.height, inst.layers
    lat = np.zeros((L, H, W))       # lateral edges, counted at their lower vertex
    via = np.zeros((L, H, W))       # via z -> z+1, counted at layer z
    for r in sol.routes:
        for u, v in r.edges:
            (ux, uy, uz), (vx, vy, vz) = u, v
            if uz != vz:
                via[min(uz, vz), uy, ux] += 1
            else:
                lat[uz, min(uy, vy), min(ux, vx)] += 1
    cap_lat = np.zeros((L, H, W))
    cap_lat[:, :, :-1] += 1          # +x edges
    cap_lat[:, :-1, :] += 1          # +y edges
    cap_via = np.zeros((L, H, W))
    cap_via[:-1] = 1

    lat_t, cap_lat_t = pool_tiles(lat, t), pool_tiles(cap_lat, t)
    via_t, cap_via_t = pool_tiles(via, t), pool_tiles(cap_via, t)
    y = np.zeros(lat_t.shape + (2,), dtype=np.float32)
    mask = np.zeros_like(y)
    ok = cap_lat_t > 0
    y[..., 0][ok] = lat_t[ok] / cap_lat_t[ok]
    mask[..., 0] = ok
    ok = cap_via_t > 0
    y[..., 1][ok] = via_t[ok] / cap_via_t[ok]
    mask[..., 1] = ok
    return y.reshape(-1, 2), mask.reshape(-1, 2)


# ---------------------------------------------------------------------------
# processing a split
# ---------------------------------------------------------------------------
def process_one(job: Dict) -> Dict:
    import torch
    stem, out_path, t = job["stem"], job["out"], job["tile"]
    inst = Instance.load(stem + ".inst.json")
    sol = Submission.load(stem + ".sol.json")
    g = build_graph(inst, t)
    y, y_mask = build_targets(inst, sol, t)
    item = {k: torch.from_numpy(v) for k, v in g.items()}
    item["y"] = torch.from_numpy(y)
    item["y_mask"] = torch.from_numpy(y_mask)
    item["name"] = os.path.basename(stem)
    item["tile"] = t
    tmp = out_path + ".tmp"
    torch.save(item, tmp)
    os.replace(tmp, out_path)
    return {"name": item["name"], "nodes": int(g["x"].shape[0])}


def process_split(split: str, tile: int, workers: int) -> None:
    raw = os.path.join(RAW_ROOT, split)
    out = proc_dir(tile, split)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(raw, "index.jsonl")) as fh:
        names = sorted({json.loads(l)["instance_file"][:-len(".inst.json")]
                        for l in fh if l.strip() and json.loads(l).get("ok")})
    jobs = [{"stem": os.path.join(raw, n), "out": os.path.join(out, n + ".pt"), "tile": tile}
            for n in names if not os.path.exists(os.path.join(out, n + ".pt"))]
    print(f"[features] {split}: {len(names)} samples, {len(jobs)} to process -> {out}")
    t0 = time.time()
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(process_one, jobs, chunksize=8), 1):
            if k % 200 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(out, "features.json"), "w") as fh:
        json.dump({"tile": tile, "version": FEATURE_VERSION, "features": FEATURE_NAMES,
                   "targets": TARGET_NAMES}, fh,
                  indent=1)


# ---------------------------------------------------------------------------
# visualization: does the input "look like" the answer?
# ---------------------------------------------------------------------------
def visualize(sample: str, tile: int, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stem = os.path.join(RAW_ROOT, sample)
    inst = Instance.load(stem + ".inst.json")
    sol = Submission.load(stem + ".sol.json")
    g = build_graph(inst, tile)
    y, _ = build_targets(inst, sol, tile)
    L, TH, TW = g["shape"]
    x = g["x"].reshape(L, TH, TW, N_FEATURES)
    y = y.reshape(L, TH, TW, 2)

    fig, axes = plt.subplots(3, L, figsize=(2.6 * L, 7.8), squeeze=False)
    for z in range(L):
        panels = [(x[z, :, :, FEATURE_NAMES.index("rudy_all")], "input: RUDY", "viridis"),
                  (y[z, :, :, 0], "target: lateral use", "magma"),
                  (y[z, :, :, 1], "target: via use", "magma")]
        for row, (img, title, cmap) in enumerate(panels):
            ax = axes[row][z]
            ax.imshow(img, origin="lower", cmap=cmap, vmin=0,
                      vmax=1 if row else max(1e-6, x[..., 9].max()))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"z={z}  delay={inst.layer_delay[z]}", fontsize=9)
            if z == 0:
                ax.set_ylabel(title, fontsize=9)
    fig.suptitle(f"{inst.name}: {inst.width}x{inst.height}x{inst.layers}, "
                 f"{len(inst.nets)} nets, tile={tile}  (RUDY is 2D: same on every layer)",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110)
    print(f"[features] wrote {out_png}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train")
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--dataset", default="", help="'engine' = use gnn/data/raw_engine")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--viz", metavar="SPLIT/SAMPLE",
                    help="plot features vs targets for one sample, e.g. train/sample_000004")
    args = ap.parse_args(argv)
    if args.viz:
        out = os.path.join(_ROOT, "gnn", "data", "viz",
                           args.viz.replace("/", "_") + f"_tile{args.tile}.png")
        visualize(args.viz, args.tile, out)
        return
    set_dataset(args.dataset)
    process_split(args.split, args.tile, args.workers)


if __name__ == "__main__":
    main()
