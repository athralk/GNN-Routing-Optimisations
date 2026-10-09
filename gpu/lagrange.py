"""GPU Lagrangian router: every net routes at once under shared resource prices.

Each grid vertex and edge carries a price (in delay units). One iteration:

1. pick a subset of nets: every net touching an overused resource with probability
   ``p_over``, every other net with probability ``p_rest`` (rerouting only some nets
   per round is what stops all nets jumping to the same cheap spot together);
2. GPU: all picked nets compute their shortest-path trees simultaneously under
   ``cost = delay + price_e + price_v`` (``gpu/bellman.py``);
3. count how many nets use each resource (GPU scatter-add);
4. prices go up on overused resources (``+ step * overuse``) and decay on unused
   ones; the step size shrinks slowly;
5. remember the best state seen: least total overuse first, then lowest delay.

This is PathFinder done in parallel, written as Lagrangian relaxation (prices =
multipliers on the "one net per resource" constraints). With shared tree edges the
classic Lagrangian lower bound isn't valid (the objective counts an edge once per
sink, capacity once per net), so for "how close to optimal" use the engine's
``ideal_delays`` instead.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from gpu.bellman import batched_spt
from gpu.tensors import GridTensors, batch_size_for, build
from gpu.trees import extract
from m3d.model import Instance

Tree = Tuple[np.ndarray, np.ndarray, float]


class LagrangeRouter:
    def __init__(self, inst: Instance, device: torch.device, step0: float = 0.5,
                 step_decay: float = 0.99, unused_decay: float = 0.95,
                 p_over: float = 0.5, p_rest: float = 0.05, seed: int = 0,
                 batch: int = 0):
        self.inst = inst
        self.gt: GridTensors = build(inst, device)
        gt = self.gt
        self.dev = device
        self.N = len(gt.net_ids)
        self.price_e = torch.zeros(3 * gt.n, device=device)
        self.price_v = torch.zeros(gt.n, device=device)
        self.step = step0 * float(np.mean(gt.ldelay))   # prices live in delay units
        self.step_decay, self.unused_decay = step_decay, unused_decay
        self.p_over, self.p_rest = p_over, p_rest
        self.rng = np.random.default_rng(seed)
        self.batch = batch or batch_size_for(gt)
        self.trees: Dict[int, Tree] = {}
        self.best: Optional[Dict] = None
        self.history = []

    # -- routing -------------------------------------------------------------
    def _route(self, idx: np.ndarray) -> bool:
        for s in range(0, len(idx), self.batch):
            chunk = idx[s:s + self.batch]
            nets = torch.as_tensor(chunk, dtype=torch.long, device=self.dev)
            _, pred = batched_spt(self.gt, nets, self.price_e, self.price_v)
            for i, tree in extract(self.gt, chunk, pred).items():
                if tree[2] < 0:
                    return False                 # a sink is walled in by pins
                self.trees[i] = tree
        return True

    def _usage(self):
        gt = self.gt
        allv = np.concatenate([t[0] for t in self.trees.values()])
        alle = np.concatenate([t[1] for t in self.trees.values()])
        occ_v = torch.bincount(torch.as_tensor(allv, device=self.dev), minlength=gt.n)
        occ_e = torch.bincount(torch.as_tensor(alle, device=self.dev), minlength=3 * gt.n)
        return occ_v, occ_e

    def _overused_nets(self, occ_v, occ_e) -> np.ndarray:
        bad_v = (occ_v > 1).cpu().numpy()
        bad_e = (occ_e > 1).cpu().numpy()
        return np.array([i for i, (tv, te, _) in self.trees.items()
                         if bad_v[tv].any() or bad_e[te].any()], dtype=np.int64)

    # -- main loop -----------------------------------------------------------
    def run(self, iters: int = 300, time_limit: float = 120.0, patience: int = 30,
            verbose: bool = False) -> Dict:
        t0 = time.time()
        if not self._route(np.arange(self.N)):
            return {"ok": False, "reason": "unreachable sink"}
        since_best = 0
        for it in range(iters):
            occ_v, occ_e = self._usage()
            over_v = (occ_v - 1).clamp(min=0).float()
            over_e = (occ_e - 1).clamp(min=0).float()
            overuse = int(over_v.sum().item() + over_e.sum().item())
            delay = float(sum(t[2] for t in self.trees.values()))
            self.history.append((it, overuse, delay))
            key = (overuse, delay)
            if self.best is None or key < self.best["key"]:
                self.best = {"key": key, "it": it,
                             "trees": {i: (tv.copy(), te.copy(), d)
                                       for i, (tv, te, d) in self.trees.items()}}
                since_best = 0
            else:
                since_best += 1
            if verbose and it % 10 == 0:
                print(f"  it {it:4d}  overuse {overuse:6d}  delay {delay:10.0f}  "
                      f"step {self.step:.3f}  {time.time() - t0:.1f}s", flush=True)
            if (overuse == 0 and since_best >= patience) or time.time() - t0 > time_limit:
                break

            # prices: up where overused, decay where unused
            self.price_v += self.step * over_v
            self.price_e += self.step * over_e
            self.price_v[occ_v == 0] *= self.unused_decay
            self.price_e[occ_e == 0] *= self.unused_decay
            self.step *= self.step_decay

            # reroute a random subset, biased toward nets in conflict
            bad = set(self._overused_nets(occ_v, occ_e).tolist())
            pick = [i for i in range(self.N)
                    if self.rng.random() < (self.p_over if i in bad else self.p_rest)]
            if not pick:
                pick = [int(self.rng.integers(self.N))]
            if not self._route(np.asarray(pick, dtype=np.int64)):
                return {"ok": False, "reason": "unreachable sink"}
        b = self.best
        return {"ok": True, "iterations": len(self.history), "best_it": b["it"],
                "overuse": b["key"][0], "delay": b["key"][1],
                "seconds": time.time() - t0}

    def best_routes(self) -> Dict[int, Tree]:
        """Best state's trees keyed by *net id* (what the CPU engine expects)."""
        return {self.gt.net_ids[i]: t for i, t in self.best["trees"].items()}
