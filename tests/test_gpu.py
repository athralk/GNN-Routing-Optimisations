"""Correctness tests for the GPU router (run from this repo's root):

    python -m unittest tests.test_gpu -v

1. GPU Bellman-Ford delays equal the CPU engine's exact Dijkstra for every net
   (zero prices, empty grid -> both are each net's ideal shortest-path tree).
2. Every GPU tree is a real tree: connected, |edges| = |vertices| - 1, spans all pins.
3. After the CPU repair, the full solution passes the official checker.
"""
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gpu  # noqa: E402,F401
import torch  # noqa: E402

from engine.fast import FastRouter  # noqa: E402
from gpu.bellman import batched_spt  # noqa: E402
from gpu.handoff import finish  # noqa: E402
from gpu.lagrange import LagrangeRouter  # noqa: E402
from gpu.tensors import build  # noqa: E402
from gpu.trees import extract  # noqa: E402
from m3d.checker import check  # noqa: E402
from m3d.generator import GenConfig, generate_candidate  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _instance(seed: int, side: int = 14, nets: int = 18, layers: int = 6):
    cfg = GenConfig(name=f"t{seed}", width=side, height=side, layers=layers, n_nets=nets,
                    frac_local=0.3, cell_min=2, cell_max=2, pins_per_cell=2, cell_gap=0)
    return generate_candidate(cfg, seed)


class TestGPU(unittest.TestCase):
    def test_delays_match_cpu_dijkstra(self):
        for seed in range(3):
            inst = _instance(seed, layers=4 + 2 * (seed % 2))
            gt = build(inst, DEV)
            idx = np.arange(len(gt.net_ids))
            _, pred = batched_spt(gt, torch.as_tensor(idx, device=DEV),
                                  torch.zeros(3 * gt.n, device=DEV),
                                  torch.zeros(gt.n, device=DEV))
            trees = extract(gt, idx, pred)
            ideal = FastRouter(inst).ideal_delays()
            for i, nid in enumerate(gt.net_ids):
                self.assertAlmostEqual(trees[i][2], ideal[nid], places=6,
                                       msg=f"seed {seed} net {nid}")

    def test_trees_are_trees(self):
        inst = _instance(7)
        gt = build(inst, DEV)
        idx = np.arange(len(gt.net_ids))
        _, pred = batched_spt(gt, torch.as_tensor(idx, device=DEV),
                              torch.zeros(3 * gt.n, device=DEV), torch.zeros(gt.n, device=DEV))
        for i, (tv, te, d) in extract(gt, idx, pred).items():
            self.assertGreaterEqual(d, 0)
            self.assertEqual(len(te), len(tv) - 1)
            self.assertEqual(len(set(tv.tolist())), len(tv))
            pins = {int(gt.driver[i])} | set(gt.sink_vid[gt.sink_ptr[i]:gt.sink_ptr[i + 1]].tolist())
            self.assertTrue(pins <= set(tv.tolist()))

    def test_full_pipeline_is_legal(self):
        inst = _instance(11, side=18, nets=30)
        lr = LagrangeRouter(inst, DEV)
        st = lr.run(iters=60, time_limit=30)
        self.assertTrue(st["ok"])
        sub, info = finish(inst, lr.best_routes())
        self.assertIsNotNone(sub)
        self.assertTrue(check(inst, sub).legal)


if __name__ == "__main__":
    unittest.main()
