"""Predecessor directions -> per-net trees (vertex ids, edge ids, true delay).

Runs on the CPU with Numba: following pointers is sequential, but cheap (a few
hundred steps per sink). Output arrays use the same vertex/edge ids as the CPU engine
(``engine/fast.py``), so trees can be handed straight to it.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numba import njit

from gpu.tensors import GridTensors


@njit(cache=True)
def _trace(pred, drivers, sink_ptr, sink_vid, net_idx, W, H, L, ldelay, via):
    """pred: [B, n] int8. Returns flat vertex/edge arrays with per-net pointers and the
    true (price-free) delay of each net; delay = -1 if a sink is unreachable."""
    WH = W * H
    B = pred.shape[0]
    n = pred.shape[1]
    mark = np.full(n, -1, dtype=np.int64)
    vs = [np.int64(0)]; vs.pop()
    es = [np.int64(0)]; es.pop()
    vptr = np.zeros(B + 1, dtype=np.int64)
    eptr = np.zeros(B + 1, dtype=np.int64)
    delay = np.zeros(B, dtype=np.float64)
    for b in range(B):
        i = net_idx[b]
        drv = drivers[i]
        mark[drv] = b
        vs.append(drv)
        ok = True
        total = 0.0
        for j in range(sink_ptr[i], sink_ptr[i + 1]):
            # true delay of this sink = sum of step delays along its full path
            v = sink_vid[j]
            while v != drv:
                k = pred[b, v]
                if k == 6:
                    ok = False
                    break
                z = v // WH
                if k == 0:
                    u = v - 1; e = 3 * u; w = ldelay[z]
                elif k == 1:
                    u = v + 1; e = 3 * v; w = ldelay[z]
                elif k == 2:
                    u = v - W; e = 3 * u + 1; w = ldelay[z]
                elif k == 3:
                    u = v + W; e = 3 * v + 1; w = ldelay[z]
                elif k == 4:
                    u = v - WH; e = 3 * u + 2; w = via
                else:
                    u = v + WH; e = 3 * v + 2; w = via
                total += w
                if mark[v] != b:          # first time this net's tree reaches v
                    mark[v] = b
                    vs.append(v)
                    es.append(e)
                v = u
            if not ok:
                break
        delay[b] = total if ok else -1.0
        vptr[b + 1] = len(vs)
        eptr[b + 1] = len(es)
    return np.array(vs, dtype=np.int64), vptr, np.array(es, dtype=np.int64), eptr, delay


def extract(gt: GridTensors, net_idx: np.ndarray, pred_dir) -> Dict[int, Tuple]:
    """Trees for a batch: {net index: (vertex ids, edge ids, true delay)}."""
    pred = pred_dir.reshape(len(net_idx), -1).cpu().numpy()
    drivers = gt.driver.cpu().numpy()
    vs, vptr, es, eptr, delay = _trace(pred, drivers, gt.sink_ptr, gt.sink_vid,
                                       np.asarray(net_idx, dtype=np.int64), gt.W, gt.H,
                                       gt.L, gt.ldelay, gt.via)
    out = {}
    for b, i in enumerate(net_idx):
        out[int(i)] = (vs[vptr[b]:vptr[b + 1]].copy(), es[eptr[b]:eptr[b + 1]].copy(),
                       float(delay[b]))
    return out
