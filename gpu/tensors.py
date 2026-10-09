"""Instance -> GPU tensors.

Grid ids match ``m3d/grid.py`` and ``engine/fast.py``: ``vid = (z*H + y)*W + x``, so a
tensor shaped ``[L, H, W]`` flattened in C order is indexed by vid. Edge ids match the
engine too: the edge between u < v is ``eid = 3*u + d`` with d = 0 (+x), 1 (+y), 2 (+z),
so an edge-price vector of length ``3n`` viewed as ``[L, H, W, 3]`` gives the x/y/z
edge prices at their lower endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from m3d.model import Instance


@dataclass
class GridTensors:
    W: int
    H: int
    L: int
    n: int                      # number of grid vertices
    ldelay: np.ndarray          # [L] step delay per layer (CPU, float64)
    via: float
    cx: torch.Tensor            # [L, H, W-1] delay of +x edges
    cy: torch.Tensor            # [L, H-1, W] delay of +y edges
    cz: torch.Tensor            # [L-1, H, W] delay of vias
    pin_owner: torch.Tensor     # [L, H, W] net index owning a pin there, -1 if none
    net_ids: List[int]          # net index -> net id in the instance
    driver: torch.Tensor        # [N] driver vid per net index
    sink_ptr: np.ndarray        # [N+1] CSR pointers into sink_vid (CPU)
    sink_vid: np.ndarray        # [S] sink vids, grouped by net index (CPU)
    device: torch.device


def build(inst: Instance, device: torch.device) -> GridTensors:
    W, H, L = inst.width, inst.height, inst.layers
    n = W * H * L
    ld = np.asarray(inst.layer_delay, dtype=np.float64)
    lt = torch.tensor(ld, dtype=torch.float32, device=device)
    cx = lt.view(L, 1, 1).expand(L, H, W - 1).contiguous()
    cy = lt.view(L, 1, 1).expand(L, H - 1, W).contiguous()
    cz = torch.full((max(L - 1, 0), H, W), float(inst.via_delay), device=device)

    vid = lambda p: (p.z * H + p.y) * W + p.x
    pin_v = {p.id: vid(p) for p in inst.pins}
    nets = sorted(inst.nets, key=lambda nt: nt.id)
    owner = np.full(n, -1, dtype=np.int64)
    drivers, ptr, sinks = [], [0], []
    for i, nt in enumerate(nets):
        drivers.append(pin_v[nt.driver])
        for pid in nt.pins():
            owner[pin_v[pid]] = i
        s = sorted({pin_v[x] for x in nt.sinks})
        sinks.extend(s)
        ptr.append(len(sinks))
    return GridTensors(
        W=W, H=H, L=L, n=n, ldelay=ld, via=float(inst.via_delay), cx=cx, cy=cy, cz=cz,
        pin_owner=torch.tensor(owner, device=device).view(L, H, W),
        net_ids=[nt.id for nt in nets],
        driver=torch.tensor(drivers, dtype=torch.long, device=device),
        sink_ptr=np.asarray(ptr, dtype=np.int64), sink_vid=np.asarray(sinks, dtype=np.int64),
        device=device)


def batch_size_for(gt: GridTensors, budget_bytes: float = 1.5e9) -> int:
    """How many nets fit in one batched shortest-path call (~14 bytes per cell per net:
    distance, convergence copy, best/argmin scratch, blocked mask, direction)."""
    return max(1, int(budget_bytes // (14 * gt.n)))
