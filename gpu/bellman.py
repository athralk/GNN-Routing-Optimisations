"""Batched shortest paths on the GPU: Bellman-Ford as a 3D min-plus stencil.

On a regular grid, one Bellman-Ford step is "every cell takes the minimum over its
6 neighbours of (neighbour distance + edge cost)". That is a stencil -- like a
convolution with ``min`` instead of ``+`` -- and we run it for a whole batch of nets
at once on a tensor ``dist[B, L, H, W]``. (It's also exactly message passing with
min-aggregation: a hand-written GNN.)

Cost of entering vertex v from neighbour u over edge e:

    delay(e) + price_e[e] + price_v[v]

Other nets' pins are blocked (distance forced to +inf). Updates are done in place, one
direction after another, so information can travel several cells per sweep
(Gauss-Seidel-style); it still converges to the exact shortest-path distances.

After convergence each cell records the direction it was reached from (``pred_dir``,
int8): 0 from x-1, 1 from x+1, 2 from y-1, 3 from y+1, 4 from z-1, 5 from z+1, 6 =
none (driver, or unreachable). Following those pointers from any sink leads back to the
driver along a shortest path, and the union over a net's sinks is a tree.
"""
from __future__ import annotations

from typing import Tuple

import torch

from gpu.tensors import GridTensors

NO_PRED = 6
INF = float("inf")


def _edge_costs(gt: GridTensors, price_e: torch.Tensor):
    """Per-direction edge cost tensors (delay + price), shared by every net."""
    L, H, W = gt.L, gt.H, gt.W
    pe = price_e.view(L, H, W, 3)
    ex = gt.cx + pe[:, :, :-1, 0]
    ey = gt.cy + pe[:, :-1, :, 1]
    ez = gt.cz + pe[:-1, :, :, 2]
    return ex, ey, ez


def _sweep(dist: torch.Tensor, ex, ey, ez, pv) -> None:
    """One in-place relaxation sweep over all 6 directions."""
    # +x: reach x from x-1 ; -x: reach x from x+1
    torch.minimum(dist[..., 1:], dist[..., :-1] + ex + pv[..., 1:], out=dist[..., 1:])
    torch.minimum(dist[..., :-1], dist[..., 1:] + ex + pv[..., :-1], out=dist[..., :-1])
    # +y / -y
    torch.minimum(dist[..., 1:, :], dist[..., :-1, :] + ey + pv[:, 1:, :], out=dist[..., 1:, :])
    torch.minimum(dist[..., :-1, :], dist[..., 1:, :] + ey + pv[:, :-1, :], out=dist[..., :-1, :])
    # +z / -z (vias)
    if dist.shape[1] > 1:
        torch.minimum(dist[:, 1:], dist[:, :-1] + ez + pv[1:], out=dist[:, 1:])
        torch.minimum(dist[:, :-1], dist[:, 1:] + ez + pv[:-1], out=dist[:, :-1])


@torch.no_grad()
def batched_spt(gt: GridTensors, nets: torch.Tensor, price_e: torch.Tensor,
                price_v: torch.Tensor, check_every: int = 4,
                max_sweeps: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Shortest-path distances and predecessor directions for a batch of nets.

    nets:    [B] net indices (into ``gt.driver``)
    price_e: [3n] additive edge prices;  price_v: [n] additive vertex prices
    returns: dist [B, L, H, W] (float32), pred_dir [B, L, H, W] (int8)
    """
    B, L, H, W = len(nets), gt.L, gt.H, gt.W
    dev = gt.device
    ex, ey, ez = _edge_costs(gt, price_e)
    pv = price_v.view(L, H, W)

    # other nets' pins are walls
    owner = gt.pin_owner.unsqueeze(0)
    blocked = (owner >= 0) & (owner != nets.view(B, 1, 1, 1))

    dist = torch.full((B, L, H, W), INF, device=dev)
    dist.view(B, -1)[torch.arange(B, device=dev), gt.driver[nets]] = 0.0

    max_sweeps = max_sweeps or gt.n          # Bellman-Ford never needs more than n
    prev = torch.empty_like(dist)
    for sweep in range(max_sweeps):
        if sweep % check_every == 0:
            prev.copy_(dist)
        _sweep(dist, ex, ey, ez, pv)
        dist.masked_fill_(blocked, INF)
        if sweep % check_every == check_every - 1 and torch.equal(prev, dist):
            break

    # predecessor direction = the neighbour that achieves the minimum
    best = torch.full_like(dist, INF)
    pred = torch.full(dist.shape, NO_PRED, dtype=torch.int8, device=dev)

    def consider(k: int, region, cand):
        b = best[region]
        better = cand < b
        best[region] = torch.where(better, cand, b)
        p = pred[region]
        p[better] = k
        pred[region] = p

    s = slice(None)
    consider(0, (s, s, s, slice(1, None)), dist[..., :-1] + ex + pv[..., 1:])
    consider(1, (s, s, s, slice(None, -1)), dist[..., 1:] + ex + pv[..., :-1])
    consider(2, (s, s, slice(1, None), s), dist[..., :-1, :] + ey + pv[:, 1:, :])
    consider(3, (s, s, slice(None, -1), s), dist[..., 1:, :] + ey + pv[:, :-1, :])
    if L > 1:
        consider(4, (s, slice(1, None)), dist[:, :-1] + ez + pv[1:])
        consider(5, (s, slice(None, -1)), dist[:, 1:] + ez + pv[:-1])
    # the driver has no predecessor; unreachable cells neither
    pred.view(B, -1)[torch.arange(B, device=dev), gt.driver[nets]] = NO_PRED
    pred[torch.isinf(dist)] = NO_PRED
    return dist, pred
