"""GNN that predicts per tile-layer routing usage (Version 1: congestion prediction).

Architecture (all standard pieces):

    x ──Linear──► h ──[ GATv2 message passing + residual ] x n_layers ──MLP──► sigmoid ──► usage

* ``GATv2Conv``: each node updates itself from a learned, attention-weighted mix of
  its neighbours' messages. Edge features ``[is_via, step_delay]`` feed into the
  attention, so the model can treat "move sideways on a slow layer" differently
  from "go up a via".
* Pre-norm residual blocks (``h = h + conv(norm(h))``) keep deep stacks trainable.
* ``n_layers = 0`` turns it into a plain per-node MLP that never looks at
  neighbours -- the "does message passing help at all?" ablation.

Receptive field: after k layers a node has heard from tiles up to k hops away, so
with 4x4 tiles and 8 layers it "sees" ~32 grid steps sideways, and every layer of
the stack (6 layers deep needs only 5 vertical hops).
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class CongestionGNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 2, hidden: int = 128,
                 n_layers: int = 8, heads: int = 4, edge_dim: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        assert hidden % heads == 0, "hidden must be divisible by heads"
        self.config: Dict = dict(in_dim=in_dim, out_dim=out_dim, hidden=hidden,
                                 n_layers=n_layers, heads=heads, edge_dim=edge_dim,
                                 dropout=dropout)
        self.inp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim,
                      add_self_loops=True)
            for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))
        self.ffs = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.GELU(),
                          nn.Linear(2 * hidden, hidden))
            for _ in range(n_layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                  nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, x, edge_index, edge_attr):
        h = self.inp(x)
        for conv, norm, ff in zip(self.convs, self.norms, self.ffs):
            h = h + self.drop(conv(norm(h), edge_index, edge_attr))   # talk to neighbours
            h = h + self.drop(ff(h))                                  # think locally
        return torch.sigmoid(self.head(h))   # usage fractions live in [0, 1]

    @staticmethod
    def from_config(cfg: Dict) -> "CongestionGNN":
        return CongestionGNN(**cfg)
