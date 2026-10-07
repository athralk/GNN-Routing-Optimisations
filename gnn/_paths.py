"""Locate the M3D routing-challenge repo and put its ``m3d`` toolkit on ``sys.path``.

This project reuses the challenge's own code for everything that must match the
official rules exactly: the instance generator, the reference routers, and the
independent legality checker/scorer. We don't copy that code here; we import it
from a checkout of https://github.com/partcleda/eda-3d-routing-challenge.

Lookup order:

1. the ``M3D_REPO`` environment variable, if set
2. a sibling folder named ``Partcl-Competition`` or ``eda-3d-routing-challenge``
   next to this repo (e.g. ``D:/Partcl-Competition`` beside
   ``D:/GNN_Routing_Optimisation``)
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SIBLING_NAMES = ("Partcl-Competition", "eda-3d-routing-challenge")


def _find_repo() -> str:
    candidates = []
    if os.environ.get("M3D_REPO"):
        candidates.append(os.environ["M3D_REPO"])
    parent = os.path.dirname(_HERE)
    candidates += [os.path.join(parent, name) for name in _SIBLING_NAMES]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "m3d", "checker.py")):
            return os.path.abspath(c)
    raise SystemExit(
        "Could not find the M3D challenge repo (needed for m3d/ and the benchmarks).\n"
        "Clone https://github.com/partcleda/eda-3d-routing-challenge next to this repo,\n"
        "or set M3D_REPO=/path/to/eda-3d-routing-challenge")


M3D_REPO = _find_repo()
if M3D_REPO not in sys.path:
    # append (not prepend) so this repo's own `gnn` package always wins
    sys.path.append(M3D_REPO)


def repo_path(p: str) -> str:
    """Resolve a benchmark path like ``benchmarks_hard`` against the challenge repo
    when it doesn't exist relative to the current directory."""
    return p if os.path.exists(p) else os.path.join(M3D_REPO, p)
