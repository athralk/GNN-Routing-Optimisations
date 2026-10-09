"""Fast exact M3D router (Numba): shortest-path-tree nets + negotiation + exact refinement.

Three ideas, each a big step over the reference routers:

1. **Exact per-net trees (SPT).** The score adds up the delay to *every sink*. One
   Dijkstra from the driver gives each sink its minimum-delay path, and the
   predecessor pointers already form a tree. (The reference routers instead attach
   each sink to the *nearest* tree point, which saves wire but not delay.)
2. **Negotiated congestion (PathFinder)** to reach legality, exactly as in
   ``m3d/negotiated.py`` (present + history costs), optionally seeded with the GNN's
   congestion prior (same fading formula as ``gnn/route.py``).
3. **Exact refinement.** Once legal, rip up one net at a time and reroute it with an
   SPT on pure delay, treating every other net as a hard obstacle. The old tree is
   still available, so the new one can never be worse; repeat until nothing improves.

Grid ids match ``m3d/grid.py``: ``vid = (z*H + y)*W + x``. An edge between u < v is
``eid = 3*u + d`` with d = 0 (+x), 1 (+y), 2 (+z).
"""
from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from numba import njit

from m3d.model import Instance, NetRoute, Submission

INF = 1e30


@njit(cache=True)
def _spt(driver, sinks, W, H, L, ldelay, via, pin_owner, net, occ_v, occ_e, hist_v,
         hist_e, pres_fac, vcong, hard, prior_lat, prior_via, prior_scale, dist, tdist,
         pred, stamp, cur):
    """Dijkstra from ``driver``; returns (tree vertices, tree edges, true delay sum).

    Cost of stepping u->v over edge e (for negotiation; ``hard`` forbids any resource
    used by another net instead):
        delay(e) * (1 + prior + hist_e + pres_fac * occ_e) + vcong * (hist_v + pres_fac * occ_v)
    ``tdist`` tracks the *true* delay along the chosen paths.
    """
    WH = W * H
    n_sinks = sinks.shape[0]
    is_sink = 0
    heap = [(0.0, driver)]
    stamp[driver] = cur
    dist[driver] = 0.0
    tdist[driver] = 0.0
    pred[driver] = -1
    # mark sinks with a negative stamp offset trick: use a small local set via sort
    sk = np.sort(sinks)
    found = 0
    done = np.zeros(n_sinks, dtype=np.bool_)
    while heap and found < n_sinks:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        # sink?
        lo, hi = 0, n_sinks
        while lo < hi:
            mid = (lo + hi) // 2
            if sk[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        if lo < n_sinks and sk[lo] == u and not done[lo]:
            done[lo] = True
            found += 1
        z = u // WH
        r = u - z * WH
        y = r // W
        x = r - y * W
        for k in range(6):
            if k == 0:
                if x + 1 >= W:
                    continue
                v = u + 1; e = 3 * u; w = ldelay[z]; p = prior_lat[u]
            elif k == 1:
                if x == 0:
                    continue
                v = u - 1; e = 3 * v; w = ldelay[z]; p = prior_lat[v]
            elif k == 2:
                if y + 1 >= H:
                    continue
                v = u + W; e = 3 * u + 1; w = ldelay[z]; p = prior_lat[u]
            elif k == 3:
                if y == 0:
                    continue
                v = u - W; e = 3 * v + 1; w = ldelay[z]; p = prior_lat[v]
            elif k == 4:
                if z + 1 >= L:
                    continue
                v = u + WH; e = 3 * u + 2; w = via; p = prior_via[u]
            else:
                if z == 0:
                    continue
                v = u - WH; e = 3 * v + 2; w = via; p = prior_via[v]
            po = pin_owner[v]
            if po >= 0 and po != net:
                continue
            if hard:
                if occ_v[v] > 0 or occ_e[e] > 0:
                    continue
                c = w
            else:
                c = w * (1.0 + prior_scale * p + hist_e[e] + pres_fac * occ_e[e]) \
                    + vcong * (hist_v[v] + pres_fac * occ_v[v])
            nd = d + c
            if stamp[v] != cur or nd < dist[v]:
                stamp[v] = cur
                dist[v] = nd
                tdist[v] = tdist[u] + w
                pred[v] = u
                heapq.heappush(heap, (nd, v))
    if found < n_sinks:
        return np.empty(0, np.int64), np.empty(0, np.int64), -1.0
    # union of driver->sink paths = a tree
    tv = [driver]
    te = [np.int64(0)]
    te.pop()
    intree_stamp = cur  # reuse stamp array semantics: mark tree vertices with -cur
    stamp[driver] = -cur
    total = 0.0
    for i in range(n_sinks):
        s = sinks[i]
        total += tdist[s]
        v = s
        while stamp[v] != -cur:
            u = pred[v]
            stamp[v] = -cur
            tv.append(v)
            a, b = (u, v) if u < v else (v, u)
            dd = b - a
            if dd == 1:
                te.append(3 * a)
            elif dd == W:
                te.append(3 * a + 1)
            else:
                te.append(3 * a + 2)
            v = u
    return np.array(tv, dtype=np.int64), np.array(te, dtype=np.int64), total


@njit(cache=True)
def _grow(driver, sinks, W, H, L, ldelay, via, pin_owner, net, occ_v, occ_e, hist_v,
          hist_e, pres_fac, vcong, prior_lat, prior_via, prior_scale, beta, dist, tdist,
          pred, stamp, cur, tmark, tcur, ttree):
    """Grow a tree sink by sink (like ``m3d/negotiated.py``), with source-aware seeding.

    Each round is a multi-source Dijkstra from every tree vertex v, started at cost
    ``beta * true_delay(driver -> v)``. beta = 0 attaches each sink to the nearest tree
    point (compact trees that share wire, so negotiation converges); beta = 1
    approaches exact per-sink shortest paths. Returns (verts, edges, true delay, cur).
    """
    WH = W * H
    n_sinks = sinks.shape[0]
    remaining = np.ones(n_sinks, dtype=np.bool_)
    left = n_sinks
    tv = [driver]
    te = [np.int64(0)]
    te.pop()
    tmark[driver] = tcur
    ttree[driver] = 0.0
    sk = np.sort(sinks)
    while left > 0:
        cur += 1
        heap = [(0.0, driver)]
        heap.pop()
        for s in tv:
            stamp[s] = cur
            dist[s] = beta * ttree[s]
            tdist[s] = ttree[s]
            pred[s] = -1
            heapq.heappush(heap, (dist[s], s))
        found = -1
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            if tmark[u] != tcur:
                lo, hi = 0, n_sinks
                while lo < hi:
                    mid = (lo + hi) // 2
                    if sk[mid] < u:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < n_sinks and sk[lo] == u and remaining[lo]:
                    remaining[lo] = False
                    left -= 1
                    found = u
                    break
            z = u // WH
            r = u - z * WH
            y = r // W
            x = r - y * W
            for k in range(6):
                if k == 0:
                    if x + 1 >= W:
                        continue
                    v = u + 1; e = 3 * u; w = ldelay[z]; p = prior_lat[u]
                elif k == 1:
                    if x == 0:
                        continue
                    v = u - 1; e = 3 * v; w = ldelay[z]; p = prior_lat[v]
                elif k == 2:
                    if y + 1 >= H:
                        continue
                    v = u + W; e = 3 * u + 1; w = ldelay[z]; p = prior_lat[u]
                elif k == 3:
                    if y == 0:
                        continue
                    v = u - W; e = 3 * v + 1; w = ldelay[z]; p = prior_lat[v]
                elif k == 4:
                    if z + 1 >= L:
                        continue
                    v = u + WH; e = 3 * u + 2; w = via; p = prior_via[u]
                else:
                    if z == 0:
                        continue
                    v = u - WH; e = 3 * v + 2; w = via; p = prior_via[v]
                if tmark[v] == tcur:
                    continue          # never re-enter our own tree (would form a cycle)
                po = pin_owner[v]
                if po >= 0 and po != net:
                    continue
                c = w * (1.0 + prior_scale * p + hist_e[e] + pres_fac * occ_e[e]) \
                    + vcong * (hist_v[v] + pres_fac * occ_v[v])
                nd = d + c
                if stamp[v] != cur or nd < dist[v]:
                    stamp[v] = cur
                    dist[v] = nd
                    tdist[v] = tdist[u] + w
                    pred[v] = u
                    heapq.heappush(heap, (nd, v))
        if found < 0:
            return np.empty(0, np.int64), np.empty(0, np.int64), -1.0, cur
        v = found
        while tmark[v] != tcur:
            u = pred[v]
            tmark[v] = tcur
            ttree[v] = tdist[v]
            tv.append(v)
            a, b = (u, v) if u < v else (v, u)
            dd = b - a
            if dd == 1:
                te.append(3 * a)
            elif dd == W:
                te.append(3 * a + 1)
            else:
                te.append(3 * a + 2)
            v = u
    total = 0.0
    for i in range(n_sinks):
        total += ttree[sinks[i]]
    return np.array(tv, dtype=np.int64), np.array(te, dtype=np.int64), total, cur


class FastRouter:
    """Negotiate to legality with SPT nets, then refine each net exactly."""

    def __init__(self, inst: Instance, prior: Optional[np.ndarray] = None, tile: int = 4,
                 alpha: float = 1.0, gamma: float = 2.0, decay: float = 1.0):
        self.inst = inst
        W, H, L = inst.width, inst.height, inst.layers
        self.W, self.H, self.L = W, H, L
        self.n = n = W * H * L
        self.ldelay = np.asarray(inst.layer_delay, dtype=np.float64)
        self.via = float(inst.via_delay)
        vid = lambda p: (p.z * H + p.y) * W + p.x
        pin_v = {p.id: vid(p) for p in inst.pins}
        self.pin_owner = np.full(n, -1, dtype=np.int64)
        self.driver: Dict[int, int] = {}
        self.sinks: Dict[int, np.ndarray] = {}
        for net in inst.nets:
            self.driver[net.id] = pin_v[net.driver]
            self.sinks[net.id] = np.array(sorted({pin_v[s] for s in net.sinks}), dtype=np.int64)
            for pid in net.pins():
                self.pin_owner[pin_v[pid]] = net.id
        self.occ_v = np.zeros(n, dtype=np.int64)
        self.occ_e = np.zeros(3 * n, dtype=np.int64)
        self.hist_v = np.zeros(n, dtype=np.float64)
        self.hist_e = np.zeros(3 * n, dtype=np.float64)
        # scratch arrays reused by every Dijkstra call
        self.dist = np.zeros(n, dtype=np.float64)
        self.tdist = np.zeros(n, dtype=np.float64)
        self.pred = np.zeros(n, dtype=np.int64)
        self.stamp = np.zeros(n, dtype=np.int64)
        self.cur = 0
        self.tmark = np.zeros(n, dtype=np.int64)
        self.ttree = np.zeros(n, dtype=np.float64)
        self.tcur = 0
        self.beta = 0.0
        self.routes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.delay: Dict[int, float] = {}
        # GNN prior per vertex: lateral edge / upward via starting there
        self.prior_lat = np.zeros(n, dtype=np.float64)
        self.prior_via = np.zeros(n, dtype=np.float64)
        self.decay = decay
        if prior is not None and alpha > 0:
            pr = alpha * np.clip(prior, 0.0, 1.0) ** gamma
            z, y, x = np.meshgrid(np.arange(L), np.arange(H), np.arange(W), indexing="ij")
            t = pr[z, y // tile, x // tile]
            self.prior_lat = np.ascontiguousarray(t[..., 0].ravel(), dtype=np.float64)
            self.prior_via = np.ascontiguousarray(t[..., 1].ravel(), dtype=np.float64)

    # -- occupancy -----------------------------------------------------------
    def _add(self, nid, tv, te, d):
        self.occ_v[tv] += 1
        self.occ_e[te] += 1
        self.routes[nid] = (tv, te)
        self.delay[nid] = d

    def _remove(self, nid):
        tv, te = self.routes.pop(nid)
        self.occ_v[tv] -= 1
        self.occ_e[te] -= 1
        self.delay.pop(nid, None)

    def _route(self, nid, pres_fac, hard, prior_scale=0.0):
        if not hard:   # negotiation: grown, source-aware trees
            self.tcur += 1
            tv, te, d, self.cur = _grow(
                self.driver[nid], self.sinks[nid], self.W, self.H, self.L, self.ldelay,
                self.via, self.pin_owner, nid, self.occ_v, self.occ_e, self.hist_v,
                self.hist_e, pres_fac, 1.0, self.prior_lat, self.prior_via, prior_scale,
                self.beta, self.dist, self.tdist, self.pred, self.stamp, self.cur,
                self.tmark, self.tcur, self.ttree)
            return tv, te, d
        self.cur += 1
        return _spt(self.driver[nid], self.sinks[nid], self.W, self.H, self.L, self.ldelay,
                    self.via, self.pin_owner, nid, self.occ_v, self.occ_e, self.hist_v,
                    self.hist_e, pres_fac, 1.0, hard, self.prior_lat, self.prior_via,
                    prior_scale, self.dist, self.tdist, self.pred, self.stamp, self.cur)

    def _order(self) -> List[int]:
        def bbox(nid):
            vs = np.concatenate([[self.driver[nid]], self.sinks[nid]])
            z, r = np.divmod(vs, self.W * self.H)
            y, x = np.divmod(r, self.W)
            return int(np.ptp(x) + np.ptp(y) + np.ptp(z))
        return sorted(self.driver, key=lambda i: (-bbox(i), i))

    # -- phases --------------------------------------------------------------
    def load_routes(self, routes: Dict[int, Tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Start from given trees ({net id: (vertex ids, edge ids, true delay)}), e.g.
        from the GPU router. Overlaps are allowed here; ``repair`` removes them."""
        for nid, (tv, te, d) in routes.items():
            self._add(nid, np.asarray(tv, dtype=np.int64), np.asarray(te, dtype=np.int64),
                      float(d))

    def repair(self, max_iters=100, pres0=0.5, mult=1.7, hist_fac=0.5) -> bool:
        """Negotiate from the *current* routes: only nets on overused resources are
        rerouted, so a nearly-legal start (from the GPU) stays mostly intact."""
        return self.negotiate(max_iters, pres0, mult, hist_fac, from_current=True)

    def negotiate(self, max_iters=100, pres0=0.5, mult=1.7, hist_fac=0.5,
                  from_current: bool = False) -> bool:
        order = self._order()
        pf = pres0
        if not from_current:
            for nid in order:
                tv, te, d = self._route(nid, pf, False, 1.0)
                if d < 0:
                    return False
                self._add(nid, tv, te, d)
        for it in range(max_iters):
            ov = np.nonzero(self.occ_v > 1)[0]
            oe = np.nonzero(self.occ_e > 1)[0]
            if ov.size == 0 and oe.size == 0:
                self.iterations = it
                return True
            self.hist_v[ov] += hist_fac * (self.occ_v[ov] - 1)
            self.hist_e[oe] += hist_fac * (self.occ_e[oe] - 1)
            pf = min(pf * mult, 1e6)
            scale = (pres0 / pf) ** self.decay
            bad_v = np.zeros(self.n, dtype=np.bool_); bad_v[ov] = True
            bad_e = np.zeros(3 * self.n, dtype=np.bool_); bad_e[oe] = True
            for nid in order:
                tv, te = self.routes[nid]
                if not (bad_v[tv].any() or bad_e[te].any()):
                    continue
                self._remove(nid)
                tv, te, d = self._route(nid, pf, False, scale)
                if d < 0:
                    return False
                self._add(nid, tv, te, d)
        self.iterations = max_iters
        return bool((self.occ_v <= 1).all() and (self.occ_e <= 1).all())

    def refine(self, max_passes=20, time_limit=None) -> int:
        """Exact per-net reroute on pure delay with all other nets as obstacles."""
        t0 = time.time()
        improved_total = 0
        for _ in range(max_passes):
            improved = 0
            for nid in sorted(self.delay, key=lambda i: -self.delay[i]):
                old = self.routes[nid]
                old_d = self.delay[nid]
                self._remove(nid)
                tv, te, d = self._route(nid, 0.0, True)
                if d >= 0 and d < old_d - 1e-9:
                    self._add(nid, tv, te, d)
                    improved += 1
                else:
                    self._add(nid, old[0], old[1], old_d)
                if time_limit and time.time() - t0 > time_limit:
                    return improved_total + improved
            improved_total += improved
            if improved == 0:
                break
        return improved_total

    # -- large neighbourhood search ---------------------------------------
    def ideal_delays(self) -> Dict[int, float]:
        """Each net's exact SPT delay on an empty grid (only other pins block): its
        best possible delay. ``delay - ideal`` = how much a net is being squeezed."""
        zv, ze = np.zeros_like(self.occ_v), np.zeros_like(self.occ_e)
        saved = self.occ_v, self.occ_e
        self.occ_v, self.occ_e = zv, ze
        ideal = {}
        for nid in self.driver:
            _, _, d = self._route(nid, 0.0, True)
            ideal[nid] = d
        self.occ_v, self.occ_e = saved
        return ideal

    def prepare_lns(self, margin: int = 2) -> None:
        """Static data for LNS: ideal delays, bounding boxes, neighbour lists."""
        if getattr(self, "_nets", None) is not None:
            return
        self._ideal = self.ideal_delays()
        nets = np.array(sorted(self.driver))
        bb = np.zeros((len(nets), 4), dtype=np.int64)
        WH = self.W * self.H
        cross = np.zeros(len(nets))
        for i, nid in enumerate(nets):
            vs = np.concatenate([[self.driver[nid]], self.sinks[nid]])
            z, r = np.divmod(vs, WH)
            y, x = np.divmod(r, self.W)
            bb[i] = (x.min(), x.max(), y.min(), y.max())
            cross[i] = float(z.min() != z.max())
        self._nets, self._bb, self._cross = nets, bb, cross
        self._nbrs = []
        for i in range(len(nets)):
            x0, x1, y0, y1 = bb[i]
            ov = np.nonzero((bb[:, 0] <= x1 + margin) & (bb[:, 1] >= x0 - margin) &
                            (bb[:, 2] <= y1 + margin) & (bb[:, 3] >= y0 - margin))[0]
            self._nbrs.append(ov[ov != i])

    def _move(self, s: int, rng, k: int, commit: bool):
        """Rip up net index s plus up to k-1 random overlapping nets, reroute exactly
        in random order. commit=True keeps the result if not worse; commit=False
        always restores (a pure "what if"). Returns the delay gain, or None if the
        group could not be rerouted."""
        nb = self._nbrs[s]
        pick = [s] + list(rng.choice(nb, size=min(k - 1, len(nb)), replace=False))
        group = [int(self._nets[i]) for i in pick]
        old = {n: (self.routes[n], self.delay[n]) for n in group}
        old_sum = sum(d for _, d in old.values())
        for n in group:
            self._remove(n)
        new_sum, ok = 0.0, True
        for n in rng.permutation(group):
            tv, te, d = self._route(int(n), 0.0, True)
            if d < 0:
                ok = False
                break
            self._add(int(n), tv, te, d)
            new_sum += d
        gain = (old_sum - new_sum) if ok else None
        if commit and ok and new_sum <= old_sum + 1e-9:
            return gain
        for n in group:                        # restore
            if n in self.routes:
                self._remove(n)
        for n, ((tv, te), d) in old.items():
            self._add(n, tv, te, d)
        return gain

    # -- LNS v2: negotiated group repair, window moves, annealing -------------
    def _route_soft(self, nid, pres_fac):
        """Exact SPT where other nets are a *soft* obstacle (congestion cost)."""
        self.cur += 1
        return _spt(self.driver[nid], self.sinks[nid], self.W, self.H, self.L, self.ldelay,
                    self.via, self.pin_owner, nid, self.occ_v, self.occ_e, self.hist_v,
                    self.hist_e, pres_fac, 1.0, False, self.prior_lat, self.prior_via,
                    0.0, self.dist, self.tdist, self.pred, self.stamp, self.cur)

    def _negotiate_group(self, group, rng, iters=10, pres0=0.5, mult=2.0, hist_fac=1.0):
        """Mini-PathFinder on an already-removed group: the group's nets may overlap
        each other (and, softly, the fixed nets) and negotiate until legal. Then each
        group net gets an exact hard-capacity polish. Returns the group's new delay
        sum, or None (group left removed) if it can't be made legal."""
        hv, he = self.hist_v, self.hist_e
        touched_v, touched_e = [], []
        pf = pres0
        for n in rng.permutation(group):
            tv, te, d = self._route_soft(int(n), pf)
            if d < 0:
                break
            self._add(int(n), tv, te, d)
        ok = all(n in self.routes for n in group)
        if ok:
            ok = False
            for _ in range(iters):
                bad = []
                for n in group:
                    tv, te = self.routes[n]
                    ov, oe = tv[self.occ_v[tv] > 1], te[self.occ_e[te] > 1]
                    if ov.size or oe.size:
                        bad.append(n)
                        hv[ov] += hist_fac; he[oe] += hist_fac
                        touched_v.append(ov); touched_e.append(oe)
                if not bad:
                    ok = True
                    break
                pf *= mult
                for n in bad:
                    self._remove(n)
                    tv, te, d = self._route_soft(n, pf)
                    if d < 0:
                        break
                    self._add(n, tv, te, d)
                if not all(n in self.routes for n in group):
                    break
        for a in touched_v:          # local history only: reset what we touched
            hv[a] = 0.0
        for a in touched_e:
            he[a] = 0.0
        if not ok:
            for n in group:
                if n in self.routes:
                    self._remove(n)
            return None
        for n in group:              # exact polish against everything else
            old = self.routes[n], self.delay[n]
            self._remove(n)
            tv, te, d = self._route(n, 0.0, True)
            if d >= 0 and d < old[1] - 1e-9:
                self._add(n, tv, te, d)
            else:
                self._add(n, old[0][0], old[0][1], old[1])
        return float(sum(self.delay[n] for n in group))

    def _try_group(self, group, rng, negotiated: bool, threshold: float):
        """Rip up ``group``, rebuild it (greedy exact or negotiated), keep if the delay
        rises by at most ``threshold``. Returns the gain (positive = better) or None."""
        old = {n: (self.routes[n], self.delay[n]) for n in group}
        old_sum = sum(d for _, d in old.values())
        for n in group:
            self._remove(n)
        new_sum = None
        if negotiated:
            new_sum = self._negotiate_group(group, rng)
        else:
            s, ok = 0.0, True
            for n in rng.permutation(group):
                tv, te, d = self._route(int(n), 0.0, True)
                if d < 0:
                    ok = False
                    break
                self._add(int(n), tv, te, d)
                s += d
            new_sum = s if ok else None
        if new_sum is not None and new_sum <= old_sum + threshold:
            return old_sum - new_sum
        for n in group:
            if n in self.routes:
                self._remove(n)
        for n, ((tv, te), d) in old.items():
            self._add(n, tv, te, d)
        return None

    def _window_group(self, s: int, rng, kmax: int = 16):
        """All nets whose bounding box meets a random window around net s's box."""
        x0, x1, y0, y1 = self._bb[s]
        cx = int(rng.integers(x0, x1 + 1)); cy = int(rng.integers(y0, y1 + 1))
        r = int(rng.integers(3, 9))
        bb = self._bb
        m = np.nonzero((bb[:, 0] <= cx + r) & (bb[:, 1] >= cx - r) &
                       (bb[:, 2] <= cy + r) & (bb[:, 3] >= cy - r))[0]
        m = m[m != s]
        pick = [s] + list(rng.choice(m, size=min(kmax - 1, len(m)), replace=False))
        return [int(self._nets[i]) for i in pick]

    def _restore(self, routes, delay) -> None:
        self.occ_v[:] = 0
        self.occ_e[:] = 0
        self.routes, self.delay = {}, {}
        for n, (tv, te) in routes.items():
            self._add(n, tv, te, delay[n])

    def regret_probs(self) -> np.ndarray:
        reg = np.array([self.delay[n] - self._ideal[n] for n in self._nets]) + 1.0
        return reg / reg.sum()

    def lns(self, time_limit: float, k: int = 8, seed: int = 0, margin: int = 2,
            policy=None, refresh: int = 500, explore: float = 0.2,
            anneal: float = 0.1, mix=(1.0, 0.0, 0.0), stall_kick: float = 2.0) -> Dict:
        """Ruin-and-recreate with three move types (probabilities ``mix``):

        * greedy:     seed net + k-1 random overlapping nets, rerouted exactly in order
        * negotiated: same kind of group (6-14 nets), rebuilt by a mini-PathFinder
        * window:     every net meeting a random window around the seed, negotiated

        The seed net is drawn from ``policy(self)`` (the destroy GNN; refreshed every
        ``refresh`` moves) mixed with ``explore`` of regret sampling, or regret only.
        Annealing: a move may raise delay by up to ``anneal`` x (mean net delay),
        shrinking linearly to 0 over the run; the best state seen is restored at the
        end. ``stall_kick``: when no move has improved for this many seconds, the next
        move is a negotiated one (they succeed ~10x more often but cost ~20x more, so
        they're used to escape stalls). ``anneal=0, stall_kick=0`` is the original LNS.
        """
        self.prepare_lns(margin)
        self.hist_v[:] = 0.0          # negotiation history is not used by LNS costs
        self.hist_e[:] = 0.0
        rng = np.random.default_rng(seed)
        t0 = time.time()
        moves = 0
        counts = {"greedy": [0, 0], "negotiated": [0, 0], "window": [0, 0]}
        start = total = self.total_delay()
        best_total, best = total, (dict(self.routes), dict(self.delay))
        mean_net = total / max(1, len(self._nets))
        mix = np.asarray(mix, dtype=np.float64) / np.sum(mix)
        probs = None
        last_gain_t = t0
        while True:
            el = time.time() - t0
            if el >= time_limit:
                break
            if moves % 50 == 0 or probs is None:
                reg = self.regret_probs()
                if policy is not None and (moves % refresh == 0 or probs is None):
                    pol = policy(self)
                probs = reg if policy is None else (1 - explore) * pol + explore * reg
            s = rng.choice(len(self._nets), p=probs)
            kind = ("greedy", "negotiated", "window")[rng.choice(3, p=mix)]
            if stall_kick > 0 and time.time() - last_gain_t > stall_kick:
                kind = "negotiated"
                last_gain_t = time.time()     # one kick per stall period
            if kind == "window":
                group = self._window_group(s, rng)
            else:
                kk = k if kind == "greedy" else int(rng.integers(6, 15))
                nb = self._nbrs[s]
                pick = [s] + list(rng.choice(nb, size=min(kk - 1, len(nb)), replace=False))
                group = [int(self._nets[i]) for i in pick]
            thr = anneal * mean_net * (1.0 - el / time_limit)
            gain = self._try_group(group, rng, kind != "greedy", thr)
            moves += 1
            counts[kind][0] += 1
            if gain is not None:
                total -= gain
                if gain > 1e-9:
                    counts[kind][1] += 1
                    last_gain_t = time.time()
                if total < best_total - 1e-9:
                    best_total, best = total, (dict(self.routes), dict(self.delay))
        if self.total_delay() > best_total + 1e-9:
            self._restore(*best)
        return {"moves": moves, "improving": sum(c[1] for c in counts.values()),
                "moves_by_kind": {k_: c[0] for k_, c in counts.items()},
                "improving_by_kind": {k_: c[1] for k_, c in counts.items()},
                "start": start, "end": self.total_delay(),
                "ideal": sum(self._ideal.values())}

    # -- destroy-policy data -------------------------------------------------
    NET_FEATURES = ["regret_ratio", "log_regret", "log_fanout", "log_hpwl", "cross_die",
                    "log_degree", "local_occupancy", "detour", "log_ideal_per_sink"]

    def net_features(self):
        """Per-net features + net-overlap graph for the destroy GNN (size-agnostic)."""
        self.prepare_lns()
        N = len(self._nets)
        occ2d = (self.occ_v.reshape(self.L, self.H, self.W) > 0).sum(0).astype(np.float64)
        ps = np.zeros((self.H + 1, self.W + 1))
        ps[1:, 1:] = occ2d.cumsum(0).cumsum(1)
        x = np.zeros((N, len(self.NET_FEATURES)), dtype=np.float32)
        for i, nid in enumerate(self._nets):
            nid = int(nid)
            d, ideal = self.delay[nid], self._ideal[nid]
            x0, x1, y0, y1 = self._bb[i]
            area = (x1 - x0 + 1) * (y1 - y0 + 1)
            occ = ps[y1 + 1, x1 + 1] - ps[y0, x1 + 1] - ps[y1 + 1, x0] + ps[y0, x0]
            ns = len(self.sinks[nid])
            hp = (x1 - x0) + (y1 - y0)
            x[i] = [(d - ideal) / max(ideal, 1.0), np.log1p(d - ideal), np.log1p(ns),
                    np.log1p(hp), self._cross[i], np.log1p(len(self._nbrs[i])),
                    occ / (area * self.L), len(self.routes[nid][1]) / (hp + 1.0),
                    np.log1p(ideal / ns)]
        src = np.concatenate([np.full(len(nb), i) for i, nb in enumerate(self._nbrs)] or
                             [np.zeros(0)])
        dst = np.concatenate(self._nbrs or [np.zeros(0)])
        return x, np.stack([src, dst]).astype(np.int64)

    def snapshot_labels(self, tries: int, rng, k: int = 8):
        """For every net: try ``tries`` random moves seeded at it (never committed).
        y = 1 if any improved delay; rate = fraction of tries that improved (a much
        less noisy target than y); best = largest gain seen."""
        self.prepare_lns()
        N = len(self._nets)
        y = np.zeros(N, dtype=np.float32)
        rate = np.zeros(N, dtype=np.float32)
        best = np.zeros(N, dtype=np.float32)
        for i in range(N):
            for _ in range(tries):
                g = self._move(i, rng, k, commit=False)
                if g is not None and g > 1e-9:
                    y[i] = 1.0
                    rate[i] += 1.0 / tries
                    best[i] = max(best[i], g)
        return y, best, rate

    def total_delay(self) -> float:
        return float(sum(self.delay.values()))

    def to_submission(self) -> Submission:
        W, WH = self.W, self.W * self.H

        def coord(v):
            z, r = divmod(int(v), WH)
            y, x = divmod(r, W)
            return (x, y, z)
        step = (1, W, WH)
        routes = []
        for net in self.inst.nets:
            _tv, te = self.routes[net.id]
            edges = []
            for e in te:
                a, d = divmod(int(e), 3)
                edges.append((coord(a), coord(a + step[d])))
            routes.append(NetRoute(net=net.id, edges=edges))
        return Submission(instance=self.inst.name, routes=routes)


def build_router(inst: Instance, prior=None, tile=4, alpha=1.0, max_iters=100,
                 betas=(0.5, 0.25, 0.0), refine_passes=20) -> Optional["FastRouter"]:
    """Negotiate to legality (first beta that works) and refine. None if it fails."""
    for beta in betas:
        r = FastRouter(inst, prior, tile, alpha)
        r.beta = beta
        if r.negotiate(max_iters=max_iters):
            r.refine(refine_passes)
            return r
    return None


def solve(inst: Instance, prior=None, tile=4, alpha=1.0, max_iters=100,
          refine_passes=20, betas=(0.5, 0.25, 0.0), lns_seconds: float = 0.0,
          lns_k: int = 8, seed: int = 0, policy=None, anneal: float = 0.1,
          mix=(1.0, 0.0, 0.0), stall_kick: float = 2.0) -> Tuple[Optional[Submission], Dict]:
    """Negotiate with the first beta that reaches legality, then refine exactly."""
    t0 = time.time()
    for beta in betas:
        r = FastRouter(inst, prior, tile, alpha)
        r.beta = beta
        ok = r.negotiate(max_iters=max_iters)
        if ok:
            break
    stats = {"legal_negotiated": ok, "iterations": getattr(r, "iterations", -1),
             "beta": beta, "t_negotiate": time.time() - t0}
    if not ok:
        return None, stats
    stats["delay_negotiated"] = r.total_delay()
    t1 = time.time()
    stats["refined_nets"] = r.refine(refine_passes)
    stats["t_refine"] = time.time() - t1
    stats["delay_refined"] = r.total_delay()
    if lns_seconds > 0:
        st = r.lns(lns_seconds, k=lns_k, seed=seed, policy=policy, anneal=anneal, mix=mix,
                   stall_kick=stall_kick)
        stats.update({f"lns_{k_}": v for k_, v in st.items()})
    stats["delay"] = r.total_delay()
    return r.to_submission(), stats
