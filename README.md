# GNN Routing Optimisation

**A fast exact router, guided by graph neural networks, for the
[Partcl M3D Routing Challenge](https://github.com/partcleda/eda-3d-routing-challenge).**
The task: route signal wires through a two-die, monolithic-3D chip stack with the lowest total
signal delay.

## Results (official challenge scorer, every case legal)

| Tier | Score | vs baseline | Approx. leaderboard position (Oct 2026) | Files |
|---|---:|---:|---|---|
| **hard** (9 cases) | **1.3138** | **+31 %** | ~8th of 19 (top: 1.4136) | `results/hard_final3/` |
| **designs** (3 real circuits) | **1.3561** | **+36 %** | ~7th of 12 (top: 1.4673) | `results/designs_final2/` |
| intro (20 cases), early version | 1.0145 | +1.5 % | (first approach only) | `results/intro/` |

Positions are estimated against the 7 Oct 2026 leaderboard snapshot. Best single cases:
hard `case_07` **1.4017**, designs `int2float` **1.5672**.

A score is the geometric mean of `baseline_delay / our_delay` over a tier's cases, so 1.0 equals
the challenge's reference router and higher is better.

**How we got here** (hard tier):

```
1.018  GNN congestion prior + the challenge's stock router           (day 1)
1.063  our own fast exact router (Numba), no search
1.190  + 60 s large neighbourhood search (LNS) per case
1.244  + 20 min LNS per case
1.255  best per case over three 20-min runs (two of them GNN-assisted)
1.301  LNS v2 (annealing + negotiated "stall kicks") + destroy GNN v2, 3 h per case
1.313  + another 3 h, warm-started from the best solutions (same algorithm, same GNN)
1.314  best per case including the Phase 3 A/B runs   <- final
```

Destroy GNN versions v3 (blocking graph + partner head) and v4 (gain ranking, ensemble) each got
clearly better *offline*, but **tied v2 in equal-time A/B tests**. See
[Part D2](#part-d2--phases-0-3-pushing-the-destroy-gnn-v3-v4) for why.

**13 Oct 2026: column generation (Part F).** We tried picking the best combination of candidate
trees exactly (integer programming over pools of 8–17k trees per case), plus exact solves on
small groups of nets and restarts guided by the relaxed solution. **None of it improved any hard
case.** What it did give is evidence: our solutions can't be improved by rearranging any group of
~15–20 neighbouring nets among the trees we tried. Further gains need a different basin, not a
better local move. See [Part F](#part-f--column-generation-13-oct-no-gain-strong-evidence).

---

## Contents

1. [The competition in plain terms](#1-the-competition-in-plain-terms)
2. [The big picture: what the GNNs do and what the router does](#2-the-big-picture)
3. [Glossary](#3-glossary)
4. [Part A: the congestion GNN (first approach)](#part-a--the-congestion-gnn-first-approach)
5. [Part B: the fast exact router (`engine/`)](#part-b--the-fast-exact-router-engine)
6. [Part C: LNS, where most of the score comes from](#part-c--lns-where-most-of-the-score-comes-from)
7. [Part D: the destroy GNN, which picks what LNS rips up](#part-d--the-destroy-gnn-which-picks-what-lns-rips-up)
8. [Part D2: Phases 0–3, pushing the destroy GNN (v3, v4)](#part-d2--phases-0-3-pushing-the-destroy-gnn-v3-v4)
9. [Part E: GPU Lagrangian router (experimental, unfinished)](#part-e--gpu-lagrangian-router-experimental-unfinished)
10. [Part F: column generation (13 Oct): no gain, strong evidence](#part-f--column-generation-13-oct-no-gain-strong-evidence)
11. [Every experiment and its result](#every-experiment-and-its-result)
12. [Repository layout: every file and what it does](#repository-layout-every-file-and-what-it-does)
13. [Setup and reproducing everything](#setup-and-reproducing-everything)
14. [Submitting to the leaderboard](#submitting-to-the-leaderboard)
15. [Lessons learned](#lessons-learned)
16. [Next steps](#next-steps)
17. [References](#references)

---

## 1. The competition in plain terms

**The chip.** Two flat layers of circuitry, the **dies**, are stacked face to face. Between them
are several thin **routing layers** (usually 6), the floors where wires run. This is
**monolithic 3D (M3D)** integration.

**The task.** Small logic blocks (**cells**) have connection points (**pins**). Pins are grouped
into **nets**: one **driver** (sends the signal) and one or more **sinks** (receive it). Every net
needs a wire **tree** (connected, no loops) from its driver to all its sinks.

**The grid.** Space is a 3D grid of points `(x, y, z)`. A wire steps one square north/south/east/
west on a floor, or hops one floor up or down at the same spot (a **via**). Bottom-die pins sit
on floor 0, top-die pins on the top floor.

**Legality (all or nothing).** No grid point or edge may be used by two nets (a **short**), and
no wire may pass through another net's pin. One illegal net in one case zeroes the whole tier.

**The objective (delay).** Each floor has a cost per step. The middle floors are **fast** and the
floors next to the dies are **slow**; vias cost extra. For example:

```
floor (z):   0    1    2    3    4    5        via = 3
step cost:   6    4    2    2    4    6
```

**Total delay** = for every sink of every net, the cost of its path from the driver. Long wires
want the fast middle floors, so those floors get crowded (**congestion**). Balancing "each net
wants the fast floors" against "nets can't share space" is the whole problem.

**Tiers.** intro (16–92, easy), **hard** (24–40, crowded), congested (64–116), **designs**
(77–105, real circuits), scale (100–156), stress (530). The baseline for hard, congested and
designs is the challenge's negotiated-congestion router; for intro and scale it's a simpler
router.

---

## 2. The big picture

```
                        ┌──────────────────────────────── learned (GNNs) ────────────────────────────────┐
                        │  congestion GNN: "where will wires go?"    destroy GNN: "what should we redo?" │
                        └───────────────┬───────────────────────────────────────────────┬──────────────┘
                                        │ starting costs                                 │ which nets to rip up
                                        ▼                                                 ▼
 problem ──► negotiation (legal) ──► exact refinement ──► LNS: rip up a few nets, reroute, keep if better ──► submission
             (PathFinder)            (per-net optimal)     (millions of moves, hours)                           │
                                                                                                             ▼
                                                                                         official checker + scorer
```

* **The router guarantees correctness.** Every output is legal by construction, and is re-checked
  with the challenge's own checker. A neural network can't promise "zero shorts", so it never
  draws wires directly.
* **The GNNs make the search smarter.** They predict where congestion will be, and which nets are
  worth reworking.
* **Most of the score comes from search (LNS).** Without the GNN, the same LNS scores lower; see
  [the experiments](#every-experiment-and-its-result).

---

## 3. Glossary

| Term | Meaning |
|---|---|
| **GNN** | Graph neural network: nodes pass messages to their neighbours; k layers = each node "hears" k hops away |
| **GATv2** | A GNN layer where each node weighs its neighbours' messages with learned attention |
| **Dijkstra** | The classic shortest-path algorithm (sequential; what the CPU router uses) |
| **Bellman-Ford** | Another shortest-path algorithm: repeatedly "take the minimum over neighbours". On a grid this is a stencil and runs well on GPUs. |
| **SPT** | Shortest-path tree: one Dijkstra from the driver gives every sink its minimum-delay path; together these paths form a tree |
| **PathFinder / negotiation** | Route all nets allowing overlaps, then make crowded spots more expensive (present + history cost) and reroute, until nothing overlaps |
| **Refinement** | Rip up one net, reroute it with an exact SPT around everyone else, keep it if better |
| **LNS** | Large neighbourhood search: rip up a *group* of nets, rebuild them, keep the result if better; repeat millions of times |
| **Destroy / repair** | The two halves of an LNS move: choose what to rip up (destroy), then rebuild it (repair) |
| **Regret** | How far a net is from its ideal delay (`current − ideal`); the hand-made rule for picking what to rip up |
| **Annealing** | Sometimes accepting a slightly worse move early on, to escape local optima |
| **Ideal ratio** | `baseline / sum of every net's ideal delay with no other nets present`: an optimistic ceiling for a case |
| **Tile** | A 4×4 patch of the grid; one node per tile per layer in the congestion GNN |
| **RUDY** | Rectangular Uniform wire DensitY: a classic quick congestion estimate |
| **R²** | Prediction quality: 0 = no better than guessing the average, 1 = perfect |
| **AUC** | Ranking quality: 0.5 = random, 1.0 = perfect |
| **Numba** | Compiles Python loops to machine code (50–100× faster) |
| **Ablation / control** | Remove one ingredient (or replace it with a no-information version) to prove what it contributes |
| **Warm start** | Begin a run from a previously saved solution instead of from scratch, so search time accumulates across runs |
| **A/B test** | Run two policies with the same time budget and seeds, from the same starting point, and compare |
| **Blocking edge** | In the net graph: "net j currently occupies part of net i's ideal route", the direct cause of i's lost delay |
| **Partner head** | Part of destroy GNN v3/v4 that chooses *which neighbours* to rip up together with the seed net |
| **Lift** | Expected gain of the GNN's choices ÷ expected gain of random choices (1.0 = no better than random) |
| **Listwise ranking loss** | Train the softmax over choices to match where the gains are, rather than yes/no per item |
| **Ensemble** | Several models trained with different seeds; their predictions are averaged |

---

## Part A: the congestion GNN (first approach)

**Question it answers:** given only the problem (pins, nets, floor costs), how heavily will a good
router use each region of each floor?

**1. Data (`gnn/datagen.py`).** Random problems from the challenge's own generator, with random
grid size, **layer count (4/6/8)**, floor and via costs, density, fanout and cross-die share.
Varying the layers and costs forces the model to learn from delay values rather than memorize
"layer 2 is fast". Every sample is solved, checked with the official checker, and saved as a JSON
pair. Two labelling modes:

| Dataset | Labelled by | Samples | Notes |
|---|---|---|---|
| `gnn/data/raw/` | challenge's reference routers | 3,000 small (16–40) + 70 large (48–100) + 200 val | first datasets |
| `gnn/data/raw_engine/` | **our fast engine** (negotiate + refine + short LNS) | 2,983 small + 400 large (48–120) + 192 val + 40 val_large | **better labels**, same problems (same seeds), much faster to make |

**2. Graph (`gnn/features.py`).** One node per 4×4 tile per layer; edges to the 4 sideways
neighbours and to the tiles above and below. **19 size-agnostic node features**: layer physics
(normalized cost, fastness rank, via cost, relative height, die layer), pin densities, RUDY split
by net length, cross-die and fanout, cross-die pin columns, and border distance. **Targets**: the
fraction of each tile-layer's edges (and vias) that the solution uses.

**3. Model (`gnn/model.py`).** 8 GATv2 layers with residual connections, about 836k parameters,
sigmoid output. `--layers 0` gives the same net with no message passing (an ablation).

**4. Training (`gnn/train.py`).** Masked MSE, AdamW, dropout 0.1, early stopping. It prints
"always predict the average" and "linear model" scores first, so the GNN's number means
something. About 11–42 min on a laptop RTX 4060.

| Model | Labels | Val lateral R² | Val via R² | Large-grid lateral / via R² |
|---|---|---:|---:|---|
| linear, no graph | — | 0.57 | 0.40 | |
| MLP, no message passing | reference | 0.68 | 0.68 | |
| v1 (15 features) | reference | 0.73 | 0.69 | |
| v2 (19 features, +large) | reference | 0.73 | 0.70 | |
| **v3** (`models/v3_engine.pt`) | **engine** | **0.75** | **0.73** | **0.77 / 0.60** |

It **generalizes to sizes it never saw**: trained on grids up to 40 (plus some up to 120), it
still predicts usefully on the 530×530 stress case and on real circuits (`gnn/evaluate.py`).

**5. Use in routing (`gnn/route.py`, `engine/fast.py`).** The prediction becomes an extra
starting cost, `alpha × predicted_usage²`, which fades as negotiation proceeds.

**Findings:**
* On the *stock* router it helped: +1.8% on hard (1.0180).
* A **control experiment** with the same predictions shuffled across positions scored −1.2%, so
  the GNN's spatial information is real, not just a lucky perturbation.
* Inside our much stronger engine with long LNS, the prior **ties** (1.2423 vs 1.2437). LNS
  erases the head start. It still won 3 of the 9 cases in the best-per-case combination.

![features vs targets](docs/figures/features_vs_target.png)
*Input RUDY (top) is the same on every floor; the real routing (middle) uses the fast floors
z=2,3. Learning that difference is the GNN's job.*

![prediction vs truth](docs/figures/prediction_hard_case05.png)

---

## Part B: the fast exact router (`engine/`)

The challenge's routers are pure Python and build each net's tree by attaching every sink to the
*nearest* point of the tree, which saves wire but not delay. Our engine (`engine/fast.py`, Numba)
fixes both:

1. **Exact per-net trees (`_spt`).** One Dijkstra from the driver gives every sink its minimum
   possible delay, and the predecessor pointers already form a tree. This is optimal for one net
   given fixed obstacles.
2. **Negotiation (`negotiate`, `_grow`).** PathFinder-style, like the challenge's router, but
   compiled. During negotiation, trees are grown with a **blend** `beta` between "attach to the
   nearest tree point" (compact, converges) and exact SPT. It tries `beta = 0.5, 0.25, 0` until
   one becomes legal. (Pure SPT trees used too much space, and 3 of 9 hard cases never became
   legal.)
3. **Exact refinement (`refine`).** Once legal, each net is ripped up and rerouted with an exact
   SPT around the others. It can never get worse, and it repeats until nothing improves.

**Speed:** about **30× faster** than the reference router (1–3 s vs 15–90 s per hard case).
**Score without any search:** 1.0626 on hard, all legal.

---

## Part C: LNS, where most of the score comes from

**LNS v1 (`FastRouter._move`):** pick a squeezed net (by regret) plus up to 7 overlapping nets,
rip them up, reroute them one by one with exact SPTs in random order, and keep the result if the
total delay didn't rise. 60 s → **1.1896**; 20 min → **1.2437**. But after 20 min, only 196 of
500,000 moves improved anything: **the search was stuck in a local optimum**.

**LNS v2 (`FastRouter.lns`, `_try_group`, `_negotiate_group`):**

| Ingredient | What it does | Measured effect (case_05, 90 s) |
|---|---|---|
| **Annealing** (`anneal=0.1`) | A move may raise delay by up to 0.1 × (mean net delay), shrinking to 0 by the end; the best state is restored at the end | 1.2056 → **1.2109** |
| **Negotiated group repair** | The ripped group may overlap internally and negotiates (mini-PathFinder) until legal, then gets an exact polish. It finds rearrangements greedy rerouting can't. | 10% of moves improve (vs 1%), but each is ~20× slower |
| **Stall kicks** (`stall_kick=2.0`) | Use a negotiated move *only* when nothing has improved for 2 s, so the expensive move becomes the escape hatch | the combination that worked |
| Window moves | Rip up every net in a random region | too slow, **disabled** (`mix=(1,0,0)`) |

2-minute check on 3 hard cases: old LNS 1.2039, v2 with fixed move mix 1.1720 (worse),
**v2 with stall kicks 1.2295**. That last one already beat the old 20-minute results on two of
the three cases.

**3-hour run (LNS v2 + destroy GNN v2):** about **45,000** improving moves on case_01, against
196 before. **Hard 1.3011, designs 1.3385**; it won every case.

---

## Part D: the destroy GNN, which picks what LNS rips up

**Problem:** most LNS moves are wasted. **Idea:** learn which nets are worth ripping up *right now*.

* **Graph:** one node per net; nets with overlapping bounding boxes are connected (they compete
  for space).
* **Node features (`FastRouter.NET_FEATURES`):** regret ratio, log regret, fanout, wire length
  (HPWL), cross-die, number of overlapping nets, local occupancy, detour ratio, ideal delay per
  sink.
* **Labels, free from the engine (`snapshot_labels`):** on *generated* problems only (never the
  benchmarks), take snapshots at increasing LNS depth. At each one, try several rip-ups seeded at
  *every* net without keeping them, and record how often each net's rip-up improved delay.
* **Model (`gnn/destroy.py`):** 4 GATv2 layers, hidden size 64. It outputs a logit per net, and
  LNS samples seed nets in proportion to `softmax(logit)`, mixed with 20% regret sampling for
  exploration.

| Version | Labels | Val AUC | Hit-rate of top 10% | vs regret ranking |
|---|---|---:|---:|---|
| v1 (`models/destroy_v1.pt`) | 4 tries per net, yes/no | 0.545 | 0.174 | regret: 0.531 / 0.133 |
| **v2** (`models/destroy_v2.pt`) | **16 tries per net, improvement rate** | **0.583** | **0.326** | regret: 0.522 / 0.265 (random 0.256) |

v1's labels were too noisy: whether a move helps depends mostly on random neighbour and order
choices. In a 20-minute run, v1 beat the regret-only run on **7 of 9 cases** but tied overall
(1.2434 vs 1.2437). v2 clears the bar the overnight script requires (AUC +0.01 over regret) and
powered the final 3-hour run.

---

## Part D2: Phases 0–3, pushing the destroy GNN (v3, v4)

**Goal:** reach 1.4 on hard by improving **only the GNN**, keeping the routing algorithm (exact SPT
router, negotiation, refinement, annealed greedy LNS with negotiated stall kicks) unchanged.

### Phase 0: measurement harness
| Piece | What it does | Where |
|---|---|---|
| **Warm start** | `--warm <folder>`: load saved solutions (`routes_from_submission`), check legality, skip negotiation, continue searching. Lets search time accumulate across nights. | `engine/fast.py`, `engine/run.py` |
| **A/B runner** | Same time budget, same seeds, same warm start for each policy; reports mean score and *improving moves per minute* (an early signal) | `gnn/ab.py` |
| **JSON output** | `--json-out` per-case results, for the A/B runner | `engine/run.py` |

### Phase 1: a better picture for the GNN (v3)
* **Blocking edges:** the engine computes every net's *ideal* route (alone on the grid) and keeps
  an owner map of who sits on which grid point. Edge j→i means "j occupies part of i's ideal
  route". This is the actual cause of lost delay, where the old graph only said "boxes overlap".
* **5 edge features:** overlap, j blocks i, i blocks j, fraction of the ideal route blocked,
  bounding-box overlap fraction.
* **18 node features** (`NET_FEATURES_V3`): the 9 v2 features + fast-layer share of the route,
  blocked fraction, number of blockers, how many nets it blocks, **search history** (times tried,
  win rate, moves since its last improvement, tracked live inside LNS), and the **congestion GNN's
  predictions** pooled over the net's region (so the two GNNs work together).

### Phase 2: let the GNN choose the whole neighbourhood (v3, partly)
* **Partner head:** besides scoring seed nets, the GNN scores each (seed, neighbour) edge. LNS then
  draws the rip-up partners from those scores (80%) plus uniform exploration (20%)
  (`FastRouter._pick_group`).
* Labels (`snapshot_labels_v3`): at snapshots 10 s, 70 s and 250 s into the search, every net is
  tried as a seed 8 times with random partners; record success per seed *and* per (seed, partner)
  pair.
* Model (`make_model_v3`): 6 GATv2 layers with edge features inside the attention, hidden size
  128, a seed head and a partner head; train/validation split by instance.
* *Not done:* order head and move-type head.

### Phase 3: learn *how much* a move gains (v4)
* **Gain-size labels** (`snapshot_labels_v4`): how much delay each trial saved, from **late
  states only** (snapshots 30 s, 3.5 min and 10.5 min into the search), 12 tries per net, 200
  generated problems. The data stage had a hard stop at 04:05 and saved as it went.
* **Listwise ranking loss** (`train_v4`): the softmax over nets (and over each seed's partners) is
  trained to match where the gains are. LNS samples from that same softmax, so training and use
  line up.
* **Sweep + ensemble:** hidden 64/128, 4/6 layers, dropout 0.1, weight decay 1e-3; best config
  (hidden 64, 4 layers) trained 3× with different seeds; logits averaged.
* **Model selection by "lift"**: expected gain of the policy's choices ÷ random choices.

### Results

| Version | Offline quality | Equal-time A/B on hard (10 min/case, 2 seeds, warm start) | Improving moves/min |
|---|---|---|---:|
| v2 (seed only) | AUC 0.583 | 1.3043 (from 1.3011) / **1.3131** (from 1.3125) | 51.0 / 50.0 |
| v3 (blocking graph + partner head) | seed AUC 0.562 (regret 0.450), partner AUC 0.548 (block rule 0.366) | 1.3043, a **tie** | **55.8 (+9%)** |
| v4 (gain ranking, ensemble) | **seed lift 1.98** (regret 0.885), partner lift 1.05–1.07 | 1.3126, a **tie** | **58.6 (+17%)** |

**What this means.** Each version is clearly better *offline*, and v3/v4 do find more improving
moves. But the score doesn't move: starting from ~1.31, the best possible rip-up choice still only
yields tiny gains. **The bottleneck has moved from *what to rip up* (destroy) to *how it gets
rebuilt* (repair)**, which is the algorithm side we chose not to change. Smaller notes:
v3 overfit after 4 epochs (150 instances); v4's ensemble members were uneven (lift 1.98, 1.83,
1.10), so the average was weaker than the best single model.

---

## Part E: GPU Lagrangian router (experimental, unfinished)

**Idea:** on a regular grid, shortest paths can be computed by **Bellman-Ford as a 3D min-plus
stencil** (like a convolution with `min` instead of `+`), for *every net at once* on the GPU. Put
a price on every resource and let all nets reroute simultaneously under those prices, raising
prices where resources are overused. That's PathFinder in parallel, written as Lagrangian
relaxation. The GPU entry `drama3d-portfolio` scores 1.4058 on hard, so the approach can work.

**Status:**
* ✅ The batched shortest-path kernel (`gpu/bellman.py`) works: all nets of a small test
  problem in 0.2 s (20 sweeps).
* ✅ Tree extraction (`gpu/trees.py`) works (about 1 ms).
* ❌ The price loop (`gpu/lagrange.py`) **slows down sharply as prices grow**, so the tests in
  `tests/test_gpu.py` don't finish yet. The root cause isn't pinned down; growing sweep counts or
  a tracing issue are the suspects.
* Note: a true Lagrangian *lower bound* is not valid here, because the score counts a shared
  wire once per sink while capacity counts it once per net.

---

## Part F: column generation (13 Oct): no gain, strong evidence

**Motivation.** Two papers shaped the day: GPU-parallel A\* (Zhou & Zeng 2015) and routing by
Monte Carlo tree search (He & Bao 2020). Before building GPU search, we asked how big the search
space really is, and whether an *exact* method could beat our LNS solutions.

### F1. How many ways can a case be wired?

| | hard `case_09` (40×40×6, 104 nets, 219 sinks) | designs `router` (105×105×6, 344 nets, 544 sinks) |
|---|---:|---:|
| every sink on a shortest-length route | ~10^2134 | ~10^5278 |
| including detours up to 2× length | ~10^10193 | ~10^32795 |
| net orders alone | 10^166 | 10^725 |

Listing them all is impossible (the universe has ~10^80 atoms). But **most of those routes have
the same delay.** Delay depends only on how many steps run on each layer and how many vias there
are, not on where the bends sit. `cg/delay_classes.py` counts the distinct delay values per sink:

| | raw shortest paths per sink (median) | distinct delays within +10 % | within +30 % |
|---|---:|---:|---:|
| `case_09` | 10^8.7 | **5** | 13 |
| `router` | 10^7.4 | **5** | 13 |

So *choosing a delay* is easy. The hard part is **placement**: fitting every net's tree into cells
that no other net uses.

### F2. The method: column generation

A case's score is the sum of every sink's driver-to-sink delay (`m3d/checker.py`). So a case's
total delay is simply the sum of each net's own tree delay. Nets interact only through one rule:
a cell belongs to at most one net. That makes routing an exact integer program over candidate
trees ("columns"):

```
min  Σ delay[n,k] · x[n,k]
s.t. Σ_k x[n,k] = 1                          every net picks one tree
     Σ_{(n,k) uses cell v} x[n,k] ≤ 1        no cell is shared
     x ∈ {0,1}
```

Column generation (as GRIP did for 2D global routing) solves this without listing every tree:

1. **Pool:** every net's tree from every saved legal solution (up to 19 per case). A legal answer
   always exists, and nets from different runs can be recombined.
2. **Relaxed solve:** let each net take fractions of several trees, and solve that easier problem
   (an LP). It gives each cell a **price π_v**: how much the total delay would improve if that
   cell could hold one more net. We use HiGHS's interior-point method without the final
   "crossover" step, which keeps these prices informative.
3. **Pricing:** for each net, find shortest-path trees on `delay + s·π_v` (several scales `s`,
   plus grown trees). A tree joins the pool if it would improve the relaxed solve, meaning its
   reduced cost `delay + Σπ_v − μ_n` is negative (μ_n is the price of giving net n any tree).
4. **Repeat**, then pick exactly one tree per net (the 0/1 solve).

We found and fixed two bugs along the way:
* Cells used by only one net had no price at all, so the pathfinder saw other nets' wires as free.
* Prices from the default simplex method were uninformative at the starting point, where every
  net already has a whole tree. The relaxed solve didn't move for 6 rounds, until we switched to
  interior-point prices.

### F3. Results

**The relaxed solve improves a lot; the legal solutions don't improve at all.**

| hard | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|
| our best (legal) | 1.3294 | 1.3799 | 1.2733 | 1.2825 | 1.2812 | 1.3102 | 1.4017 | 1.3408 | 1.2338 |
| relaxed solve after column generation | 1.4264 | 1.5170 | 1.3897 | 1.4447 | 1.4433 | 1.4245* | 1.4907* | 1.4456* | 1.3377* |
| best legal after selection | = | = | = | = | = | – | – | – | = |

\* stopped before round 50 (06 at round 45, 07 at 40, 08 at 37, 09 at 40). Pools held 8.7k–16.9k
trees. "=" means the selection returned exactly our previous best; "–" means no selection ran.

Why the relaxed numbers don't turn into legal gains:

* **The relaxed solve is far too optimistic.** It lets two nets each take half of a cell, which
  real wires can't do. Our pricing is also heuristic, so these numbers aren't proven limits in
  either direction. Treat them as optimistic estimates, not targets.
* **The full 0/1 problem is too big.** HiGHS found no solution in 5–10 minutes with 10–16k
  yes/no choices. Later runs were started from our best, so they returned it unchanged.
* **Monte Carlo rounding** (draw a tree per net weighted by its relaxed fraction, reroute clashes,
  refine) produced **zero legal samples** in 3 minutes per case. Rounding the fractions creates
  too many clashes.

**Exact solves on small groups (`cg/select.py`).** Free one net plus the 15–20 nets sitting on
cells its preferred trees want, and hold every other net fixed. Add candidate trees for the freed
nets that treat the fixed nets as walls and put penalties of 0/2/6/20 on each other's cells.
Then solve that small 0/1 problem exactly. On `case_01`: about **3,300 exact solves in 5 minutes,
0 improvements.** Each group *does* have better trees in isolation: ignoring clashes, the best
trees sum to 8–12 % less than the current ones. But **every better combination clashes**, so the
current choice is the best legal one. A tiny hand-made test confirmed the solver works.

**Restarts guided by the relaxed solution (`cg/restart.py`).** Start PathFinder from each net's
preferred tree in the relaxed solution instead of from our best, then repair the clashes, refine,
and run 10 min of LNS. The repair throws away most of the relaxed solution's advantage (scores
fall to ~1.00 right after repair). After 10 minutes: `case_01` 1.2668 (best 1.3294), `case_06`
1.2219 (best 1.3102), and `case_02` sample 0 couldn't be repaired at all. None of the 12 runs beat
its case's best. The other 9 exact numbers were lost: 12 processes appending to one log file
overwrote each other on Windows. Next time, give each process its own log file.

### F4. What this tells us

1. **Our solutions can't be improved by rearranging any group of 15–20 neighbouring nets**, among
   all the candidate trees we tried. Hours of LNS already found everything available at that
   scale. This also explains why every destroy-GNN version tied (Part D2).
2. **The headroom that's left (ideal ratio 1.6–1.9) is global.** Reaching it means changing many
   nets at once, which means finding a different basin.
3. **Starting near the relaxed solution doesn't land in a better basin** once PathFinder makes
   the routing legal.

---

## Every experiment and its result

All numbers are on the **hard** tier unless noted, from the official scorer or the identical
checker inside the runners.

| # | Setup | Score | Folder / log |
|---|---|---:|---|
| 1 | Congestion GNN v2 prior + stock router, keep the better of guided/plain per case | 1.0180 | `results/hard`, `results/logs/route_v2_hard_submission.log` |
| 1b | Same on **intro** | 1.0145 | `results/intro` |
| 2 | Control: GNN prior vs shuffled prior (v1, legal cases only) | +2.2% vs −1.2% | `results/logs/route_v1_hard_*_it100.log` |
| 3 | Fast engine, no LNS | 1.0626 | — |
| 4 | Engine + 60 s LNS | 1.1896 | — |
| 5 | Engine + 20 min LNS (**control**) | 1.2437 | `results/hard_engine`, `results/logs/overnight.log` |
| 6 | Engine + congestion GNN v3 prior + 20 min LNS | 1.2423 | `results/hard_engine_gnn` |
| 7 | Engine + destroy GNN v1 + 20 min LNS | 1.2434 (won 7/9 cases vs #5) | `results/hard_destroy_gnn`, `results/logs/hard_destroy.log` |
| 8 | Best per case of #5–#7 | 1.2550 | `results/hard_best` |
| 9 | **designs**: engine + v3 prior + 20 min LNS | 1.2749 | `results/designs_engine_gnn` |
| 10 | 2-min check: LNS v1 / v2 fixed mix / **v2 stall kicks** (3 cases) | 1.2039 / 1.1720 / **1.2295** | `results/logs/sanity_*.log` |
| 11 | **LNS v2 + destroy GNN v2, 3 h per case** | **1.3011** | `results/hard_lns3h`, `results/logs/hard_lns3h.log` |
| 12 | **designs**: same, 3 h | **1.3385** | `results/designs_lns3h`, `results/logs/designs_lns3h.log` |
| 13 | Best per case over #5–#12 | hard 1.3011, designs 1.3385 | `results/hard_final`, `results/designs_final`, `results/MORNING_SUMMARY.md` |
| 14 | Destroy GNN v3 trained (Phases 1–2) | seed AUC 0.562, partner AUC 0.548 | `results/logs/destroy_v3_train.log` |
| 15 | A/B v2 vs v3, warm from #13, 10 min × 2 seeds | 1.3043 vs 1.3043 (tie; v3 +9% improving moves) | `results/ab3/`, `results/logs/ab3.log` |
| 16 | v2, another 3 h warm-started from #13 | hard **1.3122**, designs **1.3561** | `results/hard_v3_3h`, `results/designs_v3_3h` (folder name only; policy was v2) |
| 17 | Best per case over #13–#16 | hard 1.3125, designs 1.3561 | `results/hard_final2`, `results/designs_final2`, `results/MORNING_SUMMARY3.md` |
| 18 | Destroy GNN v4 trained (Phase 3) | seed lift 1.98, partner lift 1.05 | `results/logs/destroy_v4_train.log` |
| 19 | A/B v2 vs v4, warm from #17, 10 min × 2 seeds | 1.3131 vs 1.3126 (tie; v4 +17% improving moves) | `results/ab4/`, `results/PHASE3_SUMMARY.md` |
| 20 | **Best per case over everything (final)** | hard **1.3138**, designs **1.3561** | `results/hard_final3`, `results/designs_final2` |
| 21 | Column generation, 37–50 rounds per case (relaxed solve) | 1.34–1.52 per case (optimistic, not legal) | `results/cg_hard/pools/`, `results/logs/cg_hard_live.log`, `results/logs/cg_case09.log` |
| 22 | Full 0/1 selection (HiGHS, 5–10 min) + Monte Carlo rounding | no improvement (01–05, 09); 0 legal Monte Carlo samples | same |
| 23 | Exact solves on 15–20-net groups, `case_01` | ~3,300 solves, 0 improvements | `cg/select.py` |
| 24 | Restart from the relaxed solution + 10 min LNS, cases 01–06 × 2 samples | none beat the best (e.g. 1.2668 vs 1.3294) | `results/logs/cg_restart_live.log` (partly overwritten) |

**Per case (final):**

| hard | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|
| score | 1.3294 | 1.3799 | 1.2733 | 1.2825 | 1.2812 | 1.3102 | **1.4017** | 1.3408 | 1.2338 |
| ideal ratio (ceiling) | 1.67 | 1.90 | 1.68 | 1.76 | 1.74 | 1.77 | 1.90 | 1.86 | 1.60 |

| designs | ctrl | int2float | router |
|---|---|---|---|
| score | 1.3926 | **1.5672** | 1.1428 |
| ideal ratio | 1.80 | 2.20 | 1.50 |

`router` is the weak spot: 344 nets, so moves are slow (252k in 3 h versus 435k for int2float).
`case_09` stopped improving after the first 3-hour run.

---

## Repository layout: every file and what it does

```
GNN_Routing_Optimisation/
├── README.md                 this file
├── requirements.txt          torch, torch_geometric, numpy, matplotlib (+ numba)
├── overnight.ps1             run #1: features → train congestion GNN v3 → 20-min LNS runs → scores
├── overnight2.sh             run #2: wait for destroy GNN v2 → pick policy → 3-h LNS → best-of → summary
├── overnight3.sh             run #3 (Phases 0–2): v3 data → train v3 → A/B v2 vs v3 → 3-h warm runs → summary
├── overnight4.sh             run #4 (Phase 3): v4 data (hard stop 04:05) → sweep + ensemble → A/B v2 vs v4
├── gnn/                      learned parts + data
│   ├── _paths.py             finds the challenge repo (its m3d toolkit + benchmarks)
│   ├── datagen.py            random problems + labels (reference routers or --labeler engine)
│   ├── features.py           problem → tile graph (19 features); solution → usage targets; --dataset engine
│   ├── model.py              congestion GNN (8 × GATv2)
│   ├── train.py              training loop, baselines, early stopping, size filters, extra val set
│   ├── evaluate.py           R² of congestion models on the official tiers
│   ├── predict.py            prediction-vs-truth pictures for any case
│   ├── route.py              congestion prior inside the *stock* router (first approach) + shuffle control
│   ├── destroy.py            destroy GNN v1–v4: data collection, models (seed + partner heads),
│   │                         training (BCE / listwise ranking, sweep, ensemble), LNS policy
│   └── ab.py                 equal-time A/B runner (score + improving moves/min)
├── engine/                   fast exact router (Numba)
│   ├── fast.py               SPT trees, negotiation, refinement, LNS v1/v2, warm start, owner map,
│   │                         net graphs (v2 overlap / v3 blocking), history counters, labels v2–v4
│   ├── run.py                route a tier (--ckpt prior, --destroy, --warm, --lns-seconds, --anneal,
│   │                         --stall-kick, --json-out; congestion features auto-loaded for v3+ policies)
│   └── best_of.py            combine runs: best legal route per case, re-verified, runtimes summed
├── gpu/                      GPU Lagrangian router (experimental, unfinished)
│   ├── tensors.py            problem → GPU tensors
│   ├── bellman.py            batched Bellman-Ford min-plus stencil
│   ├── trees.py              predecessor directions → trees (Numba)
│   ├── lagrange.py           price-based parallel routing loop
│   ├── handoff.py            GPU result → CPU engine repair/refine/LNS
│   └── run.py                command-line runner
├── cg/                       column generation (Part F)
│   ├── delay_classes.py      distinct delay values per sink vs raw path counts
│   ├── colgen.py             pool from saved solutions → relaxed solve (interior-point prices) →
│   │                         pricing → Monte Carlo rounding → warm-started 0/1 solve; saves the pool
│   ├── select.py             exact 0/1 solves on 15–20-net groups over a saved pool
│   ├── restart.py            restart from the relaxed solution: its trees → PathFinder repair → refine → LNS
│   └── run_hard.sh           colgen on all 9 hard cases in parallel, one live log, best-of
├── tests/test_gpu.py         GPU correctness tests (delays = Dijkstra, valid trees, legal pipeline)
├── models/
│   ├── v1_gat8.pt, v2_gat8.pt   congestion GNN on reference labels
│   ├── v3_engine.pt             congestion GNN on engine labels (best)
│   ├── destroy_v1.pt            destroy GNN, noisy labels
│   ├── destroy_v2.pt            destroy GNN, clean labels (used for every final run)
│   ├── destroy_v3.pt            blocking graph + partner head (Phases 1–2)
│   └── destroy_v4.pt            gain ranking, ensemble of 3 (Phase 3)
├── results/
│   ├── hard_final3/, designs_final2/ ⭐ the submissions (solutions + runtime.json)
│   ├── hard_final/, hard_final2/, designs_final/   earlier best-per-case snapshots
│   ├── hard_v3_3h/, designs_v3_3h/   second 3-hour runs (warm-started, policy v2)
│   ├── hard_lns3h/, designs_lns3h/   first 3-hour runs
│   ├── ab3/, ab4/                    A/B runs (solutions, JSON, AB_REPORT.md)
│   ├── hard_engine/, hard_engine_gnn/, hard_destroy_gnn/, hard_best/, designs_engine_gnn/   20-min runs
│   ├── hard/, intro/                 first approach (GNN prior + stock router)
│   ├── cg_hard/pools/                column-generation pools for hard 01–06 (pickle, ~9–17k trees each)
│   ├── MORNING_SUMMARY.md, MORNING_SUMMARY3.md, PHASE3_SUMMARY.md   overnight summaries
│   └── logs/                         training metrics and every routing log cited above
└── docs/figures/             README images
```

Git-ignored (large, reproducible): `gnn/data/` (all datasets), `gnn/checkpoints/`, `gnn/runs/`,
`__pycache__/` (including Numba caches). `results/congested_engine_gnn/` is an empty leftover
from a run that was stopped on purpose.

---

## Setup and reproducing everything

**1. Put both repos side by side.** This project imports the challenge's `m3d` toolkit and reads
its benchmarks. It looks for a sibling folder named `Partcl-Competition` or
`eda-3d-routing-challenge`, or the path in `M3D_REPO`.

```
D:/
├── Partcl-Competition/          git clone https://github.com/partcleda/eda-3d-routing-challenge
└── GNN_Routing_Optimisation/    this repo
```

**2. Install** (Python 3.9+, the CUDA build of PyTorch first):

```bash
pip install -r requirements.txt numba
```

**3. Run from this repo's root.** `python -m gnn.<x>` needs the root as the current directory.

```bash
# ---- the final pipeline (what produced 1.3011 / 1.3385) ----
# destroy GNN v2: labels on generated problems (~1 h on 12 cores), then training (~5 min)
python -m gnn.datagen --split train --n 3000 --labeler engine --lns-seconds 3 --out gnn/data/raw_engine
python -m gnn.destroy data --n 150 --tries 16 --workers 12 --data-dir gnn/data/destroy_v2
python -m gnn.destroy train --epochs 60 --data-dir gnn/data/destroy_v2 --out models/destroy_v2.pt
# 3-hour LNS on hard and designs (or just: bash overnight2.sh)
python -m engine.run --suite benchmarks_hard    --workers 9 --lns-seconds 10800 --destroy models/destroy_v2.pt --out-dir results/hard_lns3h
python -m engine.run --suite benchmarks_designs --workers 3 --lns-seconds 10800 --destroy models/destroy_v2.pt --out-dir results/designs_lns3h
python -m engine.best_of --suite benchmarks_hard --runs results/hard_engine results/hard_lns3h --out results/hard_final
# continue from the best solutions (warm start), then combine again
python -m engine.run --suite benchmarks_hard --workers 9 --lns-seconds 10800 --destroy models/destroy_v2.pt --anneal 0.05 --warm results/hard_final --out-dir results/hard_v3_3h
python -m engine.best_of --suite benchmarks_hard --runs results/hard_final results/hard_v3_3h --out results/hard_final2

# ---- Phases 0-3: destroy GNN v3 / v4 and A/B tests (or: bash overnight3.sh / overnight4.sh) ----
python -m gnn.destroy data  --version 3 --n 150 --tries 8  --waits 10 60 180  --workers 14 --data-dir gnn/data/destroy_v3
python -m gnn.destroy train --version 3 --epochs 60 --data-dir gnn/data/destroy_v3 --out models/destroy_v3.pt
python -m gnn.destroy data  --version 4 --n 200 --tries 12 --waits 30 180 420 --workers 14 --data-dir gnn/data/destroy_v4
python -m gnn.destroy train --version 4 --epochs 40 --data-dir gnn/data/destroy_v4 --out models/destroy_v4.pt
python -m gnn.ab --suite benchmarks_hard --seconds 600 --seeds 1 2 --warm results/hard_final2 \
    --config v2="--destroy models/destroy_v2.pt --anneal 0.05" --config v4="--destroy models/destroy_v4.pt --anneal 0.05" \
    --out results/ab4 --report results/ab4/AB_REPORT.md

# ---- Part F: column generation (needs scipy + highspy) ----
python -m cg.delay_classes benchmarks_hard/case_09.json benchmarks_designs/router.json
python -m cg.colgen  --case case_09 --rounds 50 --mc-seconds 180 --ilp-seconds 600 --out results/cg_hard
bash cg/run_hard.sh                                   # all 9 hard cases in parallel
python -m cg.select  --case case_01 --seconds 900     # exact solves on small groups over the saved pool
python -m cg.restart --case case_01 --sample 0 --lns-seconds 600

# ---- quick checks ----
python -m engine.run --suite benchmarks_hard --workers 9 --lns-seconds 60          # ~1.19 in 1 min
python -m engine.run --suite benchmarks_hard --anneal 0 --stall-kick 0 ...         # LNS v1 behaviour

# ---- congestion GNN (Part A) ----
python -m gnn.features --dataset engine --split train
python -m gnn.train --dataset engine --epochs 80 --run v3_engine
python -m gnn.evaluate --ckpt models/v2_gat8.pt models/v3_engine.pt
python -m gnn.predict --ckpt models/v3_engine.pt --case benchmarks_hard/case_05.json --sol benchmarks_hard/reference/case_05.sol.json
```

Score any folder with the official scorer, from the challenge repo:

```bash
python -m m3d.cli score-suite --suite benchmarks_hard --submission-dir ../GNN_Routing_Optimisation/results/hard_final
```

**Hardware used:** a laptop with a 16-core CPU and an RTX 4060 (8 GB). Routing is CPU-bound;
GNN training and inference use the GPU.

---

## Submitting to the leaderboard

1. In your fork of the challenge repo, copy `results/hard_final3/*` to `submissions/hard/<name>/`
   (and `results/designs_final2/*` to `submissions/designs/<name>/`).
2. Add a `meta.json`, for example:
   ```json
   {"author": "<you>", "url": "https://github.com/<you>/GNN_Routing_Optimisation",
    "method": "Numba exact-SPT router: negotiation + exact refinement + annealed LNS with negotiated stall kicks; GATv2 destroy GNN picks what LNS rips up (best of runs)",
    "date": "2026-10-13"}
   ```
3. Run `python -m m3d.cli leaderboard-all` and `python scripts/verify_submissions.py`.
4. Commit only `submissions/...`, `LEADERBOARD.md` and the README's generated leaderboard block,
   then open the PR.

`runtime.json` in the final folders sums **every** run that contributed (honest Pareto reporting).

---

## Lessons learned

1. **A GNN alone can't route this problem.** Legality is all or nothing, so learning has to
   guide an exact router.
2. **Search beats starting points.** A better start (the congestion prior) was erased by long LNS.
   Steering the *search itself* (the destroy GNN) is where learning pays off.
3. **Label noise kills learning.** Destroy GNN v1 (4 tries per net) was about as good as a coin
   flip; v2 (16 tries, a rate target) beat the hand-made rule clearly.
4. **Measure before scaling up.** The first LNS v2 mix was *worse*; a 90-second ablation showed
   why (negotiated moves are 10× more successful but 20× slower), and "stall kicks" fixed it.
5. **Always run controls.** Shuffled priors and equal-time comparisons kept us from believing lucky
   numbers.
6. **The ceiling is visible.** The ideal ratio (1.6–2.2) shows how far each case could still go.
7. **Offline metrics can lie.** v4's choices were ~2× better than random on held-out data and
   found 17% more improving moves, yet tied v2 on the score. Only the equal-time A/B on the real
   benchmarks tells the truth.
8. **Find the real bottleneck.** Once the search plateaus (~1.31), choosing *what* to rip up stops
   mattering, because even good rip-ups only yield tiny gains. The limit is now the *repair*
   (how the ripped nets are rebuilt) and the basin the search is stuck in.
9. **Warm starts are free score.** Continuing from the best saved solution added +1.1 points on
   hard and +1.8 on designs with no change to the method.
10. **Count what actually differs.** ~10^2134 routings sounds hopeless, but each sink has only
    ~5 genuinely different delays near its best. The problem is placement, not path choice.
11. **A relaxed solve can look great and mean little.** Column generation's relaxed solve said
    1.42–1.52, but only because it lets nets share cells half-and-half. Exact solves on small
    groups then showed our solutions were already the best possible for every local rearrangement
    we tested. Always turn a promising estimate into a *legal* answer before believing it.

## Next steps

Part F showed that local moves are exhausted. The remaining options are about **reaching a
different basin**, and about speed (more independent restarts per hour).

0. **Massively parallel restarts on the GPU (CUDA).** Batched A\* per net: one thread block per
   search, thousands in flight across nets, samples and cases. That would build and legalize
   hundreds of diverse full routings per hour, each followed by LNS. Part F's restarts reached
   only 1.22–1.27 after 10 minutes, so this pays off only if restarts are both many *and* long.
   A cheap test first: a few 1-hour LNS runs from new basins.
1. **Monte Carlo / soft-min GPU router (changes the algorithm).** Replace `min`
   with log-sum-exp in the GPU Bellman-Ford stencil: every net then has a probability
   distribution over *all* its routes, `P(route) ∝ exp(−(delay + price)/τ)`, and one autograd
   pass gives the exact probability of every edge being used by every net. Anneal τ and the
   prices so the nets' clouds jointly avoid each other, then draw many concrete routings
   ("perturb-and-MAP" Monte Carlo samples), legalize each with the CPU engine and run LNS from
   the best. The samples land in *different basins*, which targets lesson 8 directly. The
   congestion GNN can predict the initial prices and temperature. Risks: rounding to one legal
   route per net, and same-net sharing being overcounted. A ~3 h test would tell whether any
   sample basin beats 1.3138 after LNS.
2. **Let the GNN steer the repair, not just the destroy:** per-net cost hints when rerouting a
   ripped group (still learning, not a new search algorithm).
3. **Remaining Phase 2 heads:** reroute order and move type (greedy vs negotiated).
4. **Stabilize v4:** more instances, a better ensemble (drop weak members), then re-test from a
   less-optimized starting state, where destroy choices matter more.
5. **Finish the GPU router** (`gpu/`): profile the price-loop slowdown.
6. Run the remaining tiers (congested, scale, intro) with the final pipeline.

---

## References

* Partcl, **M3D Routing Challenge**: https://github.com/partcleda/eda-3d-routing-challenge (MIT)
* L. McMurchie, C. Ebeling, **PathFinder: A Negotiation-Based Performance-Driven Router for FPGAs** (FPGA 1995)
* P. Shaw, **Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems** (CP 1998), the origin of LNS
* J. Song et al., **A General Large Neighborhood Search Framework for Solving Integer Linear Programs** (NeurIPS 2020), learned destroy policies
* S. Brody, U. Alon, E. Yahav, **How Attentive are Graph Attention Networks?** (ICLR 2022), GATv2
* R. Kirby et al., **CongestionNet: Routing Congestion Prediction Using Deep Graph Neural Networks** (VLSI-SoC 2019)
* X. Du et al., **HubRouter: Learning Global Routing via Hub Generation and Pin-hub Connection** (NeurIPS 2023)
* P. Veličković et al., **Neural Execution of Graph Algorithms** (ICLR 2020), GNNs and Bellman-Ford
* Q. Cappart et al., **Combinatorial Optimization and Reasoning with Graph Neural Networks** (JMLR 2023)
* T.-H. Wu, A. Davoodi, J. T. Linderoth, **GRIP: Scalable 3D Global Routing Using Integer Programming** (DAC 2009), column generation for routing
* M. E. Lübbecke, J. Desrosiers, **Selected Topics in Column Generation** (Operations Research 2005)
* Q. Huangfu, J. A. J. Hall, **Parallelizing the dual revised simplex method** (Math. Prog. Computation 2018), the HiGHS solver
* Y. Zhou, J. Zeng, **Massively Parallel A\* Search on a GPU** (AAAI 2015)
* Y. He, F. S. Bao, **Circuit Routing Using Monte Carlo Tree Search and Deep Neural Networks** (arXiv 2006.13607, 2020)
* A. B. Kahng, J. Lienig, I. L. Markov, J. Hu, **VLSI Physical Design: From Graph Partitioning to Timing Closure** (Springer)

---

*Built with help from Claude Code. The `m3d` toolkit and all benchmark data belong to the
challenge repository and are not redistributed here.*
