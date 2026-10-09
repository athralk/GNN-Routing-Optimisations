# GNN Routing Optimisation

**A fast exact router, guided by graph neural networks, for the
[Partcl M3D Routing Challenge](https://github.com/partcleda/eda-3d-routing-challenge).**
The task: route signal wires through a two-die, monolithic-3D chip stack with the lowest total
signal delay.

## Results (official challenge scorer, every case legal)

| Tier | Score | vs baseline | Approx. leaderboard position (Oct 2026) | Files |
|---|---:|---:|---|---|
| **hard** (9 cases) | **1.3011** | **+30 %** | ~9th of 19 (top: 1.4136) | `results/hard_final/` |
| **designs** (3 real circuits) | **1.3385** | **+34 %** | ~7th of 12 (top: 1.4673) | `results/designs_final/` |
| intro (20 cases), early version | 1.0145 | +1.5 % | (first approach only) | `results/intro/` |

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
```

---

## Contents

1. [The competition in plain terms](#1-the-competition-in-plain-terms)
2. [The big picture: what the GNNs do and what the router does](#2-the-big-picture)
3. [Glossary](#3-glossary)
4. [Part A: the congestion GNN (first approach)](#part-a--the-congestion-gnn-first-approach)
5. [Part B: the fast exact router (`engine/`)](#part-b--the-fast-exact-router-engine)
6. [Part C: LNS, where most of the score comes from](#part-c--lns-where-most-of-the-score-comes-from)
7. [Part D: the destroy GNN, which picks what LNS rips up](#part-d--the-destroy-gnn-which-picks-what-lns-rips-up)
8. [Part E: GPU Lagrangian router (experimental, unfinished)](#part-e--gpu-lagrangian-router-experimental-unfinished)
9. [Every experiment and its result](#every-experiment-and-its-result)
10. [Repository layout: every file and what it does](#repository-layout-every-file-and-what-it-does)
11. [Setup and reproducing everything](#setup-and-reproducing-everything)
12. [Submitting to the leaderboard](#submitting-to-the-leaderboard)
13. [Lessons learned](#lessons-learned)
14. [Next steps](#next-steps)
15. [References](#references)

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
| 13 | Best per case over every run (final) | hard **1.3011**, designs **1.3385** | `results/hard_final`, `results/designs_final`, `results/MORNING_SUMMARY.md` |

**Per case (final):**

| hard | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 |
|---|---|---|---|---|---|---|---|---|---|
| score | 1.2837 | 1.3725 | 1.2630 | 1.2798 | 1.2725 | 1.3020 | **1.3852** | 1.3255 | 1.2338 |
| ideal ratio (ceiling) | 1.67 | 1.90 | 1.68 | 1.76 | 1.74 | 1.77 | 1.90 | 1.86 | 1.60 |

| designs | ctrl | int2float | router |
|---|---|---|---|
| score | 1.3723 | **1.5428** | 1.1327 |
| ideal ratio | 1.80 | 2.20 | 1.50 |

`router` is the weak spot: 344 nets, so moves are slow (252k in 3 h versus 435k for int2float).

---

## Repository layout: every file and what it does

```
GNN_Routing_Optimisation/
├── README.md                 this file
├── requirements.txt          torch, torch_geometric, numpy, matplotlib (+ numba)
├── overnight.ps1             run #1: features → train congestion GNN v3 → 20-min LNS runs → scores
├── overnight2.sh             run #2: wait for destroy GNN v2 → pick policy → 3-h LNS → best-of → summary
├── gnn/                      learned parts + data
│   ├── _paths.py             finds the challenge repo (its m3d toolkit + benchmarks)
│   ├── datagen.py            random problems + labels (reference routers or --labeler engine)
│   ├── features.py           problem → tile graph (19 features); solution → usage targets; --dataset engine
│   ├── model.py              congestion GNN (8 × GATv2)
│   ├── train.py              training loop, baselines, early stopping, size filters, extra val set
│   ├── evaluate.py           R² of congestion models on the official tiers
│   ├── predict.py            prediction-vs-truth pictures for any case
│   ├── route.py              congestion prior inside the *stock* router (first approach) + shuffle control
│   └── destroy.py            destroy GNN: data collection, model, training, LNS policy
├── engine/                   fast exact router (Numba)
│   ├── fast.py               SPT trees, negotiation, refinement, LNS v1/v2, net features, labels
│   ├── run.py                route a tier (options: --ckpt prior, --destroy, --lns-seconds, --anneal, --stall-kick)
│   └── best_of.py            combine runs: best legal route per case, re-verified, runtimes summed
├── gpu/                      GPU Lagrangian router (experimental, unfinished)
│   ├── tensors.py            problem → GPU tensors
│   ├── bellman.py            batched Bellman-Ford min-plus stencil
│   ├── trees.py              predecessor directions → trees (Numba)
│   ├── lagrange.py           price-based parallel routing loop
│   ├── handoff.py            GPU result → CPU engine repair/refine/LNS
│   └── run.py                command-line runner
├── tests/test_gpu.py         GPU correctness tests (delays = Dijkstra, valid trees, legal pipeline)
├── models/
│   ├── v1_gat8.pt, v2_gat8.pt   congestion GNN on reference labels
│   ├── v3_engine.pt             congestion GNN on engine labels (best)
│   ├── destroy_v1.pt            destroy GNN, noisy labels
│   └── destroy_v2.pt            destroy GNN, clean labels (used for the final results)
├── results/
│   ├── hard_final/, designs_final/   ⭐ the submissions (solutions + runtime.json)
│   ├── hard_lns3h/, designs_lns3h/   the 3-hour runs
│   ├── hard_engine/, hard_engine_gnn/, hard_destroy_gnn/, hard_best/, designs_engine_gnn/   20-min runs
│   ├── hard/, intro/                 first approach (GNN prior + stock router)
│   ├── MORNING_SUMMARY.md            overnight run #2 summary
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

1. In your fork of the challenge repo, copy `results/hard_final/*` to `submissions/hard/<name>/`
   (and `results/designs_final/*` to `submissions/designs/<name>/`).
2. Add a `meta.json`, for example:
   ```json
   {"author": "<you>", "url": "https://github.com/<you>/GNN_Routing_Optimisation",
    "method": "Numba exact-SPT router: negotiation + exact refinement + annealed LNS with negotiated stall kicks; GATv2 destroy GNN picks what LNS rips up (best of runs)",
    "date": "2026-10-09"}
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

## Next steps

1. **More seeds, best per case:** 2–3 more 3-hour runs, likely +1–2 points.
2. **Destroy GNN v3:** collect labels from *late* search states (v2's came from the first ~30 s
   of LNS), and predict which neighbours to rip up as well as the seed net.
3. **Designs `router`:** more time; speed up its moves (bounded-region Dijkstra).
4. **Finish the GPU router:** profile sweeps per call as prices grow; then try GPU-parallel LNS
   (many disjoint moves at once).
5. Run the remaining tiers (congested, scale, intro) with the final pipeline.

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
* A. B. Kahng, J. Lienig, I. L. Markov, J. Hu, **VLSI Physical Design: From Graph Partitioning to Timing Closure** (Springer)

---

*Built with help from Claude Code. The `m3d` toolkit and all benchmark data belong to the
challenge repository and are not redistributed here.*
