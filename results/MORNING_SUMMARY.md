# Overnight results (2026-10-09 05:15)

LNS: 10800s per case, annealing + stall-triggered negotiated moves. Seed policy: destroy GNN v2 (AUC 0.583 vs regret 0.522).

| Run | Official score |
|---|---|
| hard, 3 h LNS (this run) | complete=True  legal=9/9  AGGREGATE=1.3011 |
| **hard, best per case over all runs** (`results/hard_final`) | complete=True  legal=9/9  AGGREGATE=1.3011 |
| designs, 3 h LNS (this run) | complete=True  legal=3/3  AGGREGATE=1.3385 |
| **designs, best per case** (`results/designs_final`) | complete=True  legal=3/3  AGGREGATE=1.3385 |

Previous bests: hard 1.2550, designs 1.2749. Leaderboard top: hard 1.4136, designs 1.4673.

## Per-case picks
```
  case_01: 1.2837  from hard_lns3h
  case_02: 1.3725  from hard_lns3h
  case_03: 1.263  from hard_lns3h
  case_04: 1.2798  from hard_lns3h
  case_05: 1.2725  from hard_lns3h
  case_06: 1.302  from hard_lns3h
  case_07: 1.3852  from hard_lns3h
  case_08: 1.3255  from hard_lns3h
  case_09: 1.2338  from hard_lns3h
  ctrl: 1.3723  from designs_lns3h
  int2float: 1.5428  from designs_lns3h
  router: 1.1327  from designs_lns3h
```

## Raw run logs
```
gnn/runs/hard_lns3h.log:  case_01: ratio 1.2837  (after negotiation 1.0496)  iters 10  refined 27 nets  lns 4565485 moves (45629 improving), ideal ratio 1.6683  10802.2s
gnn/runs/hard_lns3h.log:  case_04: ratio 1.2798  (after negotiation 1.0288)  iters 12  refined 34 nets  lns 2962508 moves (26502 improving), ideal ratio 1.7631  10803.1s
gnn/runs/hard_lns3h.log:  case_05: ratio 1.2725  (after negotiation 1.0303)  iters 12  refined 55 nets  lns 2783512 moves (31005 improving), ideal ratio 1.7356  10803.4s
gnn/runs/hard_lns3h.log:  case_02: ratio 1.3725  (after negotiation 0.9999)  iters 38  refined 59 nets  lns 5707533 moves (37704 improving), ideal ratio 1.8955  10804.0s
gnn/runs/hard_lns3h.log:  case_06: ratio 1.3020  (after negotiation 0.9985)  iters 13  refined 63 nets  lns 2746497 moves (25988 improving), ideal ratio 1.7698  10804.2s
gnn/runs/hard_lns3h.log:  case_07: ratio 1.3852  (after negotiation 0.9798)  iters 13  refined 74 nets  lns 2508710 moves (30453 improving), ideal ratio 1.9040  10804.9s
gnn/runs/hard_lns3h.log:  case_03: ratio 1.2630  (after negotiation 1.0167)  iters 10  refined 57 nets  lns 3536979 moves (30825 improving), ideal ratio 1.6839  10805.1s
gnn/runs/hard_lns3h.log:  case_08: ratio 1.3255  (after negotiation 1.0346)  iters 19  refined 74 nets  lns 2037243 moves (29159 improving), ideal ratio 1.8633  10805.2s
gnn/runs/hard_lns3h.log:  case_09: ratio 1.2338  (after negotiation 1.0293)  iters 9  refined 47 nets  lns 1419158 moves (24111 improving), ideal ratio 1.5974  10805.3s
gnn/runs/hard_lns3h.log:[engine] legal 9/9  aggregate 1.3011  (legal-only 1.3011)  total 97237.4s
gnn/runs/designs_lns3h.log:  ctrl: ratio 1.3723  (after negotiation 1.1337)  iters 9  refined 130 nets  lns 301778 moves (13930 improving), ideal ratio 1.8026  10825.5s
gnn/runs/designs_lns3h.log:  router: ratio 1.1327  (after negotiation 1.0091)  iters 9  refined 229 nets  lns 252349 moves (9307 improving), ideal ratio 1.4975  10842.2s
gnn/runs/designs_lns3h.log:  int2float: ratio 1.5428  (after negotiation 1.2079)  iters 12  refined 248 nets  lns 434656 moves (23315 improving), ideal ratio 2.2011  10849.0s
gnn/runs/designs_lns3h.log:[engine] legal 3/3  aggregate 1.3385  (legal-only 1.3385)  total 32516.7s
```
