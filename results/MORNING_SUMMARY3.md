# Overnight run #3: destroy GNN v3 (2026-10-13 01:09)

## Destroy GNN v3 training
```
epoch   2  seed AUC 0.549 hit@10% 0.176 | partner AUC 0.543  * best
epoch   3  seed AUC 0.555 hit@10% 0.181 | partner AUC 0.547  * best
epoch   4  seed AUC 0.562 hit@10% 0.209 | partner AUC 0.548  * best
[destroy3] saved models/destroy_v3.pt
```

## A/B (warm-started from results/hard_final, 600s per case, 2 seeds)
| config | mean aggregate | per seed | improving moves/min |
|---|---:|---|---:|
| v2 | 1.3043 | 1.3042, 1.3045 | 51.0 |
| v3 | 1.3043 | 1.3038, 1.3049 | 55.8 |

winner: v2

## Scores (official)

| Run | Score |
|---|---|
| hard, 3 h with v2, warm start | complete=True  legal=9/9  AGGREGATE=1.3122 |
| **hard, best per case** (`results/hard_final2`) | complete=True  legal=9/9  AGGREGATE=1.3125 |
| designs, 3 h with v2, warm start | complete=True  legal=3/3  AGGREGATE=1.3561 |
| **designs, best per case** (`results/designs_final2`) | complete=True  legal=3/3  AGGREGATE=1.3561 |

Previous: hard 1.3011, designs 1.3385.

## Per-case picks
```
  case_01: 1.3206  from hard_v3_3h
  case_02: 1.3799  from hard_v3_3h
  case_03: 1.2733  from hard_v3_3h
  case_04: 1.2825  from v3_s2
  case_05: 1.2787  from hard_v3_3h
  case_06: 1.3102  from hard_v3_3h
  case_07: 1.4017  from hard_v3_3h
  case_08: 1.3408  from hard_v3_3h
  case_09: 1.2338  from hard_final
  ctrl: 1.3926  from designs_v3_3h
  int2float: 1.5672  from designs_v3_3h
  router: 1.1428  from designs_v3_3h
```
