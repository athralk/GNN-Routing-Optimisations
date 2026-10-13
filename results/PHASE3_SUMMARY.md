# Phase 3 results (2026-10-13 04:56)

Destroy GNN v4: gain-size labels from late search states (30/210/630 s into LNS),
listwise ranking loss, sweep of 3 configs, ensemble of 3.

## Training (lift = expected gain of the policy's choices / random choices; 1.0 = random)
```
[destroy4] 509 train / 84 val snapshots from 200 instances
[ref]   lift (1.0 = uniform): regret seed 0.885 | random partners 1.000
[sweep] hidden 64 layers 4 dropout 0.1: seed lift 1.982 partner lift 1.045 (epoch 24)  90s
[sweep] hidden 128 layers 4 dropout 0.1: seed lift 1.082 partner lift 1.049 (epoch 5)  125s
[sweep] hidden 64 layers 6 dropout 0.1: seed lift 1.138 partner lift 1.042 (epoch 12)  117s
[ensemble] member 1: seed lift 1.830 partner lift 1.070
[ensemble] member 2: seed lift 1.100 partner lift 1.058
[destroy4] saved models/destroy_v4.pt: config {'version': 3, 'in_dim': 18, 'edge_dim': 5, 'hidden': 64, 'layers': 4, 'dropout': 0.1}, ensemble of 3
```

## A/B on hard (warm-started from results/hard_final2 = 1.3125, 10 min per case, 2 seeds)
| config | mean aggregate | per seed | improving moves/min |
|---|---:|---|---:|
| v2 | 1.3131 | 1.3135, 1.3128 | 50.0 |
| v4 | 1.3126 | 1.3125, 1.3127 | 58.6 |

winner: v2
