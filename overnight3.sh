#!/usr/bin/env bash
# Overnight run #3 -- destroy GNN v3 (Phases 0-2). Run from this repo's root (Git Bash):
#   bash overnight3.sh
# 1. destroy-v3 labels on generated problems (blocking graph, history + congestion
#    features, seed AND partner labels, deeper search states)        ~60-70 min, CPU
# 2. train destroy GNN v3 (seed head + partner head)                  ~10 min, GPU
# 3. equal-time A/B, warm-started from the current best solutions:   ~45 min
#    v2 (seed only) vs v3 (seed + partners), 10 min x 2 seeds
# 4. winner: 3-hour warm-started runs on hard + designs               ~3 h
# 5. best-per-case over everything, official scores -> results/MORNING_SUMMARY3.md
LNS=${LNS:-10800}
AB=${AB:-600}
M3D=${M3D_REPO:-../Partcl-Competition}
LOG=gnn/runs/overnight3.log
ROOTW=$(pwd -W)
mkdir -p gnn/runs
say() { echo "=== $(date +%H:%M) $*" | tee -a $LOG; }
say "start"

python -m gnn.destroy data --version 3 --n 150 --tries 8 --waits 10 60 180 --workers 14 \
  --data-dir gnn/data/destroy_v3 > gnn/runs/destroy_v3_data.log 2>&1
say "data done: $(grep -ac improvable gnn/runs/destroy_v3_data.log) instances"

python -m gnn.destroy train --version 3 --epochs 60 --hidden 128 --layers 6 \
  --data-dir gnn/data/destroy_v3 --out models/destroy_v3.pt > gnn/runs/destroy_v3_train.log 2>&1
say "trained: $(grep -a '\[ref\]' gnn/runs/destroy_v3_train.log) | best: $(grep -a '\* best' gnn/runs/destroy_v3_train.log | tail -1)"

python -m gnn.ab --suite benchmarks_hard --seconds $AB --seeds 1 2 --warm results/hard_final \
  --config v2="--destroy models/destroy_v2.pt --anneal 0.05" \
  --config v3="--destroy models/destroy_v3.pt --anneal 0.05" \
  --out results/ab3 --report results/ab3/AB_REPORT.md > gnn/runs/ab3.log 2>&1
WIN=$(grep -a "winner:" gnn/runs/ab3.log | awk '{print $3}')
POLICY="--destroy models/destroy_${WIN}.pt"
say "A/B winner: $WIN"; cat results/ab3/AB_REPORT.md >> $LOG

python -m engine.run --suite benchmarks_hard --workers 9 --lns-seconds $LNS $POLICY --anneal 0.05 \
  --warm results/hard_final --out-dir results/hard_v3_3h > gnn/runs/hard_v3_3h.log 2>&1 &
python -m engine.run --suite benchmarks_designs --workers 3 --lns-seconds $LNS $POLICY --anneal 0.05 \
  --warm results/designs_final --out-dir results/designs_v3_3h > gnn/runs/designs_v3_3h.log 2>&1 &
wait
say "long runs done"

python -m engine.best_of --suite benchmarks_hard --out results/hard_final2 --runs results/hard_final \
  results/hard_v3_3h results/ab3/v2_s1 results/ab3/v2_s2 results/ab3/v3_s1 results/ab3/v3_s2 >> $LOG 2>&1
python -m engine.best_of --suite benchmarks_designs --out results/designs_final2 \
  --runs results/designs_final results/designs_v3_3h >> $LOG 2>&1

score() { (cd "$M3D" && python -m m3d.cli score-suite --suite "$1" --submission-dir "$ROOTW/$2" 2>&1 | grep -a AGGREGATE); }
{
  echo "# Overnight run #3: destroy GNN v3 ($(date '+%Y-%m-%d %H:%M'))"
  echo
  echo "## Destroy GNN v3 training"
  echo '```'; grep -a -E "\[destroy3\]|\[ref\]|\* best" gnn/runs/destroy_v3_train.log | tail -4; echo '```'
  echo
  echo "## A/B (warm-started from results/hard_final, ${AB}s per case, 2 seeds)"
  cat results/ab3/AB_REPORT.md
  echo
  echo "## Scores (official)"
  echo
  echo "| Run | Score |"
  echo "|---|---|"
  echo "| hard, 3 h with ${WIN}, warm start | $(score benchmarks_hard results/hard_v3_3h) |"
  echo "| **hard, best per case** (\`results/hard_final2\`) | $(score benchmarks_hard results/hard_final2) |"
  echo "| designs, 3 h with ${WIN}, warm start | $(score benchmarks_designs results/designs_v3_3h) |"
  echo "| **designs, best per case** (\`results/designs_final2\`) | $(score benchmarks_designs results/designs_final2) |"
  echo
  echo "Previous: hard 1.3011, designs 1.3385."
  echo
  echo "## Per-case picks"
  echo '```'; grep -a "from" $LOG | grep -a -E "case_|ctrl|router|int2float"; echo '```'
} > results/MORNING_SUMMARY3.md
say "done -> results/MORNING_SUMMARY3.md"
