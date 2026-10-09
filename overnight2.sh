#!/usr/bin/env bash
# Overnight run #2 (run from this repo's root, in Git Bash):  bash overnight2.sh
# 1. wait for destroy-GNN v2 training to finish
# 2. use the GNN destroy policy only if it beats regret ranking on held-out data
# 3. 3-hour LNS (annealing + stall-triggered negotiated moves) on hard and designs, in parallel
# 4. best-per-case across every run so far, official scores, summary -> results/MORNING_SUMMARY.md
LNS=${LNS:-10800}
M3D=${M3D_REPO:-../Partcl-Competition}
LOG=gnn/runs/overnight2.log
ROOTW=$(pwd -W)   # Windows-style path (D:/...) for the Windows Python
mkdir -p gnn/runs
echo "=== $(date +%H:%M) start (LNS=${LNS}s per case)" | tee -a $LOG

until grep -aq "\[destroy\] saved" gnn/runs/destroy_v2.log 2>/dev/null; do sleep 30; done
REF=$(grep -a "\[ref\]" gnn/runs/destroy_v2.log | grep -oE "AUC [0-9.]+" | head -1 | cut -d' ' -f2)
BEST=$(grep -a "\* best" gnn/runs/destroy_v2.log | tail -1 | grep -oE "AUC [0-9.]+" | cut -d' ' -f2)
if python -c "import sys; sys.exit(0 if float('$BEST') >= float('$REF') + 0.01 else 1)"; then
  POLICY="--destroy models/destroy_v2.pt"; PNAME="destroy GNN v2 (AUC $BEST vs regret $REF)"
else
  POLICY=""; PNAME="regret ranking (destroy GNN v2 AUC $BEST did not beat regret $REF)"
fi
echo "=== $(date +%H:%M) LNS seed policy: $PNAME" | tee -a $LOG

python -m engine.run --suite benchmarks_hard --workers 9 --lns-seconds $LNS $POLICY \
  --out-dir results/hard_lns3h > gnn/runs/hard_lns3h.log 2>&1 &
python -m engine.run --suite benchmarks_designs --workers 3 --lns-seconds $LNS $POLICY \
  --out-dir results/designs_lns3h > gnn/runs/designs_lns3h.log 2>&1 &
wait
echo "=== $(date +%H:%M) long runs done" | tee -a $LOG

python -m engine.best_of --suite benchmarks_hard \
  --runs results/hard_engine results/hard_engine_gnn results/hard_destroy_gnn results/hard_lns3h \
  --out results/hard_final >> $LOG 2>&1
python -m engine.best_of --suite benchmarks_designs \
  --runs results/designs_engine_gnn results/designs_lns3h --out results/designs_final >> $LOG 2>&1

score() { (cd "$M3D" && python -m m3d.cli score-suite --suite "$1" --submission-dir "$ROOTW/$2" 2>&1 | grep -a AGGREGATE); }
{
  echo "# Overnight results ($(date '+%Y-%m-%d %H:%M'))"
  echo
  echo "LNS: ${LNS}s per case, annealing + stall-triggered negotiated moves. Seed policy: $PNAME."
  echo
  echo "| Run | Official score |"
  echo "|---|---|"
  echo "| hard, 3 h LNS (this run) | $(score benchmarks_hard results/hard_lns3h) |"
  echo "| **hard, best per case over all runs** (\`results/hard_final\`) | $(score benchmarks_hard results/hard_final) |"
  echo "| designs, 3 h LNS (this run) | $(score benchmarks_designs results/designs_lns3h) |"
  echo "| **designs, best per case** (\`results/designs_final\`) | $(score benchmarks_designs results/designs_final) |"
  echo
  echo "Previous bests: hard 1.2550, designs 1.2749. Leaderboard top: hard 1.4136, designs 1.4673."
  echo
  echo "## Per-case picks"
  echo '```'
  grep -a -E "case_0|ctrl|router|int2float" $LOG | grep -a "from"
  echo '```'
  echo
  echo "## Raw run logs"
  echo '```'
  grep -a -E "ratio|aggregate" gnn/runs/hard_lns3h.log gnn/runs/designs_lns3h.log
  echo '```'
} > results/MORNING_SUMMARY.md
echo "=== $(date +%H:%M) done -> results/MORNING_SUMMARY.md" | tee -a $LOG
