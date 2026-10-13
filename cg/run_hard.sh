#!/usr/bin/env bash
# Column generation on every hard case, in parallel, one combined live log.
# Run from this repo's root (Git Bash):  bash cg/run_hard.sh
ROUNDS=${ROUNDS:-50}
ILP=${ILP:-600}
MC=${MC:-180}
OUT=results/cg_hard
LOG=results/logs/cg_hard_live.log
mkdir -p $OUT results/logs
echo "=== $(date +%H:%M) starting cases 01-09 ($ROUNDS rounds, MC ${MC}s, ILP ${ILP}s each)" >> $LOG
for c in 01 02 03 04 05 06 07 08 09; do
  ( python -u -m cg.colgen --case case_$c --rounds $ROUNDS --ilp-seconds $ILP --mc-seconds $MC --out $OUT 2>&1 \
      | sed -u "s/^/[case_$c] /" >> $LOG ) &
done
wait
echo "=== $(date +%H:%M) all cases done; best of (previous best, column generation):" >> $LOG
python -m engine.best_of --suite benchmarks_hard --runs results/hard_final3 $OUT \
  --out results/hard_final4 2>&1 | tee -a $LOG | python -c "
import sys, math
r = [float(l.split()[1]) for l in sys.stdin if ': ' in l and 'from' in l]
print(f'=== aggregate (geomean of {len(r)} cases): {math.exp(sum(map(math.log, r)) / len(r)):.4f}  (was 1.3138)')
" >> $LOG
echo "=== $(date +%H:%M) ALL DONE -> results/hard_final4" >> $LOG
