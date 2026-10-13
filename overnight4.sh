#!/usr/bin/env bash
# Phase 3 (destroy GNN v4): gain-size labels from late search states, listwise ranking
# loss, small sweep + ensemble of 3, then an equal-time A/B vs v2.
# Hard time limits so everything is done before 06:00:
#   data: until 04:05 (killed if still running) | train: fixed epochs | A/B: 4 x 10 min
# Run from this repo's root (Git Bash):  bash overnight4.sh
DATA_STOP=${DATA_STOP:-0405}
LOG=gnn/runs/overnight4.log
mkdir -p gnn/runs results
say() { echo "=== $(date +%H:%M) $*" | tee -a $LOG; }
say "start (data hard stop at $DATA_STOP)"

python -m gnn.destroy data --version 4 --n 200 --tries 12 --waits 30 180 420 --workers 14 \
  --data-dir gnn/data/destroy_v4 > gnn/runs/destroy_v4_data.log 2>&1 &
DPID=$!
while kill -0 $DPID 2>/dev/null; do
  if [ $((10#$(date +%H%M))) -ge $((10#$DATA_STOP)) ] && [ $((10#$(date +%H%M))) -lt 1200 ]; then
    powershell -NoProfile -Command "\$m = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -match 'gnn\.destroy data' }; \$ids = @(\$m.ProcessId); Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$ids -contains \$_.ParentProcessId } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }; foreach (\$i in \$ids) { Stop-Process -Id \$i -Force -ErrorAction SilentlyContinue }"
    say "data hard stop reached"
    break
  fi
  sleep 30
done
wait $DPID 2>/dev/null
say "data: $(ls gnn/data/destroy_v4/*.pt 2>/dev/null | wc -l) instance files"

python -m gnn.destroy train --version 4 --epochs 40 --data-dir gnn/data/destroy_v4 \
  --out models/destroy_v4.pt > gnn/runs/destroy_v4_train.log 2>&1
say "trained"; grep -a -E "\[ref\]|\[sweep\]|\[ensemble\]|saved" gnn/runs/destroy_v4_train.log | tee -a $LOG

python -m gnn.ab --suite benchmarks_hard --seconds 600 --seeds 1 2 --warm results/hard_final2 \
  --config v2="--destroy models/destroy_v2.pt --anneal 0.05" \
  --config v4="--destroy models/destroy_v4.pt --anneal 0.05" \
  --out results/ab4 --report results/ab4/AB_REPORT.md > gnn/runs/ab4.log 2>&1
say "A/B done"; cat results/ab4/AB_REPORT.md | tee -a $LOG

{
  echo "# Phase 3 results ($(date '+%Y-%m-%d %H:%M'))"
  echo
  echo "Destroy GNN v4: gain-size labels from late search states (30/210/630 s into LNS),"
  echo "listwise ranking loss, sweep of 3 configs, ensemble of 3."
  echo
  echo "## Training (lift = expected gain of the policy's choices / random choices; 1.0 = random)"
  echo '```'; grep -a -E "\[destroy4\]|\[ref\]|\[sweep\]|\[ensemble\]" gnn/runs/destroy_v4_train.log; echo '```'
  echo
  echo "## A/B on hard (warm-started from results/hard_final2 = 1.3125, 10 min per case, 2 seeds)"
  cat results/ab4/AB_REPORT.md
} > results/PHASE3_SUMMARY.md
say "done -> results/PHASE3_SUMMARY.md"
