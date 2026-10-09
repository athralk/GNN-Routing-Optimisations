# Overnight pipeline: features -> train GNN v3 on engine labels -> long LNS runs -> official scores.
# Run from this repo's root in PowerShell:   .\overnight.ps1
# (If scripts are blocked: powershell -ExecutionPolicy Bypass -File .\overnight.ps1)
# Total: roughly 3 hours on a 16-core laptop with an RTX 4060. Everything logs to gnn/runs/overnight.log.

$LNS = 1200          # LNS seconds per case (20 min). Raise for better scores, lower to finish sooner.
$log = "gnn/runs/overnight.log"
New-Item -ItemType Directory -Force gnn/runs | Out-Null
function Step($title, $cmd) {
    "`n===== $(Get-Date -Format 'HH:mm:ss')  $title =====" | Out-File -Append -Encoding utf8 $log
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  $title"
    & powershell -NoProfile -Command "$cmd 2>&1 | Out-File -Append -Encoding utf8 $log"
}

# 1. features for the engine-labelled data (~3 min)
Step "features train"     "python -m gnn.features --dataset engine --split train"
Step "features val"       "python -m gnn.features --dataset engine --split val"
Step "features val_large" "python -m gnn.features --dataset engine --split val_large"

# 2. train the congestion GNN on the better labels (~15-25 min)
Step "train v3_engine"    "python -m gnn.train --dataset engine --epochs 80 --run v3_engine"
Copy-Item gnn/checkpoints/v3_engine/best.pt models/v3_engine.pt -Force

# 3. long LNS runs (each case runs in parallel; ~35 min per tier)
Step "hard, engine only (control)" "python -m engine.run --suite benchmarks_hard --workers 9 --lns-seconds $LNS --out-dir results/hard_engine"
Step "hard, engine + GNN prior"    "python -m engine.run --suite benchmarks_hard --workers 9 --lns-seconds $LNS --ckpt models/v3_engine.pt --out-dir results/hard_engine_gnn"
Step "designs, engine + GNN prior" "python -m engine.run --suite benchmarks_designs --workers 3 --lns-seconds $LNS --ckpt models/v3_engine.pt --out-dir results/designs_engine_gnn"
Step "congested, engine + GNN prior" "python -m engine.run --suite benchmarks_congested --workers 4 --lns-seconds $LNS --ckpt models/v3_engine.pt --out-dir results/congested_engine_gnn"

# 4. official scores (the challenge repo's own scorer)
$here = (Get-Location).Path
$m3d = if ($env:M3D_REPO) { $env:M3D_REPO } else { Join-Path (Split-Path $here -Parent) "Partcl-Competition" }
Push-Location $m3d
foreach ($pair in @("benchmarks_hard:hard_engine", "benchmarks_hard:hard_engine_gnn",
                    "benchmarks_designs:designs_engine_gnn", "benchmarks_congested:congested_engine_gnn")) {
    $suite, $dir = $pair.Split(":")
    "`n===== official score: $dir =====" | Out-File -Append -Encoding utf8 (Join-Path $here $log)
    python -m m3d.cli score-suite --suite $suite --submission-dir (Join-Path $here "results/$dir") 2>&1 |
        Select-String "AGGREGATE" | Out-File -Append -Encoding utf8 (Join-Path $here $log)
}
Pop-Location
Write-Host "done - see $log (search for 'official score')"
