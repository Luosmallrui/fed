#!/usr/bin/env bash
# Symmetric learning-rate check for gev0 (it had inherited FFA-LoRA's rates; every baseline was swept):
#   1. gev0_a_dense sweep over {1e-4 .. 1e-2} (configs/roberta_gev0_dev_lr_*.yaml), plus one
#      confirm_main gev0 run re-executed in this code tree (must reproduce confirm_main exactly),
#   2. pick rates with the baselines' rule; rerun gev0 on the confirmation protocol for every task
#      whose rate changed (configs/roberta_gev0_retuned.yaml, written only if needed),
#   3. pooled summary with the reruns replacing the original gev0 runs of those tasks.
set -u
ROOT=/data/home/202412012250/bishe_base
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
LOG=$ROOT/logs/gev0_lr_pipeline.log
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

pids=()
for tag in 1e-4 3e-4 1e-3 3e-3 1e-2; do
  name=roberta_gev0_dev_lr_$tag
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  ( cd "$ROOT" && "$PY" scripts/run_parallel.py --config "$ROOT/configs/$name.yaml" --workers 1 --threads-per-run 1 \
      >> "$ROOT/logs/run_$name.log" 2>&1; rc=$?; echo "[$(date '+%F %T')] DONE $name rc=$rc" >> "$LOG" ) &
  pids+=($!)
done
echo "[$(date '+%F %T')] START repro" >> "$LOG"
( cd "$ROOT" && "$PY" scripts/run_parallel.py --config "$ROOT/configs/roberta_gev0_repro.yaml" --workers 1 --threads-per-run 1 \
    --tasks sst2 --methods gev0_a_ef --seeds 1001 >> "$ROOT/logs/run_roberta_gev0_repro.log" 2>&1; rc=$?
  echo "[$(date '+%F %T')] DONE repro rc=$rc" >> "$LOG" ) &
pids+=($!)
wait "${pids[@]}"

cd "$ROOT"
"$PY" - >> "$LOG" 2>&1 <<'EOF'
import json
run = "sst2__gev0_a_ef__seed-1001__wire-05"
a = json.load(open(f"/data/home/202412012250/bishe_confirm/docs/results/confirm_main/{run}/summary.json"))
b = json.load(open(f"/data/home/202412012250/bishe_base/docs/results/gev0_repro/{run}/summary.json"))
ma = [r["primary_metric"] for r in a["rounds"]]
mb = [r["primary_metric"] for r in b["rounds"]]
same_bytes = a["final"]["wire"]["total_delivered_bytes"] == b["final"]["wire"]["total_delivered_bytes"]
diff = max(abs(x - y) for x, y in zip(ma, mb))
print(f"REPRO {run}: max metric diff {diff:.2e}, bytes equal {same_bytes}, "
      f"basis hash equal {a['invariants']['before']['fixed_basis_hash'] == b['invariants']['before']['fixed_basis_hash']}")
EOF

"$PY" scripts/make_baseline_configs.py gev0main "$ROOT/docs/results" >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] GEV0_MAIN rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && { echo "[$(date '+%F %T')] ABORT" >> "$LOG"; exit 1; }

if [ -f "$ROOT/configs/roberta_gev0_retuned.yaml" ]; then
  echo "[$(date '+%F %T')] START retuned" >> "$LOG"
  cd "$ROOT"
  "$PY" scripts/run_parallel.py --config "$ROOT/configs/roberta_gev0_retuned.yaml" --workers 5 --threads-per-run 1 \
    >> "$ROOT/logs/run_roberta_gev0_retuned.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE retuned rc=$rc" >> "$LOG"
  cd "$ROOT"
  "$PY" scripts/summarize_baselines.py --out "$ROOT/docs/results/baselines_summary_retuned" \
    --override "$ROOT/docs/results/gev0_retuned" \
    /data/home/202412012250/bishe_confirm/docs/results/confirm_main "$ROOT/docs/results/baselines_main" \
    >> "$ROOT/logs/run_roberta_gev0_retuned.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] SUMMARY rc=$rc" >> "$LOG"
fi
echo "[$(date '+%F %T')] GEV0_LR_DONE" >> "$LOG"
