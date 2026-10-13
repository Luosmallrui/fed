#!/usr/bin/env bash
# Published federated LoRA baselines on the RoBERTa confirmation protocol:
#   1. learning-rate sweep (configs/roberta_baselines_dev_lr_*.yaml, train-holdout, seed 42),
#   2. pick rates by best last-3 mean and write configs/roberta_baselines_main.yaml,
#   3. confirmation runs (official validation, seeds 1001-1003, 144 runs),
#   4. report and the pooled comparison with confirm_main (gev0 / FedIT / FFA-LoRA).
set -u
ROOT=/data/home/202412012250/bishe_base
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
LOG=$ROOT/logs/baselines_pipeline.log
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKERS=${WORKERS:-5}
cd "$ROOT"

for tag in 3e-4 1e-3 3e-3 1e-2; do
  name=roberta_baselines_dev_lr_$tag
  echo "[$(date '+%F %T')] START $name" >> "$LOG"
  cd "$ROOT"
  "$PY" scripts/run_parallel.py --config "$ROOT/configs/$name.yaml" --workers "$WORKERS" --threads-per-run 1 \
    >> "$ROOT/logs/run_$name.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE $name rc=$rc" >> "$LOG"
done

cd "$ROOT"
"$PY" scripts/make_baseline_configs.py main "$ROOT/docs/results" >> "$LOG" 2>&1
rc=$?
echo "[$(date '+%F %T')] MAKE_MAIN rc=$rc" >> "$LOG"
[ $rc -ne 0 ] && { echo "[$(date '+%F %T')] ABORT" >> "$LOG"; exit 1; }

echo "[$(date '+%F %T')] START main" >> "$LOG"
cd "$ROOT"
"$PY" scripts/run_parallel.py --config "$ROOT/configs/roberta_baselines_main.yaml" --workers "$WORKERS" --threads-per-run 1 \
  >> "$ROOT/logs/run_baselines_main.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE main rc=$rc" >> "$LOG"

cd "$ROOT"
"$PY" scripts/build_dcef_report.py --config "$ROOT/configs/roberta_baselines_main.yaml" >> "$ROOT/logs/run_baselines_main.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] REPORT rc=$rc" >> "$LOG"
"$PY" scripts/summarize_baselines.py --out "$ROOT/docs/results/baselines_summary" \
  /data/home/202412012250/bishe_confirm/docs/results/confirm_main "$ROOT/docs/results/baselines_main" \
  >> "$ROOT/logs/run_baselines_main.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] SUMMARY rc=$rc" >> "$LOG"
echo "[$(date '+%F %T')] BASELINES_DONE" >> "$LOG"
