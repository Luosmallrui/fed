#!/usr/bin/env bash
# Normalization as a general fix (configs/roberta_normfix_main.yaml, 81 runs): EVA, EVA + normalization,
# Fed-SB + normalization on the confirmation protocol, then the pooled comparison with confirm_main
# (random / grad-energy / normalized grad-energy / ours) and baselines_main (Fed-SB).
set -u
ROOT=/data/home/202412012250/bishe_norm
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
LOG=$ROOT/logs/normfix_pipeline.log
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WORKERS=${WORKERS:-6}
cd "$ROOT"
echo "[$(date '+%F %T')] START normfix_main" >> "$LOG"
"$PY" scripts/run_parallel.py --config "$ROOT/configs/roberta_normfix_main.yaml" --workers "$WORKERS" --threads-per-run 1 \
  >> "$ROOT/logs/run_normfix_main.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE normfix_main rc=$rc" >> "$LOG"
cd "$ROOT"
"$PY" scripts/build_dcef_report.py --config "$ROOT/configs/roberta_normfix_main.yaml" >> "$ROOT/logs/run_normfix_main.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] REPORT rc=$rc" >> "$LOG"
"$PY" scripts/summarize_normfix.py --out "$ROOT/docs/results/normfix_summary" \
  /data/home/202412012250/bishe_confirm/docs/results/confirm_main \
  /data/home/202412012250/bishe_base/docs/results/baselines_main \
  "$ROOT/docs/results/normfix_main" >> "$ROOT/logs/run_normfix_main.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] SUMMARY rc=$rc" >> "$LOG"
echo "[$(date '+%F %T')] NORMFIX_DONE" >> "$LOG"
