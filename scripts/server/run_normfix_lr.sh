#!/usr/bin/env bash
# Fed-SB-normalized learning-rate check.  Normalization shrinks the input scale of Fed-SB's right basis,
# so the rate swept for unnormalized Fed-SB is not necessarily its rate.  Same rule as the gev0 check:
# sweep fedsb_norm (configs/roberta_normfix_dev_lr_*.yaml, train-holdout, seed 42, 10% budget), pick the
# best last-3 mean per task, and rerun the confirmation protocol for every task whose pick differs from
# Fed-SB's rate, whichever way the result goes.  Then re-summarize with the reruns replacing the old runs.
set -u
ROOT=/data/home/202412012250/bishe_norm
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
LOG=$ROOT/logs/normfix_lr_pipeline.log
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
echo "[$(date '+%F %T')] START normfix_dev" >> "$LOG"
pids=()
for tag in 3e-4 1e-3 3e-3 1e-2 3e-2; do
  "$PY" scripts/run_parallel.py --config "$ROOT/configs/roberta_normfix_dev_lr_${tag}.yaml" --workers 1 \
    --threads-per-run 1 >> "$ROOT/logs/run_normfix_dev_lr_${tag}.log" 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
echo "[$(date '+%F %T')] DONE normfix_dev rc=$rc" >> "$LOG"
"$PY" scripts/make_baseline_configs.py normfixretune "$ROOT/docs/results" >> "$ROOT/logs/run_normfix_retune.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] RETUNE_CONFIG rc=$rc" >> "$LOG"
if [ $rc -eq 0 ] && [ -f "$ROOT/configs/roberta_normfix_retuned.yaml" ] && ! grep -q UNCHANGED "$ROOT/logs/run_normfix_retune.log"; then
  "$PY" scripts/run_parallel.py --config "$ROOT/configs/roberta_normfix_retuned.yaml" --workers 6 --threads-per-run 1 \
    >> "$ROOT/logs/run_normfix_retuned.log" 2>&1
  echo "[$(date '+%F %T')] DONE normfix_retuned rc=$?" >> "$LOG"
  "$PY" scripts/build_dcef_report.py --config "$ROOT/configs/roberta_normfix_retuned.yaml" >> "$ROOT/logs/run_normfix_retuned.log" 2>&1
  "$PY" scripts/summarize_normfix.py --out "$ROOT/docs/results/normfix_summary_retuned" \
    --override "$ROOT/docs/results/normfix_retuned" \
    /data/home/202412012250/bishe_confirm/docs/results/confirm_main \
    /data/home/202412012250/bishe_base/docs/results/baselines_main \
    "$ROOT/docs/results/normfix_main" >> "$ROOT/logs/run_normfix_retuned.log" 2>&1
  echo "[$(date '+%F %T')] SUMMARY rc=$?" >> "$LOG"
fi
echo "[$(date '+%F %T')] NORMFIX_LR_DONE" >> "$LOG"
