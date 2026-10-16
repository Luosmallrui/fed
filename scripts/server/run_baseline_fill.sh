#!/usr/bin/env bash
# Fill the baseline learning-rate sweep to the full 5-point grid {1e-4 .. 1e-2} that gev0 was swept over
# (configs/roberta_baselines_dev3_lr_*.yaml, 19 runs, train-holdout, seed 42), then report whether any
# selected rate changes (logs/baseline_lr_choice_full.json).  If one does, its confirmation runs are redone.
# Waits for the Fed-SB-normalized reruns, which share the GPU.
set -u
ROOT=/data/home/202412012250/bishe_base
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
LOG=$ROOT/logs/baseline_fill_pipeline.log
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
echo "[$(date '+%F %T')] WAIT for NORMFIX_LR2_DONE" >> "$LOG"
until grep -q NORMFIX_LR2_DONE /data/home/202412012250/bishe_norm/logs/normfix_lr_pipeline.log 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] START baseline_fill" >> "$LOG"
pids=()
for spec in "1e-4 3" "1e-2 2" "1e-2_mq 2"; do
  set -- $spec
  "$PY" "$ROOT/scripts/run_parallel.py" --config "$ROOT/configs/roberta_baselines_dev3_lr_$1.yaml" --workers "$2" \
    --threads-per-run 1 >> "$ROOT/logs/run_baselines_dev3_lr_$1.log" 2>&1 &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
echo "[$(date '+%F %T')] DONE baseline_fill rc=$rc" >> "$LOG"
"$PY" "$ROOT/scripts/make_baseline_configs.py" sweepcheck "$ROOT/docs/results" >> "$ROOT/logs/run_baseline_sweepcheck.log" 2>&1
echo "[$(date '+%F %T')] SWEEPCHECK rc=$? $(tail -1 "$ROOT/logs/run_baseline_sweepcheck.log")" >> "$LOG"
echo "[$(date '+%F %T')] BASELINE_FILL_DONE" >> "$LOG"
