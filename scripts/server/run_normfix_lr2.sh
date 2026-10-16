#!/usr/bin/env bash
# Second half of the Fed-SB-normalized learning-rate check (after run_normfix_lr.sh's sweep).
# The sweep picked 3e-3 on MNLI (Fed-SB: 1e-3), 1e-3 on SST-2 (unchanged) and the lowest rate, 3e-4, on QNLI.
# Two branches in parallel: MNLI is rerun on the confirmation protocol at its pick; QNLI first extends the
# sweep to 1e-4 and is rerun if its final pick differs from Fed-SB's rate.  Then the reruns replace the old
# fedsb_norm runs in the normfix summary.
set -u
ROOT=/data/home/202412012250/bishe_norm
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
LOG=$ROOT/logs/normfix_lr_pipeline.log
GEN=$ROOT/scripts/make_baseline_configs.py
RUN=$ROOT/scripts/run_parallel.py
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
echo "[$(date '+%F %T')] START normfix_lr2" >> "$LOG"

mnli_branch() {
  "$PY" "$GEN" normfixretune "$ROOT/docs/results" mnli >> "$ROOT/logs/run_normfix_retune_mnli.log" 2>&1
  echo "[$(date '+%F %T')] RETUNE_CONFIG mnli rc=$?" >> "$LOG"
  [ -f "$ROOT/configs/roberta_normfix_retuned_mnli.yaml" ] || return 0
  "$PY" "$RUN" --config "$ROOT/configs/roberta_normfix_retuned_mnli.yaml" --workers 5 --threads-per-run 1 \
    >> "$ROOT/logs/run_normfix_retuned_mnli.log" 2>&1
  echo "[$(date '+%F %T')] DONE normfix_retuned_mnli rc=$?" >> "$LOG"
}

qnli_branch() {
  "$PY" "$RUN" --config "$ROOT/configs/roberta_normfix_dev2_lr_1e-4.yaml" --workers 1 --threads-per-run 1 \
    >> "$ROOT/logs/run_normfix_dev2_lr_1e-4.log" 2>&1
  echo "[$(date '+%F %T')] DONE normfix_dev2 rc=$?" >> "$LOG"
  "$PY" "$GEN" normfixretune "$ROOT/docs/results" qnli >> "$ROOT/logs/run_normfix_retune_qnli.log" 2>&1
  echo "[$(date '+%F %T')] RETUNE_CONFIG qnli rc=$?" >> "$LOG"
  [ -f "$ROOT/configs/roberta_normfix_retuned_qnli.yaml" ] || return 0
  "$PY" "$RUN" --config "$ROOT/configs/roberta_normfix_retuned_qnli.yaml" --workers 4 --threads-per-run 1 \
    >> "$ROOT/logs/run_normfix_retuned_qnli.log" 2>&1
  echo "[$(date '+%F %T')] DONE normfix_retuned_qnli rc=$?" >> "$LOG"
}

mnli_branch &
a=$!
qnli_branch &
b=$!
wait $a $b

overrides=()
for task in mnli qnli; do
  config=$ROOT/configs/roberta_normfix_retuned_${task}.yaml
  [ -f "$config" ] || continue
  "$PY" "$ROOT/scripts/build_dcef_report.py" --config "$config" >> "$ROOT/logs/run_normfix_retuned_${task}.log" 2>&1
  overrides+=(--override "$ROOT/docs/results/normfix_retuned_${task}")
done
"$PY" "$ROOT/scripts/summarize_normfix.py" --out "$ROOT/docs/results/normfix_summary_retuned" "${overrides[@]}" \
  /data/home/202412012250/bishe_confirm/docs/results/confirm_main \
  /data/home/202412012250/bishe_base/docs/results/baselines_main \
  "$ROOT/docs/results/normfix_main" >> "$ROOT/logs/run_normfix_summary_retuned.log" 2>&1
echo "[$(date '+%F %T')] SUMMARY rc=$?" >> "$LOG"
echo "[$(date '+%F %T')] NORMFIX_LR2_DONE" >> "$LOG"
