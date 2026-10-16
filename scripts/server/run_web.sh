#!/usr/bin/env bash
# Web-service tasks (banking77, moderation) in code tree bishe_web: learning-rate sweep on {3e-4, 1e-3, 3e-3}
# (train-holdout, seed 42), extension wherever a pick lies on the grid boundary, then the confirmation runs
# (official held-out sets, seeds 1001-1003, 5%/10%) with every method at its swept rate, and the summary.
set -u
ROOT=/data/home/202412012250/bishe_web
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
LOG=$ROOT/logs/web_pipeline.log
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
stamp() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

stamp "START web_dev"
pids=()
for tag in 3e-4 1e-3 3e-3; do
  "$PY" "$ROOT/scripts/run_parallel.py" --config "$ROOT/configs/roberta_web_dev_lr_${tag}.yaml" --workers 2 \
    --threads-per-run 1 >> "$ROOT/logs/run_web_dev_lr_${tag}.log" 2>&1 &
  pids+=($!)
done
rc=0; for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
stamp "DONE web_dev rc=$rc"

"$PY" "$ROOT/scripts/make_web_configs.py" dev2 "$ROOT/docs/results" > "$ROOT/logs/run_web_dev2_config.log" 2>&1
stamp "DEV2_CONFIG rc=$? $(tail -1 "$ROOT/logs/run_web_dev2_config.log")"
if ! grep -q '^NONE' "$ROOT/logs/run_web_dev2_config.log"; then
  pids=()
  for config in "$ROOT"/configs/roberta_web_dev2_lr_*.yaml; do
    "$PY" "$ROOT/scripts/run_parallel.py" --config "$config" --workers 2 --threads-per-run 1 \
      >> "$ROOT/logs/run_web_dev2.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for pid in "${pids[@]}"; do wait "$pid" || rc=$?; done
  stamp "DONE web_dev2 rc=$rc"
fi

"$PY" "$ROOT/scripts/make_web_configs.py" main "$ROOT/docs/results" > "$ROOT/logs/run_web_main_config.log" 2>&1
rc=$?
stamp "MAIN_CONFIG rc=$rc"
[ $rc -eq 0 ] || { stamp "WEB_FAILED"; exit 1; }
"$PY" "$ROOT/scripts/run_parallel.py" --config "$ROOT/configs/roberta_web_main.yaml" --workers 6 --threads-per-run 1 \
  >> "$ROOT/logs/run_web_main.log" 2>&1
stamp "DONE web_main rc=$?"
"$PY" "$ROOT/scripts/summarize_web.py" --out "$ROOT/docs/results/web_summary" "$ROOT/docs/results/web_main" \
  > "$ROOT/logs/run_web_summary.log" 2>&1
stamp "SUMMARY rc=$?"
stamp "WEB_DONE"
