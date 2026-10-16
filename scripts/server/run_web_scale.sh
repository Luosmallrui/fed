#!/usr/bin/env bash
# Client-count check: Banking77 with 50 clients (full participation) at the 10% budget, rates from the
# ten-client sweep (configs/roberta_web_scale_main.yaml, 24 runs), then the summary.
set -u
ROOT=/data/home/202412012250/bishe_web
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
LOG=$ROOT/logs/web_scale_pipeline.log
export PYTHONPATH=$ROOT/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
echo "[$(date '+%F %T')] START web_scale" >> "$LOG"
"$PY" "$ROOT/scripts/run_parallel.py" --config "$ROOT/configs/roberta_web_scale_main.yaml" --workers 6 --threads-per-run 1 \
  >> "$ROOT/logs/run_web_scale_main.log" 2>&1
echo "[$(date '+%F %T')] DONE web_scale rc=$?" >> "$LOG"
"$PY" "$ROOT/scripts/summarize_web.py" --out "$ROOT/docs/results/web_scale_summary" "$ROOT/docs/results/web_scale_main" \
  > "$ROOT/logs/run_web_scale_summary.log" 2>&1
echo "[$(date '+%F %T')] SUMMARY rc=$?" >> "$LOG"
echo "[$(date '+%F %T')] WEB_SCALE_DONE" >> "$LOG"
