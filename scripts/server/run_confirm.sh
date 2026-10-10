#!/usr/bin/env bash
# Confirmation stage (phase final, official validation, seeds 1001-1003):
# configs/roberta_confirm_main.yaml (162 runs) then roberta_confirm_eva.yaml (9 runs).
set -u
cd /data/home/202412012250/bishe_confirm
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src
for name in confirm_main confirm_eva; do
  echo "[$(date '+%F %T')] START $name" >> logs/confirm_pipeline.log
  "$PY" scripts/run_parallel.py --config "configs/roberta_${name}.yaml" --workers 4 --threads-per-run 1 \
    >> "logs/run_${name}.log" 2>&1
  echo "[$(date '+%F %T')] DONE $name rc=$?" >> logs/confirm_pipeline.log
  "$PY" scripts/build_dcef_report.py --config "configs/roberta_${name}.yaml" >> "logs/run_${name}.log" 2>&1
  echo "[$(date '+%F %T')] REPORT $name rc=$?" >> logs/confirm_pipeline.log
done
echo "[$(date '+%F %T')] CONFIRM_DONE" >> logs/confirm_pipeline.log
