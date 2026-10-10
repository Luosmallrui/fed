#!/usr/bin/env bash
# QNLI development stage (configs/roberta_qnli_dev.yaml, 19 runs, seed 42), two workers
# alongside the GEV dev stage; builds the campaign report when done.
set -u
cd /data/home/202412012250/bishe_qnli
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src
echo "[$(date '+%F %T')] START qnli_dev" >> logs/qnli_pipeline.log
"$PY" scripts/run_parallel.py --config configs/roberta_qnli_dev.yaml --workers 2 --threads-per-run 2 \
  >> logs/run_qnli_dev.log 2>&1
echo "[$(date '+%F %T')] DONE qnli_dev rc=$?" >> logs/qnli_pipeline.log
"$PY" scripts/build_dcef_report.py --help >/dev/null 2>&1 && \
  "$PY" scripts/build_dcef_report.py --config configs/roberta_qnli_dev.yaml >> logs/run_qnli_dev.log 2>&1
echo "[$(date '+%F %T')] QNLI_DEV_REPORT rc=$?" >> logs/qnli_pipeline.log
