#!/usr/bin/env bash
# GEV development stage (configs/roberta_gev_dev.yaml, 108 runs).  Starts only after
# the CAB-EF campaign (tmux session cabef) has finished, so the GPU is not oversubscribed.
set -u
cd /data/home/202412012250/bishe_gevdev
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src
while tmux has-session -t cabef 2>/dev/null; do sleep 300; done
echo "[$(date '+%F %T')] START gev_dev" >> logs/gev_dev_pipeline.log
"$PY" scripts/run_parallel.py --config configs/roberta_gev_dev.yaml --workers 3 --threads-per-run 2 \
  >> logs/run_gev_dev.log 2>&1
rc=$?
echo "[$(date '+%F %T')] DONE gev_dev rc=$rc" >> logs/gev_dev_pipeline.log
