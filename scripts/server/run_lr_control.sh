#!/usr/bin/env bash
# Learning-rate control: random_a_ef with 2x / 3x adapter LR, MNLI seed 361, wire 10%.
set -u
cd /data/home/202412012250/bishe_gevx
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for mult in 2x 3x; do
  echo "[$(date '+%F %T')] START random_a_ef lr $mult" >> logs/lr_control.log
  "$PY" scripts/run_experiment.py --config "configs/roberta_lr_control_${mult}.yaml" --task mnli \
    --method random_a_ef --seed 361 --wire-ratio 0.10 >> logs/lr_control.log 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE random_a_ef lr $mult rc=$rc" >> logs/lr_control.log
done
