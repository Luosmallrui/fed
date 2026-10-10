#!/usr/bin/env bash
# QNLI LR boundary checks (seed 42): random_a_ef at 1e-2 and fedavg_lora at 3e-3.
set -u
cd /data/home/202412012250/bishe_qnli
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
( echo "[$(date '+%F %T')] START random_a_ef lr=1e-2 0.10" >> logs/qnli_lr_c.log
  "$PY" scripts/run_experiment.py --config configs/roberta_qnli_lr_1e-2.yaml --task qnli --method random_a_ef --seed 42 --wire-ratio 0.10 >> logs/qnli_lr_c.log 2>&1
  echo "[$(date '+%F %T')] DONE random_a_ef lr=1e-2 rc=$?" >> logs/qnli_lr_c.log ) &
( echo "[$(date '+%F %T')] START fedavg_lora lr=3e-3 dense" >> logs/qnli_lr_d.log
  "$PY" scripts/run_experiment.py --config configs/roberta_qnli_lr_3e-3.yaml --task qnli --method fedavg_lora --seed 42 >> logs/qnli_lr_d.log 2>&1
  echo "[$(date '+%F %T')] DONE fedavg_lora lr=3e-3 rc=$?" >> logs/qnli_lr_d.log ) &
wait
echo "[$(date '+%F %T')] QNLI_LR_BOUNDARY_DONE" >> logs/qnli_pipeline.log
