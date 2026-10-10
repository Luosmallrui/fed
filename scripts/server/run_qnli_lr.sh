#!/usr/bin/env bash
# QNLI learning-rate sweep (configs/roberta_qnli_lr_*.yaml), seed 42, two lanes.
set -u
cd /data/home/202412012250/bishe_qnli
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run() {  # lane lr method [wire]
  local lane=$1 lr=$2 method=$3 wire=${4:-}
  local extra=()
  [ -n "$wire" ] && extra=(--wire-ratio "$wire")
  echo "[$(date '+%F %T')] START $method lr=$lr ${wire:-dense}" >> "logs/qnli_lr_${lane}.log"
  "$PY" scripts/run_experiment.py --config "configs/roberta_qnli_lr_${lr}.yaml" --task qnli \
    --method "$method" --seed 42 "${extra[@]}" >> "logs/qnli_lr_${lane}.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE $method lr=$lr ${wire:-dense} rc=$rc" >> "logs/qnli_lr_${lane}.log"
}

( run a 1e-3 random_a_ef 0.10; run a 3e-4 random_a_ef 0.10; run a 3e-3 random_a_ef 0.10 ) &
( run b 1e-3 fedavg_lora; run b 3e-4 fedavg_lora ) &
wait
echo "[$(date '+%F %T')] QNLI_LR_DONE" >> logs/qnli_pipeline.log
