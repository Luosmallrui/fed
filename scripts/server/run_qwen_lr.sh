#!/usr/bin/env bash
# Qwen2.5-0.5B learning-rate sweep (configs/qwen05b_lr_*.yaml), MNLI seed 361, three lanes
# (each run peaks at about 21 GB).
set -u
cd /data/home/202412012250/bishe_llm
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() {  # lane lr method [wire]
  local lane=$1 lr=$2 method=$3 wire=${4:-}
  local extra=()
  [ -n "$wire" ] && extra=(--wire-ratio "$wire")
  echo "[$(date '+%F %T')] START $method lr=$lr ${wire:-dense}" >> "logs/qwen_lr_${lane}.log"
  "$PY" scripts/run_experiment.py --config "configs/qwen05b_lr_${lr}.yaml" --task mnli \
    --method "$method" --seed 361 "${extra[@]}" >> "logs/qwen_lr_${lane}.log" 2>&1
  echo "[$(date '+%F %T')] DONE $method lr=$lr ${wire:-dense} rc=$?" >> "logs/qwen_lr_${lane}.log"
}
( run a 3e-4 random_a_ef 0.10; run a 1e-4 fedavg_lora ) &
( run b 1e-3 random_a_ef 0.10; run b 3e-4 fedavg_lora ) &
( run c 1e-4 random_a_ef 0.10 ) &
wait
echo "[$(date '+%F %T')] QWEN_LR_DONE" >> logs/qwen_pipeline.log
