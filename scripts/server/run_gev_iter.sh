#!/usr/bin/env bash
# Setup-cost iteration (gev0 variants v1-v3, MNLI seed 361, 5/10%) and baseline checks
# (EVA at 10%, random_a_dense, gev0_a_dense), four lanes on the otherwise idle GPU.
set -u
cd /data/home/202412012250/bishe_gev2
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

run() {  # lane config method [wire]
  local lane=$1 config=$2 method=$3 wire=${4:-}
  local extra=()
  [ -n "$wire" ] && extra=(--wire-ratio "$wire")
  echo "[$(date '+%F %T')] START $config $method ${wire:-dense}" >> "logs/iter_${lane}.log"
  "$PY" scripts/run_experiment.py --config "configs/roberta_${config}.yaml" --task mnli \
    --method "$method" --seed 361 "${extra[@]}" >> "logs/iter_${lane}.log" 2>&1
  echo "[$(date '+%F %T')] DONE $config $method ${wire:-dense} rc=$?" >> "logs/iter_${lane}.log"
}

( run a gev_cost_v1 gev0_a_ef 0.05; run a gev_cost_v1 gev0_a_ef 0.10 ) &
( run b gev_cost_v2 gev0_a_ef 0.05; run b gev_cost_v2 gev0_a_ef 0.10 ) &
( run c gev_cost_v3 gev0_a_ef 0.05; run c gev_cost_v3 gev0_a_ef 0.10 ) &
( run d gev_checks eva_a_ef 0.10; run d gev_checks random_a_dense; run d gev_checks gev0_a_dense ) &
wait
echo "[$(date '+%F %T')] ITER_DONE" >> logs/iter_pipeline.log
