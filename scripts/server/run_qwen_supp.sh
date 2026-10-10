#!/usr/bin/env bash
# Qwen2.5-0.5B supplementary runs (configs/qwen05b_supp.yaml + LR boundary check); lanes d/e
# start at once, lane f after the main comparison frees its GPU slot.
set -u
cd /data/home/202412012250/bishe_llm2
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() {  # lane config method seed [wire]
  local extra=()
  [ -n "${5:-}" ] && extra=(--wire-ratio "$5")
  echo "[$(date "+%F %T")] START $2 $3 $4 ${5:-dense}" >> "logs/qwen_supp_$1.log"
  "$PY" scripts/run_experiment.py --config "configs/$2.yaml" --task mnli --method "$3" --seed "$4" "${extra[@]}" >> "logs/qwen_supp_$1.log" 2>&1
  echo "[$(date "+%F %T")] DONE $2 $3 $4 ${5:-dense} rc=$?" >> "logs/qwen_supp_$1.log"
}
( run d qwen05b_supp gev0_a_ef 361 0.05; run d qwen05b_supp gev0_a_ef 362 0.10; run d qwen05b_supp gev0_a_ef 363 0.10; run d qwen05b_supp cab0wh_a_ef 361 0.05 ) &
( run e qwen05b_supp random_a_ef 361 0.05; run e qwen05b_supp random_a_ef 362 0.10; run e qwen05b_supp random_a_ef 363 0.10; run e qwen05b_lr_3e-3 random_a_ef 361 0.10 ) &
( until grep -q QWEN_MAIN_DONE logs/qwen_main.log 2>/dev/null; do sleep 120; done
  run f qwen05b_supp fedavg_lora 362; run f qwen05b_supp fedavg_lora 363 ) &
wait
echo "[$(date "+%F %T")] QWEN_SUPP_DONE" >> logs/qwen_supp.log
