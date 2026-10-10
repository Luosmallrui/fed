#!/usr/bin/env bash
# Retry of the four supplementary runs that hit CUDA OOM on 2026-10-10 (a dry-run basis
# check shared the GPU).  Starts after the main comparison frees its slot and 15 minutes
# after lane f, with expandable CUDA segments to limit fragmentation.
set -u
cd /data/home/202412012250/bishe_llm2
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q QWEN_MAIN_DONE logs/qwen_main.log 2>/dev/null; do sleep 120; done
sleep 900
run() {  # config method seed wire
  echo "[$(date "+%F %T")] START $1 $2 $3 $4" >> logs/qwen_supp_retry.log
  "$PY" scripts/run_experiment.py --config "configs/$1.yaml" --task mnli --method "$2" --seed "$3" --wire-ratio "$4" >> logs/qwen_supp_retry.log 2>&1
  status=$(ls docs/results/qwen05b_supp/mnli__$2__seed-$3__wire-$(printf %02d $(python3 -c "print(round($4*100))"))/summary.json >/dev/null 2>&1 && echo ok || echo FAILED)
  echo "[$(date "+%F %T")] DONE $1 $2 $3 $4 $status" >> logs/qwen_supp_retry.log
}
run qwen05b_supp gev0_a_ef 361 0.05
run qwen05b_supp gev0_a_ef 362 0.10
run qwen05b_supp gev0_a_ef 363 0.10
run qwen05b_supp cab0wh_a_ef 361 0.05
echo "[$(date "+%F %T")] QWEN_SUPP_RETRY_DONE" >> logs/qwen_supp_retry.log
