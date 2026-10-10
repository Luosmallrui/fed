#!/usr/bin/env bash
# Setup-cost variants L1-L3 of gev0 on Qwen2.5-0.5B (configs/qwen05b_cost_L*.yaml); starts
# once the supplementary runs and their OOM retry have finished, three lanes.
set -u
cd /data/home/202412012250/bishe_llm3
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L2DIR=/data/home/202412012250/bishe_llm2/logs
until grep -q QWEN_SUPP_DONE $L2DIR/qwen_supp.log 2>/dev/null && grep -q QWEN_SUPP_RETRY_DONE $L2DIR/qwen_supp_retry.log 2>/dev/null; do sleep 300; done
run() {
  echo "[$(date '+%F %T')] START $1" >> logs/qwen_cost.log
  "$PY" scripts/run_experiment.py --config "configs/qwen05b_cost_$1.yaml" --task mnli --method gev0_a_ef --seed 361 --wire-ratio 0.10 >> "logs/qwen_cost_$1.log" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] DONE $1 rc=$rc" >> logs/qwen_cost.log
}
run L1 & sleep 600; run L2 & sleep 600; run L3 &
wait
echo "[$(date '+%F %T')] QWEN_COST_DONE" >> logs/qwen_cost.log
