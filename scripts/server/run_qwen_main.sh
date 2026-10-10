#!/usr/bin/env bash
# Qwen2.5-0.5B main comparison: waits for the LR sweep in bishe_llm, picks the rates
# (scripts/make_qwen_main_config.py), then runs cab0 / cab0wh / gev0 (Gram-refined) at 10%
# and gev0 without refinement, three lanes.
set -u
cd /data/home/202412012250/bishe_llm2
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
until grep -q QWEN_LR_DONE /data/home/202412012250/bishe_llm/logs/qwen_pipeline.log 2>/dev/null; do sleep 120; done
"$PY" scripts/make_qwen_main_config.py /data/home/202412012250/bishe_llm/docs/results >> logs/qwen_main.log 2>&1 || exit 1
run() {  # lane config method
  echo "[$(date '+%F %T')] START $2 $3" >> "logs/qwen_main_$1.log"
  "$PY" scripts/run_experiment.py --config "configs/$2.yaml" --task mnli --method "$3" --seed 361 \
    --wire-ratio 0.10 >> "logs/qwen_main_$1.log" 2>&1
  echo "[$(date '+%F %T')] DONE $2 $3 rc=$?" >> "logs/qwen_main_$1.log"
}
( run a qwen05b_main cab0_a_ef ) &
( run b qwen05b_main cab0wh_a_ef ) &
( run c qwen05b_main gev0_a_ef; run c qwen05b_norefine gev0_a_ef ) &
wait
echo "[$(date '+%F %T')] QWEN_MAIN_DONE" >> logs/qwen_main.log
