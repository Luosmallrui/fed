#!/usr/bin/env bash
# Qwen2.5-0.5B alpha-0.1 supplement (configs/qwen05b_a01.yaml): starts after the Qwen confirmation
# pipeline finishes; gev0 on seeds 361-363 and random on 362/363, three staggered lanes.
set -u
H=/data/home/202412012250
TREE=$H/bishe_llm3
PY=$H/bishe_dcef/.venv/bin/python
LOG=$TREE/logs/qwen_a01.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q QWEN_HEAD_CONFIRM_DONE "$TREE/logs/qwen_head_confirm.log" 2>/dev/null; do sleep 300; done
run() {  # method seed
  cd "$TREE" || return
  echo "[$(date '+%F %T')] START $1 $2" >> "$LOG"
  PYTHONPATH="$TREE/src" "$PY" "$TREE/scripts/run_experiment.py" --config "$TREE/configs/qwen05b_a01.yaml" \
    --task mnli --method "$1" --seed "$2" --wire-ratio 0.10 >> "$TREE/logs/qwen05b_a01_$1_$2.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE $1 $2 rc=$rc" >> "$LOG"
}
( run gev0_a_ef 361; run random_a_ef 362 ) &
sleep 600
( run gev0_a_ef 362; run random_a_ef 363 ) &
sleep 600
( run gev0_a_ef 363 ) &
wait
echo "[$(date '+%F %T')] QWEN_A01_DONE" >> "$LOG"
