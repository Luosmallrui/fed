#!/usr/bin/env bash
# Qwen2.5-0.5B rank-2 test (configs/qwen05b_r2.yaml).  Takes the GPU slot of the deferred
# random_a_ef lr 3e-3 check in run_qwen_resume.sh: random starts when that lane's slot is
# released, gev0 when the next resume run finishes, so at most three runs share the GPU.
set -u
H=/data/home/202412012250
TREE=$H/bishe_llm3
PY=$H/bishe_dcef/.venv/bin/python
RLOG=$H/bishe_llm2/logs/qwen_resume.log
LOG=$TREE/logs/qwen_r2.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {
  cd "$TREE" || return
  echo "[$(date '+%F %T')] START $1" >> "$LOG"
  PYTHONPATH="$TREE/src" "$PY" "$TREE/scripts/run_experiment.py" --config "$TREE/configs/qwen05b_r2.yaml" \
    --task mnli --method "$1" --seed 361 --wire-ratio 0.10 >> "$TREE/logs/qwen_r2_$1.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE $1 rc=$rc" >> "$LOG"
}
until grep -q "DONE bishe_llm2 qwen05b_lr_3e-3" "$RLOG" 2>/dev/null; do sleep 120; done
run random_a_ef &
base=$(grep -c " DONE " "$RLOG")
until [ "$(grep -c " DONE " "$RLOG")" -gt "$base" ]; do sleep 120; done
run gev0_a_ef &
wait
echo "[$(date '+%F %T')] QWEN_R2_DONE" >> "$LOG"
