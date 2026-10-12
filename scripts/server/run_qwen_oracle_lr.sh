#!/usr/bin/env bash
# After the rank-2 test and the resumed runs: Qwen2.5-0.5B oracle gev0 and the matched learning-rate
# grid (gev0 at 1e-4 / 3e-4 / 3e-3, random at the deferred 3e-3), three staggered lanes.
set -u
H=/data/home/202412012250
PY=$H/bishe_dcef/.venv/bin/python
LOG=$H/bishe_llm3/logs/qwen_oracle_lr.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # tree config method
  local tree=$H/$1
  cd "$tree" || return
  echo "[$(date '+%F %T')] START $1 $2 $3" >> "$LOG"
  PYTHONPATH="$tree/src" "$PY" "$tree/scripts/run_experiment.py" --config "$tree/configs/$2.yaml" \
    --task mnli --method "$3" --seed 361 --wire-ratio 0.10 >> "$tree/logs/$2_$3.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE $1 $2 $3 rc=$rc" >> "$LOG"
}
until grep -q QWEN_R2_DONE $H/bishe_llm3/logs/qwen_r2.log 2>/dev/null && grep -q QWEN_RESUME_DONE $H/bishe_llm2/logs/qwen_resume.log 2>/dev/null; do sleep 300; done
mv -n $H/bishe_llm2/configs/qwen05b_lr_3e-3.yaml.deferred $H/bishe_llm2/configs/qwen05b_lr_3e-3.yaml
( run bishe_llm3 qwen05b_oracle gev0x_a_ef; run bishe_llm3 qwen05b_gev0_lr_1e-4 gev0_a_ef ) &
sleep 600
( run bishe_llm3 qwen05b_gev0_lr_3e-4 gev0_a_ef; run bishe_llm2 qwen05b_lr_3e-3 random_a_ef ) &
sleep 600
( run bishe_llm3 qwen05b_gev0_lr_3e-3 gev0_a_ef ) &
wait
echo "[$(date '+%F %T')] QWEN_ORACLE_LR_DONE" >> "$LOG"
