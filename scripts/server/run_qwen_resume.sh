#!/usr/bin/env bash
# Resume of the Qwen2.5-0.5B runs lost on 2026-10-10 when the long-lived launcher shells
# lost their working directory (network filesystem); every run re-enters its tree by
# absolute path and status checks use absolute paths.  Three staggered lanes.
set -u
H=/data/home/202412012250
PY=$H/bishe_dcef/.venv/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$H/bishe_llm2/logs/qwen_resume.log

run() {  # tree config method seed [wire]
  local tree=$H/$1 extra=() wire_tag=dense
  if [ -n "${5:-}" ]; then extra=(--wire-ratio "$5"); wire_tag=$5; fi
  cd "$tree" || { echo "[$(date '+%F %T')] CD_FAILED $1" >> "$LOG"; return; }
  echo "[$(date '+%F %T')] START $1 $2 $3 $4 $wire_tag" >> "$LOG"
  PYTHONPATH="$tree/src" "$PY" "$tree/scripts/run_experiment.py" --config "$tree/configs/$2.yaml" \
    --task mnli --method "$3" --seed "$4" "${extra[@]}" >> "$tree/logs/qwen_resume_$3_$4.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE $1 $2 $3 $4 $wire_tag rc=$rc" >> "$LOG"
}

( run bishe_llm2 qwen05b_supp gev0_a_ef 362 0.10; run bishe_llm2 qwen05b_supp random_a_ef 363 0.10; run bishe_llm3 qwen05b_cost_L1 gev0_a_ef 361 0.10 ) &
sleep 600
( run bishe_llm2 qwen05b_supp gev0_a_ef 363 0.10; run bishe_llm2 qwen05b_supp fedavg_lora 363; run bishe_llm3 qwen05b_cost_L2 gev0_a_ef 361 0.10 ) &
sleep 600
( run bishe_llm3 qwen05b_cost_L3 gev0_a_ef 361 0.10; run bishe_llm2 qwen05b_supp cab0wh_a_ef 361 0.05; run bishe_llm2 qwen05b_lr_3e-3 random_a_ef 361 0.10 ) &
wait
echo "[$(date '+%F %T')] QWEN_RESUME_DONE" >> "$LOG"
