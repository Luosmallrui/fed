#!/usr/bin/env bash
# (B) Qwen2.5-0.5B headroom check -- random fixed A vs full LoRA on QNLI, SST-2 and MNLI alpha 0.1 --
# then (A) the Qwen confirmation stage (configs/qwen05b_confirm.yaml, 21 runs) with three workers.
set -u
H=/data/home/202412012250
TREE=$H/bishe_llm3
PY=$H/bishe_dcef/.venv/bin/python
LOG=$TREE/logs/qwen_head_confirm.log
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$TREE"
echo "[$(date '+%F %T')] PRETOKENIZE" >> "$LOG"
PYTHONPATH="$TREE/src" "$PY" - >> "$LOG" 2>&1 <<'PYEOF'
from transformers import AutoTokenizer
from fedpeft.config import load_config
from fedpeft.data import prepare_tokenized_dataset
for name, task in (("qwen05b_head_qnli", "qnli"), ("qwen05b_head_sst2", "sst2")):
    cfg = load_config(f"configs/{name}.yaml")
    tok = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True, use_fast=True)
    prepare_tokenized_dataset(cfg["data"]["root"], cfg["project"]["tokenized_dir"], task, tok, int(cfg["data"]["max_length"]), 4)
    print("tokenized", task, flush=True)
PYEOF
run() {  # config task method [wire]
  local extra=()
  [ -n "${4:-}" ] && extra=(--wire-ratio "$4")
  cd "$TREE" || return
  echo "[$(date '+%F %T')] START $1 $3 ${4:-dense}" >> "$LOG"
  PYTHONPATH="$TREE/src" "$PY" "$TREE/scripts/run_experiment.py" --config "$TREE/configs/$1.yaml" \
    --task "$2" --method "$3" --seed 361 "${extra[@]}" >> "$TREE/logs/$1_$3.log" 2>&1
  local rc=$?
  echo "[$(date '+%F %T')] DONE $1 $3 ${4:-dense} rc=$rc" >> "$LOG"
}
( run qwen05b_head_qnli qnli fedavg_lora; run qwen05b_head_qnli qnli random_a_ef 0.10 ) &
sleep 300
( run qwen05b_head_mnli_a01 mnli fedavg_lora; run qwen05b_head_mnli_a01 mnli random_a_ef 0.10 ) &
sleep 300
( run qwen05b_head_sst2 sst2 fedavg_lora; run qwen05b_head_sst2 sst2 random_a_ef 0.10 ) &
wait
echo "[$(date '+%F %T')] HEADROOM_DONE" >> "$LOG"
cd "$TREE"
PYTHONPATH="$TREE/src" "$PY" "$TREE/scripts/run_parallel.py" --config "$TREE/configs/qwen05b_confirm.yaml" \
  --workers 3 --threads-per-run 2 >> "$TREE/logs/run_qwen05b_confirm.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] CONFIRM_DONE rc=$rc" >> "$LOG"
PYTHONPATH="$TREE/src" "$PY" "$TREE/scripts/build_dcef_report.py" --config "$TREE/configs/qwen05b_confirm.yaml" >> "$TREE/logs/run_qwen05b_confirm.log" 2>&1
echo "[$(date '+%F %T')] QWEN_HEAD_CONFIRM_DONE" >> "$LOG"
