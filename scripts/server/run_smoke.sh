#!/bin/bash
# Stage 0: tests + implementation smoke for DCEF and CAB (not method evidence).
set -euo pipefail
cd /data/home/202412012250/bishe_dcef
. .venv/bin/activate
export PYTHONPATH=$PWD/src TOKENIZERS_PARALLELISM=false HF_HOME=$PWD/.hf
echo "[$(date)] pytest"; echo skip-pytest-already-passed
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for name in cab_smoke; do
  echo "[$(date)] preflight $name"
  python scripts/preflight.py --config configs/roberta_$name.yaml > logs/preflight_$name.log 2>&1 || { tail -30 logs/preflight_$name.log; exit 1; }
  grep -m1 "\"status\"" logs/preflight_$name.log
  echo "[$(date)] run $name"
  python scripts/run_parallel.py --config configs/roberta_$name.yaml --include-diagnostics --workers 3 2>&1 | tee logs/run_$name.log | grep -v "^tokenized"
done
echo "[$(date)] SMOKE_DONE"
