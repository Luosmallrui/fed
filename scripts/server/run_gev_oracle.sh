#!/usr/bin/env bash
# Oracle arm (configs/roberta_gev_oracle.yaml): exact-statistics gev0, MNLI seed 361.
set -u
cd /data/home/202412012250/bishe_gevx
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
for wire in 0.10 0.20; do
  echo "[$(date "+%F %T")] START gev0x_a_ef $wire" >> logs/gev_oracle.log
  "$PY" scripts/run_experiment.py --config configs/roberta_gev_oracle.yaml --task mnli \
    --method gev0x_a_ef --seed 361 --wire-ratio "$wire" >> logs/gev_oracle.log 2>&1
  echo "[$(date "+%F %T")] DONE gev0x_a_ef $wire rc=$?" >> logs/gev_oracle.log
done
