#!/usr/bin/env bash
# Activation-metric basis probe (configs/roberta_gev_probe.yaml), MNLI seed 361,
# two lanes in priority order so the decisive arms finish first.
set -u
cd /data/home/202412012250/bishe_gev
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export PYTHONPATH=src OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CFG=configs/roberta_gev_probe.yaml

lane() {
  local name=$1
  shift
  for item in "$@"; do
    local method=${item%@*} wire=${item#*@}
    echo "[$(date '+%F %T')] START $method $wire" >> "logs/gev_lane_${name}.log"
    "$PY" scripts/run_experiment.py --config "$CFG" --task mnli --method "$method" \
      --seed 361 --wire-ratio "$wire" >> "logs/gev_lane_${name}.log" 2>&1
    rc=$?
    echo "[$(date '+%F %T')] DONE $method $wire rc=$rc" >> "logs/gev_lane_${name}.log"
  done
}

lane a gev0_a_ef@0.10 gev_a_ef@0.20 gev0_a_ef@0.05 gev_a_ef@0.10 &
lane b cab0wh_a_ef@0.10 randomwh_a_ef@0.10 gev0_a_ef@0.20 gev_a_ef@0.05 &
wait
echo "[$(date '+%F %T')] ALL DONE" >> logs/gev_pipeline.log
