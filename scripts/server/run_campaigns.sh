#!/bin/bash
# DCEF dev -> report -> gamma calibration (gated) -> CAB dev -> report.
set -uo pipefail
cd /data/home/202412012250/bishe_dcef
. .venv/bin/activate
export PYTHONPATH=$PWD/src TOKENIZERS_PARALLELISM=false HF_HOME=$PWD/.hf OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
W=${WORKERS:-3}
stage() { echo "[$(date "+%F %T")] $*"; }

if [ ! -f docs/results/dcef_dev/DCEF_REPORT.md ] || [ "${RERUN_DCEF:-0}" = 1 ]; then
  stage "preflight dcef_dev"
  python scripts/preflight.py --config configs/roberta_dcef_dev.yaml > logs/preflight_dcef_dev.log 2>&1 || { stage "preflight dcef_dev FAILED"; tail -30 logs/preflight_dcef_dev.log; exit 1; }
  stage "run dcef_dev (workers=$W)"
  python scripts/run_parallel.py --config configs/roberta_dcef_dev.yaml --include-diagnostics --workers $W > logs/run_dcef_dev.log 2>&1 || stage "dcef_dev finished with failed runs (kept as outcomes)"
  stage "report dcef_dev"
  python scripts/build_dcef_report.py --config configs/roberta_dcef_dev.yaml > logs/report_dcef_dev.log 2>&1 || { stage "report dcef_dev FAILED"; exit 1; }
fi

stage "calibrate cab gamma"
python scripts/calibrate_cab_gamma.py --runs docs/results/dcef_dev/DCEF_RUNS.csv --results-dir docs/results/dcef_dev > logs/calibrate_cab_gamma.log 2>&1 || { stage "calibration FAILED"; cat logs/calibrate_cab_gamma.log; exit 1; }
cat logs/calibrate_cab_gamma.log
if grep -q OUTSIDE logs/calibrate_cab_gamma.log; then
  stage "STOP: calibrated gamma is outside the closed-form kappa region for some task/budget; needs a decision (kappa_override or budget subset)"
  exit 2
fi
python - <<"PY"
import re, sys, datetime
from pathlib import Path
sys.path.insert(0, "src")
from fedpeft.config import load_config
log = Path("logs/calibrate_cab_gamma.log").read_text()
gamma_line = next(line for line in log.splitlines() if line.startswith("  gamma: {"))
path = Path("configs/roberta_cab_dev.yaml"); text = path.read_text()
text = re.sub(r"  # PROVISIONAL placeholders:.*\n", f"  # calibrated {datetime.date.today()} from docs/results/dcef_dev (median gamma_bar of random_a_dcef)\n", text)
text = re.sub(r"  gamma: \{.*\}\n", gamma_line + "\n", text)
text = text.replace("  gamma_source: provisional", "  gamma_source: dcef_dev")
path.write_text(text)
load_config(path)
print("applied", gamma_line.strip())
PY
[ $? -eq 0 ] || { stage "applying gamma FAILED"; exit 1; }

stage "preflight cab_dev"
python scripts/preflight.py --config configs/roberta_cab_dev.yaml > logs/preflight_cab_dev.log 2>&1 || { stage "preflight cab_dev FAILED"; tail -30 logs/preflight_cab_dev.log; exit 1; }
stage "run cab_dev (workers=$W)"
python scripts/run_parallel.py --config configs/roberta_cab_dev.yaml --include-diagnostics --workers $W > logs/run_cab_dev.log 2>&1 || stage "cab_dev finished with failed runs (kept as outcomes)"
stage "report cab_dev"
python scripts/build_dcef_report.py --config configs/roberta_cab_dev.yaml > logs/report_cab_dev.log 2>&1 || { stage "report cab_dev FAILED"; exit 1; }
stage "ALL_DONE"
