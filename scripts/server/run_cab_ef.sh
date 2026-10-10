#!/bin/bash
# After the DCEF campaign ends: (1) re-run DCEF runs whose source hash differs from the
# campaign hash (moved, not deleted), rebuild the DCEF report; (2) run the CAB-EF dev stage.
set -uo pipefail
D=/data/home/202412012250/bishe_dcef; C=/data/home/202412012250/bishe_cab
PY=$D/.venv/bin/python
export HF_HOME=$D/.hf TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
EXPECTED=683d6d515148ad382fb16da3750d584c4f1d170b468e6f5eeb84b07970a84bad
stage() { echo "[$(date "+%F %T")] $*"; }

stage "waiting for the DCEF campaign session to end"
while tmux has-session -t campaigns 2>/dev/null; do sleep 60; done
stage "DCEF campaign ended"

cd $D
$PY - "$EXPECTED" <<"PY"
import json, shutil, sys
from pathlib import Path
expected = sys.argv[1]
src = Path("docs/results/dcef_dev"); dst = Path("docs/results/dcef_dev_superseded"); dst.mkdir(exist_ok=True)
moved = []
for run in sorted(p for p in src.iterdir() if p.is_dir()):
    for name in ("summary.json", "failure.json"):
        f = run / name
        if f.exists():
            h = json.loads(f.read_text()).get("source_hash")
            if h != expected:
                shutil.move(str(run), str(dst / run.name)); moved.append((run.name, h))
            break
print("moved runs with a different source hash:", moved)
PY
stage "re-run superseded DCEF runs"
PYTHONPATH=src $PY scripts/run_parallel.py --config configs/roberta_dcef_dev.yaml --include-diagnostics --workers 3 > logs/run_dcef_dev_rerun.log 2>&1 || stage "rerun had failed runs"
PYTHONPATH=src $PY scripts/build_dcef_report.py --config configs/roberta_dcef_dev.yaml > logs/report_dcef_dev.log 2>&1 && stage "DCEF report rebuilt"

cd $C
stage "preflight cab_ef_dev"
PYTHONPATH=src $PY scripts/preflight.py --config configs/roberta_cab_ef_dev.yaml > logs/preflight_cab_ef_dev.log 2>&1 || { stage "preflight cab_ef_dev FAILED"; tail -30 logs/preflight_cab_ef_dev.log; exit 1; }
stage "run cab_ef_dev"
PYTHONPATH=src $PY scripts/run_parallel.py --config configs/roberta_cab_ef_dev.yaml --workers 3 > logs/run_cab_ef_dev.log 2>&1 || stage "cab_ef_dev finished with failed runs (kept as outcomes)"
PYTHONPATH=src $PY scripts/build_dcef_report.py --config configs/roberta_cab_ef_dev.yaml > logs/report_cab_ef_dev.log 2>&1 || { stage "report cab_ef_dev FAILED"; exit 1; }
stage "ALL_DONE"
