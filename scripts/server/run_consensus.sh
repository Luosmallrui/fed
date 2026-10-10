#!/bin/bash
# Consensus measurement: 5 configs, one run at a time beside the CAB-EF campaign.
set -uo pipefail
cd /data/home/202412012250/bishe_consensus
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
export HF_HOME=/data/home/202412012250/bishe_dcef/.hf TOKENIZERS_PARALLELISM=false PYTHONPATH=src
for tag in a0.3 a0.1 a1000 a1 a0.3_b512; do
  echo "[$(date "+%F %T")] $tag"
  $PY scripts/run_parallel.py --config configs/roberta_consensus_$tag.yaml --workers 1 --threads-per-run 1 > logs/run_consensus_$tag.log 2>&1 || echo "  $tag had failed runs"
  grep -E "DONE|FAIL" logs/run_consensus_$tag.log
done
$PY scripts/summarize_consensus.py docs/results/exploration/consensus_* > logs/consensus_summary.txt 2>&1
echo "[$(date "+%F %T")] CONSENSUS_DONE"
