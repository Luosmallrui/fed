#!/usr/bin/env bash
# Fetch the raw files of the Web-service tasks (approved by the user on 2026-10-15) into
# bishe_dcef/datasets/raw_web, then build the DatasetDicts with scripts/prepare_web_tasks.py.
#   Banking77            PolyAI task-specific-datasets (CC BY 4.0), the files the HF loader of PolyAI/banking77 reads
#   Civil Comments       google/civil_comments parquet via hf-mirror (CC0)
#   TweetEval-offensive  cardiffnlp/tweet_eval offensive parquet via hf-mirror
set -euo pipefail
ROOT=/data/home/202412012250/bishe_web
DATA=/data/home/202412012250/bishe_dcef/datasets
RAW=$DATA/raw_web
PY=/data/home/202412012250/bishe_dcef/.venv/bin/python
MIRROR=https://hf-mirror.com
fetch() {  # url dest
  mkdir -p "$(dirname "$2")"
  [ -s "$2" ] && { echo "have $2"; return 0; }
  curl -fL --retry 5 --retry-delay 5 --connect-timeout 30 -o "$2.part" "$1" && mv "$2.part" "$2"
  echo "got $2 $(stat -c %s "$2") bytes"
}
fetch https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv $RAW/banking77/train.csv
fetch https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv $RAW/banking77/test.csv
for f in train-00000-of-00002 train-00001-of-00002 test-00000-of-00001; do
  fetch $MIRROR/datasets/google/civil_comments/resolve/main/data/$f.parquet $RAW/civil_comments/$f.parquet
done
for s in train validation test; do
  fetch $MIRROR/datasets/cardiffnlp/tweet_eval/resolve/main/offensive/$s-00000-of-00001.parquet $RAW/tweet_eval_offensive/$s-00000-of-00001.parquet
done
"$PY" "$ROOT/scripts/prepare_web_tasks.py" --raw "$RAW" --out "$DATA"
echo WEB_DATA_DONE
