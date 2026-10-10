#!/usr/bin/env bash
set -euo pipefail

BASE=/data/home/202422011138/bishe
EXEC_ROOT="$BASE/artifacts/formal_ablation_source_7d3aa3d6476289aad3260ffceadc76f8bdca9cd03a0b3b53646579248f9a0e15"
CONFIG="$BASE/configs/roberta_fixed_a_ablation.yaml"
OUTPUT="$BASE/docs/results/fixed_a_ablation/formal"
EXPECTED_SOURCE=7d3aa3d6476289aad3260ffceadc76f8bdca9cd03a0b3b53646579248f9a0e15
EXPECTED_CONFIG=ca5b3fb56f5b61fc070e44448a32813f816e5b95d9967dc915d0bbc38d652d5b

export PYTHONPATH="$EXEC_ROOT/src:$EXEC_ROOT/.compat:$EXEC_ROOT/.deps"
export TOKENIZERS_PARALLELISM=false

actual_config=$(sha256sum "$CONFIG" | awk '{print $1}')
if [[ "$actual_config" != "$EXPECTED_CONFIG" ]]; then
  echo "config hash mismatch: $actual_config" >&2
  exit 2
fi

actual_source=$(python3 - "$EXEC_ROOT" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted((root / "src").rglob("*.py")):
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
print(digest.hexdigest())
PY
)
if [[ "$actual_source" != "$EXPECTED_SOURCE" ]]; then
  echo "formal source hash mismatch: $actual_source" >&2
  exit 3
fi

python3 - "$CONFIG" "$OUTPUT/preflight.json" <<'PY'
import json
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
expected = (
    len(config["data"]["tasks"])
    * len(config["data"]["partition_seeds"])
    * (
        len(config["matrix"]["dense_methods"])
        + len(config["matrix"]["sparse_methods"])
        * len(config["communication"]["wire_ratios"])
    )
)
if expected != 54 or config["project"].get("expected_runs") != 54:
    raise SystemExit(f"expected matrix is not exactly 54 runs: {expected}")
preflight = json.load(open(sys.argv[2], encoding="utf-8"))
if preflight.get("status") != "PASS":
    raise SystemExit("preflight is not PASS")
PY

mkdir -p "$OUTPUT"
LOCK="$OUTPUT/campaign.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "campaign lock already exists: $LOCK" >&2
  exit 75
fi
cleanup() {
  rm -f "$LOCK/owner"
  rmdir "$LOCK" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
printf '%s:%s\n' "$(hostname)" "$$" > "$LOCK/owner"

if [[ -e "$OUTPUT/campaign_events.jsonl" || -e "$OUTPUT/campaign_config.json" || -e "$OUTPUT/campaign_summary.json" ]]; then
  echo "refusing to overwrite an existing campaign" >&2
  exit 4
fi
if find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d -name '*__*' -print -quit | grep -q .; then
  echo "refusing to mix with existing run directories" >&2
  exit 5
fi

cd "$EXEC_ROOT"
python3 scripts/run_matrix.py --config "$CONFIG" --fail-fast
