#!/usr/bin/env bash
set -euo pipefail

BASE=/data/home/202422011138/bishe
SOURCE_SHA=5423f99f735cf4dd5bea547a43be11d7ad7af601b6a06d0f59f5cd9d317824b6
CONFIG_SHA=b62eada60ed4a6410d582903dca3a66087cb1e7669d544f215b35e795305decb
PROTOCOL_SHA=0fac36aa12df3d452458418eb40b91d748a595bd2433d068edde891c089f9205
PREFLIGHT_SOURCE_SHA=6a232aaf4a0bc3bbac912b91e0ce5ae991586ba4eeb3494eb1c94fcec0abe8a1
PREFLIGHT_SHA=d5190875218a3ab372283c73b519cb58474b73bce220eb075b9f867b00974df0
SMOKE_SUMMARY_SHA=a1231d3bad5a0b7617a5884cc2a58a01b9d208befb2cb66a155fd199eb53273c
SMOKE_RUNS_SHA=1e9adf0abd9fab9f6b6a22a964de0871d474a1ef2cb63efde357483a831f368f
EXEC_ROOT="$BASE/artifacts/full_lora_noef_source_$SOURCE_SHA"
CONFIG="$BASE/configs/roberta_full_lora_noef.yaml"
PROTOCOL="$BASE/docs/results/exploration/FORMAL_FULL_LORA_NOEF_PLAN.md"
OUTPUT="$BASE/docs/results/full_lora_noef/formal"
SMOKE="$BASE/docs/results/full_lora_noef/smoke_seed311"

export PYTHONPATH="$EXEC_ROOT:$EXEC_ROOT/src:$EXEC_ROOT/.compat:$EXEC_ROOT/.deps"
export TOKENIZERS_PARALLELISM=false

test "$(sha256sum "$CONFIG" | awk '{print $1}')" = "$CONFIG_SHA"
test "$(sha256sum "$PROTOCOL" | awk '{print $1}')" = "$PROTOCOL_SHA"
test "$(sha256sum "$OUTPUT/preflight.json" | awk '{print $1}')" = "$PREFLIGHT_SHA"
test "$(sha256sum "$SMOKE/campaign_summary.json" | awk '{print $1}')" = "$SMOKE_SUMMARY_SHA"
test "$(sha256sum "$SMOKE/RUNS.csv" | awk '{print $1}')" = "$SMOKE_RUNS_SHA"

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
test "$actual_source" = "$SOURCE_SHA"

python3 - "$OUTPUT/preflight.json" "$SMOKE" "$CONFIG" \
  "$SOURCE_SHA" "$PREFLIGHT_SOURCE_SHA" "$CONFIG_SHA" "$PROTOCOL_SHA" <<'PY'
import csv
import json
import pathlib
import sys
import yaml

preflight = json.load(open(sys.argv[1], encoding="utf-8"))
if preflight.get("status") != "PASS":
    raise SystemExit("formal preflight is not PASS")
checks = preflight.get("checks", {})
if checks.get("expected_runs") != 18 or checks.get("required_runs") != 18:
    raise SystemExit("formal preflight is not the exact 18-run matrix")
if checks.get("config_file_hash") != sys.argv[6]:
    raise SystemExit("formal preflight config hash mismatch")
if checks.get("source_hash") != sys.argv[5]:
    raise SystemExit("formal preflight source hash mismatch")
config = yaml.safe_load(open(sys.argv[3], encoding="utf-8"))
metadata = config.get("full_lora_noef", {})
if metadata.get("execution_source_sha256") != sys.argv[4]:
    raise SystemExit("config execution source binding mismatch")
if metadata.get("protocol_sha256") != sys.argv[7]:
    raise SystemExit("config protocol binding mismatch")
if config.get("matrix", {}).get("sparse_methods") != ["full_lora_topk"]:
    raise SystemExit("formal config does not contain only full_lora_topk")
smoke = pathlib.Path(sys.argv[2])
campaign = json.load((smoke / "campaign_summary.json").open(encoding="utf-8"))
if campaign.get("status") != "COMPLETE" or len(campaign.get("completed_runs", [])) != 2:
    raise SystemExit("two-run smoke is not COMPLETE")
if campaign.get("failed_runs"):
    raise SystemExit("smoke contains failed runs")
with (smoke / "RUNS.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 2 or any(row.get("audit_status") != "VALID" for row in rows):
    raise SystemExit("smoke report is not exactly 2/2 VALID")
summaries = sorted(smoke.glob("*__*/summary.json"))
if len(summaries) != 2:
    raise SystemExit("smoke does not contain exactly two summaries")
for path in summaries:
    summary = json.load(path.open(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        raise SystemExit(f"incomplete smoke summary: {path}")
    if summary.get("run", {}).get("method") != "full_lora_topk":
        raise SystemExit(f"unexpected smoke method: {path}")
    if summary.get("source_hash") != sys.argv[4]:
        raise SystemExit(f"smoke source hash mismatch: {path}")
    invariants = summary.get("invariants", {})
    if invariants.get("ef_audit") != "disabled":
        raise SystemExit(f"EF audit is not disabled: {path}")
    for row in summary.get("rounds", []):
        diagnostics = row.get("diagnostics", {})
        for key in (
            "server_residual_norm",
            "mean_client_residual_norm",
            "residual_conservation_max",
            "residual_conservation_absolute_max",
        ):
            if float(diagnostics.get(key, -1.0)) != 0.0:
                raise SystemExit(f"nonzero no-EF diagnostic {key}: {path}")
PY

if [[ -e "$OUTPUT/campaign_events.jsonl" || -e "$OUTPUT/campaign_config.json" || -e "$OUTPUT/campaign_summary.json" || -e "$OUTPUT/campaign.lock" ]]; then
  echo "refusing to overwrite or resume an existing formal campaign" >&2
  exit 4
fi
if find "$OUTPUT" -mindepth 1 -maxdepth 1 -type d -name '*__*' -print -quit | grep -q .; then
  echo "refusing to mix with existing formal run directories" >&2
  exit 5
fi

cd "$EXEC_ROOT"
exec python3 -u scripts/run_matrix.py --config "$CONFIG" --fail-fast
