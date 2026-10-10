#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


TASKS = ("sst2", "rte", "mnli")
SEEDS = (401, 402, 403)
RATIOS = (0.1, 0.2)
SOURCE_HASH = "7d3aa3d6476289aad3260ffceadc76f8bdca9cd03a0b3b53646579248f9a0e15"
EXPECTED_CONFIG_SHA256 = "ca5b3fb56f5b61fc070e44448a32813f816e5b95d9967dc915d0bbc38d652d5b"
EXPECTED_PROTOCOL_SHA256 = "0189b584a90db7f750af5a9826fe46bd71f42c45a54d2d82ae8be8607d42a332"
EXPECTED_PREFLIGHT_SOURCE_HASH = "6b38637b564b164e0b68fc8977dadb2201c52e31e58c5aa27181ffcf3eada28c"
THRESHOLDS = {"sst2": 0.003, "rte": 0.010, "mnli": 0.003}
NEW_METHODS = ("random_a_dense", "spectral_a_dense", "random_a_topk", "spectral_a_topk")
EF_METHODS = ("random_a_ef", "spectral_a_ef")
SUBREPORT_ARTIFACTS = (
    "RESULTS.md",
    "ANALYSIS.md",
    "RUNS.csv",
    "PAIRED_DELTAS.csv",
    "COMMUNICATION_EFFICIENCY_BY_SEED.csv",
    "COMMUNICATION_EFFICIENCY.csv",
)
COMPARISON_ORDER = (
    "dense_spectral_minus_random",
    "random_ef_minus_noef",
    "spectral_ef_minus_noef",
    "noef_spectral_minus_random",
    "basis_by_ef_difference_in_differences",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"YAML root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _summary_path(root: Path, task: str, method: str, seed: int, ratio: float | None) -> Path:
    suffix = "dense" if ratio is None else f"wire-{int(round(ratio * 100)):02d}"
    return root / f"{task}__{method}__seed-{seed}__{suffix}" / "summary.json"


def _new_run_specs() -> tuple[tuple[str, str, int, float | None], ...]:
    specs: list[tuple[str, str, int, float | None]] = []
    for task in TASKS:
        for seed in SEEDS:
            for method in NEW_METHODS:
                ratios: tuple[float | None, ...] = (None,) if method.endswith("_dense") else RATIOS
                specs.extend((task, method, seed, ratio) for ratio in ratios)
    return tuple(specs)


def _run_id(task: str, method: str, seed: int, ratio: float | None) -> str:
    suffix = "dense" if ratio is None else f"wire-{int(round(ratio * 100)):02d}"
    return f"{task}__{method}__seed-{seed}__{suffix}"


def _load_matrix(root: Path, methods: tuple[str, ...], *, ef: bool) -> dict[tuple[str, str, int, float | None], dict[str, Any]]:
    rows: dict[tuple[str, str, int, float | None], dict[str, Any]] = {}
    for task in TASKS:
        for seed in SEEDS:
            for method in methods:
                ratios: tuple[float | None, ...]
                if method.endswith("_dense"):
                    ratios = (None,)
                else:
                    ratios = RATIOS
                for ratio in ratios:
                    path = _summary_path(root, task, method, seed, ratio)
                    if not path.exists():
                        raise RuntimeError(f"missing summary: {path}")
                    summary = _read_json(path)
                    key = (task, method, seed, ratio)
                    if key in rows:
                        raise RuntimeError(f"duplicate summary key: {key}")
                    summary["_path"] = str(path)
                    rows[key] = summary
    expected = 36 if ef else 54
    if len(rows) != expected:
        raise RuntimeError(f"matrix size mismatch for {root}: {len(rows)} != {expected}")
    return rows


def _basis_hash(summary: dict[str, Any]) -> str:
    return str(summary["invariants"]["before"]["fixed_basis_hash"])


def _score(summary: dict[str, Any]) -> float:
    return float(summary["final"]["primary_metric"])


def _common_audit(summary: dict[str, Any], expected_task: str, expected_method: str, expected_seed: int, expected_ratio: float | None) -> list[str]:
    errors: list[str] = []
    expected_run = {
        "task": expected_task,
        "method": expected_method,
        "seed": expected_seed,
        "wire_ratio": expected_ratio,
    }
    if summary.get("schema_version") != 2:
        errors.append("schema_version != 2")
    if summary.get("status") != "COMPLETE":
        errors.append("status != COMPLETE")
    if summary.get("run") != expected_run:
        errors.append(f"run mismatch: {summary.get('run')} != {expected_run}")
    if summary.get("source_hash") != SOURCE_HASH:
        errors.append(f"source hash mismatch: {summary.get('source_hash')}")
    if summary.get("data", {}).get("evaluation_policy") != "official_validation":
        errors.append("evaluation policy is not official_validation")
    if summary.get("invariants", {}).get("basis_unchanged") is not True:
        errors.append("fixed basis changed")
    if summary.get("invariants", {}).get("before", {}).get("all_base_parameters_frozen") is not True:
        errors.append("base parameters are not frozen")
    if summary.get("invariants", {}).get("after", {}).get("all_base_parameters_frozen") is not True:
        errors.append("base parameters were unfrozen")
    if summary.get("invariants", {}).get("wire_cap_respected") is not True:
        errors.append("wire cap not respected")
    if not math.isfinite(_score(summary)):
        errors.append("final primary score is non-finite")
    final_round = int(summary.get("final", {}).get("round", -1))
    configured_rounds = int(summary.get("training", {}).get("rounds", -2))
    if final_round != configured_rounds:
        errors.append(f"final round mismatch: {final_round} != {configured_rounds}")
    return errors


def _equal_fields(left: dict[str, Any], right: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    for field in fields:
        if left.get(field) != right.get(field):
            errors.append(f"{field} mismatch")
    return errors


def _pair_audit(left: dict[str, Any], right: dict[str, Any], *, same_basis: bool, sparse: bool) -> list[str]:
    errors = _equal_fields(
        left,
        right,
        ("source_hash", "model_hash", "dataset_hash", "partition_hash", "layout_hash", "model", "data", "training"),
    )
    for field in ("initial_adapter_hash", "initial_head_hash"):
        if left.get("initialization", {}).get(field) != right.get("initialization", {}).get(field):
            errors.append(f"initialization.{field} mismatch")
    left_basis = _basis_hash(left)
    right_basis = _basis_hash(right)
    if same_basis and left_basis != right_basis:
        errors.append("fixed basis hash mismatch for same-basis pair")
    if not same_basis and left_basis == right_basis:
        errors.append("Random/Spectral basis hashes unexpectedly match")
    if sparse:
        if left.get("wire_budget") != right.get("wire_budget"):
            errors.append("wire_budget mismatch")
        if left.get("setup_communication") != right.get("setup_communication"):
            errors.append("setup_communication mismatch")
        for side, summary in (("left", left), ("right", right)):
            final_wire = summary.get("final", {}).get("wire", {})
            cap = summary.get("wire_budget", {}).get("message_cap_bytes")
            for key in ("max_uplink_message_bytes", "max_downlink_message_bytes"):
                value = final_wire.get(key)
                if not isinstance(value, int) or not isinstance(cap, int) or value > cap:
                    errors.append(f"{side} {key} exceeds or lacks cap")
    return errors


def _strip_scope(config: dict[str, Any]) -> dict[str, Any]:
    clean = json.loads(json.dumps(config))
    project = clean.get("project", {})
    for key in ("root", "output_dir", "expected_runs"):
        project.pop(key, None)
    clean.pop("matrix", None)
    return clean


def _normalize_manifest_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize only the output path that load_config resolves at launch time."""
    clean = json.loads(json.dumps(config))
    clean.setdefault("project", {})["output_dir"] = "<campaign-output>"
    return clean


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _audit_frozen_inputs(
    ablation_dir: Path,
    config_path: Path,
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    errors: list[str] = []
    config_hash = _sha256(config_path)
    protocol_hash = _sha256(protocol_path)
    if config_hash != EXPECTED_CONFIG_SHA256:
        errors.append(
            f"frozen config hash mismatch: {config_hash} != {EXPECTED_CONFIG_SHA256}"
        )
    if protocol_hash != EXPECTED_PROTOCOL_SHA256:
        errors.append(
            f"frozen protocol hash mismatch: {protocol_hash} != {EXPECTED_PROTOCOL_SHA256}"
        )

    archived_config = ablation_dir / "provenance" / "roberta_fixed_a_ablation.yaml"
    archived_protocol = ablation_dir / "provenance" / "FORMAL_FIXED_A_ABLATION_PLAN.md"
    archived_hashes: dict[str, str] = {}
    for label, path, expected_hash in (
        ("config", archived_config, EXPECTED_CONFIG_SHA256),
        ("protocol", archived_protocol, EXPECTED_PROTOCOL_SHA256),
    ):
        if not path.is_file():
            errors.append(f"missing archived frozen {label}: {path}")
            continue
        actual_hash = _sha256(path)
        archived_hashes[label] = actual_hash
        if actual_hash != expected_hash:
            errors.append(
                f"archived frozen {label} hash mismatch: {actual_hash} != {expected_hash}"
            )

    frozen_config = _read_yaml(config_path)
    metadata = {
        "config": {
            "source_path": str(config_path),
            "sha256": config_hash,
            "expected_sha256": EXPECTED_CONFIG_SHA256,
            "archived_path": str(archived_config),
            "archived_sha256": archived_hashes.get("config", ""),
        },
        "protocol": {
            "source_path": str(protocol_path),
            "sha256": protocol_hash,
            "expected_sha256": EXPECTED_PROTOCOL_SHA256,
            "archived_path": str(archived_protocol),
            "archived_sha256": archived_hashes.get("protocol", ""),
        },
    }
    return frozen_config, metadata, errors


def _audit_campaign(
    ablation_dir: Path,
    frozen_config: dict[str, Any],
    expected_specs: tuple[tuple[str, str, int, float | None], ...],
) -> tuple[dict[str, Any], list[str]]:
    ablation_dir = ablation_dir.resolve()
    errors: list[str] = []
    expected_ids = [_run_id(*spec) for spec in expected_specs]
    expected_summary_paths = {
        (ablation_dir / run_id / "summary.json").resolve() for run_id in expected_ids
    }

    manifest = _read_json(ablation_dir / "campaign_config.json")
    if set(manifest) != {"config", "include_diagnostics", "requested_run_ids"}:
        errors.append(f"campaign manifest keys mismatch: {sorted(manifest)}")
    if manifest.get("include_diagnostics") is not False:
        errors.append("campaign manifest include_diagnostics is not false")
    if manifest.get("requested_run_ids") != expected_ids:
        errors.append("campaign manifest requested_run_ids are not the exact frozen 54-run order")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        errors.append("campaign manifest config is not an object")
    else:
        if _normalize_manifest_config(manifest_config) != _normalize_manifest_config(frozen_config):
            errors.append("campaign manifest config does not match the frozen config")
        output_dir = manifest_config.get("project", {}).get("output_dir")
        if not isinstance(output_dir, str) or Path(output_dir).resolve() != ablation_dir:
            errors.append(f"campaign manifest output_dir mismatch: {output_dir}")

    campaign_summary = _read_json(ablation_dir / "campaign_summary.json")
    if set(campaign_summary) != {
        "completed_runs", "failed_runs", "finished_at", "requested_runs", "started_at", "status"
    }:
        errors.append(f"campaign summary keys mismatch: {sorted(campaign_summary)}")
    if campaign_summary.get("status") != "COMPLETE":
        errors.append("campaign summary status is not COMPLETE")
    if campaign_summary.get("requested_runs") != len(expected_ids):
        errors.append("campaign summary requested_runs is not 54")
    if campaign_summary.get("completed_runs") != expected_ids:
        errors.append("campaign summary completed_runs are not the exact frozen 54-run order")
    if campaign_summary.get("failed_runs") != []:
        errors.append("campaign summary failed_runs is not empty")

    events = _read_jsonl(ablation_dir / "campaign_events.jsonl")
    if len(events) != 2 * len(expected_ids):
        errors.append(f"campaign event count mismatch: {len(events)} != {2 * len(expected_ids)}")
    else:
        for index, run_id in enumerate(expected_ids, start=1):
            start = events[2 * (index - 1)]
            done = events[2 * (index - 1) + 1]
            expected_common = {"run_id": run_id, "index": index, "total": len(expected_ids)}
            for key, expected in expected_common.items():
                if start.get(key) != expected:
                    errors.append(f"campaign START {index} {key} mismatch")
                if done.get(key) != expected:
                    errors.append(f"campaign DONE {index} {key} mismatch")
            if start.get("event") != "START":
                errors.append(f"campaign event {2 * index - 1} is not START")
            if done.get("event") != "DONE" or done.get("returncode") != 0:
                errors.append(f"campaign event {2 * index} is not successful DONE")

    if (ablation_dir / "campaign.lock").exists():
        errors.append("campaign.lock still exists")
    if (ablation_dir / "failure.json").exists():
        errors.append("failure.json exists")

    actual_summary_paths = {path.resolve() for path in ablation_dir.rglob("summary.json")}
    missing_summaries = expected_summary_paths - actual_summary_paths
    extra_summaries = actual_summary_paths - expected_summary_paths
    if missing_summaries:
        errors.append(
            "campaign is missing summaries: "
            + ", ".join(str(path) for path in sorted(missing_summaries))
        )
    if extra_summaries:
        errors.append(
            "campaign has extra summaries: "
            + ", ".join(str(path) for path in sorted(extra_summaries))
        )

    preflight_path = ablation_dir / "preflight.json"
    preflight = _read_json(preflight_path)
    checks = preflight.get("checks", {})
    if preflight.get("schema") != "experiment-preflight-v1":
        errors.append("preflight schema mismatch")
    if preflight.get("status") != "PASS" or preflight.get("failures") != []:
        errors.append("preflight is not a clean PASS")
    if not isinstance(checks, dict):
        errors.append("preflight checks is not an object")
        checks = {}
    if checks.get("config_file_hash") != EXPECTED_CONFIG_SHA256:
        errors.append("preflight config_file_hash does not bind the frozen config")
    if checks.get("source_hash") != EXPECTED_PREFLIGHT_SOURCE_HASH:
        errors.append("preflight execution source hash mismatch")
    if checks.get("expected_runs") != len(expected_ids):
        errors.append("preflight expected_runs is not 54")
    preflight_output = checks.get("output_dir")
    if not isinstance(preflight_output, str) or Path(preflight_output).resolve() != ablation_dir:
        errors.append(f"preflight output_dir mismatch: {preflight_output}")

    metadata = {
        "requested_run_ids": len(expected_ids),
        "completed_run_ids": len(campaign_summary.get("completed_runs", [])),
        "event_rows": len(events),
        "summary_files": len(actual_summary_paths),
        "lock_absent": not (ablation_dir / "campaign.lock").exists(),
        "failure_file_absent": not (ablation_dir / "failure.json").exists(),
        "manifest_sha256": _sha256(ablation_dir / "campaign_config.json"),
        "events_sha256": _sha256(ablation_dir / "campaign_events.jsonl"),
        "summary_sha256": _sha256(ablation_dir / "campaign_summary.json"),
        "preflight_sha256": _sha256(preflight_path),
        "preflight_source_hash": checks.get("source_hash", ""),
    }
    return metadata, errors


def _audit_subreport(
    ablation_dir: Path,
    expected_specs: tuple[tuple[str, str, int, float | None], ...],
    summaries: dict[tuple[str, str, int, float | None], dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    artifact_hashes: dict[str, str] = {}
    for name in SUBREPORT_ARTIFACTS:
        path = ablation_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty build_report artifact: {path}")
            continue
        artifact_hashes[name] = _sha256(path)

    runs_path = ablation_dir / "RUNS.csv"
    if not runs_path.is_file():
        return {
            "source": str(runs_path),
            "rows": 0,
            "valid_required_rows": 0,
            "artifact_sha256": artifact_hashes,
        }, errors
    with runs_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    required_fields = {
        "run_id", "role", "audit_status", "reasons", "task", "method", "seed",
        "wire_target", "schema_version", "declared_status", "final_primary",
        "source_hash", "model_hash", "dataset_hash", "partition_hash", "layout_hash",
        "initial_head_hash", "summary_path",
    }
    missing_fields = required_fields - fields
    if missing_fields:
        errors.append(f"RUNS.csv missing fields: {sorted(missing_fields)}")

    expected_ids = [_run_id(*spec) for spec in expected_specs]
    if len(rows) != len(expected_specs):
        errors.append(f"RUNS.csv row count mismatch: {len(rows)} != {len(expected_specs)}")
    if [row.get("run_id") for row in rows] != expected_ids:
        errors.append("RUNS.csv run_ids are not the exact frozen 54-run order")

    valid_required = 0
    for row_index, (spec, row) in enumerate(zip(expected_specs, rows), start=1):
        task, method, seed, ratio = spec
        run_id = _run_id(*spec)
        expected_wire = "dense" if ratio is None else str(ratio)
        expected_summary = summaries.get(spec)
        row_errors: list[str] = []
        for field, expected in (
            ("run_id", run_id),
            ("role", "required"),
            ("audit_status", "VALID"),
            ("reasons", ""),
            ("task", task),
            ("method", method),
            ("seed", str(seed)),
            ("wire_target", expected_wire),
            ("schema_version", "2"),
            ("declared_status", "COMPLETE"),
            ("source_hash", SOURCE_HASH),
        ):
            if row.get(field) != expected:
                row_errors.append(f"{field}={row.get(field)!r}, expected {expected!r}")
        summary_source = Path(row.get("summary_path", ""))
        if summary_source.name != "summary.json" or summary_source.parent.name != run_id:
            row_errors.append(f"summary_path does not identify {run_id}/summary.json")
        if expected_summary is None:
            row_errors.append("corresponding loaded summary is missing")
        else:
            try:
                if not math.isclose(
                    float(row.get("final_primary", "nan")),
                    _score(expected_summary),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    row_errors.append("final_primary does not match summary")
            except (TypeError, ValueError):
                row_errors.append("final_primary is not finite numeric evidence")
            for csv_field, summary_field in (
                ("model_hash", "model_hash"),
                ("dataset_hash", "dataset_hash"),
                ("partition_hash", "partition_hash"),
                ("layout_hash", "layout_hash"),
            ):
                if row.get(csv_field) != str(expected_summary.get(summary_field, "")):
                    row_errors.append(f"{csv_field} does not match summary")
            if row.get("initial_head_hash") != str(
                expected_summary.get("initialization", {}).get("initial_head_hash", "")
            ):
                row_errors.append("initial_head_hash does not match summary")
        if row_errors:
            errors.extend(f"RUNS.csv row {row_index}/{run_id}: {reason}" for reason in row_errors)
        else:
            valid_required += 1

    metadata = {
        "source": str(runs_path),
        "rows": len(rows),
        "valid_required_rows": valid_required,
        "artifact_sha256": artifact_hashes,
    }
    return metadata, errors


def _judgement(deltas: list[float], threshold: float) -> str:
    mean = statistics.mean(deltas)
    if abs(mean) < threshold:
        return "TIE"
    if all(value > 0 for value in deltas) and mean >= threshold:
        return "SUPPORTED_POSITIVE"
    if all(value < 0 for value in deltas) and mean <= -threshold:
        return "SUPPORTED_NEGATIVE"
    return "MIXED"


def _csv_text(rows: list[dict[str, Any]], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _fmt(value: float) -> str:
    return f"{value:+.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", required=True)
    parser.add_argument("--formal-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir).resolve()
    formal_dir = Path(args.formal_dir).resolve()
    protocol_path = Path(args.protocol).resolve()
    config_path = Path(args.config).resolve()
    expected_specs = _new_run_specs()
    if len(expected_specs) != 54:
        raise RuntimeError(f"internal frozen matrix is not 54 runs: {len(expected_specs)}")

    frozen_config, frozen_inputs, frozen_input_errors = _audit_frozen_inputs(
        ablation_dir, config_path, protocol_path
    )
    campaign_audit, campaign_errors = _audit_campaign(
        ablation_dir, frozen_config, expected_specs
    )
    new = _load_matrix(ablation_dir, NEW_METHODS, ef=False)
    old = _load_matrix(formal_dir, EF_METHODS, ef=True)
    subreport_audit, subreport_errors = _audit_subreport(
        ablation_dir, expected_specs, new
    )

    errors: list[str] = [
        *frozen_input_errors,
        *campaign_errors,
        *subreport_errors,
    ]
    for (task, method, seed, ratio), summary in {**new, **old}.items():
        errors.extend(
            f"{summary['_path']}: {reason}"
            for reason in _common_audit(summary, task, method, seed, ratio)
        )

    new_campaign = _read_json(ablation_dir / "campaign_config.json")["config"]
    old_campaign = _read_json(formal_dir / "campaign_config.json")["config"]
    normalized_config_equal = _strip_scope(new_campaign) == _strip_scope(old_campaign)
    if not normalized_config_equal:
        errors.append("normalized formal and ablation configs differ outside project/matrix scope")

    by_seed: list[dict[str, Any]] = []

    def append_pair(comparison: str, task: str, seed: int, ratio: float | None, left: dict[str, Any], right: dict[str, Any], *, same_basis: bool, sparse: bool) -> None:
        pair_errors = _pair_audit(left, right, same_basis=same_basis, sparse=sparse)
        if pair_errors:
            errors.extend(f"{comparison}/{task}/{seed}/{ratio}: {reason}" for reason in pair_errors)
        left_score = _score(left)
        right_score = _score(right)
        by_seed.append(
            {
                "comparison": comparison,
                "task": task,
                "wire_ratio": "dense" if ratio is None else ratio,
                "seed": seed,
                "left_method": left["run"]["method"],
                "right_method": right["run"]["method"],
                "left_score": left_score,
                "right_score": right_score,
                "delta": left_score - right_score,
                "threshold": THRESHOLDS[task],
                "comparable": not pair_errors,
                "left_summary": left["_path"],
                "right_summary": right["_path"],
            }
        )

    for task in TASKS:
        for seed in SEEDS:
            append_pair(
                "dense_spectral_minus_random", task, seed, None,
                new[(task, "spectral_a_dense", seed, None)],
                new[(task, "random_a_dense", seed, None)],
                same_basis=False, sparse=False,
            )
            for ratio in RATIOS:
                random_ef = old[(task, "random_a_ef", seed, ratio)]
                spectral_ef = old[(task, "spectral_a_ef", seed, ratio)]
                random_noef = new[(task, "random_a_topk", seed, ratio)]
                spectral_noef = new[(task, "spectral_a_topk", seed, ratio)]
                append_pair(
                    "random_ef_minus_noef", task, seed, ratio,
                    random_ef, random_noef, same_basis=True, sparse=True,
                )
                append_pair(
                    "spectral_ef_minus_noef", task, seed, ratio,
                    spectral_ef, spectral_noef, same_basis=True, sparse=True,
                )
                append_pair(
                    "noef_spectral_minus_random", task, seed, ratio,
                    spectral_noef, random_noef, same_basis=False, sparse=True,
                )
                four_errors = []
                four_errors.extend(_pair_audit(spectral_ef, random_ef, same_basis=False, sparse=True))
                four_errors.extend(_pair_audit(spectral_noef, random_noef, same_basis=False, sparse=True))
                if four_errors:
                    errors.extend(
                        f"basis_by_ef_difference_in_differences/{task}/{seed}/{ratio}: {reason}"
                        for reason in four_errors
                    )
                ef_basis = _score(spectral_ef) - _score(random_ef)
                noef_basis = _score(spectral_noef) - _score(random_noef)
                by_seed.append(
                    {
                        "comparison": "basis_by_ef_difference_in_differences",
                        "task": task,
                        "wire_ratio": ratio,
                        "seed": seed,
                        "left_method": "spectral_minus_random_with_ef",
                        "right_method": "spectral_minus_random_without_ef",
                        "left_score": ef_basis,
                        "right_score": noef_basis,
                        "delta": ef_basis - noef_basis,
                        "threshold": THRESHOLDS[task],
                        "comparable": not four_errors,
                        "left_summary": f"{spectral_ef['_path']} | {random_ef['_path']}",
                        "right_summary": f"{spectral_noef['_path']} | {random_noef['_path']}",
                    }
                )

    if errors:
        raise RuntimeError("joint ablation audit failed:\n" + "\n".join(errors))

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in by_seed:
        grouped[(str(row["comparison"]), str(row["task"]), str(row["wire_ratio"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (comparison, task, wire), rows in grouped.items():
        rows.sort(key=lambda row: int(row["seed"]))
        deltas = [float(row["delta"]) for row in rows]
        summary_rows.append(
            {
                "comparison": comparison,
                "task": task,
                "wire_ratio": wire,
                "n": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "sample_sd": statistics.stdev(deltas),
                "min_delta": min(deltas),
                "max_delta": max(deltas),
                "positive": sum(value > 0 for value in deltas),
                "negative": sum(value < 0 for value in deltas),
                "zero": sum(value == 0 for value in deltas),
                "threshold": THRESHOLDS[task],
                "judgement": _judgement(deltas, THRESHOLDS[task]),
            }
        )
    comparison_rank = {name: index for index, name in enumerate(COMPARISON_ORDER)}
    task_rank = {name: index for index, name in enumerate(TASKS)}
    summary_rows.sort(key=lambda row: (comparison_rank[row["comparison"]], task_rank[row["task"]], str(row["wire_ratio"])))
    by_seed.sort(key=lambda row: (comparison_rank[row["comparison"]], task_rank[row["task"]], str(row["wire_ratio"]), int(row["seed"])))

    audit = {
        "status": "PASS",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "new_runs": len(new),
        "frozen_ef_runs": len(old),
        "paired_rows": len(by_seed),
        "paired_groups": len(summary_rows),
        "source_hash": SOURCE_HASH,
        "config_sha256": frozen_inputs["config"]["sha256"],
        "protocol_sha256": frozen_inputs["protocol"]["sha256"],
        "frozen_inputs": frozen_inputs,
        "campaign": campaign_audit,
        "new_subreport": subreport_audit,
        "normalized_config_equal": normalized_config_equal,
        "audit_errors": [],
    }

    lines = [
        "# Fixed-A basis and error-feedback ablation",
        "",
        f"Generated: {audit['generated_at']}",
        "",
        "## Audit",
        "",
        "- New formal ablation: **54/54 VALID**.",
        "- Frozen protocol/config hashes and archived copies: **PASS**.",
        "- Exact campaign manifest/events/summary set; no lock or failure artifact: **PASS**.",
        "- Independent `build_report.py` subreport: **54/54 required rows VALID**; artifact hashes are recorded in `JOINT_AUDIT.json`.",
        "- Frozen Random/Spectral EF controls: **36/36 COMPLETE** (read-only).",
        "- Joint pair-level provenance audit: **PASS**, 0 errors.",
        f"- Shared execution source: `{SOURCE_HASH}`.",
        "- Evaluation: official GLUE validation; seeds 401/402/403 were already exposed.",
        "",
        "The old 81-run report was not regenerated or modified. All claims below are paired descriptive evidence, not statistical significance.",
        "",
        "## Paired results",
        "",
        "| Comparison | Task | Wire | Mean delta ± SD | Signs (+/−/0) | Judgment |",
        "|---|---|---:|---:|---:|---|",
    ]
    labels = {
        "dense_spectral_minus_random": "Spectral dense − Random dense",
        "random_ef_minus_noef": "Random EF − no-EF",
        "spectral_ef_minus_noef": "Spectral EF − no-EF",
        "noef_spectral_minus_random": "Spectral no-EF − Random no-EF",
        "basis_by_ef_difference_in_differences": "Basis×EF difference-in-differences",
    }
    for row in summary_rows:
        wire = "dense" if row["wire_ratio"] == "dense" else f"{100 * float(row['wire_ratio']):.0f}%"
        lines.append(
            f"| {labels[row['comparison']]} | {row['task']} | {wire} | "
            f"{_fmt(float(row['mean_delta']))} ± {float(row['sample_sd']):.6f} | "
            f"{row['positive']}/{row['negative']}/{row['zero']} | {row['judgement']} |"
        )
    lines.extend(
        [
            "",
            "## Definitions",
            "",
            "- `Random EF − no-EF` and `Spectral EF − no-EF` isolate error feedback within a fixed basis and exact wire budget.",
            "- `Basis×EF difference-in-differences = (Spectral−Random with EF) − (Spectral−Random without EF)`.",
            "- `SUPPORTED_POSITIVE/NEGATIVE` requires all three seed deltas to share direction and the mean to exceed the preregistered practical threshold.",
            "- `TIE` means the mean lies inside the practical-equivalence interval; `MIXED` means seed directions are not uniform.",
            "",
            "Per-seed values and source paths are in `JOINT_PAIRED_BY_SEED.csv`; machine-readable audit details are in `JOINT_AUDIT.json`.",
            "",
        ]
    )

    seed_fields = [
        "comparison", "task", "wire_ratio", "seed", "left_method", "right_method",
        "left_score", "right_score", "delta", "threshold", "comparable",
        "left_summary", "right_summary",
    ]
    summary_fields = [
        "comparison", "task", "wire_ratio", "n", "mean_delta", "sample_sd",
        "min_delta", "max_delta", "positive", "negative", "zero", "threshold", "judgement",
    ]
    _atomic_write(ablation_dir / "JOINT_PAIRED_BY_SEED.csv", _csv_text(by_seed, seed_fields))
    _atomic_write(ablation_dir / "JOINT_PAIRED_SUMMARY.csv", _csv_text(summary_rows, summary_fields))
    _atomic_write(ablation_dir / "JOINT_AUDIT.json", json.dumps(audit, indent=2, sort_keys=True) + "\n")
    _atomic_write(ablation_dir / "JOINT_ABLATION_REPORT.md", "\n".join(lines))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
