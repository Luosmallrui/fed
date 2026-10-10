#!/usr/bin/env python3
"""Build the preregistered Hybrid4 + late-decay Stage-A report.

The generic report builder remains authoritative for provenance and wire
audits.  This module adds only the Generation-4 paired, seed-level decision
gate from STABILITY_V4_PLAN.md.  In particular, it never substitutes a mean
for a failed seed and never selects a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fedpeft.config import load_config, result_path, run_id
from fedpeft.provenance import verified_stability_launch_provenance


def _report_modules():
    """Load the heavyweight audit stack only for real report construction."""
    try:  # Package imports in tests and ``python -m`` execution.
        from scripts import build_report as base_module
        from scripts import build_stability_report as stability_module
    except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
        import build_report as base_module  # type: ignore[no-redef]
        import build_stability_report as stability_module  # type: ignore[no-redef]
    return base_module, stability_module


V4_PROTOCOL_SHA256 = (
    "d22a2fd4d45717bb507f5572b9a6eabbc78332b6217cefcd36b1464fdc9df04c"
)
DEVELOPMENT_SEEDS = (351, 352, 353)
TASKS = ("sst2", "mrpc", "qqp")
EXPECTED_CELLS = len(TASKS) * len(DEVELOPMENT_SEEDS)
EXPECTED_RUNS = EXPECTED_CELLS * 5
WIRE_RATIO = 0.1
REFERENCE_METHOD = "fedavg_lora"
CONSTANT_DENSE_METHOD = "hybrid4_a_dense"
DECAY_DENSE_METHOD = "hybrid4_a_decay_dense"
CONSTANT_EF_METHOD = "hybrid4_a_ef"
CANDIDATE_METHOD = "hybrid4_a_decay_ef"
DENSE_METHODS = (
    REFERENCE_METHOD,
    CONSTANT_DENSE_METHOD,
    DECAY_DENSE_METHOD,
)
SPARSE_METHODS = (CONSTANT_EF_METHOD, CANDIDATE_METHOD)
ALL_METHODS = (*DENSE_METHODS, *SPARSE_METHODS)

BY_SEED_FIELDS = (
    "task",
    "method",
    "wire_target",
    "seed",
    "audit_status",
    "technical_valid",
    "primary_metric_name",
    "target_primary_score",
    "c95_status",
    "c95_round",
    "c95_wire_bytes",
    "fedavg_c95_round",
    "fedavg_c95_wire_bytes",
    "speedup95",
    "final_primary_score",
    "dense_final_primary_score",
    "evaluation_accuracy",
    "evaluation_positive_f1",
    "final_retention",
    "final_normal_path_wire_bytes",
    "post_hit_violation_count",
    "post_hit_violation_rounds",
    "tail_drop",
    "tail_drop_limit",
    "actual_message_bytes",
    "message_cap_bytes",
    "message_cap_utilization",
    "stable_success",
    "communication_efficient_success",
    "reasons",
    "strict_stable_success",
    "speedup_gt_one",
    "stage_a_success",
    "failure_components",
)

SUMMARY_FIELDS = (
    "method",
    "expected_cells",
    "observed_cells",
    "technical_valid_cells",
    "c95_reached_cells",
    "strict_stable_success_cells",
    "speedup_gt_one_cells",
    "stage_a_success_cells",
    "mean_speedup95",
    "worst_final_retention",
    "worst_tail_drop",
    "eligible",
)

LR_TRACE_FIELDS = (
    "task",
    "method",
    "wire_target",
    "seed",
    "round",
    "schedule_name",
    "multiplier",
    "adapter_learning_rate",
    "head_learning_rate",
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, float | None]:
    wire = row.get("wire_target")
    return (
        str(row.get("task", "")),
        str(row.get("method", "")),
        int(row.get("seed", -1)),
        None if wire == "dense" else float(wire),
    )


def _expected_keys() -> set[tuple[str, str, int, float | None]]:
    dense = {
        (task, method, seed, None)
        for task in TASKS
        for seed in DEVELOPMENT_SEEDS
        for method in DENSE_METHODS
    }
    sparse = {
        (task, method, seed, WIRE_RATIO)
        for task in TASKS
        for seed in DEVELOPMENT_SEEDS
        for method in SPARSE_METHODS
    }
    return dense | sparse


def _failure_components(row: dict[str, Any]) -> list[str]:
    """Return the preregistered strict-stability failures (not Speedup95)."""
    failures: list[str] = []
    if row.get("technical_valid") is not True:
        failures.append("TECHNICAL_INVALID")
        return failures
    if row.get("c95_status") != "REACHED":
        failures.append("C95_NOT_REACHED")
    if not _finite(row.get("fedavg_c95_wire_bytes")):
        failures.append("REFERENCE_C95_NOT_REACHED")
    if not _finite(row.get("final_accuracy")) or not _finite(
        row.get("target_accuracy")
    ):
        failures.append("MISSING_ENDPOINT")
    elif float(row["final_accuracy"]) < float(row["target_accuracy"]):
        failures.append("FINAL_BELOW_TARGET")
    if not _finite(row.get("final_retention")) or float(
        row["final_retention"]
    ) < 0.95:
        failures.append("RETENTION_BELOW_0.95")
    try:
        post_hit = int(row.get("post_hit_violation_count", -1))
    except (TypeError, ValueError):
        post_hit = -1
    if post_hit != 0:
        failures.append("POST_HIT_VIOLATION")
    if not _finite(row.get("tail_drop")) or not _finite(
        row.get("tail_drop_limit")
    ):
        failures.append("MISSING_TAIL_DROP")
    elif float(row["tail_drop"]) > float(row["tail_drop_limit"]):
        failures.append("TAIL_DROP_EXCEEDED")
    return failures


def apply_v4_gate(row: dict[str, Any]) -> dict[str, Any]:
    """Copy a generic stability row and apply the stricter v4 endpoint."""
    output = dict(row)
    failures = _failure_components(output)
    speedup_gt_one = _finite(output.get("speedup95")) and float(
        output["speedup95"]
    ) > 1.0
    strict = not failures
    is_candidate = output.get("method") == CANDIDATE_METHOD
    output.update(
        {
            "strict_stable_success": strict,
            "speedup_gt_one": speedup_gt_one,
            "stage_a_success": is_candidate and strict and speedup_gt_one,
            "failure_components": ";".join(failures),
        }
    )
    # Override the legacy field so consumers cannot mistake the v2 gate for
    # the v4 retention-aware definition.
    output["stable_success"] = strict
    output["communication_efficient_success"] = strict and speedup_gt_one
    return output


def v4_rows(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Create audited rows while reusing the generic paired-C95 parser."""
    _, stability_module = _report_modules()
    records = list(records)
    references = {
        (record.spec.task, record.spec.seed): record.summary
        for record in records
        if record.valid
        and record.spec is not None
        and record.spec.method == REFERENCE_METHOD
        and record.spec.wire_ratio is None
        and record.summary is not None
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        spec = record.spec
        if spec is None or spec.method not in ALL_METHODS:
            continue
        summary = record.summary or {"run": spec.run_dict}
        row = stability_module.stability_cell(
            summary,
            references.get((spec.task, spec.seed)),
            audit_status=record.audit_status,
            audit_reasons=record.reasons,
        )
        metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
        final = summary.get("final", {}) if isinstance(summary, dict) else {}
        splits = final.get("splits", {}) if isinstance(final, dict) else {}
        split = splits.get("train_holdout", splits.get("validation", {}))
        row.update(
            {
                "primary_metric_name": metrics.get("primary_metric_name", ""),
                "target_primary_score": row.get("target_accuracy", ""),
                "final_primary_score": row.get("final_accuracy", ""),
                "dense_final_primary_score": row.get("dense_final_accuracy", ""),
                "evaluation_accuracy": split.get("accuracy", "")
                if isinstance(split, dict)
                else "",
                "evaluation_positive_f1": split.get("f1", "")
                if isinstance(split, dict)
                else "",
            }
        )
        rows.append(apply_v4_gate(row))
    return rows


def lr_trace_rows(
    summaries: dict[tuple[str, str, int, float | None], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten every audited run's immutable per-round LR trace."""
    output: list[dict[str, Any]] = []
    for key in sorted(summaries, key=lambda item: (item[0], item[2], item[1], item[3] or -1.0)):
        task, method, seed, ratio = key
        if key not in _expected_keys():
            continue
        for round_record in summaries[key].get("rounds", []):
            learning_rate = round_record.get("learning_rate", {})
            output.append(
                {
                    "task": task,
                    "method": method,
                    "wire_target": "dense" if ratio is None else ratio,
                    "seed": seed,
                    "round": round_record.get("round", ""),
                    "schedule_name": learning_rate.get("schedule_name", ""),
                    "multiplier": learning_rate.get("multiplier", ""),
                    "adapter_learning_rate": learning_rate.get(
                        "adapter_learning_rate", ""
                    ),
                    "head_learning_rate": learning_rate.get(
                        "head_learning_rate", ""
                    ),
                }
            )
    return output


def _summarize_method(
    index: dict[tuple[str, str, int, float | None], dict[str, Any]],
    method: str,
) -> dict[str, Any]:
    ratio = None if method in DENSE_METHODS else WIRE_RATIO
    expected = [
        (task, method, seed, ratio)
        for task in TASKS
        for seed in DEVELOPMENT_SEEDS
    ]
    rows = [index[key] for key in expected if key in index]
    speedups = [
        float(row["speedup95"])
        for row in rows
        if _finite(row.get("speedup95"))
    ]
    retentions = [
        float(row["final_retention"])
        for row in rows
        if _finite(row.get("final_retention"))
    ]
    tails = [
        float(row["tail_drop"])
        for row in rows
        if _finite(row.get("tail_drop"))
    ]
    stage_successes = sum(row.get("stage_a_success") is True for row in rows)
    return {
        "method": method,
        "expected_cells": len(expected),
        "observed_cells": len(rows),
        "technical_valid_cells": sum(
            row.get("technical_valid") is True for row in rows
        ),
        "c95_reached_cells": sum(
            row.get("c95_status") == "REACHED" for row in rows
        ),
        "strict_stable_success_cells": sum(
            row.get("strict_stable_success") is True for row in rows
        ),
        "speedup_gt_one_cells": sum(
            row.get("speedup_gt_one") is True for row in rows
        ),
        "stage_a_success_cells": stage_successes,
        "mean_speedup95": statistics.mean(speedups)
        if len(speedups) == len(expected)
        else "",
        "worst_final_retention": min(retentions)
        if len(retentions) == len(expected)
        else "",
        "worst_tail_drop": max(tails) if len(tails) == len(expected) else "",
        "eligible": method == CANDIDATE_METHOD
        and len(rows) == len(expected)
        and stage_successes == len(expected),
    }


def _nested_number(value: Any, path: tuple[str, ...]) -> float | None:
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if _finite(value) else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    x_mean = (len(values) - 1) / 2
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator == 0:
        return None
    y_mean = statistics.mean(values)
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(values)
    ) / denominator


def _negative_fraction(values: list[float]) -> float | None:
    return sum(value < 0 for value in values) / len(values) if values else None


def _diagnostic_evidence(summary: dict[str, Any] | None) -> dict[str, Any]:
    """Evaluate the frozen equal-window directional mechanism gates."""
    if not isinstance(summary, dict):
        return {"complete": False, "reason": "candidate summary unavailable"}
    rounds = [row for row in summary.get("rounds", []) if isinstance(row, dict)]
    if len(rounds) < 5:
        return {"complete": False, "reason": "too few diagnostic rounds"}
    total = max(int(row.get("round", 0)) for row in rounds)
    start_round = math.floor(0.8 * total)
    late_length = total - start_round
    previous_start = start_round - late_length + 1
    previous = [
        row
        for row in rounds
        if previous_start <= int(row.get("round", 0)) <= start_round
    ]
    late = [row for row in rounds if int(row.get("round", 0)) > start_round]

    def series(rows_: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
        values: list[float] = []
        for row in rows_:
            value = _nested_number(
                row,
                ("diagnostics", "stability_diagnostics", *path),
            )
            if value is not None:
                values.append(value)
        return values

    def direct_series(rows_: list[dict[str, Any]], key: str) -> list[float]:
        values: list[float] = []
        for row in rows_:
            value = _nested_number(row, ("diagnostics", key))
            if value is not None:
                values.append(value)
        return values

    residual_ratio_path = (
        "server",
        "residual",
        "overall",
        "residual_to_update_ratio",
    )
    aggregate_cosine_path = ("updates", "aggregate_to_transmitted_cosine")
    consecutive_cosine_path = ("updates", "consecutive_aggregate_cosine")
    previous_residual_ratio = series(previous, residual_ratio_path)
    late_residual_ratio = series(late, residual_ratio_path)
    late_server_residual_norm = direct_series(late, "server_residual_norm")
    late_aggregate_to_transmitted = series(late, aggregate_cosine_path)
    previous_consecutive = series(previous, consecutive_cosine_path)
    late_consecutive = series(late, consecutive_cosine_path)
    previous_aggregate_norm = direct_series(previous, "adapter_update_norm")
    late_aggregate_norm = direct_series(late, "adapter_update_norm")
    diagnostic_complete = all(
        len(values) == late_length
        for values in (
            previous_residual_ratio,
            late_residual_ratio,
            late_server_residual_norm,
            late_aggregate_to_transmitted,
            previous_consecutive,
            late_consecutive,
            previous_aggregate_norm,
            late_aggregate_norm,
        )
    )
    residual_norm_slope = _slope(late_server_residual_norm)
    previous_residual_median = _median(previous_residual_ratio)
    late_residual_median = _median(late_residual_ratio)
    aggregate_to_transmitted_slope = _slope(late_aggregate_to_transmitted)
    previous_negative_fraction = _negative_fraction(previous_consecutive)
    late_negative_fraction = _negative_fraction(late_consecutive)
    previous_norm_median = _median(previous_aggregate_norm)
    late_norm_median = _median(late_aggregate_norm)
    norm_ratio = (
        late_norm_median / previous_norm_median
        if late_norm_median is not None
        and previous_norm_median is not None
        and previous_norm_median > 0
        else None
    )
    residual_gate = bool(
        diagnostic_complete
        and residual_norm_slope is not None
        and residual_norm_slope > 0
        and late_residual_median is not None
        and previous_residual_median is not None
        and late_residual_median > previous_residual_median
        and aggregate_to_transmitted_slope is not None
        and aggregate_to_transmitted_slope < 0
    )
    damping_gate = bool(
        diagnostic_complete
        and previous_negative_fraction is not None
        and late_negative_fraction is not None
        and late_negative_fraction > previous_negative_fraction
        and norm_ratio is not None
        and norm_ratio > 1.0
    )
    return {
        "complete": diagnostic_complete,
        "previous_window": [previous_start, start_round],
        "late_window": [start_round + 1, total],
        "late_server_residual_norm_slope": residual_norm_slope,
        "previous_server_residual_to_update_median": previous_residual_median,
        "late_server_residual_to_update_median": late_residual_median,
        "late_aggregate_to_transmitted_cosine_slope": aggregate_to_transmitted_slope,
        "previous_negative_consecutive_aggregate_cosine_fraction": previous_negative_fraction,
        "late_negative_consecutive_aggregate_cosine_fraction": late_negative_fraction,
        "late_to_previous_aggregate_norm_ratio": norm_ratio,
        "residual_control_gate": residual_gate,
        "server_damping_gate": damping_gate,
    }


def _failure_occurs_in_late_window(row: dict[str, Any]) -> bool:
    total = 30 if row.get("task") == "sst2" else 50
    first_late = math.floor(0.8 * total) + 1
    raw_rounds = row.get("post_hit_violation_rounds", "")
    if isinstance(raw_rounds, str):
        tokens = raw_rounds.replace(",", ";").split(";")
    elif isinstance(raw_rounds, list):
        tokens = raw_rounds
    else:
        tokens = []
    violation_rounds: list[int] = []
    for token in tokens:
        try:
            violation_rounds.append(int(token))
        except (TypeError, ValueError):
            continue
    first_violation_is_late = bool(
        violation_rounds and min(violation_rounds) >= first_late
    )
    final_failures = {
        "C95_NOT_REACHED",
        "FINAL_BELOW_TARGET",
        "RETENTION_BELOW_0.95",
        "TAIL_DROP_EXCEEDED",
        "MISSING_ENDPOINT",
        "MISSING_TAIL_DROP",
    }
    final_failure_is_late = bool(
        final_failures.intersection(_failure_components(row))
    )
    return first_violation_is_late or final_failure_is_late


def _mechanism_routing(
    index: dict[tuple[str, str, int, float | None], dict[str, Any]],
    summaries: dict[tuple[str, str, int, float | None], dict[str, Any]],
    *,
    matrix_complete: bool,
) -> dict[str, Any]:
    if not matrix_complete:
        return {
            "classification": "BLOCKED",
            "recommended_mechanism": "NONE",
            "future_research_direction": "NONE",
            "reason": f"the {EXPECTED_RUNS}-run audited matrix is incomplete or invalid",
        }
    candidate_rows = [
        index[(task, CANDIDATE_METHOD, seed, WIRE_RATIO)]
        for task in TASKS
        for seed in DEVELOPMENT_SEEDS
    ]
    strict_failures = [
        row for row in candidate_rows if row.get("strict_stable_success") is not True
    ]
    if not strict_failures:
        if all(row.get("stage_a_success") is True for row in candidate_rows):
            return {
                "classification": "NO_ADDITIONAL_MECHANISM",
                "recommended_mechanism": "NONE",
                "future_research_direction": "NONE",
                "reason": "candidate passed all 9/9 strict stability and Speedup95 gates",
            }
        return {
            "classification": "COMMUNICATION_EFFICIENCY_FAILURE",
            "recommended_mechanism": "NONE",
            "future_research_direction": "COMMUNICATION_EFFICIENCY_STUDY",
            "reason": "stability passed 9/9; at least one Speedup95 was not strictly above one",
        }

    matched: list[dict[str, Any]] = []
    any_shared_failure = False
    dense_all_stable = all(
        index[(task, DECAY_DENSE_METHOD, seed, None)].get(
            "strict_stable_success"
        )
        is True
        for task in TASKS
        for seed in DEVELOPMENT_SEEDS
    )
    for candidate in strict_failures:
        task, _, seed, _ = _row_key(candidate)
        dense = index[(task, DECAY_DENSE_METHOD, seed, None)]
        candidate_failures = set(_failure_components(candidate))
        dense_failures = set(_failure_components(dense))
        shared = sorted(candidate_failures & dense_failures)
        any_shared_failure = any_shared_failure or bool(shared)
        matched.append(
            {
                "task": task,
                "seed": seed,
                "candidate_failures": sorted(candidate_failures),
                "decay_dense_failures": sorted(dense_failures),
                "shared_failures": shared,
            }
        )
    if any_shared_failure:
        return {
            "classification": "HYBRID_BASIS_OR_OPTIMIZATION_SIDE",
            "recommended_mechanism": "NONE",
            "future_research_direction": "HYBRID_BASIS_OR_OPTIMIZATION_STUDY",
            "reason": (
                "a matched decay-dense arm shares a strict failure component; "
                "compression is not isolated as the cause"
            ),
            "matched_failed_cells": matched,
        }
    if not dense_all_stable:
        return {
            "classification": "INCONCLUSIVE",
            "recommended_mechanism": "NONE",
            "future_research_direction": "UNRESOLVED",
            "reason": (
                "decay-dense is not 9/9 stable, but its failures do not match "
                "the failed EF components"
            ),
            "matched_failed_cells": matched,
        }

    late_discordant = [
        candidate
        for candidate in strict_failures
        if _failure_occurs_in_late_window(candidate)
    ]
    evidence: list[dict[str, Any]] = []
    for candidate in late_discordant:
        key = _row_key(candidate)
        item = _diagnostic_evidence(summaries.get(key))
        item.update({"task": key[0], "seed": key[2]})
        evidence.append(item)
    if len(late_discordant) < 2:
        return {
            "classification": "INCONCLUSIVE",
            "recommended_mechanism": "NONE",
            "future_research_direction": "INSUFFICIENT_CROSS_CELL_EVIDENCE",
            "reason": (
                "fewer than two fresh cells have decay-dense success paired "
                "with a decay-EF failure in the frozen late window"
            ),
            "matched_failed_cells": matched,
            "late_discordant_cells": len(late_discordant),
            "diagnostic_evidence": evidence,
        }

    complete = bool(evidence) and all(
        item.get("complete") is True for item in evidence
    )
    residual_consistent = complete and all(
        item.get("residual_control_gate") is True for item in evidence
    )
    damping_consistent = complete and all(
        item.get("server_damping_gate") is True for item in evidence
    )
    if residual_consistent:
        future_direction = "RESIDUAL_CONTROL"
        reason = (
            "all late discordant EF cells pass the frozen residual-direction "
            "diagnostic gate"
        )
    elif damping_consistent:
        future_direction = "SERVER_DAMPING"
        reason = (
            "the residual gate is not consistent while all late discordant EF "
            "cells pass the frozen damping-direction diagnostic gate"
        )
    else:
        future_direction = "UNRESOLVED_COMPRESSION_DYNAMICS"
        reason = (
            "compression-specific failure is isolated, but neither frozen "
            "diagnostic gate is consistent across the late discordant cells"
        )
    return {
        "classification": "COMPRESSION_SPECIFIC",
        "recommended_mechanism": "NONE",
        "future_research_direction": future_direction,
        "reason": reason,
        "matched_failed_cells": matched,
        "late_discordant_cells": len(late_discordant),
        "diagnostic_evidence": evidence,
    }


def evaluate_stage_a(
    rows: Iterable[dict[str, Any]],
    summaries: dict[tuple[str, str, int, float | None], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gated = [apply_v4_gate(row) for row in rows]
    index: dict[tuple[str, str, int, float | None], dict[str, Any]] = {}
    duplicate_keys: list[str] = []
    for row in gated:
        key = _row_key(row)
        if key in index:
            duplicate_keys.append(str(key))
        else:
            index[key] = row
    expected = _expected_keys()
    valid = sum(
        key in index and index[key].get("technical_valid") is True
        for key in expected
    )
    matrix_complete = (
        not duplicate_keys
        and len(gated) == EXPECTED_RUNS
        and set(index) == expected
        and valid == EXPECTED_RUNS
    )
    summaries_by_method = [_summarize_method(index, method) for method in ALL_METHODS]
    candidate = next(
        row for row in summaries_by_method if row["method"] == CANDIDATE_METHOD
    )
    status = (
        "INCOMPLETE_OR_INVALID"
        if not matrix_complete
        else "PASS"
        if candidate["eligible"] is True
        else "NEGATIVE_RESULT"
    )
    return {
        "stage": "stage_a",
        "status": status,
        "candidate": CANDIDATE_METHOD,
        "selected_candidate": CANDIDATE_METHOD if status == "PASS" else None,
        "primary_matrix_valid": valid,
        "primary_matrix_expected": EXPECTED_RUNS,
        "duplicate_keys": duplicate_keys,
        "selection_rule": (
            "sole candidate must pass 9/9: technical VALID, C95-DenseScore reached, "
            "fixed-final >= target, zero post-hit violations, retention >= "
            "0.95, task TailDrop gate, and Speedup95 > 1"
        ),
        "summaries": summaries_by_method,
        "candidate_summary": candidate,
        "mechanism_routing": _mechanism_routing(
            index, summaries or {}, matrix_complete=matrix_complete
        ),
    }


def _csv_text(rows: list[dict[str, Any]], fields: Iterable[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _fmt(value: Any, digits: int = 6) -> str:
    if value == "" or value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown(
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    audit_errors: list[str],
    evaluation_policy: str,
) -> str:
    routing = decision["mechanism_routing"]
    policy_description = (
        "pooled train-only holdout"
        if evaluation_policy == "train_holdout"
        else "official validation"
    )
    lines = [
        "# Hybrid4 + Late-Decay Stability-v4 Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Protocol SHA-256: `{V4_PROTOCOL_SHA256}`",
        f"- Stage-A decision: **{decision['status']}**",
        f"- Audited matrix: {decision['primary_matrix_valid']}/{EXPECTED_RUNS} VALID",
        f"- Mechanism classification: `{routing['classification']}`",
        f"- Frozen-protocol next mechanism: `{routing['recommended_mechanism']}`",
        "- Descriptive future-study direction: "
        f"`{routing.get('future_research_direction', 'NONE')}`",
        f"- Evaluation policy: `{evaluation_policy}` ({policy_description})",
        f"- Technical audit errors: {len(audit_errors)}",
        "",
        "The sole candidate passes only when every one of nine paired cells "
        "passes strict stability and has `Speedup95 > 1`. Dense controls are "
        "mechanism controls and cannot replace the candidate.",
        "`Evaluation` below always means the split selected by the printed "
        "evaluation policy; it must not be read as official validation when "
        "the policy is `train_holdout`.",
        "",
        "## Per-seed cells",
        "",
        "| Task | Method | Wire | Seed | Metric | Target score | C95 round | C95 bytes | Speedup95 | Final score | Evaluation accuracy | Evaluation positive F1 | Retention | Post-hit | TailDrop | Strict stable | Stage-A pass |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=_row_key):
        lines.append(
            f"| {row.get('task', '')} | {row.get('method', '')} | "
            f"{row.get('wire_target', '')} | {row.get('seed', '')} | "
            f"{row.get('primary_metric_name', '')} | "
            f"{_fmt(row.get('target_primary_score'))} | {_fmt(row.get('c95_round'))} | "
            f"{_fmt(row.get('c95_wire_bytes'), 0)} | {_fmt(row.get('speedup95'), 3)} | "
            f"{_fmt(row.get('final_primary_score'))} | "
            f"{_fmt(row.get('evaluation_accuracy'))} | "
            f"{_fmt(row.get('evaluation_positive_f1'))} | "
            f"{_fmt(row.get('final_retention'))} | "
            f"{_fmt(row.get('post_hit_violation_count'))} | {_fmt(row.get('tail_drop'))} | "
            f"{'PASS' if row.get('strict_stable_success') else 'FAIL'} | "
            f"{'PASS' if row.get('stage_a_success') else '—'} |"
        )
    lines += [
        "",
        "## Method summary",
        "",
        "| Method | VALID | C95 | Strict stable | Speedup > 1 | Stage-A | Mean Speedup95 | Worst retention | Worst TailDrop | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in decision["summaries"]:
        lines.append(
            f"| {row['method']} | {row['technical_valid_cells']}/{EXPECTED_CELLS} | "
            f"{row['c95_reached_cells']}/{EXPECTED_CELLS} | {row['strict_stable_success_cells']}/{EXPECTED_CELLS} | "
            f"{row['speedup_gt_one_cells']}/{EXPECTED_CELLS} | {row['stage_a_success_cells']}/{EXPECTED_CELLS} | "
            f"{_fmt(row['mean_speedup95'], 3)} | "
            f"{_fmt(row['worst_final_retention'])} | "
            f"{_fmt(row['worst_tail_drop'])} | {row['eligible']} |"
        )
    lines += [
        "",
        "## Descriptive future-study routing",
        "",
        f"**{routing['classification']}** — {routing['reason']}",
        "",
        "The frozen V4 action remains `NONE`. The future-study direction is "
        f"`{routing.get('future_research_direction', 'NONE')}` and is descriptive "
        "only. It cannot trigger tuning, another V4 run, task/seed replacement, "
        "or mechanism selection on seeds 351/352/353.",
        "",
        "The complete audited per-run, per-round learning-rate schedule is in "
        "`STABILITY_V4_LR_TRACE.csv`.",
    ]
    evidence = routing.get("diagnostic_evidence", [])
    if evidence:
        lines += [
            "",
            "### Frozen late-window diagnostic evidence",
            "",
            "| Task | Seed | Complete | Previous / late window | Residual norm slope | Residual/update median (prev → late) | Tx cosine slope | Negative consecutive cosine (prev → late) | Update-norm ratio | Residual gate | Damping gate |",
            "|---|---:|---:|---|---:|---|---:|---|---:|---:|---:|",
        ]
        for item in evidence:
            lines.append(
                f"| {item.get('task', '')} | {item.get('seed', '')} | "
                f"{item.get('complete', False)} | "
                f"{item.get('previous_window', '—')} / {item.get('late_window', '—')} | "
                f"{_fmt(item.get('late_server_residual_norm_slope'))} | "
                f"{_fmt(item.get('previous_server_residual_to_update_median'))} → "
                f"{_fmt(item.get('late_server_residual_to_update_median'))} | "
                f"{_fmt(item.get('late_aggregate_to_transmitted_cosine_slope'))} | "
                f"{_fmt(item.get('previous_negative_consecutive_aggregate_cosine_fraction'))} → "
                f"{_fmt(item.get('late_negative_consecutive_aggregate_cosine_fraction'))} | "
                f"{_fmt(item.get('late_to_previous_aggregate_norm_ratio'))} | "
                f"{item.get('residual_control_gate', False)} | "
                f"{item.get('server_damping_gate', False)} |"
            )
    if audit_errors:
        lines += ["", "## Audit errors", ""] + [f"- {item}" for item in audit_errors]
    lines.append("")
    return "\n".join(lines)


def _protocol_errors(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    path = (
        Path(cfg["project"]["root"])
        / "docs"
        / "results"
        / "exploration"
        / "STABILITY_V4_PLAN.md"
    )
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return [f"cannot read v4 protocol: {exc}"]
    if actual != V4_PROTOCOL_SHA256:
        errors.append(f"v4 protocol file hash mismatch: {actual}")
    section = cfg.get("stability_v4")
    if not isinstance(section, dict):
        section = cfg.get("stability")
    if isinstance(section, dict):
        declared = section.get("protocol_sha256") or section.get(
            "v4_protocol_sha256"
        )
        if declared is not None and declared != V4_PROTOCOL_SHA256:
            errors.append("config v4 protocol hash differs from frozen plan")
    return errors


def _campaign_errors(cfg: dict[str, Any]) -> list[str]:
    """Require one complete, unmodified 45-run Stage-A campaign."""
    errors: list[str] = []
    output = Path(cfg["project"]["output_dir"])
    expected_launch_provenance: dict[str, str] | None = None
    try:
        expected_launch_provenance = verified_stability_launch_provenance(cfg, output)
    except (KeyError, RuntimeError) as exc:
        errors.append(f"preflight/launch provenance audit failed: {exc}")
    # run_matrix orders dense then sparse within every task/seed, rather than
    # concatenating the two method families globally.
    expected_ids = []
    for task in TASKS:
        for seed in DEVELOPMENT_SEEDS:
            expected_ids.extend(run_id(task, method, seed, None) for method in DENSE_METHODS)
            expected_ids.extend(
                run_id(task, method, seed, WIRE_RATIO) for method in SPARSE_METHODS
            )

    def load_json(path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {path.name}: {type(exc).__name__}: {exc}")
            return None
        if not isinstance(value, dict):
            errors.append(f"{path.name} is not a JSON object")
            return None
        return value

    campaign_config = load_json(output / "campaign_config.json")
    if campaign_config is not None:
        clean = {key: value for key, value in cfg.items() if not key.startswith("_")}
        if campaign_config.get("config") != clean:
            errors.append("campaign_config.json config differs from the frozen config")
        if campaign_config.get("requested_run_ids") != expected_ids:
            errors.append("campaign_config.json does not request the exact ordered 45 runs")
        if campaign_config.get("include_diagnostics") is not False:
            errors.append("campaign_config.json unexpectedly enables diagnostics")
        launch_provenance = campaign_config.get("launch_provenance")
        if not isinstance(launch_provenance, dict):
            errors.append("campaign_config.json lacks launch_provenance")
        elif (
            expected_launch_provenance is not None
            and launch_provenance != expected_launch_provenance
        ):
            errors.append(
                "campaign_config.json launch_provenance differs from the bound "
                "preflight/current bytes"
            )

    campaign_summary = load_json(output / "campaign_summary.json")
    if campaign_summary is not None:
        if campaign_summary.get("status") != "COMPLETE":
            errors.append("campaign_summary.json status is not COMPLETE")
        if campaign_summary.get("requested_runs") != EXPECTED_RUNS:
            errors.append("campaign_summary.json requested_runs is not 45")
        if campaign_summary.get("completed_runs") != expected_ids:
            errors.append("campaign_summary.json completed_runs differs from the frozen order")
        if campaign_summary.get("failed_runs") != []:
            errors.append("campaign_summary.json contains failed runs")

    events_path = output / "campaign_events.jsonl"
    events: list[dict[str, Any]] = []
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("event is not an object")
            events.append(value)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        errors.append(
            f"cannot read campaign_events.jsonl: {type(exc).__name__}: {exc}"
        )
    if not events:
        errors.append("campaign_events.jsonl is empty")
    else:
        expected_index = {identifier: index for index, identifier in enumerate(expected_ids, start=1)}
        next_run = 0
        active_run: str | None = None
        for event in events:
            identifier = event.get("run_id")
            kind = event.get("event")
            if identifier not in expected_index:
                errors.append(f"campaign event contains unexpected run_id: {identifier}")
                continue
            common_valid = (
                event.get("index") == expected_index[identifier]
                and event.get("total") == EXPECTED_RUNS
            )
            if not common_valid:
                errors.append(f"campaign event metadata differs for {identifier}")
            if kind == "START":
                expected = expected_ids[next_run] if next_run < len(expected_ids) else None
                if active_run is not None:
                    errors.append(
                        "campaign events are not globally serial: "
                        f"START for {identifier} precedes DONE for {active_run}"
                    )
                elif identifier != expected:
                    errors.append(
                        "campaign START order differs from the frozen 45-run order: "
                        f"expected {expected}, observed {identifier}"
                    )
                else:
                    active_run = str(identifier)
            elif kind == "RESUME":
                if active_run != identifier:
                    errors.append(
                        "RESUME without the globally active unfinished START: "
                        f"active={active_run}, observed={identifier}"
                    )
            elif kind == "DONE":
                if active_run != identifier or event.get("returncode") != 0:
                    errors.append(f"invalid or duplicate DONE for {identifier}")
                else:
                    active_run = None
                    next_run += 1
            else:
                errors.append(f"campaign contains prohibited event {kind!r}")
        if active_run is not None or next_run != len(expected_ids):
            errors.append("campaign event chain is incomplete")

    run_artifact_patterns = (
        "*/summary.json",
        "*/failure.json",
        "*/checkpoint.pt",
        "*/run_config.json",
        "*/launcher.log",
        "*/.lock",
        "*/round-*.json",
    )
    run_artifact_dirs = {
        path.parent.name
        for pattern in run_artifact_patterns
        for path in output.glob(pattern)
    }
    unexpected_artifact_dirs = sorted(run_artifact_dirs - set(expected_ids))
    if unexpected_artifact_dirs:
        errors.append(
            "output contains unexpected run-artifact directories: "
            f"{unexpected_artifact_dirs}"
        )
    summary_paths = sorted(output.glob("*/summary.json"))
    if len(summary_paths) != EXPECTED_RUNS:
        errors.append(
            f"output contains {len(summary_paths)} summary files, expected {EXPECTED_RUNS}"
        )
    if list(output.glob("*/failure.json")):
        errors.append("output contains failure.json")
    if list(output.glob("*/.lock")):
        errors.append("output contains live or stale .lock files")
    if list(output.rglob("campaign.lock")):
        errors.append("output contains live or stale campaign.lock")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the preregistered Stability-v4 Stage-A report."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", help="directory under docs/results")
    args = parser.parse_args()

    base, _ = _report_modules()
    cfg = load_config(args.config)
    if cfg.get("stability", {}).get("generation") != 4 or cfg.get(
        "stability", {}
    ).get("stage") != "stage_a":
        raise ValueError("this report accepts exactly one stability-v4 Stage-A config")
    required = base._specs(cfg, diagnostic=False, role="required")
    optional = base._specs(cfg, diagnostic=True, role="optional")
    context = base._build_context(cfg, [*required, *optional])
    records, _, _ = base._collect_records(cfg, context)
    audit_errors: list[str] = list(context.errors)
    audit_errors.extend(
        f"{record.run_id}: {'; '.join(record.reasons)}"
        for record in records
        if record.audit_status != "VALID"
    )
    audit_errors.extend(_protocol_errors(cfg))
    audit_errors.extend(_campaign_errors(cfg))

    deduplicated: dict[str, base.AuditRecord] = {}
    for record in records:
        prior = deduplicated.get(record.run_id)
        if prior is None:
            deduplicated[record.run_id] = record
        elif prior.audit_status != record.audit_status or prior.summary != record.summary:
            audit_errors.append(f"conflicting duplicate run: {record.run_id}")
    rows = v4_rows(deduplicated.values())
    summaries = {
        (record.spec.task, record.spec.method, record.spec.seed, record.spec.wire_ratio): record.summary
        for record in deduplicated.values()
        if record.valid and record.spec is not None and record.summary is not None
    }
    decision = evaluate_stage_a(rows, summaries)
    learning_rate_rows = lr_trace_rows(summaries)
    if audit_errors:
        decision["status"] = "INVALID"
        decision["selected_candidate"] = None
        decision["audit_errors"] = audit_errors
        decision["mechanism_routing"] = {
            "classification": "BLOCKED",
            "recommended_mechanism": "NONE",
            "future_research_direction": "NONE",
            "reason": "technical audit errors prohibit progression",
        }

    output = Path(args.output) if args.output else Path(cfg["project"]["output_dir"])
    output = result_path(cfg, output)
    outputs = {
        "STABILITY_V4_BY_SEED.csv": _csv_text(rows, BY_SEED_FIELDS),
        "STABILITY_V4_SUMMARY.csv": _csv_text(
            decision["summaries"], SUMMARY_FIELDS
        ),
        "STABILITY_V4_LR_TRACE.csv": _csv_text(
            learning_rate_rows, LR_TRACE_FIELDS
        ),
        "STABILITY_V4_DECISION.json": json.dumps(
            decision, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        "STABILITY_V4_REPORT.md": _markdown(
            rows,
            decision,
            audit_errors,
            str(cfg.get("evaluation", {}).get("policy", "official_validation")),
        ),
    }
    for name, content in outputs.items():
        path = result_path(cfg, output / name)
        base._atomic_write(path, content)
        print(path)


if __name__ == "__main__":
    main()
