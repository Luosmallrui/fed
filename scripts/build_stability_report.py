#!/usr/bin/env python3
"""Build the preregistered second-generation stability report.

The generic report builder remains the authority for provenance, layout,
invariant, and exact-wire validation.  This module consumes those audited
records and applies the stronger seed-level stability gates from
STABILITY_PLAN.md.  It never chooses a checkpoint or averages away a failed
seed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from fedpeft.config import (
    STABILITY_CANDIDATES,
    STABILITY_PROTOCOL_SHA256,
    load_config,
    result_path,
)
try:  # Package import in tests; sibling import for direct script execution.
    from scripts import build_report as base
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI smoke test
    import build_report as base  # type: ignore[no-redef]


TAIL_DROP_LIMITS = {
    "sst2": 0.01,
    "rte": 0.02,
    "mnli": 0.01,
    "mrpc": 0.02,
    "qqp": 0.01,
}
DEVELOPMENT_SEEDS = (321, 322, 323)
CONFIRMATION_SEEDS = (501, 502, 503)
TASKS = ("sst2", "rte", "mnli")
WIRE_RATIOS = (0.1, 0.2)
ORIGINAL_METHOD = "spectral_a_ef"
REFERENCE_METHOD = "fedavg_lora"
MATCHED_RANDOM_CONTROL = "random_block_ef"

STABILITY_BY_SEED_FIELDS = (
    "task",
    "method",
    "wire_target",
    "seed",
    "audit_status",
    "technical_valid",
    "target_accuracy",
    "c95_status",
    "c95_round",
    "c95_wire_bytes",
    "fedavg_c95_round",
    "fedavg_c95_wire_bytes",
    "speedup95",
    "final_accuracy",
    "dense_final_accuracy",
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
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _evaluations(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in summary.get("rounds", [])
        if isinstance(row, dict) and row.get("primary_metric") is not None
    ]


def _first_hit(
    summary: dict[str, Any], target: float
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    evaluations = _evaluations(summary)
    for index, row in enumerate(evaluations):
        if float(row["primary_metric"]) >= target:
            return row, evaluations[index + 1 :]
    return None, []


def stability_cell(
    summary: dict[str, Any] | None,
    reference: dict[str, Any] | None,
    *,
    audit_status: str = "VALID",
    audit_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    """Compute one paired seed-level stability cell.

    A technical audit failure is terminal for the cell even when parseable
    metric values happen to be present.
    """
    declared = (
        summary.get("run", {}) if isinstance(summary, dict) else {}
    )
    task = str(declared.get("task", ""))
    method = str(declared.get("method", ""))
    seed = declared.get("seed", "")
    ratio = declared.get("wire_ratio")
    reasons = list(audit_reasons)
    technical_valid = audit_status == "VALID" and summary is not None
    reference_valid = reference is not None
    values: dict[str, Any] = {
        "task": task,
        "method": method,
        "wire_target": "dense" if ratio is None else float(ratio),
        "seed": seed,
        "audit_status": audit_status,
        "technical_valid": technical_valid,
        "target_accuracy": "",
        "c95_status": "INVALID" if not technical_valid else "MISSING_REFERENCE",
        "c95_round": "",
        "c95_wire_bytes": "",
        "fedavg_c95_round": "",
        "fedavg_c95_wire_bytes": "",
        "speedup95": "",
        "final_accuracy": "",
        "dense_final_accuracy": "",
        "final_retention": "",
        "final_normal_path_wire_bytes": "",
        "post_hit_violation_count": "",
        "post_hit_violation_rounds": "",
        "tail_drop": "",
        "tail_drop_limit": TAIL_DROP_LIMITS.get(task, ""),
        "actual_message_bytes": "",
        "message_cap_bytes": "",
        "message_cap_utilization": "",
        "stable_success": False,
        "communication_efficient_success": False,
        "reasons": "; ".join(reasons),
    }
    if not technical_valid:
        if not reasons:
            reasons.append("technical audit did not pass")
        values["reasons"] = "; ".join(reasons)
        return values
    assert summary is not None
    try:
        final_accuracy = float(summary["final"]["primary_metric"])
        final_wire = summary["final"]["wire"]
        actual_message = max(
            int(final_wire["max_uplink_message_bytes"]),
            int(final_wire["max_downlink_message_bytes"]),
        )
        message_cap = int(summary["wire_budget"]["message_cap_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        values["technical_valid"] = False
        values["audit_status"] = "INVALID"
        values["reasons"] = f"malformed audited endpoint: {exc}"
        return values
    values.update(
        {
            "final_accuracy": final_accuracy,
            "final_normal_path_wire_bytes": base._normal_path_wire_bytes(summary),
            "actual_message_bytes": actual_message,
            "message_cap_bytes": message_cap,
            "message_cap_utilization": actual_message / message_cap
            if message_cap > 0
            else "",
        }
    )
    if not reference_valid:
        values["reasons"] = "paired dense FedAvg-LoRA reference is unavailable"
        return values
    assert reference is not None
    try:
        dense_final = float(reference["final"]["primary_metric"])
    except (KeyError, TypeError, ValueError) as exc:
        values["reasons"] = f"paired dense endpoint is malformed: {exc}"
        return values
    if not math.isfinite(dense_final) or dense_final <= 0.0:
        values["reasons"] = "paired dense endpoint is non-positive or non-finite"
        return values

    target = 0.95 * dense_final
    hit, later = _first_hit(summary, target)
    dense_hit, _ = _first_hit(reference, target)
    evaluations = _evaluations(summary)
    tail = (
        max(float(row["primary_metric"]) for row in evaluations[-3:])
        - final_accuracy
        if evaluations
        else math.inf
    )
    violations = [
        int(row["round"])
        for row in later
        if float(row["primary_metric"]) < target
    ]
    retention = final_accuracy / dense_final
    values.update(
        {
            "target_accuracy": target,
            "dense_final_accuracy": dense_final,
            "final_retention": retention,
            "post_hit_violation_count": len(violations),
            "post_hit_violation_rounds": ",".join(map(str, violations)),
            "tail_drop": tail,
        }
    )
    if dense_hit is not None:
        values["fedavg_c95_round"] = int(dense_hit["round"])
        values["fedavg_c95_wire_bytes"] = base._normal_path_wire_bytes(
            reference, dense_hit
        )
    if hit is None:
        values["c95_status"] = "NOT_REACHED"
        reasons.append("C95 target was never reached")
    else:
        values["c95_status"] = "REACHED"
        values["c95_round"] = int(hit["round"])
        values["c95_wire_bytes"] = base._normal_path_wire_bytes(summary, hit)
        if dense_hit is not None and float(values["c95_wire_bytes"]) > 0.0:
            values["speedup95"] = float(
                values["fedavg_c95_wire_bytes"]
            ) / float(values["c95_wire_bytes"])
    if final_accuracy < target:
        reasons.append("final accuracy is below the paired C95 target")
    if violations:
        reasons.append("a scheduled evaluation fell below target after first hit")
    limit = TAIL_DROP_LIMITS.get(task)
    if limit is None:
        reasons.append(f"no TailDrop threshold is registered for task {task}")
    elif not math.isfinite(tail) or tail > limit:
        reasons.append(f"TailDrop exceeds {limit:.3f}")
    stable = (
        hit is not None
        and dense_hit is not None
        and final_accuracy >= target
        and not violations
        and limit is not None
        and tail <= limit
    )
    speedup = values["speedup95"]
    values["stable_success"] = stable
    values["communication_efficient_success"] = (
        stable and _finite(speedup) and float(speedup) > 1.0
    )
    values["reasons"] = "; ".join(reasons)
    return values


def stability_rows(records: Iterable[base.AuditRecord]) -> list[dict[str, Any]]:
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
    output: list[dict[str, Any]] = []
    for record in records:
        if record.spec is None:
            continue
        summary = record.summary
        if summary is None:
            summary = {"run": record.spec.run_dict}
        reference = references.get((record.spec.task, record.spec.seed))
        output.append(
            stability_cell(
                summary,
                reference,
                audit_status=record.audit_status,
                audit_reasons=record.reasons,
            )
        )
    return output


def _cell_index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int, float | None], dict[str, Any]]:
    output: dict[tuple[str, str, int, float | None], dict[str, Any]] = {}
    for row in rows:
        wire = row.get("wire_target")
        ratio = None if wire == "dense" else float(wire)
        try:
            key = (
                str(row["task"]),
                str(row["method"]),
                int(row["seed"]),
                ratio,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if key in output:
            raise ValueError(f"duplicate stability cell: {key}")
        output[key] = row
    return output


def _worst_cell(cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not cells:
        return None

    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        reached = row.get("c95_status") == "REACHED"
        c95_round = float(row["c95_round"]) if _finite(row.get("c95_round")) else math.inf
        retention = (
            float(row["final_retention"])
            if _finite(row.get("final_retention"))
            else -math.inf
        )
        tail = float(row["tail_drop"]) if _finite(row.get("tail_drop")) else math.inf
        return (
            not bool(row.get("technical_valid")),
            not bool(row.get("stable_success")),
            not reached,
            c95_round,
            -retention,
            tail,
            str(row.get("task")),
            int(row.get("seed", -1)),
        )

    return max(cells, key=key)


def _candidate_summary(
    index: dict[tuple[str, str, int, float | None], dict[str, Any]],
    method: str,
    *,
    tasks: Iterable[str],
    seeds: Iterable[int],
    ratios: Iterable[float],
) -> dict[str, Any]:
    expected = [
        (task, method, seed, ratio)
        for task in tasks
        for seed in seeds
        for ratio in ratios
    ]
    cells = [index[key] for key in expected if key in index]
    stable = [row for row in cells if row.get("stable_success") is True]
    efficient = [
        row for row in cells if row.get("communication_efficient_success") is True
    ]
    worst = _worst_cell(cells)
    reached_rounds = [
        int(row["c95_round"])
        for row in cells
        if row.get("c95_status") == "REACHED" and _finite(row.get("c95_round"))
    ]
    retentions = [
        float(row["final_retention"])
        for row in cells
        if _finite(row.get("final_retention"))
    ]
    tails = [
        float(row["tail_drop"])
        for row in cells
        if _finite(row.get("tail_drop"))
    ]
    speedups = [
        float(row["speedup95"])
        for row in cells
        if _finite(row.get("speedup95"))
    ]
    return {
        "method": method,
        "expected_cells": len(expected),
        "observed_cells": len(cells),
        "technical_valid_cells": sum(
            row.get("technical_valid") is True for row in cells
        ),
        "stable_success_cells": len(stable),
        "communication_efficient_cells": len(efficient),
        "eligible": len(cells) == len(expected) and len(stable) == len(expected),
        "communication_claim_ready": len(cells) == len(expected)
        and len(efficient) == len(expected),
        "worst_case_c95_round": max(reached_rounds)
        if len(reached_rounds) == len(expected)
        else "",
        "worst_case_final_retention": min(retentions)
        if len(retentions) == len(expected)
        else "",
        "worst_case_tail_drop": max(tails)
        if len(tails) == len(expected)
        else "",
        "mean_speedup95": statistics.mean(speedups)
        if len(speedups) == len(expected)
        else "",
        "worst_task": "" if worst is None else worst.get("task", ""),
        "worst_seed": "" if worst is None else worst.get("seed", ""),
        "worst_wire_target": ""
        if worst is None
        else worst.get("wire_target", ""),
    }


def evaluate_stage_a(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    index = _cell_index(rows)
    expected_matrix = {
        *((task, REFERENCE_METHOD, seed, None) for task in ("sst2", "rte") for seed in DEVELOPMENT_SEEDS),
        *((task, "spectral_a_dense", seed, None) for task in ("sst2", "rte") for seed in DEVELOPMENT_SEEDS),
        *((task, ORIGINAL_METHOD, seed, 0.1) for task in ("sst2", "rte") for seed in DEVELOPMENT_SEEDS),
        *((task, method, seed, 0.1) for task in ("sst2", "rte") for seed in DEVELOPMENT_SEEDS for method in STABILITY_CANDIDATES),
    }
    valid_matrix = sum(
        key in index and index[key].get("technical_valid") is True
        for key in expected_matrix
    )
    matrix_complete = len(expected_matrix) == 36 and valid_matrix == 36
    candidates = [
        _candidate_summary(
            index,
            method,
            tasks=("sst2", "rte"),
            seeds=DEVELOPMENT_SEEDS,
            ratios=(0.1,),
        )
        for method in STABILITY_CANDIDATES
    ]
    eligible = [row for row in candidates if row["eligible"]]
    priority = {method: rank for rank, method in enumerate(STABILITY_CANDIDATES)}
    eligible.sort(
        key=lambda row: (
            int(row["worst_case_c95_round"]),
            -float(row["worst_case_final_retention"]),
            float(row["worst_case_tail_drop"]),
            priority[str(row["method"])],
        )
    )
    winner = (
        None
        if not eligible or not matrix_complete
        else str(eligible[0]["method"])
    )
    status = (
        "INCOMPLETE_OR_INVALID"
        if not matrix_complete
        else "PASS"
        if winner is not None
        else "NEGATIVE_RESULT"
    )
    return {
        "stage": "stage_a",
        "status": status,
        "selected_candidate": winner,
        "primary_matrix_valid": valid_matrix,
        "primary_matrix_expected": 36,
        "selection_rule": (
            "6/6 stable success; then min worst C95 round, max worst final "
            "retention, min worst TailDrop, fixed P1/P2/P3 priority"
        ),
        "candidates": candidates,
    }


def evaluate_stage_b(
    rows: Iterable[dict[str, Any]], winner: str | None
) -> dict[str, Any]:
    rows = list(rows)
    stage_a = evaluate_stage_a(rows)
    if winner not in STABILITY_CANDIDATES:
        return {
            "stage": "stage_b",
            "status": "BLOCKED",
            "winner": winner,
            "reason": "no valid frozen Stage-A winner",
        }
    if stage_a.get("selected_candidate") != winner:
        return {
            "stage": "stage_b",
            "status": "BLOCKED",
            "winner": winner,
            "reason": "declared winner does not equal the deterministic Stage-A selection",
            "stage_a": stage_a,
        }
    index = _cell_index(rows)
    summary = _candidate_summary(
        index,
        winner,
        tasks=TASKS,
        seeds=DEVELOPMENT_SEEDS,
        ratios=WIRE_RATIOS,
    )
    expected_additions = {
        *((task, winner, seed, 0.2) for task in ("sst2", "rte") for seed in DEVELOPMENT_SEEDS),
        *(("mnli", REFERENCE_METHOD, seed, None) for seed in DEVELOPMENT_SEEDS),
        *(("mnli", winner, seed, ratio) for seed in DEVELOPMENT_SEEDS for ratio in WIRE_RATIOS),
    }
    valid_additions = sum(
        key in index and index[key].get("technical_valid") is True
        for key in expected_additions
    )
    additions_complete = len(expected_additions) == 15 and valid_additions == 15
    passed = (
        stage_a.get("primary_matrix_valid") == 36
        and additions_complete
        and summary["eligible"] is True
    )
    status = (
        "INCOMPLETE_OR_INVALID"
        if not additions_complete
        else "PASS"
        if passed
        else "NEGATIVE_RESULT"
    )
    return {
        "stage": "stage_b",
        "status": status,
        "winner": winner,
        "frozen_for_confirmation": passed,
        "stage_b_additions_valid": valid_additions,
        "stage_b_additions_expected": 15,
        "winner_summary": summary,
        "stage_a": stage_a,
    }


def evaluate_confirmation(
    rows: Iterable[dict[str, Any]], winner: str | None
) -> dict[str, Any]:
    rows = list(rows)
    if winner not in STABILITY_CANDIDATES:
        return {
            "stage": "confirmation",
            "status": "BLOCKED",
            "winner": winner,
            "reason": "confirmation requires a frozen Stage-B winner",
        }
    development_gate = evaluate_stage_b(rows, winner)
    if (
        development_gate.get("status") != "PASS"
        or development_gate.get("frozen_for_confirmation") is not True
    ):
        return {
            "stage": "confirmation",
            "status": "BLOCKED",
            "winner": winner,
            "reason": "confirmation requires the complete deterministic Stage-A/Stage-B freeze",
            "development_gate": development_gate,
        }
    index = _cell_index(rows)
    winner_summary = _candidate_summary(
        index,
        winner,
        tasks=TASKS,
        seeds=CONFIRMATION_SEEDS,
        ratios=WIRE_RATIOS,
    )
    expected_primary = {
        *((task, REFERENCE_METHOD, seed, None) for task in TASKS for seed in CONFIRMATION_SEEDS),
        *((task, ORIGINAL_METHOD, seed, ratio) for task in TASKS for seed in CONFIRMATION_SEEDS for ratio in WIRE_RATIOS),
        *((task, winner, seed, ratio) for task in TASKS for seed in CONFIRMATION_SEEDS for ratio in WIRE_RATIOS),
    }
    valid_primary = sum(
        key in index and index[key].get("technical_valid") is True
        for key in expected_primary
    )
    matrix_complete = len(expected_primary) == 45 and valid_primary == 45
    stable_pass = winner_summary["eligible"] is True
    communication_pass = winner_summary["communication_claim_ready"] is True
    random_control_expected = {
        (task, MATCHED_RANDOM_CONTROL, seed, ratio)
        for task in TASKS
        for seed in CONFIRMATION_SEEDS
        for ratio in WIRE_RATIOS
    }
    random_control_valid = sum(
        key in index and index[key].get("technical_valid") is True
        for key in random_control_expected
    )
    basis_claim_ready = winner != "spectral_block_ef" or random_control_valid == 18
    passed = matrix_complete and stable_pass
    status = (
        "INCOMPLETE_OR_INVALID"
        if not matrix_complete
        else "PASS"
        if stable_pass
        else "FAIL"
    )
    return {
        "stage": "confirmation",
        "status": status,
        "winner": winner,
        "primary_matrix_valid": valid_primary,
        "primary_matrix_expected": 45,
        "winner_summary": winner_summary,
        "stable_claim_ready": passed,
        "communication_efficiency_claim_ready": passed and communication_pass,
        "matched_random_control_valid": random_control_valid,
        "matched_random_control_expected": 18
        if winner == "spectral_block_ef"
        else 0,
        "basis_vs_random_claim_ready": passed and basis_claim_ready,
        "development_gate": development_gate,
    }


def _confirmation_config_errors(
    configs: Iterable[dict[str, Any]], winner: str | None
) -> list[str]:
    """Require the complete development evidence alongside confirmation.

    This is a CLI input gate in addition to the row-level Stage-A/Stage-B
    decision above.  It prevents a confirmation-only invocation from being
    labelled PASS even if its 45 rows are internally valid.
    """

    configs = list(configs)
    sections = [cfg.get("stability") for cfg in configs]
    errors: list[str] = []
    stage_a = [
        value for value in sections if isinstance(value, dict) and value.get("stage") == "stage_a"
    ]
    stage_b = [
        value for value in sections if isinstance(value, dict) and value.get("stage") == "stage_b"
    ]
    confirmation = [
        value
        for value in sections
        if isinstance(value, dict) and value.get("stage") == "confirmation"
    ]
    if len(stage_a) != 1:
        errors.append("confirmation requires exactly one Stage-A config")
    matching_stage_b = [
        value for value in stage_b if value.get("winner") == winner
    ]
    substages = [str(value.get("substage")) for value in matching_stage_b]
    if len(stage_b) != 2 or sorted(substages) != ["mnli_10_20", "sst_rte_20"]:
        errors.append(
            "confirmation requires exactly one of each frozen Stage-B substage "
            "config for its winner"
        )
    if len(confirmation) != 1:
        errors.append("confirmation requires exactly one confirmation config")
    elif confirmation[0].get("winner") != winner:
        errors.append("confirmation config winner differs from the requested winner")
    if any(value.get("winner") != winner for value in stage_b):
        errors.append("a Stage-B config declares a different winner")
    return errors


def _csv_text(rows: list[dict[str, Any]], fields: Iterable[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
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
    stage: str,
    rows: list[dict[str, Any]],
    decision: dict[str, Any],
    audit_errors: list[str],
) -> str:
    lines = [
        "# Spectral-A+EF Stability Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Stage: `{stage}`",
        f"- Protocol SHA-256: `{STABILITY_PROTOCOL_SHA256}`",
        f"- Decision: **{decision['status']}**",
        f"- Technical audit errors: {len(audit_errors)}",
        "",
        "A stable success requires technical validity, reaching the paired dense "
        "95% target, retaining it through the final checkpoint without any "
        "post-hit violation, and passing the preregistered TailDrop limit.",
        "",
        "## Per-seed cells",
        "",
        "| Task | Method | Wire | Seed | C95 | Speedup95 | Final acc. | Final retention | Post-hit | TailDrop | Message/cap | Stable |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda value: (
            str(value.get("task")),
            str(value.get("method")),
            str(value.get("wire_target")),
            int(value.get("seed", -1)),
        ),
    ):
        lines.append(
            f"| {row.get('task', '')} | {row.get('method', '')} | "
            f"{row.get('wire_target', '')} | {row.get('seed', '')} | "
            f"{_fmt(row.get('c95_round'))} | {_fmt(row.get('speedup95'), 3)} | "
            f"{_fmt(row.get('final_accuracy'))} | "
            f"{_fmt(row.get('final_retention'))} | "
            f"{_fmt(row.get('post_hit_violation_count'))} | "
            f"{_fmt(row.get('tail_drop'))} | "
            f"{_fmt(row.get('actual_message_bytes'), 0)}/{_fmt(row.get('message_cap_bytes'), 0)} | "
            f"{'PASS' if row.get('stable_success') else 'FAIL'} |"
        )
    candidate_rows = decision.get("candidates")
    if not isinstance(candidate_rows, list):
        candidate = decision.get("winner_summary")
        candidate_rows = [candidate] if isinstance(candidate, dict) else []
    if candidate_rows:
        lines += [
            "",
            "## Gate summary",
            "",
            "| Method | Stable | Eligible | Worst C95 | Worst retention | Worst TailDrop | Worst cell |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in candidate_rows:
            lines.append(
                f"| {row['method']} | {row['stable_success_cells']}/{row['expected_cells']} | "
                f"{row['eligible']} | {_fmt(row['worst_case_c95_round'])} | "
                f"{_fmt(row['worst_case_final_retention'])} | "
                f"{_fmt(row['worst_case_tail_drop'])} | "
                f"{row['worst_task']}/seed-{row['worst_seed']}/{row['worst_wire_target']} |"
            )
    if audit_errors:
        lines += ["", "## Audit errors", ""] + [
            f"- {value}" for value in audit_errors
        ]
    lines += [
        "",
        "The confirmation set is never used to change the frozen method. A "
        "basis-vs-random claim for `spectral_block_ef` additionally requires all "
        "18 matched `random_block_ef` controls.",
        "",
    ]
    return "\n".join(lines)


def _protocol_file_error(cfg: dict[str, Any]) -> str | None:
    path = (
        Path(cfg["project"]["root"])
        / "docs"
        / "results"
        / "exploration"
        / "STABILITY_PLAN.md"
    )
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"cannot read preregistered protocol: {exc}"
    if digest != STABILITY_PROTOCOL_SHA256:
        return f"protocol file hash mismatch: {digest}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument(
        "--stage", choices=("stage_a", "stage_b", "confirmation")
    )
    parser.add_argument("--winner", choices=STABILITY_CANDIDATES)
    parser.add_argument("--output", help="directory under docs/results")
    args = parser.parse_args()

    configs = [load_config(path) for path in args.config]
    protocol_sections = [cfg.get("stability") for cfg in configs]
    if any(value != protocol_sections[0] for value in protocol_sections[1:]):
        # Stage/winner legitimately change; immutable fields may not.
        immutable = ("protocol_sha256", "diagnostic_recipe_version", "candidates")
        reference = {
            key: protocol_sections[0].get(key) for key in immutable  # type: ignore[union-attr]
        }
        for value in protocol_sections[1:]:
            if not isinstance(value, dict) or {
                key: value.get(key) for key in immutable
            } != reference:
                raise SystemExit("stability protocol metadata differs across configs")

    all_records: list[base.AuditRecord] = []
    audit_errors: list[str] = []
    for cfg in configs:
        required = base._specs(cfg, diagnostic=False, role="required")
        optional = base._specs(cfg, diagnostic=True, role="optional")
        context = base._build_context(cfg, [*required, *optional])
        records, _, _ = base._collect_records(cfg, context)
        all_records.extend(records)
        audit_errors.extend(context.errors)
        audit_errors.extend(
            f"{record.run_id}: {'; '.join(record.reasons)}"
            for record in records
            if record.audit_status != "VALID" and record.role == "required"
        )
        protocol_error = _protocol_file_error(cfg)
        if protocol_error is not None:
            audit_errors.append(protocol_error)

    deduplicated: dict[str, base.AuditRecord] = {}
    for record in all_records:
        prior = deduplicated.get(record.run_id)
        if prior is None:
            deduplicated[record.run_id] = record
        elif prior.audit_status != record.audit_status or prior.summary != record.summary:
            audit_errors.append(f"conflicting duplicate run: {record.run_id}")
    rows = stability_rows(deduplicated.values())
    last_stability = configs[-1]["stability"]
    stage = args.stage or str(last_stability["stage"])
    winner = args.winner or last_stability.get("winner")
    if stage == "stage_a":
        decision = evaluate_stage_a(rows)
    elif stage == "stage_b":
        decision = evaluate_stage_b(rows, winner)
    elif stage == "confirmation":
        audit_errors.extend(_confirmation_config_errors(configs, winner))
        decision = evaluate_confirmation(rows, winner)
    else:
        raise SystemExit("smoke configurations do not produce a selection report")
    if audit_errors and decision.get("status") == "PASS":
        decision["status"] = "INVALID"
        decision["audit_errors"] = audit_errors

    cfg = configs[-1]
    output = (
        Path(args.output)
        if args.output
        else Path(cfg["project"]["output_dir"])
    )
    output = result_path(cfg, output)
    summary_rows = decision.get("candidates")
    if not isinstance(summary_rows, list):
        winner_summary = decision.get("winner_summary")
        summary_rows = [winner_summary] if isinstance(winner_summary, dict) else []
    outputs = {
        "STABILITY_BY_SEED.csv": _csv_text(rows, STABILITY_BY_SEED_FIELDS),
        "STABILITY_SUMMARY.csv": _csv_text(
            summary_rows,
            (
                "method",
                "expected_cells",
                "observed_cells",
                "technical_valid_cells",
                "stable_success_cells",
                "communication_efficient_cells",
                "eligible",
                "communication_claim_ready",
                "worst_case_c95_round",
                "worst_case_final_retention",
                "worst_case_tail_drop",
                "mean_speedup95",
                "worst_task",
                "worst_seed",
                "worst_wire_target",
            ),
        ),
        "STABILITY_DECISION.json": json.dumps(
            decision, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        "STABILITY_REPORT.md": _markdown(stage, rows, decision, audit_errors),
    }
    for name, content in outputs.items():
        path = result_path(cfg, output / name)
        base._atomic_write(path, content)
        print(path)


if __name__ == "__main__":
    main()
