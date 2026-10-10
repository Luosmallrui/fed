#!/usr/bin/env python3
"""Merge audited phase tables without regenerating frozen phase reports."""

from __future__ import annotations

import csv
import math
import os
import tempfile
from io import StringIO
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
FORMAL_SOURCE = "7d3aa3d6476289aad3260ffceadc76f8bdca9cd03a0b3b53646579248f9a0e15"
PHASES = (
    ("formal_main", FORMAL_SOURCE, RESULTS / "roberta_screen"),
    (
        "fedavg_wire_supplement",
        "0ea826d7f46452c6e9c914a58bec135db4a5e50c4ee6e0f890fd3cf726adbb15",
        RESULTS / "fedavg_lora_wire",
    ),
    ("fixed_a_ablation", FORMAL_SOURCE, RESULTS / "fixed_a_ablation" / "formal"),
)
TASKS = ("sst2", "rte", "mnli")
SEEDS = (401, 402, 403)
RATIOS = (0.1, 0.2)
RESERVED_FIELDS = {"experiment_phase", "source_revision"}

LEGACY_RUNS_FIELDS = (
    "run_id",
    "role",
    "audit_status",
    "reasons",
    "task",
    "method",
    "seed",
    "wire_target",
    "schema_version",
    "declared_status",
    "round_count",
    "adapter_learning_rate",
    "head_learning_rate",
    "final_primary",
    "flasc_dense_server_primary",
    "mnli_matched",
    "mnli_mismatched",
    "flasc_dense_server_mnli_matched",
    "flasc_dense_server_mnli_mismatched",
    "delivered_wire_bytes",
    "actual_reference_ratio",
    "data_prepare_seconds",
    "spectral_seconds",
    "train_seconds",
    "compression_seconds",
    "eval_seconds",
    "runtime_seconds",
    "resume_count",
    "peak_gpu_memory_bytes",
    "basis_handshake_bytes_per_client",
    "basis_handshake_bytes_total",
    "fallback_basis_bytes_per_client",
    "fallback_basis_bytes_total",
    "config_hash",
    "source_hash",
    "model_hash",
    "dataset_hash",
    "partition_hash",
    "layout_hash",
    "initial_head_hash",
    "summary_path",
)
FIXED_A_RUNS_FIELDS = (
    "run_id",
    "role",
    "audit_status",
    "reasons",
    "task",
    "method",
    "seed",
    "wire_target",
    "schema_version",
    "declared_status",
    "round_count",
    "adapter_learning_rate",
    "head_learning_rate",
    "final_primary",
    "primary_metric_name",
    "task_accuracy",
    "task_positive_f1",
    "flasc_dense_server_primary",
    "mnli_matched",
    "mnli_mismatched",
    "flasc_dense_server_mnli_matched",
    "flasc_dense_server_mnli_mismatched",
    "delivered_wire_bytes",
    "actual_reference_ratio",
    "data_prepare_seconds",
    "spectral_seconds",
    "train_seconds",
    "compression_seconds",
    "eval_seconds",
    "runtime_seconds",
    "resume_count",
    "peak_gpu_memory_bytes",
    "stability_stage",
    "protocol_sha256",
    "diagnostic_recipe_version",
    "compressor_kind",
    "block_quota_hash",
    "diagnostic_layout_fingerprint",
    "basis_handshake_bytes_per_client",
    "basis_handshake_bytes_total",
    "fallback_basis_bytes_per_client",
    "fallback_basis_bytes_total",
    "config_hash",
    "source_hash",
    "model_hash",
    "dataset_hash",
    "partition_hash",
    "layout_hash",
    "initial_head_hash",
    "summary_path",
)
LEGACY_COMMUNICATION_BY_SEED_FIELDS = (
    "task",
    "method",
    "wire_target",
    "seed",
    "metric_semantics",
    "reference_method",
    "reference_status",
    "interpretation",
    "accuracy",
    "reference_accuracy",
    "delta_accuracy_vs_fedavg",
    "accuracy_retention_vs_fedavg",
    "normal_path_wire_bytes",
    "reference_normal_path_wire_bytes",
    "wire_fraction_vs_fedavg",
    "saving_vs_fedavg",
    "descriptive_cer",
    "c95_dense_target_accuracy",
    "c95_dense_status",
    "c95_dense_round",
    "c95_dense_wire_bytes",
    "fedavg_c95_dense_round",
    "fedavg_c95_dense_wire_bytes",
    "speedup95_vs_fedavg",
)
FIXED_A_COMMUNICATION_BY_SEED_FIELDS = (
    "task",
    "method",
    "wire_target",
    "seed",
    "primary_metric_name",
    "primary_score",
    "reference_primary_score",
    "delta_primary_score_vs_fedavg",
    "primary_score_retention_vs_fedavg",
    "metric_semantics",
    "reference_method",
    "reference_status",
    "interpretation",
    "accuracy",
    "reference_accuracy",
    "delta_accuracy_vs_fedavg",
    "accuracy_retention_vs_fedavg",
    "normal_path_wire_bytes",
    "reference_normal_path_wire_bytes",
    "wire_fraction_vs_fedavg",
    "saving_vs_fedavg",
    "descriptive_cer",
    "c95_dense_target_accuracy",
    "c95_dense_target_primary_score",
    "c95_dense_status",
    "c95_dense_round",
    "c95_dense_wire_bytes",
    "fedavg_c95_dense_round",
    "fedavg_c95_dense_wire_bytes",
    "speedup95_vs_fedavg",
)
LEGACY_COMMUNICATION_FIELDS = (
    "task",
    "method",
    "wire_target",
    "n",
    "expected_n",
    "complete",
    "metric_semantics",
    "reference_method",
    "paired_reference_n",
    "reference_complete",
    "interpretation",
    "mean_accuracy",
    "sd_accuracy",
    "mean_normal_path_wire_bytes",
    "sd_normal_path_wire_bytes",
    "mean_wire_fraction_vs_fedavg",
    "sd_wire_fraction_vs_fedavg",
    "mean_saving_vs_fedavg",
    "sd_saving_vs_fedavg",
    "mean_delta_accuracy_vs_fedavg",
    "sd_delta_accuracy_vs_fedavg",
    "mean_accuracy_retention_vs_fedavg",
    "sd_accuracy_retention_vs_fedavg",
    "mean_descriptive_cer",
    "sd_descriptive_cer",
    "c95_dense_reached_n",
    "mean_c95_dense_round",
    "sd_c95_dense_round",
    "mean_c95_dense_wire_bytes",
    "sd_c95_dense_wire_bytes",
    "mean_fedavg_c95_dense_wire_bytes",
    "sd_fedavg_c95_dense_wire_bytes",
    "mean_speedup95_vs_fedavg",
    "sd_speedup95_vs_fedavg",
    "pareto_status",
)
FIXED_A_COMMUNICATION_FIELDS = (
    "task",
    "method",
    "wire_target",
    "n",
    "expected_n",
    "complete",
    "metric_semantics",
    "reference_method",
    "paired_reference_n",
    "reference_complete",
    "interpretation",
    "primary_metric_name",
    "mean_primary_score",
    "sd_primary_score",
    "mean_accuracy",
    "sd_accuracy",
    "mean_normal_path_wire_bytes",
    "sd_normal_path_wire_bytes",
    "mean_wire_fraction_vs_fedavg",
    "sd_wire_fraction_vs_fedavg",
    "mean_saving_vs_fedavg",
    "sd_saving_vs_fedavg",
    "mean_delta_accuracy_vs_fedavg",
    "sd_delta_accuracy_vs_fedavg",
    "mean_delta_primary_score_vs_fedavg",
    "sd_delta_primary_score_vs_fedavg",
    "mean_accuracy_retention_vs_fedavg",
    "sd_accuracy_retention_vs_fedavg",
    "mean_primary_score_retention_vs_fedavg",
    "sd_primary_score_retention_vs_fedavg",
    "mean_descriptive_cer",
    "sd_descriptive_cer",
    "c95_dense_reached_n",
    "mean_c95_dense_round",
    "sd_c95_dense_round",
    "mean_c95_dense_wire_bytes",
    "sd_c95_dense_wire_bytes",
    "mean_fedavg_c95_dense_wire_bytes",
    "sd_fedavg_c95_dense_wire_bytes",
    "mean_speedup95_vs_fedavg",
    "sd_speedup95_vs_fedavg",
    "pareto_status",
)
PAIRED_FIELDS = (
    "task",
    "wire_ratio",
    "seed",
    "comparison",
    "left_method",
    "right_method",
    "left_metric_semantics",
    "right_metric_semantics",
    "pair_status",
    "comparable",
    "interpretation",
    "left_score",
    "right_score",
    "delta",
    "threshold",
)
JOINT_FIELDS = (
    "comparison",
    "task",
    "wire_ratio",
    "seed",
    "left_method",
    "right_method",
    "left_score",
    "right_score",
    "delta",
    "threshold",
    "comparable",
    "left_summary",
    "right_summary",
)
LEGACY_TABLE_FIELDS = {
    "RUNS.csv": LEGACY_RUNS_FIELDS,
    "COMMUNICATION_EFFICIENCY_BY_SEED.csv": LEGACY_COMMUNICATION_BY_SEED_FIELDS,
    "COMMUNICATION_EFFICIENCY.csv": LEGACY_COMMUNICATION_FIELDS,
}
FIXED_A_TABLE_FIELDS = {
    "RUNS.csv": FIXED_A_RUNS_FIELDS,
    "COMMUNICATION_EFFICIENCY_BY_SEED.csv": FIXED_A_COMMUNICATION_BY_SEED_FIELDS,
    "COMMUNICATION_EFFICIENCY.csv": FIXED_A_COMMUNICATION_FIELDS,
}
PHASE_TABLE_FIELDS = {
    "formal_main": LEGACY_TABLE_FIELDS,
    "fedavg_wire_supplement": LEGACY_TABLE_FIELDS,
    "fixed_a_ablation": FIXED_A_TABLE_FIELDS,
}
# Root schemas are deliberately declared, ordered supersets.  Do not replace
# these with a runtime union: an unexpected phase column must remain a hard
# schema failure instead of silently becoming part of the published contract.
OUTPUT_TABLE_FIELDS = {
    "RUNS.csv": FIXED_A_RUNS_FIELDS,
    "COMMUNICATION_EFFICIENCY_BY_SEED.csv": FIXED_A_COMMUNICATION_BY_SEED_FIELDS,
    "COMMUNICATION_EFFICIENCY.csv": FIXED_A_COMMUNICATION_FIELDS,
}
# Backward-compatible names for callers constructing the two legacy phases.
RUNS_FIELDS = LEGACY_RUNS_FIELDS
COMMUNICATION_BY_SEED_FIELDS = LEGACY_COMMUNICATION_BY_SEED_FIELDS
COMMUNICATION_FIELDS = LEGACY_COMMUNICATION_FIELDS
PHASE_METHODS = {
    "formal_main": (
        ("fedavg_lora", "ffa_lora", "lora_xs"),
        ("flasc", "random_a_ef", "spectral_a_ef"),
    ),
    "fedavg_wire_supplement": (
        ("fedavg_lora",),
        ("full_lora_ef",),
    ),
    "fixed_a_ablation": (
        ("random_a_dense", "spectral_a_dense"),
        ("random_a_topk", "spectral_a_topk"),
    ),
}
EXPECTED_TOTALS = {
    "RUNS.csv": 162,
    "COMMUNICATION_EFFICIENCY_BY_SEED.csv": 162,
    "COMMUNICATION_EFFICIENCY.csv": 54,
    "PAIRED_DELTAS.csv": 135,
}


def _read(path: Path, expected_fields: Iterable[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual = tuple(reader.fieldnames or ())
        expected = tuple(expected_fields)
        if actual != expected:
            raise RuntimeError(f"schema mismatch for {path}: {actual} != {expected}")
        if RESERVED_FIELDS.intersection(actual):
            raise RuntimeError(f"reserved merge fields appear in {path}")
        return list(reader)


def _csv_text(fields: Iterable[str], rows: Iterable[dict[str, str]]) -> str:
    field_list = list(fields)
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=field_list, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in field_list} for row in rows)
    return buffer.getvalue()


def _run_id(task: str, method: str, seed: int, ratio: float | None) -> str:
    suffix = "dense" if ratio is None else f"wire-{int(round(ratio * 100)):02d}"
    return f"{task}__{method}__seed-{seed}__{suffix}"


def _expected_runs(phase: str) -> dict[str, tuple[str, str, str, str]]:
    dense, sparse = PHASE_METHODS[phase]
    output: dict[str, tuple[str, str, str, str]] = {}
    for task in TASKS:
        for seed in SEEDS:
            for method in dense:
                identifier = _run_id(task, method, seed, None)
                output[identifier] = (task, method, str(seed), "dense")
            for method in sparse:
                for ratio in RATIOS:
                    identifier = _run_id(task, method, seed, ratio)
                    output[identifier] = (task, method, str(seed), str(ratio))
    return output


def _validate_unique(
    rows: Iterable[dict[str, str]], fields: tuple[str, ...], label: str
) -> set[tuple[str, ...]]:
    keys: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if key in keys:
            raise RuntimeError(f"duplicate {label} key: {key}")
        keys.add(key)
    return keys


def _validate_runs(
    phase: str, source_revision: str, rows: list[dict[str, str]]
) -> set[tuple[str, str, str, str]]:
    expected = _expected_runs(phase)
    actual_ids = _validate_unique(rows, ("run_id",), f"{phase} RUNS")
    if {key[0] for key in actual_ids} != set(expected):
        raise RuntimeError(f"{phase} RUNS run_id set differs from the frozen matrix")
    identities: set[tuple[str, str, str, str]] = set()
    for row in rows:
        identifier = row["run_id"]
        identity = (row["task"], row["method"], row["seed"], row["wire_target"])
        if identity != expected[identifier]:
            raise RuntimeError(f"{phase} RUNS identity mismatch for {identifier}")
        if row["role"] != "required":
            raise RuntimeError(f"{phase} contains non-required run {identifier}")
        if row["audit_status"] != "VALID" or row["declared_status"] != "COMPLETE":
            raise RuntimeError(f"{phase} contains non-VALID run {identifier}")
        if row["source_hash"] != source_revision:
            raise RuntimeError(f"{phase} source hash mismatch for {identifier}")
        identities.add(identity)
    return identities


def _validate_generic_phase(
    phase: str,
    source_revision: str,
    directory: Path,
) -> dict[str, list[dict[str, str]]]:
    try:
        phase_fields = PHASE_TABLE_FIELDS[phase]
    except KeyError as exc:
        raise RuntimeError(f"no declared table schemas for phase {phase}") from exc
    tables = {
        name: _read(directory / name, fields)
        for name, fields in phase_fields.items()
    }
    run_identities = _validate_runs(phase, source_revision, tables["RUNS.csv"])
    by_seed = tables["COMMUNICATION_EFFICIENCY_BY_SEED.csv"]
    by_seed_keys = _validate_unique(
        by_seed,
        ("task", "method", "seed", "wire_target"),
        f"{phase} communication-by-seed",
    )
    if by_seed_keys != run_identities:
        raise RuntimeError(f"{phase} communication-by-seed identities differ from RUNS")

    efficiency = tables["COMMUNICATION_EFFICIENCY.csv"]
    efficiency_keys = _validate_unique(
        efficiency,
        ("task", "method", "wire_target"),
        f"{phase} communication summary",
    )
    expected_groups = {(task, method, wire) for task, method, _, wire in run_identities}
    if efficiency_keys != expected_groups:
        raise RuntimeError(f"{phase} communication groups differ from RUNS")
    for row in efficiency:
        if row["n"] != "3" or row["expected_n"] != "3" or row["complete"] != "True":
            raise RuntimeError(f"{phase} contains an incomplete communication group")
    return tables


def build_generic_outputs(
    phases: Iterable[tuple[str, str, Path]] = PHASES,
) -> dict[str, str]:
    phase_tables: list[
        tuple[str, str, dict[str, list[dict[str, str]]]]
    ] = []
    for phase, source_revision, directory in phases:
        phase_tables.append(
            (
                phase,
                source_revision,
                _validate_generic_phase(phase, source_revision, directory),
            )
        )
    outputs: dict[str, str] = {}
    for name, output_fields in OUTPUT_TABLE_FIELDS.items():
        merged: list[dict[str, str]] = []
        for phase, source_revision, tables in phase_tables:
            merged.extend(
                {
                    **row,
                    "experiment_phase": phase,
                    "source_revision": source_revision,
                }
                for row in tables[name]
            )
        merged_fields = ("experiment_phase", "source_revision", *output_fields)
        outputs[name] = _csv_text(merged_fields, merged)
    return outputs


def _expected_formal_pair_keys() -> set[tuple[str, str, str, str]]:
    output = {
        (task, str(ratio), str(seed), comparison)
        for task in TASKS
        for ratio in RATIOS
        for seed in SEEDS
        for comparison in (
            "spectral_vs_random",
            "proposed_vs_flasc",
            "random_ef_vs_noef",
            "spectral_ef_vs_noef",
        )
    }
    output.update(
        (task, "", str(seed), "dense_spectral_vs_random")
        for task in TASKS
        for seed in SEEDS
    )
    return output


def _expected_joint_keys() -> set[tuple[str, str, str, str]]:
    output = {
        ("dense_spectral_minus_random", task, "dense", str(seed))
        for task in TASKS
        for seed in SEEDS
    }
    output.update(
        (comparison, task, str(ratio), str(seed))
        for comparison in (
            "random_ef_minus_noef",
            "spectral_ef_minus_noef",
            "noef_spectral_minus_random",
            "basis_by_ef_difference_in_differences",
        )
        for task in TASKS
        for ratio in RATIOS
        for seed in SEEDS
    )
    return output


def _validate_joint_rows(rows: list[dict[str, str]]) -> None:
    keys = _validate_unique(
        rows,
        ("comparison", "task", "wire_ratio", "seed"),
        "fixed-A joint pair",
    )
    if keys != _expected_joint_keys():
        raise RuntimeError("fixed-A joint pair keys differ from the frozen 81-row set")
    expected_methods = {
        "dense_spectral_minus_random": ("spectral_a_dense", "random_a_dense"),
        "random_ef_minus_noef": ("random_a_ef", "random_a_topk"),
        "spectral_ef_minus_noef": ("spectral_a_ef", "spectral_a_topk"),
        "noef_spectral_minus_random": ("spectral_a_topk", "random_a_topk"),
        "basis_by_ef_difference_in_differences": (
            "spectral_minus_random_with_ef",
            "spectral_minus_random_without_ef",
        ),
    }
    for row in rows:
        comparison = row["comparison"]
        if (row["left_method"], row["right_method"]) != expected_methods[comparison]:
            raise RuntimeError(f"joint method direction differs for {comparison}")
        if row["comparable"] != "True":
            raise RuntimeError(f"joint pair is not comparable: {comparison}")
        if not row["left_summary"] or not row["right_summary"]:
            raise RuntimeError(f"joint pair lacks source summaries: {comparison}")
        try:
            left = float(row["left_score"])
            right = float(row["right_score"])
            delta = float(row["delta"])
            threshold = float(row["threshold"])
        except ValueError as exc:
            raise RuntimeError(f"joint pair has a non-numeric score: {comparison}") from exc
        if not all(math.isfinite(value) for value in (left, right, delta, threshold)):
            raise RuntimeError(f"joint pair has a non-finite score: {comparison}")
        if not math.isclose(delta, left - right, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"joint pair delta is inconsistent: {comparison}")


def build_paired_output(results: Path = RESULTS) -> str:
    formal = _read(results / "roberta_screen" / "PAIRED_DELTAS.csv", PAIRED_FIELDS)
    supplement = _read(
        results / "fedavg_lora_wire" / "PAIRED_DELTAS.csv", PAIRED_FIELDS
    )
    formal_keys = _validate_unique(
        formal, ("task", "wire_ratio", "seed", "comparison"), "formal pair"
    )
    if formal_keys != _expected_formal_pair_keys():
        raise RuntimeError("formal paired rows differ from the frozen 81-row set")
    supplement_keys = _validate_unique(
        supplement,
        ("task", "wire_ratio", "seed", "comparison"),
        "supplement pair",
    )
    expected_supplement = {
        (task, str(ratio), str(seed), "full_lora_ef_vs_fedavg")
        for task in TASKS
        for ratio in RATIOS
        for seed in SEEDS
    }
    if supplement_keys != expected_supplement:
        raise RuntimeError("supplement paired rows differ from the frozen 18-row set")
    if any(row["pair_status"] != "COMPLETE" for row in supplement):
        raise RuntimeError("supplement contains a non-COMPLETE pair")

    missing_mechanism = {
        "random_ef_vs_noef",
        "spectral_ef_vs_noef",
        "dense_spectral_vs_random",
    }
    retained_formal: list[dict[str, str]] = []
    removed = 0
    for row in formal:
        if row["comparison"] in missing_mechanism:
            if row["pair_status"] != "MISSING":
                raise RuntimeError("formal mechanism placeholder is not MISSING")
            removed += 1
        else:
            if row["pair_status"] != "COMPLETE":
                raise RuntimeError("retained formal pair is not COMPLETE")
            retained_formal.append(row)
    if removed != 45 or len(retained_formal) != 36:
        raise RuntimeError("formal paired replacement boundary is not 45/36")

    joint = _read(
        results / "fixed_a_ablation" / "formal" / "JOINT_PAIRED_BY_SEED.csv",
        JOINT_FIELDS,
    )
    _validate_joint_rows(joint)
    merged: list[dict[str, str]] = [
        {
            **row,
            "experiment_phase": "formal_main",
            "source_revision": FORMAL_SOURCE,
        }
        for row in retained_formal
    ]
    merged.extend(
        {
            **row,
            "experiment_phase": "fedavg_wire_supplement",
            "source_revision": PHASES[1][1],
        }
        for row in supplement
    )

    name_map = {
        "dense_spectral_minus_random": "dense_spectral_vs_random",
        "random_ef_minus_noef": "random_ef_vs_noef",
        "spectral_ef_minus_noef": "spectral_ef_vs_noef",
        "noef_spectral_minus_random": "noef_spectral_vs_random",
        "basis_by_ef_difference_in_differences": "basis_by_ef_difference_in_differences",
    }
    for row in joint:
        comparison = name_map[row["comparison"]]
        is_did = comparison == "basis_by_ef_difference_in_differences"
        wire = "" if row["wire_ratio"] == "dense" else row["wire_ratio"]
        merged.append(
            {
                "experiment_phase": "fixed_a_joint",
                "source_revision": FORMAL_SOURCE,
                "task": row["task"],
                "wire_ratio": wire,
                "seed": row["seed"],
                "comparison": comparison,
                "left_method": row["left_method"],
                "right_method": row["right_method"],
                "left_metric_semantics": (
                    "paired_basis_delta" if is_did else "communicated_global"
                ),
                "right_metric_semantics": (
                    "paired_basis_delta" if is_did else "communicated_global"
                ),
                "pair_status": "COMPLETE",
                "comparable": "True",
                "interpretation": (
                    "DIFFERENCE_IN_DIFFERENCES" if is_did else "COMPARABLE"
                ),
                "left_score": row["left_score"],
                "right_score": row["right_score"],
                "delta": row["delta"],
                "threshold": row["threshold"],
            }
        )
    final_keys = _validate_unique(
        merged,
        ("experiment_phase", "task", "wire_ratio", "seed", "comparison"),
        "combined pair",
    )
    if len(final_keys) != EXPECTED_TOTALS["PAIRED_DELTAS.csv"]:
        raise RuntimeError("combined paired row count differs from 135")
    fields = ("experiment_phase", "source_revision", *PAIRED_FIELDS)
    return _csv_text(fields, merged)


def build_outputs(results: Path = RESULTS) -> dict[str, str]:
    phases = tuple(
        (phase, source, results / directory.relative_to(RESULTS))
        for phase, source, directory in PHASES
    )
    outputs = build_generic_outputs(phases)
    outputs["PAIRED_DELTAS.csv"] = build_paired_output(results)
    counts: dict[str, int] = {}
    for name, content in outputs.items():
        counts[name] = len(list(csv.DictReader(StringIO(content))))
    if counts != EXPECTED_TOTALS:
        raise RuntimeError(f"combined table counts differ: {counts} != {EXPECTED_TOTALS}")
    return outputs


def commit_outputs(outputs: dict[str, str], results: Path = RESULTS) -> None:
    """Stage every output first; replace roots only after all writes succeed."""
    if set(outputs) != set(EXPECTED_TOTALS):
        raise RuntimeError("refusing to commit an incomplete combined report set")
    temporary_paths: dict[str, Path] = {}
    try:
        for name, content in outputs.items():
            target = results / name
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=target.parent)
            temporary_path = Path(temporary)
            temporary_paths[name] = temporary_path
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for name in EXPECTED_TOTALS:
            os.replace(temporary_paths.pop(name), results / name)
    finally:
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)


def main() -> None:
    outputs = build_outputs(RESULTS)
    commit_outputs(outputs, RESULTS)
    counts = {
        name: len(list(csv.DictReader(StringIO(content))))
        for name, content in outputs.items()
    }
    print(counts)


if __name__ == "__main__":
    main()
