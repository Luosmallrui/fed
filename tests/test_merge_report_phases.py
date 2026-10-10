from __future__ import annotations

import csv
import tempfile
import unittest
from io import StringIO
from pathlib import Path

from scripts import merge_report_phases as merge


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def blank_row(fields) -> dict[str, str]:
    return {field: "" for field in fields}


def make_generic_phase(results: Path, phase: str, source: str, relative: str) -> None:
    directory = results / relative
    expected = merge._expected_runs(phase)
    schemas = merge.PHASE_TABLE_FIELDS[phase]
    runs_fields = schemas["RUNS.csv"]
    by_seed_fields = schemas["COMMUNICATION_EFFICIENCY_BY_SEED.csv"]
    communication_fields = schemas["COMMUNICATION_EFFICIENCY.csv"]
    run_rows = []
    by_seed_rows = []
    groups: set[tuple[str, str, str]] = set()
    for identifier, (task, method, seed, wire) in expected.items():
        run = blank_row(runs_fields)
        run.update(
            {
                "run_id": identifier,
                "role": "required",
                "audit_status": "VALID",
                "task": task,
                "method": method,
                "seed": seed,
                "wire_target": wire,
                "declared_status": "COMPLETE",
                "source_hash": source,
            }
        )
        if phase == "fixed_a_ablation":
            run.update(
                {
                    "primary_metric_name": "accuracy",
                    "task_accuracy": "0.8",
                    "task_positive_f1": "",
                    "stability_stage": "",
                    "protocol_sha256": "",
                    "diagnostic_recipe_version": "",
                    "compressor_kind": "global_topk",
                    "block_quota_hash": "",
                    "diagnostic_layout_fingerprint": "layout-diagnostic",
                }
            )
        run_rows.append(run)
        by_seed = blank_row(by_seed_fields)
        by_seed.update(
            {"task": task, "method": method, "seed": seed, "wire_target": wire}
        )
        if phase == "fixed_a_ablation":
            by_seed.update(
                {
                    "primary_metric_name": "accuracy",
                    "primary_score": "0.8",
                    "reference_primary_score": "0.8",
                    "delta_primary_score_vs_fedavg": "0.0",
                    "primary_score_retention_vs_fedavg": "1.0",
                    "c95_dense_target_primary_score": "0.76",
                }
            )
        by_seed_rows.append(by_seed)
        groups.add((task, method, wire))
    summary_rows = []
    for task, method, wire in groups:
        row = blank_row(communication_fields)
        row.update(
            {
                "task": task,
                "method": method,
                "wire_target": wire,
                "n": "3",
                "expected_n": "3",
                "complete": "True",
            }
        )
        if phase == "fixed_a_ablation":
            row.update(
                {
                    "primary_metric_name": "accuracy",
                    "mean_primary_score": "0.8",
                    "sd_primary_score": "0.0",
                    "mean_delta_primary_score_vs_fedavg": "0.0",
                    "sd_delta_primary_score_vs_fedavg": "0.0",
                    "mean_primary_score_retention_vs_fedavg": "1.0",
                    "sd_primary_score_retention_vs_fedavg": "0.0",
                }
            )
        summary_rows.append(row)
    write_csv(directory / "RUNS.csv", runs_fields, run_rows)
    write_csv(
        directory / "COMMUNICATION_EFFICIENCY_BY_SEED.csv",
        by_seed_fields,
        by_seed_rows,
    )
    write_csv(
        directory / "COMMUNICATION_EFFICIENCY.csv",
        communication_fields,
        summary_rows,
    )


def make_paired_tables(results: Path) -> None:
    formal_rows = []
    missing = {
        "random_ef_vs_noef",
        "spectral_ef_vs_noef",
        "dense_spectral_vs_random",
    }
    for task, wire, seed, comparison in merge._expected_formal_pair_keys():
        row = blank_row(merge.PAIRED_FIELDS)
        row.update(
            {
                "task": task,
                "wire_ratio": wire,
                "seed": seed,
                "comparison": comparison,
                "pair_status": "MISSING" if comparison in missing else "COMPLETE",
            }
        )
        formal_rows.append(row)
    write_csv(
        results / "roberta_screen" / "PAIRED_DELTAS.csv",
        merge.PAIRED_FIELDS,
        formal_rows,
    )

    supplement_rows = []
    for task in merge.TASKS:
        for ratio in merge.RATIOS:
            for seed in merge.SEEDS:
                row = blank_row(merge.PAIRED_FIELDS)
                row.update(
                    {
                        "task": task,
                        "wire_ratio": str(ratio),
                        "seed": str(seed),
                        "comparison": "full_lora_ef_vs_fedavg",
                        "pair_status": "COMPLETE",
                    }
                )
                supplement_rows.append(row)
    write_csv(
        results / "fedavg_lora_wire" / "PAIRED_DELTAS.csv",
        merge.PAIRED_FIELDS,
        supplement_rows,
    )

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
    joint_rows = []
    for comparison, task, wire, seed in merge._expected_joint_keys():
        row = blank_row(merge.JOINT_FIELDS)
        left_method, right_method = expected_methods[comparison]
        row.update(
            {
                "comparison": comparison,
                "task": task,
                "wire_ratio": wire,
                "seed": seed,
                "left_method": left_method,
                "right_method": right_method,
                "left_score": "0.8",
                "right_score": "0.7",
                "delta": "0.1",
                "threshold": "0.003",
                "comparable": "True",
                "left_summary": "left/summary.json",
                "right_summary": "right/summary.json",
            }
        )
        joint_rows.append(row)
    write_csv(
        results / "fixed_a_ablation" / "formal" / "JOINT_PAIRED_BY_SEED.csv",
        merge.JOINT_FIELDS,
        joint_rows,
    )


def make_results(results: Path) -> None:
    make_generic_phase(
        results, "formal_main", merge.FORMAL_SOURCE, "roberta_screen"
    )
    make_generic_phase(
        results,
        "fedavg_wire_supplement",
        merge.PHASES[1][1],
        "fedavg_lora_wire",
    )
    make_generic_phase(
        results,
        "fixed_a_ablation",
        merge.FORMAL_SOURCE,
        "fixed_a_ablation/formal",
    )
    make_paired_tables(results)


class MergeReportTests(unittest.TestCase):
    def test_valid_inputs_build_exact_counts_and_commit_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            make_results(results)
            outputs = merge.build_outputs(results)
            counts = {
                name: len(list(csv.DictReader(StringIO(content))))
                for name, content in outputs.items()
            }
            self.assertEqual(counts, merge.EXPECTED_TOTALS)
            for name in (
                "RUNS.csv",
                "COMMUNICATION_EFFICIENCY_BY_SEED.csv",
                "COMMUNICATION_EFFICIENCY.csv",
            ):
                reader = csv.DictReader(StringIO(outputs[name]))
                self.assertEqual(
                    tuple(reader.fieldnames or ()),
                    (
                        "experiment_phase",
                        "source_revision",
                        *merge.OUTPUT_TABLE_FIELDS[name],
                    ),
                )
            run_rows = list(csv.DictReader(StringIO(outputs["RUNS.csv"])))
            legacy = next(
                row for row in run_rows if row["experiment_phase"] == "formal_main"
            )
            fixed_a = next(
                row
                for row in run_rows
                if row["experiment_phase"] == "fixed_a_ablation"
            )
            self.assertEqual(legacy["primary_metric_name"], "")
            self.assertEqual(fixed_a["primary_metric_name"], "accuracy")
            self.assertEqual(fixed_a["compressor_kind"], "global_topk")
            self.assertEqual(
                fixed_a["diagnostic_layout_fingerprint"], "layout-diagnostic"
            )
            merge.commit_outputs(outputs, results)
            self.assertTrue(all((results / name).exists() for name in outputs))

    def test_duplicate_run_with_same_total_is_rejected_before_root_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            make_results(results)
            target = results / "fixed_a_ablation" / "formal" / "RUNS.csv"
            with target.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[-1] = dict(rows[0])
            write_csv(target, merge.FIXED_A_RUNS_FIELDS, rows)
            sentinel = results / "RUNS.csv"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "duplicate|run_id set"):
                merge.build_outputs(results)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")

    def test_bad_source_and_noncomparable_joint_pair_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            make_results(results)
            runs = results / "fedavg_lora_wire" / "RUNS.csv"
            with runs.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["source_hash"] = "wrong"
            write_csv(runs, merge.RUNS_FIELDS, rows)
            with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
                merge.build_outputs(results)

            make_results(results)
            joint = (
                results
                / "fixed_a_ablation"
                / "formal"
                / "JOINT_PAIRED_BY_SEED.csv"
            )
            with joint.open(encoding="utf-8", newline="") as handle:
                joint_rows = list(csv.DictReader(handle))
            joint_rows[0]["comparable"] = "False"
            write_csv(joint, merge.JOINT_FIELDS, joint_rows)
            with self.assertRaisesRegex(RuntimeError, "not comparable"):
                merge.build_outputs(results)

    def test_phase_specific_schema_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary) / "results"
            make_results(results)
            target = results / "fixed_a_ablation" / "formal" / "RUNS.csv"
            with target.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            fields = (*merge.FIXED_A_RUNS_FIELDS, "unexpected_column")
            for row in rows:
                row["unexpected_column"] = "must-not-be-published"
            write_csv(target, fields, rows)
            with self.assertRaisesRegex(RuntimeError, "schema mismatch"):
                merge.build_outputs(results)


if __name__ == "__main__":
    unittest.main()
