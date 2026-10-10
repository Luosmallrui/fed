import unittest

from scripts.build_stability_v4_report import (
    ALL_METHODS,
    BY_SEED_FIELDS,
    CANDIDATE_METHOD,
    DECAY_DENSE_METHOD,
    DEVELOPMENT_SEEDS,
    EXPECTED_CELLS,
    EXPECTED_RUNS,
    TASKS,
    WIRE_RATIO,
    _csv_text,
    _markdown,
    apply_v4_gate,
    evaluate_stage_a,
)


def row(method, task, seed, *, strict=True, speedup=2.0, technical=True):
    ratio = None if method in {
        "fedavg_lora",
        "hybrid4_a_dense",
        "hybrid4_a_decay_dense",
    } else WIRE_RATIO
    return {
        "task": task,
        "method": method,
        "wire_target": "dense" if ratio is None else ratio,
        "seed": seed,
        "audit_status": "VALID" if technical else "INVALID",
        "technical_valid": technical,
        "target_accuracy": 0.76,
        "c95_status": "REACHED",
        "c95_round": 5,
        "c95_wire_bytes": 100,
        "fedavg_c95_round": 5,
        "fedavg_c95_wire_bytes": 200,
        "speedup95": speedup,
        "final_accuracy": 0.78,
        "dense_final_accuracy": 0.8,
        "final_retention": 0.975,
        "post_hit_violation_count": 0 if strict else 1,
        "post_hit_violation_rounds": "" if strict else "8",
        "tail_drop": 0.0,
        "tail_drop_limit": 0.02 if task == "mrpc" else 0.01,
        "stable_success": strict,
        "communication_efficient_success": strict and speedup > 1,
        "reasons": "",
    }


def complete_rows():
    return [
        row(method, task, seed)
        for task in TASKS
        for seed in DEVELOPMENT_SEEDS
        for method in ALL_METHODS
    ]


def diagnostic_summary(*, residual_pattern=False, damping_pattern=False):
    rounds = []
    for round_index in range(1, 11):
        late = round_index > 8
        residual_ratio = 2.0 if residual_pattern and late else 1.0
        server_residual_norm = (
            float(round_index - 8) if residual_pattern and late else 1.0
        )
        aggregate_to_transmitted = (
            1.0 - 0.2 * (round_index - 8)
            if residual_pattern and late
            else 0.8
        )
        consecutive = -0.5 if damping_pattern and late else 0.5
        aggregate_norm = 2.0 if damping_pattern and late else 1.0
        rounds.append(
            {
                "round": round_index,
                "diagnostics": {
                    "adapter_update_norm": aggregate_norm,
                    "server_residual_norm": server_residual_norm,
                    "stability_diagnostics": {
                        "server": {
                            "residual": {
                                "overall": {
                                    "residual_to_update_ratio": residual_ratio,
                                }
                            }
                        },
                        "updates": {
                            "aggregate_to_transmitted_cosine": aggregate_to_transmitted,
                            "consecutive_aggregate_cosine": consecutive,
                        },
                    },
                },
            }
        )
    return {"rounds": rounds}


class V4GateTests(unittest.TestCase):
    def test_matrix_shape_is_frozen(self):
        self.assertEqual(TASKS, ("sst2", "mrpc", "qqp"))
        self.assertEqual(DEVELOPMENT_SEEDS, (351, 352, 353))
        self.assertEqual(EXPECTED_CELLS, 9)
        self.assertEqual(EXPECTED_RUNS, 45)

    def test_export_fields_use_evaluation_not_validation_semantics(self):
        self.assertIn("evaluation_accuracy", BY_SEED_FIELDS)
        self.assertIn("evaluation_positive_f1", BY_SEED_FIELDS)
        self.assertNotIn("validation_accuracy", BY_SEED_FIELDS)
        csv_header = _csv_text([], BY_SEED_FIELDS).splitlines()[0]
        self.assertIn("evaluation_accuracy", csv_header)
        self.assertNotIn("validation_accuracy", csv_header)
        decision = evaluate_stage_a(complete_rows())
        markdown = _markdown(complete_rows(), decision, [], "train_holdout")
        self.assertIn("Evaluation accuracy", markdown)
        self.assertIn("Evaluation positive F1", markdown)
        self.assertIn("`train_holdout` (pooled train-only holdout)", markdown)

    def test_v4_gate_requires_retention_and_speedup_above_one(self):
        value = row(CANDIDATE_METHOD, "mrpc", 351, speedup=1.0)
        gated = apply_v4_gate(value)
        self.assertTrue(gated["strict_stable_success"])
        self.assertFalse(gated["stage_a_success"])

        value["speedup95"] = 1.01
        value["final_retention"] = 0.949999
        gated = apply_v4_gate(value)
        self.assertFalse(gated["strict_stable_success"])
        self.assertIn("RETENTION_BELOW_0.95", gated["failure_components"])

    def test_stage_a_passes_only_nine_of_nine_candidate_cells(self):
        decision = evaluate_stage_a(complete_rows())
        self.assertEqual(decision["primary_matrix_valid"], EXPECTED_RUNS)
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(decision["selected_candidate"], CANDIDATE_METHOD)
        self.assertEqual(
            decision["candidate_summary"]["stage_a_success_cells"],
            EXPECTED_CELLS,
        )

    def test_one_candidate_failure_is_a_negative_result_without_new_mechanism(self):
        rows = complete_rows()
        for value in rows:
            if (
                value["method"] == CANDIDATE_METHOD
                and value["task"] == "qqp"
                and value["seed"] == 353
            ):
                value["post_hit_violation_count"] = 1
        decision = evaluate_stage_a(rows)
        self.assertEqual(decision["status"], "NEGATIVE_RESULT")
        self.assertIsNone(decision["selected_candidate"])
        self.assertEqual(
            decision["mechanism_routing"]["recommended_mechanism"], "NONE"
        )

    def test_diagnostics_only_describe_a_future_residual_control_study(self):
        rows = complete_rows()
        failures = {("sst2", 351), ("mrpc", 352)}
        for value in rows:
            if value["method"] == CANDIDATE_METHOD and (
                value["task"], value["seed"]
            ) in failures:
                value["post_hit_violation_count"] = 1
                value["post_hit_violation_rounds"] = (
                    "25" if value["task"] == "sst2" else "41"
                )
        summaries = {
            (task, CANDIDATE_METHOD, seed, WIRE_RATIO): diagnostic_summary(
                residual_pattern=True
            )
            for task, seed in failures
        }
        routing = evaluate_stage_a(rows, summaries)["mechanism_routing"]
        self.assertEqual(routing["classification"], "COMPRESSION_SPECIFIC")
        self.assertEqual(routing["recommended_mechanism"], "NONE")
        self.assertEqual(routing["future_research_direction"], "RESIDUAL_CONTROL")
        self.assertEqual(len(routing["diagnostic_evidence"]), 2)

    def test_shared_dense_failure_points_to_future_basis_study_without_tuning(self):
        rows = complete_rows()
        for value in rows:
            if (
                value["task"] == "qqp"
                and value["seed"] == 353
                and value["method"] in {CANDIDATE_METHOD, DECAY_DENSE_METHOD}
            ):
                value["post_hit_violation_count"] = 1
        routing = evaluate_stage_a(rows)["mechanism_routing"]
        self.assertEqual(
            routing["classification"], "HYBRID_BASIS_OR_OPTIMIZATION_SIDE"
        )
        self.assertEqual(routing["recommended_mechanism"], "NONE")
        self.assertEqual(
            routing["future_research_direction"],
            "HYBRID_BASIS_OR_OPTIMIZATION_STUDY",
        )

    def test_missing_cell_is_incomplete(self):
        decision = evaluate_stage_a(complete_rows()[:-1])
        self.assertEqual(decision["status"], "INCOMPLETE_OR_INVALID")
        self.assertIsNone(decision["selected_candidate"])


if __name__ == "__main__":
    unittest.main()
