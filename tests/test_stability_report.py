import copy
import unittest

from scripts.build_stability_report import (
    _confirmation_config_errors,
    evaluate_confirmation,
    evaluate_stage_a,
    evaluate_stage_b,
    stability_cell,
)


def summary(method, values, wire_per_round, *, task="sst2", seed=321, ratio=0.1):
    rounds = []
    for index, value in enumerate(values, start=1):
        rounds.append(
            {
                "round": index,
                "primary_metric": value,
                "wire": {"total_delivered_bytes": wire_per_round * index},
            }
        )
    return {
        "run": {
            "task": task,
            "method": method,
            "seed": seed,
            "wire_ratio": ratio,
        },
        "rounds": rounds,
        "final": {
            **rounds[-1],
            "wire": {
                "total_delivered_bytes": wire_per_round * len(rounds),
                "max_uplink_message_bytes": wire_per_round // 20,
                "max_downlink_message_bytes": wire_per_round // 20,
            },
        },
        "wire_budget": {"message_cap_bytes": wire_per_round // 20},
        "setup_communication": {"basis_hash_handshake_bytes_per_client": 0},
        "data": {"num_clients": 10},
    }


def decision_row(
    method,
    task,
    seed,
    ratio,
    *,
    stable=True,
    c95_round=5,
    retention=0.97,
    tail=0.0,
    technical=True,
):
    return {
        "task": task,
        "method": method,
        "wire_target": "dense" if ratio is None else ratio,
        "seed": seed,
        "technical_valid": technical,
        "stable_success": stable,
        "communication_efficient_success": stable,
        "c95_status": "REACHED" if c95_round is not None else "NOT_REACHED",
        "c95_round": "" if c95_round is None else c95_round,
        "final_retention": retention,
        "tail_drop": tail,
        "speedup95": 2.0 if stable else "",
    }


class StabilityCellTests(unittest.TestCase):
    def test_stable_success_uses_paired_target_and_actual_c95_wire(self):
        dense = summary("fedavg_lora", [0.5, 0.8], 100, ratio=None)
        candidate = summary("spectral_rot_ef", [0.7, 0.76, 0.77, 0.78], 20)
        row = stability_cell(candidate, dense)
        self.assertTrue(row["stable_success"])
        self.assertEqual(row["c95_round"], 2)
        self.assertEqual(row["c95_wire_bytes"], 40)
        self.assertEqual(row["fedavg_c95_wire_bytes"], 200)
        self.assertEqual(row["speedup95"], 5.0)
        self.assertAlmostEqual(row["final_retention"], 0.975)

    def test_post_hit_drop_fails_even_if_final_recovers(self):
        dense = summary("fedavg_lora", [0.5, 0.8], 100, ratio=None)
        candidate = summary("spectral_rot_ef", [0.7, 0.77, 0.75, 0.78], 20)
        row = stability_cell(candidate, dense)
        self.assertFalse(row["stable_success"])
        self.assertEqual(row["post_hit_violation_count"], 1)
        self.assertEqual(row["post_hit_violation_rounds"], "3")

    def test_tail_drop_gate_is_task_specific(self):
        dense = summary("fedavg_lora", [0.5, 0.8], 100, ratio=None)
        candidate = summary("spectral_rot_ef", [0.76, 0.82, 0.81, 0.79], 20)
        row = stability_cell(candidate, dense)
        self.assertFalse(row["stable_success"])
        self.assertAlmostEqual(row["tail_drop"], 0.03)
        self.assertIn("TailDrop", row["reasons"])


class StabilityDecisionTests(unittest.TestCase):
    def stage_a_rows(self):
        rows = []
        for task in ("sst2", "rte"):
            for seed in (321, 322, 323):
                rows.extend(
                    [
                        decision_row("fedavg_lora", task, seed, None),
                        decision_row("spectral_a_dense", task, seed, None),
                        decision_row("spectral_a_ef", task, seed, 0.1),
                    ]
                )
        for method, c95 in (
            ("spectral_rot_ef", 5),
            ("spectral_block_ef", 6),
            ("hybrid4_a_ef", 4),
        ):
            for task in ("sst2", "rte"):
                for seed in (321, 322, 323):
                    rows.append(
                        decision_row(
                            method,
                            task,
                            seed,
                            0.1,
                            stable=not (
                                method == "hybrid4_a_ef"
                                and task == "rte"
                                and seed == 323
                            ),
                            c95_round=c95,
                        )
                    )
        return rows

    def test_stage_a_requires_six_of_six_and_uses_worst_case_order(self):
        decision = evaluate_stage_a(self.stage_a_rows())
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(decision["selected_candidate"], "spectral_rot_ef")
        values = {row["method"]: row for row in decision["candidates"]}
        self.assertTrue(values["spectral_rot_ef"]["eligible"])
        self.assertTrue(values["spectral_block_ef"]["eligible"])
        self.assertFalse(values["hybrid4_a_ef"]["eligible"])

    def test_stage_a_reports_negative_result_without_expanding_grid(self):
        rows = self.stage_a_rows()
        for row in rows:
            if row["method"] in {"spectral_rot_ef", "spectral_block_ef"}:
                row["stable_success"] = False
        decision = evaluate_stage_a(rows)
        self.assertEqual(decision["status"], "NEGATIVE_RESULT")
        self.assertIsNone(decision["selected_candidate"])

    def test_stage_b_requires_all_eighteen_winner_cells(self):
        rows = [
            row
            for row in self.stage_a_rows()
            if row["method"] != "spectral_rot_ef"
        ]
        for task in ("sst2", "rte", "mnli"):
            for seed in (321, 322, 323):
                if task == "mnli":
                    rows.append(decision_row("fedavg_lora", task, seed, None))
                for ratio in (0.1, 0.2):
                    rows.append(
                        decision_row(
                            "spectral_rot_ef", task, seed, ratio, c95_round=5
                        )
                    )
        decision = evaluate_stage_b(rows, "spectral_rot_ef")
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(
            decision["winner_summary"]["stable_success_cells"], 18
        )

        broken = copy.deepcopy(rows)
        broken[-1]["stable_success"] = False
        self.assertEqual(
            evaluate_stage_b(broken, "spectral_rot_ef")["status"],
            "NEGATIVE_RESULT",
        )

    def test_confirmation_requires_45_primary_and_p2_random_control_for_basis_claim(self):
        development = self.stage_a_rows()
        # P2 must be the deterministic Stage-A winner for this test.
        for row in development:
            if row["method"] == "spectral_rot_ef":
                row["stable_success"] = False
        for task in ("sst2", "rte"):
            for seed in (321, 322, 323):
                development.append(
                    decision_row("spectral_block_ef", task, seed, 0.2)
                )
        for seed in (321, 322, 323):
            development.append(decision_row("fedavg_lora", "mnli", seed, None))
            for ratio in (0.1, 0.2):
                development.append(
                    decision_row("spectral_block_ef", "mnli", seed, ratio)
                )

        confirmation = []
        for task in ("sst2", "rte", "mnli"):
            for seed in (501, 502, 503):
                confirmation.append(decision_row("fedavg_lora", task, seed, None))
                for ratio in (0.1, 0.2):
                    confirmation.append(
                        decision_row("spectral_a_ef", task, seed, ratio)
                    )
                    confirmation.append(
                        decision_row("spectral_block_ef", task, seed, ratio)
                    )
        self.assertEqual(
            evaluate_confirmation(confirmation, "spectral_block_ef")["status"],
            "BLOCKED",
        )
        rows = development + confirmation
        decision = evaluate_confirmation(rows, "spectral_block_ef")
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(decision["primary_matrix_valid"], 45)
        self.assertFalse(decision["basis_vs_random_claim_ready"])
        self.assertEqual(
            evaluate_confirmation(rows, "spectral_rot_ef")["status"],
            "BLOCKED",
        )

        for task in ("sst2", "rte", "mnli"):
            for seed in (501, 502, 503):
                for ratio in (0.1, 0.2):
                    rows.append(
                        decision_row("random_block_ef", task, seed, ratio)
                    )
        decision = evaluate_confirmation(rows, "spectral_block_ef")
        self.assertTrue(decision["basis_vs_random_claim_ready"])
        self.assertEqual(decision["matched_random_control_valid"], 18)

    def test_confirmation_cli_requires_stage_a_both_stage_b_and_confirmation_configs(self):
        winner = "spectral_rot_ef"
        stage_a = {"stability": {"stage": "stage_a", "winner": None}}
        stage_b_first = {
            "stability": {
                "stage": "stage_b",
                "substage": "sst_rte_20",
                "winner": winner,
            }
        }
        stage_b_second = {
            "stability": {
                "stage": "stage_b",
                "substage": "mnli_10_20",
                "winner": winner,
            }
        }
        confirmation = {
            "stability": {"stage": "confirmation", "winner": winner}
        }
        complete = [stage_a, stage_b_first, stage_b_second, confirmation]
        self.assertEqual(_confirmation_config_errors(complete, winner), [])
        self.assertTrue(
            _confirmation_config_errors(
                [stage_a, stage_b_first, confirmation], winner
            )
        )
        fake = copy.deepcopy(complete)
        fake[-1]["stability"]["winner"] = "hybrid4_a_ef"
        self.assertTrue(_confirmation_config_errors(fake, winner))
        duplicate = [
            stage_a,
            stage_b_first,
            copy.deepcopy(stage_b_first),
            stage_b_second,
            confirmation,
        ]
        reasons = _confirmation_config_errors(duplicate, winner)
        self.assertTrue(any("exactly one of each" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
