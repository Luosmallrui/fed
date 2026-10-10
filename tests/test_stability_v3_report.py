import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from fedpeft.config import run_id
from fedpeft.provenance import verified_v3_launch_provenance
from fedpeft.utils import file_sha256, source_python_sha256

from scripts.build_stability_v3_report import (
    ALL_METHODS,
    CANDIDATE_METHOD,
    DECAY_DENSE_METHOD,
    DEVELOPMENT_SEEDS,
    TASKS,
    WIRE_RATIO,
    _campaign_errors,
    apply_v3_gate,
    evaluate_stage_a,
    lr_trace_rows,
)


def row(method, task, seed, *, strict=True, speedup=2.0, technical=True):
    ratio = None if method in {
        "fedavg_lora",
        "hybrid4_a_dense",
        "hybrid4_a_decay_dense",
    } else WIRE_RATIO
    failures = [] if strict else ["POST_HIT_VIOLATION"]
    return {
        "task": task,
        "method": method,
        "wire_target": "dense" if ratio is None else ratio,
        "seed": seed,
        "audit_status": "VALID" if technical else "INVALID",
        "technical_valid": technical,
        "target_accuracy": 0.76,
        "c95_status": "REACHED" if strict else "REACHED",
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
        "tail_drop_limit": 0.01 if task == "sst2" else 0.02,
        "stable_success": strict,
        "communication_efficient_success": strict and speedup > 1,
        "reasons": ";".join(failures),
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
        metric = 0.80
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
                "primary_metric": metric,
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
                    }
                },
            }
        )
    return {"rounds": rounds}


class V3GateTests(unittest.TestCase):
    def test_v3_gate_requires_retention_and_speedup_strictly_above_one(self):
        value = row(CANDIDATE_METHOD, "sst2", 331, speedup=1.0)
        gated = apply_v3_gate(value)
        self.assertTrue(gated["strict_stable_success"])
        self.assertFalse(gated["speedup_gt_one"])
        self.assertFalse(gated["stage_a_success"])

        value["speedup95"] = 1.01
        value["final_retention"] = 0.949999
        gated = apply_v3_gate(value)
        self.assertFalse(gated["strict_stable_success"])
        self.assertIn("RETENTION_BELOW_0.95", gated["failure_components"])

    def test_stage_a_passes_only_six_of_six_candidate_cells(self):
        decision = evaluate_stage_a(complete_rows())
        self.assertEqual(decision["primary_matrix_valid"], 30)
        self.assertEqual(decision["status"], "PASS")
        self.assertEqual(decision["selected_candidate"], CANDIDATE_METHOD)
        self.assertEqual(
            decision["candidate_summary"]["stage_a_success_cells"], 6
        )
        self.assertEqual(
            decision["mechanism_routing"]["classification"],
            "NO_ADDITIONAL_MECHANISM",
        )

    def test_one_speedup_failure_is_a_negative_result_without_stability_mechanism(self):
        rows = complete_rows()
        for value in rows:
            if (
                value["method"] == CANDIDATE_METHOD
                and value["task"] == "rte"
                and value["seed"] == 333
            ):
                value["speedup95"] = 1.0
        decision = evaluate_stage_a(rows)
        self.assertEqual(decision["status"], "NEGATIVE_RESULT")
        self.assertEqual(
            decision["mechanism_routing"]["classification"],
            "COMMUNICATION_EFFICIENCY_FAILURE",
        )
        self.assertEqual(
            decision["mechanism_routing"]["recommended_mechanism"], "NONE"
        )

    def test_shared_decay_dense_failure_routes_to_basis_optimization_side(self):
        rows = complete_rows()
        for value in rows:
            if (
                value["task"] == "rte"
                and value["seed"] == 332
                and value["method"] in {CANDIDATE_METHOD, DECAY_DENSE_METHOD}
            ):
                value["post_hit_violation_count"] = 1
        decision = evaluate_stage_a(rows)
        routing = decision["mechanism_routing"]
        self.assertEqual(routing["classification"], "HYBRID_BASIS_OR_OPTIMIZATION_SIDE")
        self.assertEqual(routing["recommended_mechanism"], "NONE")
        self.assertEqual(
            routing["matched_failed_cells"][0]["shared_failures"],
            ["POST_HIT_VIOLATION"],
        )

    def test_compression_specific_failure_is_conservatively_inconclusive_without_diagnostics(self):
        rows = complete_rows()
        for value in rows:
            if (
                value["method"] == CANDIDATE_METHOD
                and value["task"] == "sst2"
                and value["seed"] == 331
            ):
                value["post_hit_violation_count"] = 1
                value["post_hit_violation_rounds"] = "25"
        decision = evaluate_stage_a(rows)
        routing = decision["mechanism_routing"]
        self.assertEqual(routing["classification"], "INCONCLUSIVE")
        self.assertEqual(routing["recommended_mechanism"], "NONE")
        self.assertEqual(routing["late_discordant_cells"], 1)

    def test_late_routing_uses_first_post_hit_violation(self):
        rows = complete_rows()
        failures = {("sst2", 331), ("sst2", 332)}
        for value in rows:
            if value["method"] == CANDIDATE_METHOD and (
                value["task"], value["seed"]
            ) in failures:
                value["post_hit_violation_count"] = 2
                value["post_hit_violation_rounds"] = "20,25"
        routing = evaluate_stage_a(rows)["mechanism_routing"]
        self.assertEqual(routing["classification"], "INCONCLUSIVE")
        self.assertEqual(routing["recommended_mechanism"], "NONE")
        self.assertEqual(routing["late_discordant_cells"], 0)

    def test_late_routing_accepts_final_failure_despite_early_violation(self):
        rows = complete_rows()
        failures = {("sst2", 331), ("sst2", 332)}
        for value in rows:
            if value["method"] == CANDIDATE_METHOD and (
                value["task"], value["seed"]
            ) in failures:
                value["post_hit_violation_count"] = 2
                value["post_hit_violation_rounds"] = "20,25"
                value["final_accuracy"] = 0.75
                value["final_retention"] = 0.9375
        routing = evaluate_stage_a(rows)["mechanism_routing"]
        self.assertEqual(routing["classification"], "COMPRESSION_SPECIFIC")
        self.assertEqual(routing["recommended_mechanism"], "NONE")
        self.assertEqual(routing["late_discordant_cells"], 2)

    def test_compression_specific_residual_pattern_routes_to_new_protocol_only(self):
        rows = complete_rows()
        failures = {("sst2", 331), ("rte", 332)}
        for value in rows:
            if value["method"] == CANDIDATE_METHOD and (
                value["task"], value["seed"]
            ) in failures:
                value["post_hit_violation_count"] = 1
                value["post_hit_violation_rounds"] = (
                    "25" if value["task"] == "sst2" else "41"
                )
        summaries = {
            ("sst2", CANDIDATE_METHOD, 331, WIRE_RATIO): diagnostic_summary(
                residual_pattern=True
            ),
            ("rte", CANDIDATE_METHOD, 332, WIRE_RATIO): diagnostic_summary(
                residual_pattern=True
            ),
        }
        routing = evaluate_stage_a(rows, summaries)["mechanism_routing"]
        self.assertEqual(routing["classification"], "COMPRESSION_SPECIFIC")
        self.assertEqual(routing["recommended_mechanism"], "RESIDUAL_CONTROL")

    def test_compression_specific_reversal_pattern_routes_to_server_damping(self):
        rows = complete_rows()
        failures = {("sst2", 331), ("rte", 333)}
        for value in rows:
            if value["method"] == CANDIDATE_METHOD and (
                value["task"], value["seed"]
            ) in failures:
                value["post_hit_violation_count"] = 1
                value["post_hit_violation_rounds"] = (
                    "25" if value["task"] == "sst2" else "41"
                )
        summaries = {
            ("sst2", CANDIDATE_METHOD, 331, WIRE_RATIO): diagnostic_summary(
                damping_pattern=True
            ),
            ("rte", CANDIDATE_METHOD, 333, WIRE_RATIO): diagnostic_summary(
                damping_pattern=True
            ),
        }
        routing = evaluate_stage_a(rows, summaries)["mechanism_routing"]
        self.assertEqual(routing["recommended_mechanism"], "SERVER_DAMPING")

    def test_decay_dense_must_be_six_of_six_before_compression_routing(self):
        rows = complete_rows()
        for value in rows:
            if (
                value["method"] == CANDIDATE_METHOD
                and value["task"] == "sst2"
                and value["seed"] == 331
            ):
                value["post_hit_violation_count"] = 1
            if (
                value["method"] == DECAY_DENSE_METHOD
                and value["task"] == "rte"
                and value["seed"] == 333
            ):
                value["tail_drop"] = 0.03
        routing = evaluate_stage_a(rows)["mechanism_routing"]
        self.assertEqual(routing["classification"], "INCONCLUSIVE")
        self.assertEqual(routing["recommended_mechanism"], "NONE")

    def test_missing_or_invalid_run_blocks_decision(self):
        rows = complete_rows()[:-1]
        decision = evaluate_stage_a(rows)
        self.assertEqual(decision["status"], "INCOMPLETE_OR_INVALID")
        self.assertEqual(decision["primary_matrix_valid"], 29)
        self.assertEqual(decision["mechanism_routing"]["classification"], "BLOCKED")

    def test_extra_known_run_blocks_exact_matrix_decision(self):
        rows = complete_rows()
        extra = row(CANDIDATE_METHOD, "sst2", 331)
        extra["wire_target"] = 0.2
        rows.append(extra)
        decision = evaluate_stage_a(rows)
        self.assertEqual(decision["status"], "INCOMPLETE_OR_INVALID")
        self.assertEqual(decision["mechanism_routing"]["classification"], "BLOCKED")

    def test_input_rows_are_not_mutated(self):
        rows = complete_rows()
        before = copy.deepcopy(rows)
        evaluate_stage_a(rows)
        self.assertEqual(rows, before)

    def test_lr_trace_rows_preserve_each_run_and_round(self):
        key = ("sst2", CANDIDATE_METHOD, 331, WIRE_RATIO)
        summaries = {
            key: {
                "rounds": [
                    {
                        "round": 24,
                        "learning_rate": {
                            "schedule_name": "late_cosine_v1",
                            "multiplier": 1.0,
                            "adapter_learning_rate": 3e-4,
                            "head_learning_rate": 3e-4,
                        },
                    },
                    {
                        "round": 30,
                        "learning_rate": {
                            "schedule_name": "late_cosine_v1",
                            "multiplier": 0.1,
                            "adapter_learning_rate": 3e-5,
                            "head_learning_rate": 3e-5,
                        },
                    },
                ]
            }
        }
        trace = lr_trace_rows(summaries)
        self.assertEqual(len(trace), 2)
        self.assertEqual(trace[0]["wire_target"], WIRE_RATIO)
        self.assertEqual(trace[0]["round"], 24)
        self.assertEqual(trace[-1]["multiplier"], 0.1)


class V3CliTests(unittest.TestCase):
    def test_campaign_audit_requires_one_exact_ordered_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            output = root / "docs" / "results" / "stability_v3" / "development"
            source = root / "src" / "fedpeft" / "module.py"
            protocol = root / "docs" / "results" / "exploration" / "STABILITY_V3_PLAN.md"
            config_path = root / "configs" / "stage_a.yaml"
            source.parent.mkdir(parents=True)
            protocol.parent.mkdir(parents=True)
            config_path.parent.mkdir(parents=True)
            output.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            protocol.write_text("# frozen protocol\n", encoding="utf-8")
            config_path.write_text("frozen: true\n", encoding="utf-8")
            protocol_hash = file_sha256(protocol)
            cfg = {
                "_config_path": str(config_path),
                "project": {"root": str(root), "output_dir": str(output)},
                "stability": {
                    "generation": 3,
                    "stage": "stage_a",
                    "protocol_sha256": protocol_hash,
                },
            }
            preflight = {
                "status": "PASS",
                "checks": {
                    "source_hash": source_python_sha256(root),
                    "config_file_hash": file_sha256(config_path),
                    "stability_protocol_hash": protocol_hash,
                    "stability_protocol_file_hash": protocol_hash,
                },
                "failures": [],
            }
            preflight_path = output / "preflight.json"
            preflight_path.write_text(json.dumps(preflight) + "\n", encoding="utf-8")
            clean_cfg = {
                key: value for key, value in cfg.items() if not key.startswith("_")
            }
            expected = []
            dense = (
                "fedavg_lora",
                "hybrid4_a_dense",
                "hybrid4_a_decay_dense",
            )
            sparse = ("hybrid4_a_ef", CANDIDATE_METHOD)
            for task in TASKS:
                for seed in DEVELOPMENT_SEEDS:
                    expected.extend(run_id(task, method, seed, None) for method in dense)
                    expected.extend(
                        run_id(task, method, seed, WIRE_RATIO) for method in sparse
                    )
            (output / "campaign_config.json").write_text(
                json.dumps(
                    {
                        "config": clean_cfg,
                        "requested_run_ids": expected,
                        "include_diagnostics": False,
                        "launch_provenance": verified_v3_launch_provenance(
                            cfg, output
                        ),
                    }
                ),
                encoding="utf-8",
            )
            (output / "campaign_summary.json").write_text(
                json.dumps(
                    {
                        "status": "COMPLETE",
                        "requested_runs": 30,
                        "completed_runs": expected,
                        "failed_runs": [],
                    }
                ),
                encoding="utf-8",
            )
            events = []
            for index, identifier in enumerate(expected, start=1):
                common = {"run_id": identifier, "index": index, "total": 30}
                events.extend(
                    [
                        {"event": "START", **common},
                        {"event": "DONE", "returncode": 0, **common},
                    ]
                )
                run_dir = output / identifier
                run_dir.mkdir()
                (run_dir / "summary.json").write_text("{}", encoding="utf-8")
            events[1]["recovered_from_summary"] = True
            events.insert(3, {**events[2], "event": "RESUME"})
            events.insert(4, {**events[2], "event": "RESUME"})
            events_path = output / "campaign_events.jsonl"

            def write_events(values):
                events_path.write_text(
                    "\n".join(json.dumps(event) for event in values) + "\n",
                    encoding="utf-8",
                )

            # A summary recovered after START may go straight to DONE, while an
            # unfinished run may have any number of serial RESUME events.
            write_events(events)
            self.assertEqual(_campaign_errors(cfg), [])

            # Starting run two before run one is DONE is a concurrency artifact,
            # even though every per-run chain would otherwise look complete.
            interleaved = [events[0], events[2], events[1], *events[3:]]
            write_events(interleaved)
            self.assertTrue(
                any("not globally serial" in error for error in _campaign_errors(cfg))
            )

            # Complete chains must also appear in the exact frozen run order.
            first_chain = events[:2]
            second_chain = events[2:6]
            out_of_order = [*second_chain, *first_chain, *events[6:]]
            write_events(out_of_order)
            self.assertTrue(
                any("START order differs" in error for error in _campaign_errors(cfg))
            )

            write_events(events)
            campaign_lock = output / "campaign.lock"
            campaign_lock.write_text("still-running", encoding="utf-8")
            self.assertTrue(
                any("campaign.lock" in error for error in _campaign_errors(cfg))
            )
            campaign_lock.unlink()
            self.assertEqual(_campaign_errors(cfg), [])

            original_preflight = preflight_path.read_bytes()
            preflight["rewritten_after_launch"] = True
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            self.assertTrue(
                any(
                    "launch_provenance differs" in error
                    for error in _campaign_errors(cfg)
                )
            )
            preflight_path.write_bytes(original_preflight)
            self.assertEqual(_campaign_errors(cfg), [])

            for path, replacement in (
                (source, "VALUE = 2\n"),
                (config_path, "frozen: false\n"),
                (protocol, "# changed protocol\n"),
            ):
                with self.subTest(current_bytes=path.name):
                    original = path.read_bytes()
                    path.write_text(replacement, encoding="utf-8")
                    self.assertTrue(
                        any(
                            "preflight/launch provenance audit failed" in error
                            for error in _campaign_errors(cfg)
                        )
                    )
                    path.write_bytes(original)
                    self.assertEqual(_campaign_errors(cfg), [])

            partial = output / "unregistered-seed-999" / "checkpoint.pt"
            partial.parent.mkdir()
            partial.write_bytes(b"partial")
            partial_errors = _campaign_errors(cfg)
            self.assertTrue(
                any("unexpected run-artifact directories" in error for error in partial_errors)
            )
            partial.unlink()
            partial.parent.rmdir()
            self.assertEqual(_campaign_errors(cfg), [])

            extra = output / "unexpected" / "summary.json"
            extra.parent.mkdir()
            extra.write_text("{}", encoding="utf-8")
            self.assertTrue(_campaign_errors(cfg))
            extra.unlink()
            for artifact_name in ("run_config.json", "launcher.log"):
                artifact = extra.parent / artifact_name
                artifact.write_text("partial\n", encoding="utf-8")
                self.assertTrue(
                    any(
                        "unexpected run-artifact directories" in error
                        for error in _campaign_errors(cfg)
                    )
                )
                artifact.unlink()

    def test_help_runs_as_direct_script(self):
        root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(root / "scripts" / "build_stability_v3_report.py"), "--help"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--config", completed.stdout)


if __name__ == "__main__":
    unittest.main()
