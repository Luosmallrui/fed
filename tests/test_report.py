import copy
import hashlib
import json
import unittest
from pathlib import Path

import torch

from fedpeft.compression import BlockTopKCompressor
from fedpeft.diagnostics import SparseDiagnostics

from scripts.build_report import (
    COMMUNICATION_EFFICIENCY_BY_SEED_FIELDS,
    COMMUNICATION_EFFICIENCY_FIELDS,
    ExpectedRun,
    FIXED_A_METHODS,
    ModelExpectations,
    STANDARD_LORA_METHODS,
    _actual_wire_ratio,
    _analysis_markdown,
    _communication_efficiency_rows,
    _communication_efficiency_seed_rows,
    _dense_fedavg_all_in_wire_bytes,
    _expected_layout,
    _expected_lr_schedule,
    _expected_round_learning_rate,
    _pair_rows,
    _specs,
    _validate_full_lora_factor_audit,
    _validate_stability_protocol,
    _validate_wire,
)


def _strict_wire_fixture(
    method: str,
    wire_ratio: float | None,
) -> tuple[dict, dict, ModelExpectations, ExpectedRun]:
    """Small, independently calculated ledger fixture for wire report tests."""

    target = "encoder.encoder.layer.0.attention.self.query"
    hidden = 4
    rank = 2
    labels = 2
    clients = 2
    rounds = 3
    if method in STANDARD_LORA_METHODS:
        adapter_entries = [
            (f"{target}.lora_A", (rank, hidden)),
            (f"{target}.lora_B", (hidden, rank)),
        ]
    elif method in FIXED_A_METHODS:
        adapter_entries = [(f"{target}.lora_B", (hidden, rank))]
    else:
        adapter_entries = [(f"{target}.lora_R", (rank, rank))]
    adapter_entries.sort(key=lambda item: item[0])
    head_entries = sorted(
        [
            ("classifier.weight", (labels, hidden)),
            ("classifier.bias", (labels,)),
        ],
        key=lambda item: item[0],
    )
    manifest = [
        {"name": name, "shape": list(shape)}
        for name, shape in adapter_entries + head_entries
    ]
    schema_bytes = len(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    ) * (1 + clients)
    adapter_parameters = sum(
        shape[0] * shape[1] for _, shape in adapter_entries
    )
    head_parameters = hidden * labels + labels
    reference_parameters = 2 * hidden * rank + head_parameters
    reference_message = 26 + 2 * reference_parameters
    dense_head = 13 + 2 * head_parameters
    if wire_ratio is None:
        cap = reference_message
        sparse_k = None
        density = None
        packet_bytes = 26 + 2 * (adapter_parameters + head_parameters)
        adapter_cap = reference_message - dense_head
    else:
        cap = int(reference_message * wire_ratio)
        adapter_cap = cap - dense_head
        sparse_k = min(adapter_parameters, (adapter_cap - 21) // 6)
        density = sparse_k / adapter_parameters
        packet_bytes = 21 + 6 * sparse_k + dense_head

    def ledger(round_index: int) -> dict[str, int]:
        uplink = packet_bytes * clients * round_index
        emitted = packet_bytes * round_index
        delivered = packet_bytes * clients * round_index
        return {
            "uplink_bytes": uplink,
            "downlink_emitted_bytes": emitted,
            "downlink_delivered_bytes": delivered,
            "uplink_messages": clients * round_index,
            "downlink_messages": round_index,
            "schema_bytes": schema_bytes,
            "max_uplink_message_bytes": packet_bytes,
            "max_downlink_message_bytes": packet_bytes,
            "total_delivered_bytes": uplink + delivered + schema_bytes,
        }

    round_records = [
        {"round": index, "wire": ledger(index)}
        for index in range(1, rounds + 1)
    ]
    summary = {
        "run": {
            "task": "sst2",
            "method": method,
            "seed": 331,
            "wire_ratio": wire_ratio,
        },
        "model": {
            "rank": rank,
            "target_modules": [target],
            "adapter_parameters": adapter_parameters,
            "head_parameters": head_parameters,
            "reference_full_lora_parameters": reference_parameters,
        },
        "data": {"num_clients": clients},
        "training": {"rounds": rounds},
        "wire_budget": {
            "requested_ratio": wire_ratio,
            "reference_message_bytes": reference_message,
            "message_cap_bytes": cap,
            "adapter_cap_bytes": adapter_cap,
            "sparse_k": sparse_k,
            "adapter_density": density,
        },
        "rounds": round_records,
        "final": copy.deepcopy(round_records[-1]),
        "setup_communication": {
            "basis_hash_handshake_bytes_per_client": (
                0 if method in STANDARD_LORA_METHODS else 32
            )
        },
    }
    cfg = {
        "model": {"rank": rank},
        "data": {"num_clients": clients},
        "training": {"rounds": {"sst2": rounds}},
        "communication": {"schema_once_per_run": True},
    }
    expectations = ModelExpectations(
        model_hash="0" * 64,
        model_path=Path("/model"),
        hidden_size=hidden,
        num_layers=1,
        target_names=(target,),
    )
    spec = ExpectedRun("sst2", method, 331, wire_ratio, "required")
    return summary, cfg, expectations, spec


class DenseBasisDiagnosticReportTests(unittest.TestCase):
    def test_v3_report_recomputes_frozen_round_learning_rates(self):
        from fedpeft.config import load_config

        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_v3_stage_a.yaml")
        spec = ExpectedRun("sst2", "hybrid4_a_decay_ef", 331, 0.1, "required")
        schedule = _expected_lr_schedule(cfg, spec.method, 30)
        self.assertEqual(schedule["start_round"], 24)
        self.assertEqual(
            _expected_round_learning_rate(cfg, spec, 24, 30)["multiplier"],
            1.0,
        )
        self.assertLess(
            float(_expected_round_learning_rate(cfg, spec, 25, 30)["multiplier"]),
            1.0,
        )
        self.assertAlmostEqual(
            float(_expected_round_learning_rate(cfg, spec, 30, 30)["multiplier"]),
            0.1,
        )

    def test_stability_smoke_has_three_required_and_one_optional_run(self):
        from fedpeft.config import load_config

        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_smoke.yaml")
        required = _specs(cfg, diagnostic=False, role="required")
        optional = _specs(cfg, diagnostic=True, role="optional")
        self.assertEqual(len(required), 3)
        self.assertEqual(len(optional), 1)
        self.assertEqual(optional[0].method, "random_block_ef")

    def test_stability_candidates_use_fixed_a_report_layout(self):
        self.assertTrue(
            {
                "spectral_rot_ef",
                "spectral_block_ef",
                "hybrid4_a_ef",
                "hybrid4_a_dense",
                "hybrid4_a_decay_dense",
                "hybrid4_a_decay_ef",
                "random_block_ef",
            }.issubset(FIXED_A_METHODS)
        )

    def test_stability_protocol_and_block_diagnostic_are_report_gates(self):
        stability = {
            "stage": "stage_a",
            "protocol_sha256": "e96302aca47dc5e5afba431ce1e02fb3d653fa6196d1ad09acf3594e980a5116",
            "diagnostic_recipe_version": "v1",
            "candidates": [
                "spectral_rot_ef",
                "spectral_block_ef",
                "hybrid4_a_ef",
            ],
            "winner": None,
        }
        cfg = {"stability": stability, "model": {"basis_seed": 20260910}}
        tensor_names = ["tensor.0.lora_B", "tensor.1.lora_B"]
        tracker = SparseDiagnostics(
            tensor_names, [(1, 2), (1, 2)], [2, 2], rank=2, num_clients=2
        )
        compressor = BlockTopKCompressor([2, 2], 39)
        decoded = []
        for client_id, raw in enumerate(
            (
                torch.tensor([4.0, 3.0, 2.0, 1.0]),
                torch.tensor([1.0, 2.0, 3.0, 4.0]),
            )
        ):
            compressed = compressor.compress(raw)
            residual = raw - compressed.reconstructed
            tracker.observe_client(
                1,
                client_id,
                compressed.packet,
                raw,
                raw,
                compressed.reconstructed,
                residual,
            )
            decoded.append(compressed.reconstructed)
        aggregate = 0.5 * (decoded[0] + decoded[1])
        server_packet = compressor.compress(aggregate)
        tracker.observe_server(
            1,
            server_packet.packet,
            aggregate,
            aggregate,
            server_packet.reconstructed,
            aggregate - server_packet.reconstructed,
        )
        diagnostic = tracker.finish_round(
            1,
            decoded,
            [1.0, 1.0],
            aggregate,
            server_packet.reconstructed,
        )
        block_numels = [2, 2]
        block_quotas = [2, 1]
        quota_hash = hashlib.sha256(
            json.dumps(
                {"block_numels": block_numels, "quotas": block_quotas},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        summary = {
            "stability_protocol": stability,
            "model": {
                "target_modules": ["tensor.0", "tensor.1"],
                "rank": 2,
                "adapter_parameters": 4,
            },
            "wire_budget": {
                "sparse_k": 3,
                "compressor_kind": "block_topk",
                "block_numels": block_numels,
                "block_quotas": block_quotas,
                "block_quota_hash": quota_hash,
                "expected_sparse_adapter_bytes": 39,
                "diagnostic_recipe_version": "v1",
                "diagnostic_layout_fingerprint": tracker.layout_fingerprint,
            },
            "data": {"num_clients": 2},
            "invariants": {
                "before": {"fixed_basis_recipe": "stabilized_spectral_svd_v1"},
                "after": {"fixed_basis_recipe": "stabilized_spectral_svd_v1"},
            },
            "rounds": [
                {
                    "diagnostics": {
                        "stability_diagnostics": diagnostic
                    }
                }
            ],
        }
        spec = ExpectedRun("sst2", "spectral_block_ef", 321, 0.1, "required")
        self.assertEqual(_validate_stability_protocol(summary, cfg, spec), [])

        bad = copy.deepcopy(summary)
        bad["rounds"][0]["diagnostics"]["stability_diagnostics"]["clients"][
            "zero_support_tensor_count"
        ]["max"] = 1
        self.assertTrue(_validate_stability_protocol(bad, cfg, spec))

        bad = copy.deepcopy(summary)
        bad["stability_protocol"]["protocol_sha256"] = "0" * 64
        self.assertTrue(_validate_stability_protocol(bad, cfg, spec))

        bad = copy.deepcopy(summary)
        bad["wire_budget"]["block_quotas"] = [1, 2]
        bad["wire_budget"]["block_quota_hash"] = hashlib.sha256(
            json.dumps(
                {"block_numels": block_numels, "quotas": [1, 2]},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        reasons = _validate_stability_protocol(bad, cfg, spec)
        self.assertTrue(any("largest-remainder" in reason for reason in reasons))

        bad = copy.deepcopy(summary)
        bad["rounds"][0]["diagnostics"]["stability_diagnostics"]["server"][
            "support"
        ]["selected_by_tensor"] = {
            "tensor.0.lora_B": 1,
            "tensor.1.lora_B": 2,
        }
        reasons = _validate_stability_protocol(bad, cfg, spec)
        self.assertIn(
            "round 1 stability_diagnostics.server.support.selected_by_tensor "
            "differs from frozen block quotas",
            reasons,
        )

        bad = copy.deepcopy(summary)
        bad["rounds"][0]["diagnostics"]["stability_diagnostics"]["clients"][
            "per_client"
        ][0]["support"]["selected_by_tensor"] = {
            "tensor.0.lora_B": 1,
            "tensor.1.lora_B": 2,
        }
        reasons = _validate_stability_protocol(bad, cfg, spec)
        self.assertIn(
            "round 1 stability_diagnostics.clients.per_client[0].support."
            "selected_by_tensor differs from frozen block quotas",
            reasons,
        )

        bad = copy.deepcopy(summary)
        bad["rounds"][0]["diagnostics"]["stability_diagnostics"]["clients"][
            "selected_by_tensor_total"
        ] = {
            "tensor.0.lora_B": 2,
            "tensor.1.lora_B": 4,
        }
        reasons = _validate_stability_protocol(bad, cfg, spec)
        self.assertIn(
            "round 1 stability_diagnostics.clients.selected_by_tensor_total "
            "differs from num_clients times frozen block quotas",
            reasons,
        )

        bad = copy.deepcopy(summary)
        fake = "f" * 64
        bad["wire_budget"]["diagnostic_layout_fingerprint"] = fake
        bad["rounds"][0]["diagnostics"]["stability_diagnostics"][
            "layout_fingerprint"
        ] = fake
        reasons = _validate_stability_protocol(bad, cfg, spec)
        self.assertTrue(
            any("independently recomputed canonical layout" in reason for reason in reasons)
        )

    def test_full_lora_ef_uses_full_ab_report_layout(self):
        self.assertIn("full_lora_ef", STANDARD_LORA_METHODS)
        cfg = {"model": {"rank": 8}}
        model = ModelExpectations(
            model_hash="0" * 64,
            model_path=Path("/tmp/model"),
            hidden_size=768,
            num_layers=1,
            target_names=("layer.query", "layer.value"),
        )
        spec = ExpectedRun("sst2", "full_lora_ef", 401, 0.1, "required")
        _, adapter_parameters, head_parameters, reference_parameters, _ = (
            _expected_layout(cfg, model, spec)
        )
        self.assertEqual(adapter_parameters, 2 * 2 * 8 * 768)
        self.assertEqual(head_parameters, 2 * 768 + 2)
        self.assertEqual(reference_parameters, adapter_parameters + head_parameters)

    def test_full_lora_ef_factor_audit_is_a_required_report_gate(self):
        spec = ExpectedRun("sst2", "full_lora_ef", 401, 0.1, "required")
        summary = {
            "model": {"adapter_parameters": 10},
            "wire_budget": {"sparse_k": 3},
            "data": {"num_clients": 2},
            "rounds": [
                {
                    "diagnostics": {
                        "adapter_factor_audit": {
                            "lora_A": {
                                "logical_numel": 5,
                                "aggregate_update_norm": 1.0,
                                "server_residual_norm": 2.0,
                                "server_transmitted_selected_count": 1,
                                "client_transmitted_selected_count_total": 2,
                                "client_transmitted_selected_count_mean": 1.0,
                                "client_residual_norm_mean": 3.0,
                            },
                            "lora_B": {
                                "logical_numel": 5,
                                "aggregate_update_norm": 4.0,
                                "server_residual_norm": 5.0,
                                "server_transmitted_selected_count": 2,
                                "client_transmitted_selected_count_total": 4,
                                "client_transmitted_selected_count_mean": 2.0,
                                "client_residual_norm_mean": 6.0,
                            },
                        }
                    }
                }
            ],
        }
        self.assertEqual(_validate_full_lora_factor_audit(summary, spec), [])

        corruptions = {
            "missing factor": lambda value: value["rounds"][0]["diagnostics"][
                "adapter_factor_audit"
            ].pop("lora_B"),
            "logical total": lambda value: value["rounds"][0]["diagnostics"][
                "adapter_factor_audit"
            ]["lora_B"].__setitem__("logical_numel", 4),
            "server count": lambda value: value["rounds"][0]["diagnostics"][
                "adapter_factor_audit"
            ]["lora_B"].__setitem__("server_transmitted_selected_count", 1),
            "client count": lambda value: value["rounds"][0]["diagnostics"][
                "adapter_factor_audit"
            ]["lora_B"].__setitem__("client_transmitted_selected_count_total", 3),
            "client mean": lambda value: value["rounds"][0]["diagnostics"][
                "adapter_factor_audit"
            ]["lora_B"].__setitem__("client_transmitted_selected_count_mean", 1.5),
            "nonfinite norm": lambda value: value["rounds"][0]["diagnostics"][
                "adapter_factor_audit"
            ]["lora_A"].__setitem__("aggregate_update_norm", float("nan")),
        }
        for label, corrupt in corruptions.items():
            with self.subTest(label=label):
                value = copy.deepcopy(summary)
                corrupt(value)
                self.assertTrue(_validate_full_lora_factor_audit(value, spec))

        fedavg_spec = ExpectedRun("sst2", "fedavg_lora", 401, None, "required")
        self.assertEqual(_validate_full_lora_factor_audit({}, fedavg_spec), [])

    def test_wire_gate_independently_recomputes_sparse_packet_and_dense_actual_size(self):
        sparse_spec = ExpectedRun("sst2", "full_lora_ef", 401, 0.1, "required")
        sparse = {
            "model": {
                "adapter_parameters": 1000,
                "head_parameters": 10,
                "reference_full_lora_parameters": 1010,
            },
            "wire_budget": {
                "requested_ratio": 0.1,
                "reference_message_bytes": 2046,
                "message_cap_bytes": 204,
                "adapter_cap_bytes": 171,
                "sparse_k": 25,
                "adapter_density": 0.025,
            },
            "final": {
                "wire": {
                    "uplink_bytes": 204,
                    "downlink_emitted_bytes": 204,
                    "downlink_delivered_bytes": 204,
                    "uplink_messages": 1,
                    "downlink_messages": 1,
                    "schema_bytes": 1,
                    "max_uplink_message_bytes": 204,
                    "max_downlink_message_bytes": 204,
                    "total_delivered_bytes": 409,
                }
            },
        }
        self.assertEqual(_validate_wire(sparse, sparse_spec), [])
        for field, bad_value in (
            ("sparse_k", 24),
            ("adapter_cap_bytes", 170),
            ("adapter_density", 0.024),
        ):
            with self.subTest(field=field):
                value = copy.deepcopy(sparse)
                value["wire_budget"][field] = bad_value
                self.assertTrue(_validate_wire(value, sparse_spec))
        value = copy.deepcopy(sparse)
        value["final"]["wire"]["max_uplink_message_bytes"] = 205
        self.assertTrue(_validate_wire(value, sparse_spec))

        dense_spec = ExpectedRun("sst2", "ffa_lora", 401, None, "required")
        dense = copy.deepcopy(sparse)
        dense["model"]["adapter_parameters"] = 500
        dense["wire_budget"] = {
            "requested_ratio": None,
            "reference_message_bytes": 2046,
            "message_cap_bytes": 2046,
            "adapter_cap_bytes": 2013,
            "sparse_k": None,
            "adapter_density": None,
        }
        dense_message = 26 + 2 * (500 + 10)
        dense["final"]["wire"].update(
            {
                "uplink_bytes": dense_message,
                "downlink_emitted_bytes": dense_message,
                "downlink_delivered_bytes": dense_message,
                "max_uplink_message_bytes": dense_message,
                "max_downlink_message_bytes": dense_message,
                "total_delivered_bytes": 2 * dense_message + 1,
            }
        )
        self.assertEqual(_validate_wire(dense, dense_spec), [])

    def test_wire_gate_reconstructs_every_round_for_all_layout_families(self):
        cases = (
            ("fedavg_lora", None),
            ("ffa_lora", None),
            ("lora_xs", None),
            ("hybrid4_a_decay_ef", 0.9),
        )
        for method, ratio in cases:
            with self.subTest(method=method):
                summary, cfg, expectations, spec = _strict_wire_fixture(
                    method, ratio
                )
                self.assertEqual(
                    _validate_wire(summary, spec, cfg, expectations),
                    [],
                )
                self.assertEqual(_validate_wire(summary, spec), [])

    def test_wire_gate_rejects_any_tampered_round_ledger_field(self):
        ledger_fields = (
            "uplink_bytes",
            "downlink_emitted_bytes",
            "downlink_delivered_bytes",
            "uplink_messages",
            "downlink_messages",
            "schema_bytes",
            "total_delivered_bytes",
            "max_uplink_message_bytes",
            "max_downlink_message_bytes",
        )
        summary, _cfg, _expectations, spec = _strict_wire_fixture(
            "hybrid4_a_decay_ef", 0.9
        )
        for field in ledger_fields:
            with self.subTest(field=field):
                tampered = copy.deepcopy(summary)
                tampered["rounds"][1]["wire"][field] += 1
                reasons = _validate_wire(tampered, spec)
                self.assertTrue(
                    any(f"round 2.wire.{field}" in reason for reason in reasons),
                    reasons,
                )

        non_monotonic = copy.deepcopy(summary)
        non_monotonic["rounds"][1]["wire"]["uplink_bytes"] = 0
        reasons = _validate_wire(non_monotonic, spec)
        self.assertIn("round 2.wire.uplink_bytes is not monotonic", reasons)

    def test_wire_gate_rejects_any_tampered_final_ledger_field(self):
        ledger_fields = (
            "uplink_bytes",
            "downlink_emitted_bytes",
            "downlink_delivered_bytes",
            "uplink_messages",
            "downlink_messages",
            "schema_bytes",
            "total_delivered_bytes",
            "max_uplink_message_bytes",
            "max_downlink_message_bytes",
        )
        summary, cfg, expectations, spec = _strict_wire_fixture(
            "fedavg_lora", None
        )
        for field in ledger_fields:
            with self.subTest(field=field):
                tampered = copy.deepcopy(summary)
                tampered["final"]["wire"][field] += 1
                reasons = _validate_wire(tampered, spec, cfg, expectations)
                self.assertTrue(
                    any(
                        f"final.wire.{field} differs from the independently "
                        "reconstructed final ledger" in reason
                        for reason in reasons
                    ),
                    reasons,
                )

    def test_actual_wire_ratio_uses_handshake_and_rebuilt_dense_fedavg(self):
        summary, _cfg, _expectations, _spec = _strict_wire_fixture(
            "ffa_lora", None
        )
        numerator = summary["final"]["wire"]["total_delivered_bytes"] + 64

        target = summary["model"]["target_modules"][0]
        dense_manifest = [
            {"name": f"{target}.lora_A", "shape": [2, 4]},
            {"name": f"{target}.lora_B", "shape": [4, 2]},
            {"name": "classifier.bias", "shape": [2]},
            {"name": "classifier.weight", "shape": [2, 4]},
        ]
        dense_schema = len(
            json.dumps(
                dense_manifest, separators=(",", ":"), sort_keys=True
            ).encode()
        ) * 3
        denominator = 78 * 2 * 3 * 2 + dense_schema
        self.assertEqual(_dense_fedavg_all_in_wire_bytes(summary), denominator)
        self.assertAlmostEqual(_actual_wire_ratio(summary), numerator / denominator)

        # The ratio must not trust the candidate's declared reference packet.
        summary["wire_budget"]["reference_message_bytes"] = 1
        self.assertAlmostEqual(_actual_wire_ratio(summary), numerator / denominator)

        no_handshake = copy.deepcopy(summary)
        no_handshake["setup_communication"][
            "basis_hash_handshake_bytes_per_client"
        ] = 0
        self.assertAlmostEqual(
            _actual_wire_ratio(summary) - _actual_wire_ratio(no_handshake),
            64 / denominator,
        )

    def test_full_lora_ef_is_paired_to_dense_fedavg_across_wire_budgets(self):
        cfg = {
            "data": {"tasks": ["sst2"], "partition_seeds": [401, 402, 403]},
            "communication": {"wire_ratios": [0.1, 0.2]},
            "matrix": {
                "dense_methods": ["fedavg_lora"],
                "sparse_methods": ["full_lora_ef"],
                "diagnostic_dense_methods": [],
                "diagnostic_sparse_methods": [],
            },
        }
        rows = []
        for seed in cfg["data"]["partition_seeds"]:
            rows.append(
                {
                    "run": {
                        "task": "sst2",
                        "seed": seed,
                        "method": "fedavg_lora",
                        "wire_ratio": None,
                    },
                    "final": {"primary_metric": 0.80},
                }
            )
            for ratio, score in ((0.1, 0.78), (0.2, 0.79)):
                rows.append(
                    {
                        "run": {
                            "task": "sst2",
                            "seed": seed,
                            "method": "full_lora_ef",
                            "wire_ratio": ratio,
                        },
                        "final": {"primary_metric": score},
                    }
                )

        pairs = [
            row
            for row in _pair_rows(cfg, rows)
            if row["comparison"] == "full_lora_ef_vs_fedavg"
        ]
        self.assertEqual(len(pairs), 6)
        self.assertTrue(all(row["pair_status"] == "COMPLETE" for row in pairs))
        self.assertTrue(all(row["comparable"] is True for row in pairs))
        self.assertTrue(all(row["right_method"] == "fedavg_lora" for row in pairs))
        self.assertTrue(
            all(row["right_metric_semantics"] == "communicated_global" for row in pairs)
        )
        ten_percent = [row for row in pairs if row["wire_ratio"] == 0.1]
        self.assertTrue(
            all(abs(float(row["delta"]) + 0.02) < 1e-12 for row in ten_percent)
        )
        analysis = _analysis_markdown(cfg, [], pairs)
        self.assertIn("Full-LoRA 压缩成本", analysis)
        self.assertIn("FedAvg-LoRA+Top-k+EF − dense FedAvg-LoRA", analysis)
        self.assertIn("-0.020000 ± 0.000000", analysis)
        self.assertNotIn("## 核心方法比较", analysis)
        self.assertNotIn("## Error-feedback 归因", analysis)

    def test_random_a_dense_uses_fixed_a_report_validation(self):
        self.assertIn("random_a_dense", FIXED_A_METHODS)

    def test_dense_spectral_random_pairs_are_reported_without_wire_ratio(self):
        cfg = {
            "data": {"tasks": ["sst2"], "partition_seeds": [401, 402, 403]},
            "communication": {"wire_ratios": [0.1, 0.2]},
            "matrix": {
                "dense_methods": [],
                "diagnostic_dense_methods": [
                    "random_a_dense",
                    "spectral_a_dense",
                ],
            },
        }
        rows = []
        for seed in cfg["data"]["partition_seeds"]:
            for method, score in (
                ("random_a_dense", 0.80),
                ("spectral_a_dense", 0.82),
            ):
                rows.append(
                    {
                        "run": {
                            "task": "sst2",
                            "seed": seed,
                            "method": method,
                            "wire_ratio": None,
                        },
                        "final": {"primary_metric": score},
                    }
                )

        pairs = _pair_rows(cfg, rows)
        dense_pairs = [
            row
            for row in pairs
            if row["comparison"] == "dense_spectral_vs_random"
        ]
        self.assertEqual(len(dense_pairs), 3)
        self.assertTrue(all(row["wire_ratio"] is None for row in dense_pairs))
        self.assertTrue(all(row["pair_status"] == "COMPLETE" for row in dense_pairs))
        self.assertTrue(all(abs(float(row["delta"]) - 0.02) < 1e-12 for row in dense_pairs))

        analysis = _analysis_markdown(cfg, [], pairs)
        self.assertIn("Spectral-A dense − Random-A dense", analysis)
        self.assertIn("SUPPORTED", analysis)

    def test_dense_section_is_absent_when_diagnostics_are_not_configured(self):
        cfg = {
            "data": {"tasks": ["sst2"], "partition_seeds": [401, 402, 403]},
            "communication": {"wire_ratios": [0.1, 0.2]},
            "matrix": {"dense_methods": [], "diagnostic_dense_methods": []},
        }
        pairs = _pair_rows(cfg, [])
        self.assertFalse(
            any(row["comparison"] == "dense_spectral_vs_random" for row in pairs)
        )
        self.assertNotIn("无压缩基底诊断", _analysis_markdown(cfg, [], pairs))

    def test_flasc_comparison_is_marked_as_semantically_non_comparable(self):
        cfg = {
            "data": {"tasks": ["sst2"], "partition_seeds": [401, 402, 403]},
            "communication": {"wire_ratios": [0.1]},
            "matrix": {"dense_methods": [], "diagnostic_dense_methods": []},
        }
        valid_rows = []
        for seed in cfg["data"]["partition_seeds"]:
            for method, score in (("spectral_a_ef", 0.8), ("flasc", 0.7)):
                valid_rows.append(
                    {
                        "run": {
                            "task": "sst2",
                            "seed": seed,
                            "method": method,
                            "wire_ratio": 0.1,
                        },
                        "final": {"primary_metric": score},
                    }
                )
        pair_rows = _pair_rows(cfg, valid_rows)
        flasc_rows = [
            row
            for row in pair_rows
            if row["comparison"] == "proposed_vs_flasc"
        ]
        self.assertTrue(flasc_rows)
        self.assertTrue(all(row["comparable"] is False for row in flasc_rows))
        self.assertTrue(
            all(row["interpretation"] == "DESCRIPTIVE_ONLY" for row in flasc_rows)
        )
        self.assertTrue(all(row["threshold"] == "" for row in flasc_rows))

        misleading_rows = [
            {
                "task": "sst2",
                "wire_ratio": 0.1,
                "seed": seed,
                "comparison": "proposed_vs_flasc",
                "pair_status": "COMPLETE",
                "delta": 0.1,
            }
            for seed in cfg["data"]["partition_seeds"]
        ]
        analysis = _analysis_markdown(cfg, [], misleading_rows)
        self.assertIn("口径不一致", analysis)
        self.assertIn("仅作透明描述", analysis)
        self.assertNotIn("**SUPPORTED**", analysis)
        self.assertNotIn("**HURT**", analysis)

    def test_development_ef_pairs_are_labeled_seed_limited_not_missing(self):
        cfg = {
            "data": {"tasks": ["sst2"], "partition_seeds": [311]},
            "communication": {"wire_ratios": [0.1]},
            "matrix": {"dense_methods": [], "diagnostic_dense_methods": []},
            "evaluation": {"policy": "train_holdout"},
        }
        pair_rows = []
        for comparison in ("random_ef_vs_noef", "spectral_ef_vs_noef"):
            pair_rows.append(
                {
                    "task": "sst2",
                    "wire_ratio": 0.1,
                    "seed": 311,
                    "comparison": comparison,
                    "pair_status": "COMPLETE",
                    "delta": 0.1,
                }
            )
        analysis = _analysis_markdown(cfg, [], pair_rows)
        self.assertIn("已有 EF/no-EF 对照，但种子不足", analysis)
        self.assertNotIn("缺少同 task、seed、wire 的 no-EF 对照", analysis)
        self.assertIn("pooled train-holdout", analysis)

    def test_communication_efficiency_uses_paired_fedavg_and_actual_wire(self):
        cfg = {
            "data": {"tasks": ["sst2"], "partition_seeds": [401, 402]},
            "communication": {"wire_ratios": [0.1]},
            "matrix": {
                "dense_methods": ["fedavg_lora", "lora_xs"],
                "sparse_methods": ["flasc", "random_a_ef", "spectral_a_ef"],
                "diagnostic_dense_methods": ["random_a_dense"],
            },
        }
        rows = []
        methods = (
            ("fedavg_lora", None, 0.80, 1000, 0),
            ("lora_xs", None, 0.70, 100, 0),
            ("flasc", 0.1, 0.81, 100, 0),
            ("random_a_ef", 0.1, 0.78, 190, 1),
            ("spectral_a_ef", 0.1, 0.79, 190, 1),
            ("random_a_dense", None, 0.76, 300, 1),
        )
        for seed in cfg["data"]["partition_seeds"]:
            for method, ratio, score, recurrent_bytes, handshake_per_client in methods:
                if method == "fedavg_lora":
                    round_scores, round_bytes = (0.75, score), (500, recurrent_bytes)
                elif method == "spectral_a_ef":
                    round_scores, round_bytes = (0.77, score), (90, recurrent_bytes)
                elif method == "flasc":
                    round_scores, round_bytes = (0.80, score), (50, recurrent_bytes)
                else:
                    round_scores, round_bytes = (score - 0.08, score), (
                        recurrent_bytes / 2,
                        recurrent_bytes,
                    )
                rows.append(
                    {
                        "run": {
                            "task": "sst2",
                            "seed": seed,
                            "method": method,
                            "wire_ratio": ratio,
                        },
                        "data": {"num_clients": 10},
                        "setup_communication": {
                            "basis_hash_handshake_bytes_per_client": handshake_per_client
                        },
                        "final": {
                            "primary_metric": score,
                            "wire": {"total_delivered_bytes": recurrent_bytes},
                        },
                        "rounds": [
                            {
                                "round": index,
                                "primary_metric": round_score,
                                "wire": {"total_delivered_bytes": round_wire},
                            }
                            for index, (round_score, round_wire) in enumerate(
                                zip(round_scores, round_bytes), start=1
                            )
                        ],
                    }
                )

        by_seed = _communication_efficiency_seed_rows(cfg, rows)
        self.assertTrue(by_seed)
        self.assertTrue(
            all(
                tuple(row) == COMMUNICATION_EFFICIENCY_BY_SEED_FIELDS
                for row in by_seed
            )
        )
        spectral_seed = next(
            row
            for row in by_seed
            if row["method"] == "spectral_a_ef" and row["seed"] == 401
        )
        self.assertAlmostEqual(spectral_seed["wire_fraction_vs_fedavg"], 0.20)
        self.assertAlmostEqual(spectral_seed["delta_accuracy_vs_fedavg"], -0.01)
        self.assertEqual(spectral_seed["c95_dense_status"], "REACHED")
        self.assertEqual(spectral_seed["c95_dense_round"], 1)
        self.assertAlmostEqual(spectral_seed["c95_dense_wire_bytes"], 100.0)
        self.assertAlmostEqual(
            spectral_seed["speedup95_vs_fedavg"], 10.0
        )
        lora_xs_seed = next(
            row
            for row in by_seed
            if row["method"] == "lora_xs" and row["seed"] == 401
        )
        self.assertEqual(lora_xs_seed["c95_dense_status"], "NOT_REACHED")
        self.assertEqual(lora_xs_seed["speedup95_vs_fedavg"], "")

        efficiency = _communication_efficiency_rows(cfg, rows)
        self.assertTrue(efficiency)
        self.assertTrue(
            all(tuple(row) == COMMUNICATION_EFFICIENCY_FIELDS for row in efficiency)
        )
        by_method = {row["method"]: row for row in efficiency}
        self.assertNotIn("random_a_dense", by_method)

        spectral = by_method["spectral_a_ef"]
        self.assertAlmostEqual(spectral["mean_normal_path_wire_bytes"], 200.0)
        self.assertAlmostEqual(spectral["mean_wire_fraction_vs_fedavg"], 0.20)
        self.assertAlmostEqual(spectral["mean_saving_vs_fedavg"], 0.80)
        self.assertAlmostEqual(spectral["mean_delta_accuracy_vs_fedavg"], -0.01)
        self.assertAlmostEqual(
            spectral["mean_accuracy_retention_vs_fedavg"], 0.9875
        )
        self.assertAlmostEqual(spectral["mean_descriptive_cer"], 4.9375)
        for field in (
            "sd_wire_fraction_vs_fedavg",
            "sd_saving_vs_fedavg",
            "sd_delta_accuracy_vs_fedavg",
            "sd_accuracy_retention_vs_fedavg",
            "sd_descriptive_cer",
            "sd_fedavg_c95_dense_wire_bytes",
            "sd_speedup95_vs_fedavg",
        ):
            self.assertEqual(spectral[field], 0.0)
        self.assertEqual(spectral["c95_dense_reached_n"], 2)
        self.assertAlmostEqual(spectral["mean_c95_dense_round"], 1.0)
        self.assertAlmostEqual(spectral["mean_c95_dense_wire_bytes"], 100.0)
        self.assertAlmostEqual(
            spectral["mean_speedup95_vs_fedavg"], 10.0
        )
        self.assertEqual(by_method["lora_xs"]["c95_dense_reached_n"], 0)
        self.assertEqual(by_method["lora_xs"]["mean_c95_dense_wire_bytes"], "")
        self.assertEqual(spectral["pareto_status"], "YES")
        self.assertEqual(by_method["random_a_ef"]["pareto_status"], "NO")
        self.assertEqual(
            by_method["flasc"]["pareto_status"], "REFERENCE_STYLE_YES"
        )
        self.assertEqual(
            by_method["flasc"]["interpretation"],
            "DESCRIPTIVE_ONLY_METRIC_STATE_MISMATCH",
        )

    def test_mrpc_legacy_accuracy_and_primary_score_are_not_conflated(self):
        cfg = {
            "data": {"tasks": ["mrpc"], "partition_seeds": [601]},
            "communication": {"wire_ratios": [0.1]},
            "matrix": {
                "dense_methods": ["fedavg_lora"],
                "sparse_methods": ["hybrid4_a_decay_ef"],
                "diagnostic_dense_methods": [],
            },
        }
        rows = []
        for method, ratio, accuracy, f1, wire in (
            ("fedavg_lora", None, 0.80, 0.60, 1000),
            ("hybrid4_a_decay_ef", 0.1, 0.75, 0.75, 100),
        ):
            primary = 0.5 * (accuracy + f1)
            rows.append(
                {
                    "run": {
                        "task": "mrpc",
                        "seed": 601,
                        "method": method,
                        "wire_ratio": ratio,
                    },
                    "data": {"num_clients": 10},
                    "setup_communication": {
                        "basis_hash_handshake_bytes_per_client": 0
                    },
                    "final": {
                        "primary_metric": primary,
                        "splits": {
                            "validation": {"accuracy": accuracy, "f1": f1}
                        },
                        "wire": {"total_delivered_bytes": wire},
                    },
                    "rounds": [
                        {
                            "round": 1,
                            "primary_metric": primary,
                            "wire": {"total_delivered_bytes": wire},
                        }
                    ],
                }
            )
        by_seed = _communication_efficiency_seed_rows(cfg, rows)
        candidate = next(
            row for row in by_seed if row["method"] == "hybrid4_a_decay_ef"
        )
        self.assertEqual(candidate["primary_metric_name"], "glue_accuracy_f1_mean")
        self.assertAlmostEqual(candidate["accuracy"], 0.75)
        self.assertAlmostEqual(candidate["primary_score"], 0.75)
        self.assertAlmostEqual(candidate["reference_accuracy"], 0.80)
        self.assertAlmostEqual(candidate["reference_primary_score"], 0.70)
        self.assertAlmostEqual(candidate["delta_accuracy_vs_fedavg"], -0.05)
        self.assertAlmostEqual(candidate["delta_primary_score_vs_fedavg"], 0.05)
        expected_primary_target = 0.95 * 0.70
        self.assertAlmostEqual(
            candidate["c95_dense_target_primary_score"],
            expected_primary_target,
        )
        # The legacy accuracy-named column is intentionally blank for
        # composite-metric tasks; only the explicitly named primary-score
        # column may carry the C95 target.
        self.assertEqual(candidate["c95_dense_target_accuracy"], "")
        self.assertEqual(candidate["c95_dense_status"], "REACHED")


if __name__ == "__main__":
    unittest.main()
