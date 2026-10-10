import unittest
import copy

import torch

from fedpeft.adapters import TensorLayout
from fedpeft.compression import BlockTopKCompressor, TopKCompressor
from fedpeft.diagnostics import DIAGNOSTIC_RECIPE_VERSION
from fedpeft.engine import (
    BLOCK_TOPK_EF_METHODS,
    DENSE_METHODS,
    EF_METHODS,
    STABILITY_CANDIDATE_METHODS,
    STABILITY_DIAGNOSTIC_METHODS,
    TOPK_NO_EF_METHODS,
    RunSpec,
    WireTracker,
    _adapter_construction_method,
    _adapter_factor_audit,
    _build_adapter_compressor,
    _build_sparse_diagnostics,
    _compressor_metadata,
    _factor_selected_counts,
    _lr_multiplier,
    _lora_factor_slices,
    _resolve_lr_schedule,
    _round_learning_rates,
    _restore_sparse_diagnostics,
    run_experiment,
)
from fedpeft.wire import encode_sparse


class EnginePathSafetyTests(unittest.TestCase):
    @staticmethod
    def fixed_a_layout() -> TensorLayout:
        return TensorLayout(
            names=["layer0.lora_B", "layer1.lora_B"],
            shapes=[(2, 2), (2, 2)],
            numels=[4, 4],
        )

    def test_rejects_path_like_method_before_creating_run_directory(self):
        cfg = {"data": {"tasks": ["sst2"]}}
        spec = RunSpec("sst2", "../../outside", 401, 0.1)
        with self.assertRaisesRegex(ValueError, "unsupported method"):
            run_experiment(cfg, spec)

    def test_random_a_dense_is_dispatched_as_dense(self):
        cfg = {
            "data": {"tasks": ["sst2"], "partition_seeds": [401]},
            "matrix": {"diagnostic_dense_methods": ["random_a_dense"]},
        }
        spec = RunSpec("sst2", "random_a_dense", 401, 0.1)
        with self.assertRaisesRegex(ValueError, "dense methods must not specify"):
            run_experiment(cfg, spec)

    def test_full_lora_ef_is_dispatched_only_as_error_feedback_sparse(self):
        self.assertIn("full_lora_ef", EF_METHODS)
        self.assertNotIn("full_lora_ef", DENSE_METHODS)
        self.assertNotIn("full_lora_ef", TOPK_NO_EF_METHODS)
        cfg = {
            "data": {"tasks": ["sst2"], "partition_seeds": [401]},
            "matrix": {"sparse_methods": ["full_lora_ef"]},
        }
        spec = RunSpec("sst2", "full_lora_ef", 401, None)
        with self.assertRaisesRegex(ValueError, "sparse methods require"):
            run_experiment(cfg, spec)

    def test_stability_methods_use_frozen_ef_and_compressor_dispatch(self):
        self.assertEqual(
            STABILITY_CANDIDATE_METHODS,
            {
                "spectral_rot_ef",
                "spectral_block_ef",
                "hybrid4_a_ef",
                "hybrid4_a_decay_ef",
            },
        )
        self.assertTrue(STABILITY_CANDIDATE_METHODS.issubset(EF_METHODS))
        self.assertTrue(
            {"spectral_a_ef", "random_block_ef"}.issubset(
                STABILITY_DIAGNOSTIC_METHODS
            )
        )
        self.assertEqual(
            BLOCK_TOPK_EF_METHODS,
            {"spectral_block_ef", "random_block_ef"},
        )
        self.assertEqual(_adapter_construction_method("random_block_ef"), "random_a_ef")
        self.assertEqual(_adapter_construction_method("spectral_rot_ef"), "spectral_rot_ef")

        layout = self.fixed_a_layout()
        cap = 21 + 6 * 4
        for method in (
            "spectral_a_ef",
            "spectral_rot_ef",
            "hybrid4_a_ef",
            "hybrid4_a_decay_ef",
        ):
            compressor = _build_adapter_compressor(method, layout, cap)
            self.assertIs(type(compressor), TopKCompressor)
            self.assertEqual(compressor.k, 4)
        for method in BLOCK_TOPK_EF_METHODS:
            compressor = _build_adapter_compressor(method, layout, cap)
            self.assertIsInstance(compressor, BlockTopKCompressor)
            self.assertEqual(compressor.quotas, (2, 2))

    def test_hybrid4_aliases_dispatch_to_the_same_basis_family(self):
        aliases = {
            "hybrid4_a_dense",
            "hybrid4_a_decay_dense",
            "hybrid4_a_ef",
            "hybrid4_a_decay_ef",
        }
        self.assertTrue({"hybrid4_a_dense", "hybrid4_a_decay_dense"}.issubset(DENSE_METHODS))
        self.assertTrue({"hybrid4_a_ef", "hybrid4_a_decay_ef"}.issubset(EF_METHODS))
        self.assertTrue(DENSE_METHODS.isdisjoint(EF_METHODS))
        for method in aliases:
            self.assertEqual(_adapter_construction_method(method), "hybrid4_a_ef")

    def test_late_cosine_schedule_is_frozen_and_synchronous(self):
        cfg = {
            "training": {
                "schedule_overrides": {
                    "hybrid4_a_decay_ef": {
                        "name": "late_cosine_v1",
                        "start_fraction": 0.8,
                        "final_multiplier": 0.1,
                        "apply_to": ["adapter", "head"],
                    }
                }
            }
        }
        for rounds, constant_end, first_decay in ((30, 24, 25), (50, 40, 41)):
            schedule = _resolve_lr_schedule(cfg, "hybrid4_a_decay_ef", rounds)
            self.assertEqual(schedule["start_round"], constant_end)
            self.assertEqual(_lr_multiplier(constant_end, schedule), 1.0)
            self.assertLess(_lr_multiplier(first_decay, schedule), 1.0)
            self.assertAlmostEqual(_lr_multiplier(rounds, schedule), 0.1)
            rates = _round_learning_rates(3e-4, 3e-4, rounds, schedule)
            self.assertAlmostEqual(float(rates["adapter_learning_rate"]), 3e-5)
            self.assertEqual(
                rates["adapter_learning_rate"], rates["head_learning_rate"]
            )

        constant = _resolve_lr_schedule(cfg, "hybrid4_a_ef", 50)
        self.assertEqual(_lr_multiplier(1, constant), 1.0)
        self.assertEqual(_lr_multiplier(50, constant), 1.0)

    def test_compressor_metadata_records_exact_sparse_and_diagnostic_recipe(self):
        layout = self.fixed_a_layout()
        compressor = _build_adapter_compressor(
            "spectral_block_ef", layout, 21 + 6 * 4
        )
        diagnostics = _build_sparse_diagnostics(
            "spectral_block_ef",
            layout,
            rank=2,
            num_clients=1,
            recipe_version=DIAGNOSTIC_RECIPE_VERSION,
        )
        self.assertIsNotNone(diagnostics)
        metadata = _compressor_metadata(compressor, diagnostics)
        self.assertEqual(metadata["compressor_kind"], "block_topk")
        self.assertEqual(metadata["block_numels"], [4, 4])
        self.assertEqual(metadata["block_quotas"], [2, 2])
        self.assertEqual(len(metadata["block_quota_hash"]), 64)
        self.assertEqual(metadata["expected_sparse_adapter_bytes"], 21 + 6 * 4)
        self.assertEqual(
            metadata["diagnostic_recipe_version"], DIAGNOSTIC_RECIPE_VERSION
        )
        self.assertEqual(
            metadata["diagnostic_layout_fingerprint"],
            diagnostics.layout_fingerprint,
        )

        global_metadata = _compressor_metadata(
            _build_adapter_compressor("spectral_rot_ef", layout, 21 + 6 * 4),
            diagnostics,
        )
        self.assertEqual(global_metadata["compressor_kind"], "global_topk")
        self.assertIsNone(global_metadata["block_quotas"])
        self.assertIsNone(global_metadata["block_quota_hash"])

        full_lora_layout = TensorLayout(
            names=["layer0.lora_A", "layer0.lora_B"],
            shapes=[(2, 2), (2, 2)],
            numels=[4, 4],
        )
        full_lora_compressor = _build_adapter_compressor(
            "full_lora_ef", full_lora_layout, 21 + 6 * 4
        )
        self.assertIs(type(full_lora_compressor), TopKCompressor)
        self.assertIsNone(
            _build_sparse_diagnostics(
                "full_lora_ef",
                full_lora_layout,
                rank=2,
                num_clients=1,
                recipe_version=DIAGNOSTIC_RECIPE_VERSION,
            )
        )

    def test_diagnostic_checkpoint_restore_is_complete_and_strict(self):
        layout = self.fixed_a_layout()
        compressor = _build_adapter_compressor(
            "spectral_block_ef", layout, 21 + 6 * 2
        )
        diagnostics = _build_sparse_diagnostics(
            "spectral_block_ef", layout, rank=2, num_clients=1, recipe_version="v1"
        )
        assert diagnostics is not None
        raw = torch.tensor([4.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
        client = compressor.compress(raw)
        client_residual = raw - client.reconstructed
        diagnostics.observe_client(
            1,
            0,
            client.packet,
            raw,
            raw,
            client.reconstructed,
            client_residual,
        )
        server = compressor.compress(client.reconstructed)
        server_residual = client.reconstructed - server.reconstructed
        diagnostics.observe_server(
            1,
            server.packet,
            client.reconstructed,
            client.reconstructed,
            server.reconstructed,
            server_residual,
        )
        diagnostics.finish_round(
            1,
            [client.reconstructed],
            [1.0],
            client.reconstructed,
            server.reconstructed,
        )
        state = diagnostics.state_dict()

        restored = _build_sparse_diagnostics(
            "spectral_block_ef", layout, rank=2, num_clients=1, recipe_version="v1"
        )
        assert restored is not None
        _restore_sparse_diagnostics(restored, state, completed_round=1)
        self.assertEqual(restored.completed_round, 1)
        self.assertEqual(restored.layout_fingerprint, diagnostics.layout_fingerprint)

        with self.assertRaisesRegex(RuntimeError, "missing complete"):
            _restore_sparse_diagnostics(restored, None, completed_round=1)
        incomplete = copy.deepcopy(state)
        del incomplete["previous_aggregate"]
        fresh = _build_sparse_diagnostics(
            "spectral_block_ef", layout, rank=2, num_clients=1, recipe_version="v1"
        )
        with self.assertRaisesRegex(RuntimeError, "checkpoint is invalid"):
            _restore_sparse_diagnostics(fresh, incomplete, completed_round=1)
        fresh = _build_sparse_diagnostics(
            "spectral_block_ef", layout, rank=2, num_clients=1, recipe_version="v1"
        )
        with self.assertRaisesRegex(RuntimeError, "round differs"):
            _restore_sparse_diagnostics(fresh, state, completed_round=2)

    def test_full_lora_wire_tracker_preserves_cumulative_delivered_bytes(self):
        clients = 10
        cases = (
            (30, 59_288, 35_615_535),
            (30, 118_580, 71_190_735),
            (50, 59_288, 59_330_735),
            (50, 118_580, 118_622_735),
            (50, 59_446, 59_488_735),
            (50, 118_888, 118_930_735),
        )
        for rounds, message_bytes, expected_total in cases:
            tracker = WireTracker(schema_bytes=42_735)
            cumulative = []
            for round_index in range(rounds):
                for _ in range(clients):
                    tracker.upload(message_bytes)
                tracker.download(message_bytes, clients)
                cumulative.append(tracker.total_delivered_bytes)
                self.assertEqual(
                    cumulative[-1],
                    42_735 + (round_index + 1) * 2 * clients * message_bytes,
                )
            self.assertEqual(cumulative, sorted(cumulative))
            self.assertEqual(tracker.total_delivered_bytes, expected_total)

    def test_factor_audit_preserves_zero_selected_factor(self):
        layout = TensorLayout(
            names=[
                "layer0.lora_A",
                "layer0.lora_B",
                "layer1.lora_A",
                "layer1.lora_B",
            ],
            shapes=[(1, 2), (1, 3), (1, 2), (1, 3)],
            numels=[2, 3, 2, 3],
        )
        factor_slices = _lora_factor_slices(layout)
        packet = encode_sparse(
            torch.tensor([0, 1, 5, 6]),
            torch.tensor([3.0, 4.0, 1.0e-12, 0.0]),
            logical_numel=layout.total_numel,
        )
        selected = _factor_selected_counts(packet, factor_slices)
        self.assertEqual(selected, {"lora_A": 4, "lora_B": 0})

        audit = _adapter_factor_audit(
            factor_slices=factor_slices,
            aggregate_update=torch.tensor(
                [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            server_residual=torch.tensor(
                [0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            ),
            client_residuals=[
                torch.tensor(
                    [6.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                ),
                torch.zeros(10),
            ],
            server_selected_counts=selected,
            client_selected_counts={"lora_A": 8, "lora_B": 0},
        )
        self.assertEqual(audit["lora_A"]["logical_numel"], 4)
        self.assertEqual(audit["lora_B"]["logical_numel"], 6)
        self.assertEqual(audit["lora_A"]["aggregate_update_norm"], 5.0)
        self.assertEqual(audit["lora_B"]["aggregate_update_norm"], 0.0)
        self.assertEqual(audit["lora_A"]["server_residual_norm"], 0.0)
        self.assertEqual(audit["lora_B"]["server_residual_norm"], 5.0)
        self.assertEqual(
            audit["lora_B"]["server_transmitted_selected_count"], 0
        )
        self.assertEqual(
            audit["lora_B"]["client_transmitted_selected_count_total"], 0
        )
        self.assertEqual(
            audit["lora_B"]["client_transmitted_selected_count_mean"], 0.0
        )
        self.assertEqual(audit["lora_A"]["client_residual_norm_mean"], 5.0)
        self.assertEqual(audit["lora_B"]["client_residual_norm_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
