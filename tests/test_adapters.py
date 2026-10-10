import copy
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from fedpeft.adapters import (
    FIXED_SPECTRAL_METHODS,
    HYBRID4_METHODS,
    STABILITY_SPECTRAL_METHODS,
    FixedALinear,
    LoRALinear,
    LoRAXSLinear,
    _hybrid4_spectral_basis,
    _rotate_spectral_basis,
    _stabilize_svd_basis,
    _sylvester_hadamard_8,
    build_model,
    model_invariants,
    trainable_parameter_count,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.query = nn.Linear(7, 5, bias=False)
        self.classifier = nn.Linear(5, 2)


class TinyRankEightClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.query = nn.Linear(11, 9, bias=False)
        self.classifier = nn.Linear(9, 2)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = nn.Linear(7, 5, bias=False)
        self.x = torch.randn(3, 7)

    def test_standard_lora_starts_as_identity_update(self):
        layer = LoRALinear(self.base, rank=3, alpha=3, dropout=0, seed=1)
        self.assertTrue(torch.equal(layer(self.x), self.base(self.x)))
        self.assertLessEqual(float(layer.lora_A.detach().abs().max()), 1 / (7**0.5) + 1e-6)

    def test_fixed_a_matches_explicit_delta(self):
        raw = torch.randn(7, 3)
        fixed_a = torch.linalg.qr(raw, mode="reduced").Q.T
        layer = FixedALinear(self.base, fixed_a, rank=3, alpha=3, dropout=0)
        with torch.no_grad():
            layer.lora_B.normal_()
        expected = self.base(self.x) + F.linear(self.x, layer.lora_B @ layer.fixed_A)
        self.assertTrue(torch.allclose(layer(self.x), expected, atol=1e-6))
        gram = layer.fixed_A @ layer.fixed_A.T
        self.assertTrue(torch.allclose(gram, torch.eye(3), atol=1e-5))

    def test_lora_xs_matches_explicit_delta_rectangular(self):
        left, singular, right = torch.linalg.svd(self.base.weight.float(), full_matrices=False)
        layer = LoRAXSLinear(
            self.base,
            left[:, :3],
            singular[:3],
            right[:3],
            rank=3,
            alpha=3,
            dropout=0,
            init_std=1e-5,
            seed=1,
        )
        with torch.no_grad():
            layer.lora_R.copy_(torch.tensor([[0.2, -0.4, 0.1], [0.3, 0.5, -0.2], [-0.1, 0.6, 0.7]]))
        # Match the official LoRA-XS factorization: B @ R @ A, where
        # B = U_r and A = diag(S_r) @ Vh_r.
        delta = layer.fixed_left @ layer.lora_R @ torch.diag(layer.fixed_singular) @ layer.fixed_right
        expected = self.base(self.x) + F.linear(self.x, delta)
        self.assertTrue(torch.allclose(layer(self.x), expected, atol=1e-6))

    def test_svd_basis_stabilization_is_orthonormal_and_sign_canonical(self):
        left, _, right = torch.linalg.svd(torch.randn(7, 5), full_matrices=False)
        stable_left, stable_right = _stabilize_svd_basis(left[:, :3], -right[:3])
        self.assertTrue(torch.allclose(stable_left.T @ stable_left, torch.eye(3), atol=1e-6))
        self.assertTrue(torch.allclose(stable_right @ stable_right.T, torch.eye(3), atol=1e-6))
        for row in stable_right:
            pivot = int(row.abs().argmax())
            self.assertGreaterEqual(float(row[pivot]), 0.0)

    def test_sylvester_hadamard_is_frozen_cpu_float64_orthonormal(self):
        hadamard = _sylvester_hadamard_8()
        self.assertEqual(hadamard.device.type, "cpu")
        self.assertEqual(hadamard.dtype, torch.float64)
        self.assertEqual(tuple(hadamard.shape), (8, 8))
        self.assertTrue(torch.allclose(hadamard @ hadamard.T, torch.eye(8, dtype=torch.float64)))
        expected_first = torch.full((8,), 1 / (8**0.5), dtype=torch.float64)
        self.assertTrue(torch.equal(hadamard[0], expected_first))

    def test_spectral_rotation_preserves_projector_and_coordinate_map(self):
        generator = torch.Generator(device="cpu").manual_seed(91)
        spectral = torch.linalg.qr(
            torch.randn(13, 8, generator=generator, dtype=torch.float64), mode="reduced"
        ).Q.T.float()
        rotated, audit = _rotate_spectral_basis(spectral)
        hadamard = _sylvester_hadamard_8()
        self.assertEqual(rotated.dtype, torch.float32)
        self.assertEqual(audit["fixed_basis_recipe"], "spectral_rot_hadamard8_cpu_float64_v1")
        self.assertLess(float(audit["spectral_rotation_projector_error"]), 1e-12)
        self.assertTrue(
            torch.allclose(rotated.T @ rotated, spectral.T @ spectral, atol=2e-7, rtol=0)
        )
        b_spectral = torch.randn(6, 8, generator=generator, dtype=torch.float64)
        b_rotated = b_spectral @ hadamard.T
        self.assertTrue(
            torch.allclose(
                b_rotated @ rotated.double(),
                b_spectral @ spectral.double(),
                atol=5e-7,
                rtol=0,
            )
        )

    def test_hybrid_basis_is_deterministic_canonical_and_orthonormal(self):
        generator = torch.Generator(device="cpu").manual_seed(92)
        spectral = torch.linalg.qr(
            torch.randn(13, 8, generator=generator, dtype=torch.float64), mode="reduced"
        ).Q.T.float()
        global_rng_before = torch.get_rng_state().clone()
        first, first_audit = _hybrid4_spectral_basis(spectral, basis_seed=17, canonical_target_index=2)
        global_rng_after = torch.get_rng_state().clone()
        second, second_audit = _hybrid4_spectral_basis(spectral, basis_seed=17, canonical_target_index=2)
        different_target, _ = _hybrid4_spectral_basis(
            spectral, basis_seed=17, canonical_target_index=3
        )
        self.assertTrue(torch.equal(global_rng_before, global_rng_after))
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_audit, second_audit)
        self.assertTrue(torch.equal(first[:4], spectral[:4]))
        self.assertFalse(torch.equal(first[4:], different_target[4:]))
        expected_seed = 17 + 104729 * 3 + 3001
        self.assertEqual(first_audit["hybrid_basis_seed"], expected_seed)
        self.assertEqual(first_audit["hybrid_leading_row_equality_max_error"], 0.0)
        self.assertLess(float(first_audit["hybrid_mutual_orthogonality_max_error"]), 1e-6)
        self.assertLess(float(first_audit["hybrid_row_orthogonality_max_error"]), 1e-6)
        expected_generator = torch.Generator(device="cpu").manual_seed(expected_seed)
        expected_complement = torch.randn(
            (spectral.shape[1], 4),
            generator=expected_generator,
            device="cpu",
            dtype=torch.float64,
        )
        expected_leading = spectral[:4].double()
        for _ in range(2):
            expected_complement = expected_complement - expected_leading.T @ (
                expected_leading @ expected_complement
            )
        expected_complement, _ = torch.linalg.qr(expected_complement, mode="reduced")
        for column in range(expected_complement.shape[1]):
            pivot = int(expected_complement[:, column].abs().argmax())
            if float(expected_complement[pivot, column]) < 0:
                expected_complement[:, column].neg_()
        expected = torch.cat((spectral[:4], expected_complement.T.float()), dim=0)
        self.assertTrue(torch.equal(first, expected))
        for row in first[4:]:
            pivot = int(row.abs().argmax())
            self.assertGreaterEqual(float(row[pivot]), 0.0)

    def test_stability_bases_require_rank_eight(self):
        spectral = torch.eye(3, 7)
        with self.assertRaisesRegex(ValueError, "rank exactly eight"):
            _rotate_spectral_basis(spectral)
        with self.assertRaisesRegex(ValueError, "rank exactly eight"):
            _hybrid4_spectral_basis(spectral, basis_seed=1, canonical_target_index=0)

    def test_stability_method_sets_and_model_basis_audits(self):
        self.assertEqual(
            STABILITY_SPECTRAL_METHODS,
            {"spectral_rot_ef", "spectral_block_ef", *HYBRID4_METHODS},
        )
        self.assertTrue(STABILITY_SPECTRAL_METHODS.issubset(FIXED_SPECTRAL_METHODS))
        template = TinyRankEightClassifier()

        def build(method: str, cache_path: str):
            with patch(
                "fedpeft.adapters.RobertaLinearClassifier",
                side_effect=lambda *args, **kwargs: copy.deepcopy(template),
            ):
                return build_model(
                    model_path="unused",
                    num_labels=2,
                    method=method,
                    rank=8,
                    alpha=8,
                    dropout=0,
                    target_suffixes=["query"],
                    basis_seed=17,
                    lora_xs_init_std=1e-5,
                    spectral_cache_path=cache_path,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        with tempfile.TemporaryDirectory() as temporary:
            cache_path = f"{temporary}/spectral.pt"
            original = build("spectral_a_ef", cache_path)
            blocked = build("spectral_block_ef", cache_path)
            rotated = build("spectral_rot_ef", cache_path)
            hybrid = build("hybrid4_a_ef", cache_path)
            hybrid_dense = build("hybrid4_a_dense", cache_path)
            hybrid_decay_dense = build("hybrid4_a_decay_dense", cache_path)
            hybrid_decay_ef = build("hybrid4_a_decay_ef", cache_path)

        original_layer = original.model.encoder.query
        blocked_layer = blocked.model.encoder.query
        rotated_layer = rotated.model.encoder.query
        hybrid_layer = hybrid.model.encoder.query
        hybrid_aliases = [hybrid, hybrid_dense, hybrid_decay_dense, hybrid_decay_ef]
        self.assertTrue(torch.equal(original_layer.fixed_A, blocked_layer.fixed_A))
        self.assertEqual(original.adapter_layout.names, blocked.adapter_layout.names)
        self.assertEqual(trainable_parameter_count(original.model), trainable_parameter_count(blocked.model))
        self.assertEqual(trainable_parameter_count(original.model), trainable_parameter_count(rotated.model))
        self.assertEqual(trainable_parameter_count(original.model), trainable_parameter_count(hybrid.model))
        self.assertTrue(torch.equal(hybrid_layer.fixed_A[:4], original_layer.fixed_A[:4]))
        for alias in hybrid_aliases[1:]:
            alias_layer = alias.model.encoder.query
            self.assertTrue(torch.equal(alias_layer.fixed_A, hybrid_layer.fixed_A))
            self.assertEqual(alias.adapter_layout.names, hybrid.adapter_layout.names)
            self.assertEqual(
                trainable_parameter_count(alias.model),
                trainable_parameter_count(hybrid.model),
            )
        self.assertTrue(
            torch.allclose(
                rotated_layer.fixed_A.T @ rotated_layer.fixed_A,
                original_layer.fixed_A.T @ original_layer.fixed_A,
                atol=2e-7,
                rtol=0,
            )
        )

        original_audit = model_invariants(original.model, "spectral_a_ef")
        blocked_audit = model_invariants(blocked.model, "spectral_block_ef")
        rotated_audit = model_invariants(rotated.model, "spectral_rot_ef")
        hybrid_audit = model_invariants(hybrid.model, "hybrid4_a_ef")
        self.assertEqual(original_audit["fixed_basis_hash"], blocked_audit["fixed_basis_hash"])
        self.assertEqual(blocked_audit["fixed_basis_recipe"], "stabilized_spectral_svd_v1")
        self.assertLess(rotated_audit["spectral_rotation_projector_error_max"], 1e-12)
        self.assertEqual(
            rotated_audit["fixed_basis_recipe"], "spectral_rot_hadamard8_cpu_float64_v1"
        )
        self.assertEqual(hybrid_audit["hybrid_leading_row_equality_max_error"], 0.0)
        self.assertLess(hybrid_audit["hybrid_mutual_orthogonality_max_error"], 1e-6)
        self.assertLess(hybrid_audit["hybrid_row_orthogonality_max_error"], 1e-6)
        expected_seed = 17 + 104729 + 3001
        self.assertEqual(hybrid_audit["hybrid_basis_seed_min"], expected_seed)
        self.assertEqual(hybrid_audit["hybrid_basis_seed_max"], expected_seed)

    def test_random_a_dense_matches_random_a_ef_adapter_construction(self):
        template = TinyClassifier()

        def build(method: str):
            with patch(
                "fedpeft.adapters.RobertaLinearClassifier",
                side_effect=lambda *args, **kwargs: copy.deepcopy(template),
            ):
                return build_model(
                    model_path="unused",
                    num_labels=2,
                    method=method,
                    rank=3,
                    alpha=3,
                    dropout=0,
                    target_suffixes=["query"],
                    basis_seed=17,
                    lora_xs_init_std=1e-5,
                    spectral_cache_path="unused.pt",
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        dense = build("random_a_dense")
        compressed = build("random_a_ef")
        dense_layer = dense.model.encoder.query
        compressed_layer = compressed.model.encoder.query
        self.assertIsInstance(dense_layer, FixedALinear)
        self.assertIsInstance(compressed_layer, FixedALinear)
        self.assertEqual(dense.adapter_layout.names, compressed.adapter_layout.names)
        self.assertTrue(torch.equal(dense_layer.fixed_A, compressed_layer.fixed_A))
        gram = dense_layer.fixed_A @ dense_layer.fixed_A.T
        self.assertTrue(torch.allclose(gram, torch.eye(3), atol=1e-6))

    def test_random_block_control_is_byte_identical_to_random_a_ef(self):
        template = TinyClassifier()

        def build(method: str):
            with patch(
                "fedpeft.adapters.RobertaLinearClassifier",
                side_effect=lambda *args, **kwargs: copy.deepcopy(template),
            ):
                return build_model(
                    model_path="unused",
                    num_labels=2,
                    method=method,
                    rank=3,
                    alpha=3,
                    dropout=0,
                    target_suffixes=["query"],
                    basis_seed=17,
                    lora_xs_init_std=1e-5,
                    spectral_cache_path="unused.pt",
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        original = build("random_a_ef")
        block_control = build("random_block_ef")
        original_layer = original.model.encoder.query
        control_layer = block_control.model.encoder.query
        self.assertIsInstance(original_layer, FixedALinear)
        self.assertIsInstance(control_layer, FixedALinear)
        self.assertEqual(original.adapter_layout.names, block_control.adapter_layout.names)
        self.assertEqual(original.adapter_layout.shapes, block_control.adapter_layout.shapes)
        self.assertTrue(torch.equal(original_layer.fixed_A, control_layer.fixed_A))
        self.assertTrue(
            torch.equal(
                original.adapter_layout.flatten(original.model),
                block_control.adapter_layout.flatten(block_control.model),
            )
        )

    def test_full_lora_ef_matches_fedavg_adapter_construction(self):
        template = TinyClassifier()

        def build(method: str):
            with patch(
                "fedpeft.adapters.RobertaLinearClassifier",
                side_effect=lambda *args, **kwargs: copy.deepcopy(template),
            ):
                return build_model(
                    model_path="unused",
                    num_labels=2,
                    method=method,
                    rank=3,
                    alpha=3,
                    dropout=0,
                    target_suffixes=["query"],
                    basis_seed=17,
                    lora_xs_init_std=1e-5,
                    spectral_cache_path="unused.pt",
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

        dense = build("fedavg_lora")
        compressed = build("full_lora_ef")
        dense_layer = dense.model.encoder.query
        compressed_layer = compressed.model.encoder.query
        self.assertIsInstance(dense_layer, LoRALinear)
        self.assertIsInstance(compressed_layer, LoRALinear)
        self.assertEqual(dense.adapter_layout.names, compressed.adapter_layout.names)
        self.assertEqual(dense.adapter_layout.shapes, compressed.adapter_layout.shapes)
        self.assertTrue(
            torch.equal(
                dense.adapter_layout.flatten(dense.model),
                compressed.adapter_layout.flatten(compressed.model),
            )
        )
        self.assertTrue(compressed_layer.lora_A.requires_grad)
        self.assertTrue(compressed_layer.lora_B.requires_grad)
        self.assertEqual(torch.count_nonzero(compressed_layer.lora_B).item(), 0)


if __name__ == "__main__":
    unittest.main()
