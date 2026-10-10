import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from fedpeft.adapters import FixedALinear, model_invariants
from fedpeft.cab import (
    ACTIVATION_METHODS,
    ACTIVATION_RECIPE,
    EVA_METHODS,
    GRADIENT_FREE_METHODS,
    WHITENED_METHODS,
    encode_factor_payload,
    CAB_KAPPA_ZERO_METHODS,
    CAB_METHODS,
    CAB_RECIPE,
    LayerStatistics,
    broadcast_basis,
    canonical_rows,
    client_weight_gradients,
    effective_clients,
    encode_client_payload,
    kappa_from_budget,
    activation_sketch,
    anisotropy,
    client_input_moments,
    generalized_top_rows,
    kfac_capture,
    orthonormalize_rows,
    scale_to_reference,
    sketch_matrix,
    subspace_overlap,
    top_eigen_rows,
    truncated_factors,
    whiten_rows,
)
from fedpeft.config import CAB_METHODS as CONFIG_CAB_METHODS
from fedpeft.config import load_config, matrix_run_count, validate_config
from fedpeft.dcef import DCEF_CONSTRUCTION, DCEF_SERVER_DISCOUNT_METHODS
from fedpeft.engine import EF_METHODS, RunSpec, WireTracker, _install_cab_basis


ROOT = Path(__file__).resolve().parents[1]


def direct_m(gradients, weights, kappa):
    p = torch.tensor(weights, dtype=torch.float64)
    p = p / p.sum()
    g = [x.double() for x in gradients]
    mean = sum(pi * gi for pi, gi in zip(p, g))
    n_eff = 1.0 / float((p * p).sum())
    het = sum(pi * pi * (gi - mean).T @ (gi - mean) for pi, gi in zip(p, g))
    return mean.T @ mean - kappa * n_eff * het


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.layer1 = FixedALinear(nn.Linear(6, 5), orthonormalize_rows(torch.randn(2, 6, generator=generator)).float(), 2, 4.0, 0.0)
        self.layer2 = FixedALinear(nn.Linear(5, 4), orthonormalize_rows(torch.randn(2, 5, generator=generator)).float(), 2, 4.0, 0.0)
        self.classifier = nn.Linear(4, 2)
        for module in (self.layer1.base, self.layer2.base):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, labels=None, x=None):
        hidden = torch.tanh(self.layer2(torch.tanh(self.layer1(x))))
        logits = self.classifier(hidden)
        return SimpleNamespace(loss=F.cross_entropy(logits, labels), logits=logits)


class TinyData:
    def __init__(self, size=120, seed=1):
        generator = torch.Generator().manual_seed(seed)
        self.x = torch.randn(size, 6, generator=generator)
        self.labels = (self.x[:, 0] + 0.3 * self.x[:, 1] > 0).long()

    def batch(self, indices):
        return {"x": self.x.index_select(0, indices), "labels": self.labels.index_select(0, indices)}


class TinyAttention(nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.q = FixedALinear(nn.Linear(6, 5), orthonormalize_rows(torch.randn(2, 6, generator=generator)).float(), 2, 4.0, 0.0)
        self.v = FixedALinear(nn.Linear(6, 3), orthonormalize_rows(torch.randn(2, 6, generator=generator)).float(), 2, 4.0, 0.0)


class TinySharedModel(nn.Module):
    """q and v read the same input, like the query/value projections of one layer."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.attn = TinyAttention(generator)
        self.classifier = nn.Linear(8, 2)
        for module in (self.attn.q.base, self.attn.v.base):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, labels=None, x=None):
        hidden = torch.tanh(torch.cat([self.attn.q(x), self.attn.v(x)], dim=-1))
        logits = self.classifier(hidden)
        return SimpleNamespace(loss=F.cross_entropy(logits, labels), logits=logits)


def tiny_clients(data):
    positive = torch.nonzero(data.labels == 1).flatten()
    negative = torch.nonzero(data.labels == 0).flatten()
    # label-skewed clients: mostly positive, mostly negative, mixed
    return [
        torch.cat([positive[:30], negative[:5]]),
        torch.cat([negative[:30], positive[30:35]]),
        torch.cat([positive[35:45], negative[30:40]]),
    ]


def tiny_cfg(**cab):
    section = {
        "gamma": 0.2,
        "gamma_source": "provisional",
        "kappa_multiplier": 1.0,
        "kappa_override": None,
        "client_rank": 5,
        "gradient_examples": 64,
        "gradient_batch_size": 16,
    }
    section.update(cab)
    return {"cab": section, "training": {"mixed_precision": "bfloat16"}}


class CabClosedFormTests(unittest.TestCase):
    def test_kappa_matches_the_stationary_formula_and_rejects_unstable_gamma(self):
        # Synthetic check value: d = 256, K = 26, N_eff = 10, gamma = 1 - 0.9^4
        self.assertAlmostEqual(kappa_from_budget(1 - 0.9**4, 256, 26, 10.0), 0.2250, places=3)
        self.assertGreater(kappa_from_budget(0.2, 147_456, 4_422, 7.0), kappa_from_budget(0.2, 147_456, 19_245, 7.0))
        with self.assertRaisesRegex(ValueError, "stability region"):
            kappa_from_budget(0.9, 147_456, 4_422, 7.0)
        with self.assertRaisesRegex(ValueError, "gamma"):
            kappa_from_budget(0.0, 10, 1, 1.0)
        self.assertAlmostEqual(effective_clients([1, 1, 1, 1]), 4.0)
        self.assertAlmostEqual(effective_clients([3, 1]), 16 / 10)

    def test_streaming_statistics_equal_direct_formula_and_full_rank_factors_are_exact(self):
        generator = torch.Generator().manual_seed(3)
        m, n, clients = 12, 9, 4
        gradients = [torch.randn(m, n, generator=generator) for _ in range(clients)]
        sizes = [5.0, 1.0, 3.0, 2.0]
        weights = [s / sum(sizes) for s in sizes]
        kappa = 0.7
        full = LayerStatistics(m, n)
        truncated = LayerStatistics(m, n)
        for g, w in zip(gradients, weights):
            full.add_exact(g, w)
            full.add_reported(*truncated_factors(g, n), w)
            truncated.add_exact(g, w)
            truncated.add_reported(*truncated_factors(g, 3), w)
        n_eff = effective_clients(sizes)
        matrices = full.matrices(kappa, n_eff)
        expected = direct_m(gradients, sizes, kappa)
        self.assertTrue(torch.allclose(matrices["exact_m"], expected, atol=1e-9))
        self.assertTrue(torch.allclose(matrices["reported_m"], expected, atol=1e-9))
        approx = truncated.matrices(kappa, n_eff)["reported_m"]
        self.assertFalse(torch.allclose(approx, expected, atol=1e-3))

    def test_truncated_factors_are_balanced_and_optimal(self):
        generator = torch.Generator().manual_seed(9)
        g = torch.randn(7, 5, generator=generator, dtype=torch.float64)
        left, right = truncated_factors(g, 2)
        self.assertEqual(tuple(left.shape), (7, 2))
        self.assertEqual(tuple(right.shape), (2, 5))
        self.assertTrue(torch.allclose(left.norm(dim=0), right.norm(dim=1), atol=1e-10))
        singular = torch.linalg.svdvals(g)
        self.assertAlmostEqual(float((g - left @ right).norm() ** 2), float((singular[2:] ** 2).sum()), places=8)

    def test_top_eigen_rows_attain_ky_fan_bound_with_canonical_signs(self):
        generator = torch.Generator().manual_seed(5)
        x = torch.randn(10, 10, generator=generator, dtype=torch.float64)
        matrix = x @ x.T - 3.0 * torch.eye(10, dtype=torch.float64)
        rows, values = top_eigen_rows(matrix, 4)
        self.assertTrue(torch.allclose(rows @ rows.T, torch.eye(4, dtype=torch.float64), atol=1e-10))
        self.assertAlmostEqual(float(torch.trace(rows @ matrix @ rows.T)), float(values[:4].sum()), places=8)
        for _ in range(200):
            q, _ = torch.linalg.qr(torch.randn(10, 4, generator=generator, dtype=torch.float64))
            self.assertLessEqual(float(torch.trace(q.T @ matrix @ q)), float(values[:4].sum()) + 1e-9)
        pivots = rows.abs().argmax(dim=1)
        self.assertTrue(bool((rows[torch.arange(4), pivots] > 0).all()))
        self.assertTrue(torch.equal(canonical_rows(-rows), rows))

    def test_wire_roundtrips_keep_small_values_and_orthonormality(self):
        tiny = [1e-7 * torch.randn(3, 5), 1e-7 * torch.randn(2, 5)]
        decoded, nbytes = encode_client_payload(tiny)
        self.assertEqual(nbytes, 13 + 2 * (1 + 25))
        for original, value in zip(tiny, decoded):
            self.assertLess(float((value - original.double()).abs().max() / original.abs().max()), 2e-3)

        rows = [orthonormalize_rows(torch.randn(2, 7)), orthonormalize_rows(torch.randn(3, 9))]
        used, nbytes = broadcast_basis(rows)
        self.assertEqual(nbytes, 13 + 2 * (14 + 27))
        for original, value in zip(rows, used):
            self.assertTrue(torch.allclose(value @ value.T, torch.eye(value.shape[0], dtype=torch.float64), atol=1e-12))
            self.assertGreater(subspace_overlap(original, value), 0.9999)


class CabGradientAndInstallTests(unittest.TestCase):
    def test_client_gradients_match_autograd_and_restore_state(self):
        model = TinyModel()
        data = TinyData()
        indices = tiny_clients(data)[0]
        flags = {name: p.requires_grad for name, p in model.named_parameters()}
        model.train()
        weights = [model.layer1.base.weight, model.layer2.base.weight]
        grads = client_weight_gradients(model, weights, data, indices, 64, 7, seed=4, mixed_precision_dtype=torch.bfloat16)

        generator = torch.Generator().manual_seed(4)
        order = torch.randperm(indices.numel(), generator=generator)[:64]
        reference = copy.deepcopy(model)
        reference.eval()
        ref_weights = [reference.layer1.base.weight, reference.layer2.base.weight]
        for weight in ref_weights:
            weight.requires_grad_(True)
        batch = data.batch(indices.index_select(0, order))
        expected = torch.autograd.grad(reference(**batch).loss, ref_weights)
        for got, want in zip(grads, expected):
            self.assertTrue(torch.allclose(got, want, atol=1e-6))
        self.assertEqual({name: p.requires_grad for name, p in model.named_parameters()}, flags)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(model.training)

    def test_install_replaces_basis_counts_bytes_and_reuses_cache(self):
        data = TinyData()
        clients = tiny_clients(data)
        sizes = [len(c) for c in clients]
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cab_basis.pt"
            model = TinyModel()
            placeholder = model.layer1.fixed_A.clone()
            kwargs = dict(
                spec=RunSpec("sst2", "cab_a_dcef", 361, 0.1),
                train_data=data,
                client_train=clients,
                client_sizes=sizes,
                sparse_k=3,
                adapter_numel=18,
                device=torch.device("cpu"),
                cache_path=cache,
                fingerprints={"config_hash": "c", "source_hash": "s"},
            )
            record, setup = _install_cab_basis(cfg=tiny_cfg(), model=model, **kwargs)
            self.assertFalse(torch.equal(model.layer1.fixed_A, placeholder))
            for layer in (model.layer1, model.layer2):
                a = layer.fixed_A.double()
                self.assertTrue(torch.allclose(a @ a.T, torch.eye(2, dtype=torch.float64), atol=1e-6))
                self.assertEqual(layer.basis_audit["fixed_basis_recipe"], CAB_RECIPE)
            expected_kappa = kappa_from_budget(0.2, 18, 3, effective_clients(sizes))
            self.assertAlmostEqual(record["kappa"], expected_kappa)
            # client_rank 5 covers both layers fully: 5 x (5 + 6) and 4 x (4 + 5) values
            uplink = 13 + 2 * (1 + 5 * (5 + 6) + 4 * (4 + 5))
            downlink = 13 + 2 * (2 * 6 + 2 * 5)
            self.assertEqual(setup, 3 * uplink + 3 * downlink)
            self.assertEqual(record["setup_bytes"], setup)
            self.assertGreater(record["summary"]["overlap_reported_vs_exact_mean"], 0.99)
            self.assertGreater(record["summary"]["reported_energy_fraction_min"], 0.999)
            # J(selected) matches the exact optimum here (full-rank factors) and beats random
            self.assertLessEqual(
                record["summary"]["objective_selected_mean"],
                record["summary"]["objective_random_mean"] + 1e-9,
            )
            self.assertTrue(cache.exists())

            reloaded = TinyModel()
            record2, setup2 = _install_cab_basis(cfg=tiny_cfg(), model=reloaded, **kwargs)
            self.assertTrue(torch.equal(reloaded.layer1.fixed_A, model.layer1.fixed_A))
            self.assertEqual((record2, setup2), (record, setup))
            with self.assertRaisesRegex(RuntimeError, "stale CAB basis cache"):
                _install_cab_basis(cfg=tiny_cfg(kappa_multiplier=2.0), model=TinyModel(), **kwargs)

    def test_kappa_zero_variant_selects_the_consensus_basis(self):
        data = TinyData()
        clients = tiny_clients(data)
        with tempfile.TemporaryDirectory() as tmp:
            record, _ = _install_cab_basis(
                cfg=tiny_cfg(),
                model=TinyModel(),
                spec=RunSpec("sst2", "cab0_a_dcef", 361, 0.1),
                train_data=data,
                client_train=clients,
                client_sizes=[len(c) for c in clients],
                sparse_k=3,
                adapter_numel=18,
                device=torch.device("cpu"),
                cache_path=Path(tmp) / "cab_basis.pt",
                fingerprints={},
            )
        self.assertEqual(record["kappa"], 0.0)
        self.assertGreater(record["summary"]["overlap_vs_consensus_basis_mean"], 0.9)


def anisotropic_problem(d=24, r=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mean = 4.0 * torch.randn(d, generator=generator, dtype=torch.float64)
    x = torch.randn(3000, d, generator=generator, dtype=torch.float64) + mean
    second = x.T @ x / x.shape[0]
    gradients = torch.randn(12, d, generator=generator, dtype=torch.float64) @ torch.diag(
        torch.linspace(1.0, 0.1, d, dtype=torch.float64)
    ) + torch.randn(12, 1, generator=generator, dtype=torch.float64) * mean[None, :]
    return gradients.T @ gradients, second, mean, r


class ActivationBasisTests(unittest.TestCase):
    def test_generalized_rows_are_metric_orthonormal_and_attain_the_kfac_optimum(self):
        matrix, second, _, r = anisotropic_problem()
        rows, values = generalized_top_rows(matrix, second, r, 1e-3)
        regular = second + 1e-3 * torch.trace(second) / second.shape[0] * torch.eye(second.shape[0], dtype=torch.float64)
        self.assertTrue(torch.allclose(rows @ regular @ rows.T, torch.eye(r, dtype=torch.float64), atol=1e-9))
        self.assertTrue(torch.all(values[:-1] >= values[1:]))
        self.assertAlmostEqual(kfac_capture(rows, matrix, second, 1e-3), 1.0, places=9)
        generator = torch.Generator().manual_seed(5)
        for _ in range(5):
            other = torch.randn(r, second.shape[0], generator=generator, dtype=torch.float64)
            self.assertLess(kfac_capture(other, matrix, second, 1e-3), 1.0)
            mix = torch.randn(r, r, generator=generator, dtype=torch.float64)
            self.assertAlmostEqual(
                kfac_capture(mix @ other, matrix, second, 1e-3), kfac_capture(other, matrix, second, 1e-3), places=9
            )
        # The Euclidean optimum is not the curvature-aware optimum on anisotropic inputs.
        self.assertLess(kfac_capture(top_eigen_rows(matrix, r)[0], matrix, second, 1e-3), 0.99)

    def test_whitening_keeps_the_row_space_and_scaling_matches_random_energy(self):
        matrix, second, mean, r = anisotropic_problem()
        euclidean, _ = top_eigen_rows(matrix, r)
        whitened = whiten_rows(euclidean, second, 1e-3)
        self.assertAlmostEqual(subspace_overlap(orthonormalize_rows(whitened), euclidean), 1.0, places=9)
        self.assertGreater(anisotropy(euclidean, second, mean)["energy_ratio"], 5.0)
        scaled = scale_to_reference(whitened, second)
        # Whitening is against the ridge-regularised metric, so the energy sits a
        # little below the random-basis reference.
        ratio = anisotropy(scaled, second, mean)["energy_ratio"]
        self.assertLessEqual(ratio, 1.0)
        self.assertGreater(ratio, 0.97)

    def test_sketch_is_exact_at_full_rank_and_keeps_the_spike(self):
        _, second, _, _ = anisotropic_problem()
        factor, rest = activation_sketch(second, second.shape[0])
        self.assertTrue(torch.allclose(sketch_matrix(factor, rest), second, atol=1e-9))
        factor, rest = activation_sketch(second, 2)
        error = torch.linalg.norm(sketch_matrix(factor, rest) - second) / torch.linalg.norm(second)
        self.assertLess(float(error), 0.1)
        self.assertTrue(torch.all(rest >= 0))

    def test_input_moments_follow_the_gradient_sample(self):
        data = TinyData()
        clients = tiny_clients(data)
        model = TinyModel()
        seconds, firsts = client_input_moments(
            model, [model.layer1, model.layer2], [0, 1], data, clients[0], 20, 8, 7, torch.bfloat16
        )
        generator = torch.Generator().manual_seed(7)
        take = torch.randperm(clients[0].numel(), generator=generator)[:20]
        x = data.x.index_select(0, clients[0].index_select(0, take)).double()
        self.assertTrue(torch.allclose(seconds[0], x.T @ x / 20, atol=1e-6))
        self.assertTrue(torch.allclose(firsts[0], x.mean(dim=0), atol=1e-6))
        self.assertEqual(tuple(seconds[1].shape), (5, 5))
        self.assertTrue(model.training)

    def test_install_activation_methods_count_sketch_bytes_and_report_predictions(self):
        data = TinyData()
        clients = tiny_clients(data)
        sizes = [len(c) for c in clients]
        # Full-rank sketch of the 6- and 5-dim inputs: 1 exponent + 6x6 + 6 + 5x5 + 5 values.
        sketch = 13 + 2 * (1 + 6 * 6 + 6 + 5 * 5 + 5)
        gradient = 13 + 2 * (1 + 5 * (5 + 6) + 4 * (4 + 5))
        downlink = 13 + 2 * (2 * 6 + 2 * 5)
        for method in sorted(ACTIVATION_METHODS):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                model = TinyModel()
                placeholder = model.layer1.fixed_A.double().clone()
                record, setup = _install_cab_basis(
                    cfg=tiny_cfg(),
                    model=model,
                    spec=RunSpec("sst2", method, 361, 0.1),
                    train_data=data,
                    client_train=clients,
                    client_sizes=sizes,
                    sparse_k=3,
                    adapter_numel=18,
                    device=torch.device("cpu"),
                    cache_path=Path(tmp) / "cab_basis.pt",
                    fingerprints={},
                )
                self.assertEqual(record["recipe"], ACTIVATION_RECIPE)
                self.assertEqual(model.layer1.basis_audit["fixed_basis_recipe"], ACTIVATION_RECIPE)
                upload = 0 if method in GRADIENT_FREE_METHODS else gradient
                self.assertEqual(setup, 3 * (upload + sketch + downlink))
                self.assertEqual(record["activation"]["uplink_sketch_bytes_per_client"], [sketch] * 3)
                summary = record["summary"]
                if method in WHITENED_METHODS:
                    self.assertAlmostEqual(summary["energy_ratio_selected_mean"], 1.0, delta=0.05)
                self.assertGreater(summary["kfac_capture_gev0_exact_mean"], 0.999)
                if method in {"gev0_a_ef", "gev0x_a_ef", "gev0_a_dense"}:
                    self.assertGreater(summary["kfac_capture_selected_mean"], 0.99)
                    self.assertGreaterEqual(
                        summary["kfac_capture_selected_mean"], summary["kfac_capture_random_mean"]
                    )
                if method == "randomwh_a_ef":
                    self.assertAlmostEqual(
                        subspace_overlap(orthonormalize_rows(model.layer1.fixed_A), orthonormalize_rows(placeholder)),
                        1.0,
                        places=3,
                    )
                if method == "cab0wh_a_ef":
                    self.assertGreater(summary["overlap_vs_consensus_basis_mean"], 0.9)
                if method == "gev0x_a_ef":
                    self.assertTrue(record["activation"]["oracle_exact_statistics"])
                    self.assertAlmostEqual(summary["overlap_reported_vs_exact_mean"], 1.0, places=6)
                if method in EVA_METHODS:
                    # EVA rows stay orthonormal and span the top input second-moment directions.
                    self.assertLess(model_invariants(model, method)["fixed_a_orthogonality_max_error"], 1e-4)
                    self.assertGreater(summary["overlap_vs_eva_mean"], 0.999)
                else:
                    self.assertEqual(model_invariants(model, method)["fixed_a_orthogonality_max_error"], 0.0)
                    self.assertGreater(model_invariants(model, "random_a_ef")["fixed_a_orthogonality_max_error"], 1e-3)

    def test_int8_payloads_count_bytes_and_bound_the_error(self):
        generator = torch.Generator().manual_seed(3)
        left = torch.randn(7, 3, generator=generator, dtype=torch.float64) * torch.tensor([10.0, 1.0, 0.01], dtype=torch.float64)
        right = torch.randn(3, 5, generator=generator, dtype=torch.float64) * 1e-4
        decoded, nbytes = encode_factor_payload([left, right], "int8")
        # left is quantised per column (3 scales), right per row (3 scales)
        self.assertEqual(nbytes, 13 + 2 + 7 * 3 + 2 * 3 + 3 * 5 + 2 * 3)
        for original, received, axis in ((left, decoded[0], 0), (right, decoded[1], 1)):
            peak = original.abs().amax(dim=axis, keepdim=True)
            self.assertTrue(torch.all((original - received).abs() <= peak / 254 * 1.01 + 1e-12))
        fp16, fp16_bytes = encode_factor_payload([left, right], "float16")
        self.assertEqual(fp16_bytes, 13 + 2 * (1 + 21 + 15))
        rows = orthonormalize_rows(torch.randn(2, 9, generator=generator, dtype=torch.float64))
        received, nbytes = broadcast_basis([rows], orthonormalize=True, dtype="int8")
        self.assertEqual(nbytes, 13 + 2 + 2 * 9 + 2 * 2)
        self.assertTrue(torch.allclose(received[0] @ received[0].T, torch.eye(2, dtype=torch.float64), atol=1e-12))
        self.assertGreater(subspace_overlap(received[0], rows), 0.999)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            encode_factor_payload([left], "bfloat16")

    def test_low_cost_protocol_charges_fewer_setup_bytes(self):
        data = TinyData()
        clients = tiny_clients(data)
        common = dict(
            spec=RunSpec("sst2", "gev0_a_ef", 361, 0.1),
            train_data=data,
            client_train=clients,
            client_sizes=[len(c) for c in clients],
            sparse_k=3,
            adapter_numel=18,
            device=torch.device("cpu"),
            fingerprints={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            _, full = _install_cab_basis(cfg=tiny_cfg(), model=TinyModel(), cache_path=Path(tmp) / "a.pt", **common)
            model = TinyModel()
            record, cheap = _install_cab_basis(
                cfg=tiny_cfg(client_rank=2, activation_rank=1, factor_dtype="int8", basis_dtype="int8"),
                model=model,
                cache_path=Path(tmp) / "b.pt",
                **common,
            )
        gradient = 13 + 2 + (2 * 5 + 2 * 2) + (2 * 6 + 2 * 2) + (2 * 4 + 2 * 2) + (2 * 5 + 2 * 2)
        sketch = 13 + 2 * (1 + 6 + 6 + 5 + 5)
        downlink = 13 + 2 + (2 * 6 + 2 * 2) + (2 * 5 + 2 * 2)
        self.assertEqual(cheap, 3 * (gradient + sketch + downlink))
        self.assertLess(cheap, full)
        self.assertEqual(record["factor_dtype"], "int8")
        self.assertTrue(torch.isfinite(model.layer1.fixed_A).all())

    def test_gram_refinement_normalises_exactly_and_charges_its_bytes(self):
        data = TinyData()
        clients = tiny_clients(data)
        common = dict(
            spec=RunSpec("sst2", "gev0_a_ef", 361, 0.1),
            train_data=data,
            client_train=clients,
            client_sizes=[len(c) for c in clients],
            sparse_k=3,
            adapter_numel=18,
            device=torch.device("cpu"),
            fingerprints={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            # a coarse sketch (rank 1) leaves the plain protocol away from the target energy
            plain, plain_bytes = _install_cab_basis(
                cfg=tiny_cfg(activation_rank=1), model=TinyModel(), cache_path=Path(tmp) / "a.pt", **common
            )
            model = TinyModel()
            refined, refined_bytes = _install_cab_basis(
                cfg=tiny_cfg(activation_rank=1, basis_refine="gram"), model=model, cache_path=Path(tmp) / "b.pt", **common
            )
        self.assertAlmostEqual(refined["summary"]["energy_ratio_installed_mean"], 1.0, delta=2e-3)
        self.assertEqual(refined["activation"]["basis_refine"], "gram")
        # per client: one FP16 packet with 2 upper triangles of 2x2 (3 + 3), 2 traces and the
        # exponent, plus the broadcast of two 2x2 mixes
        upload = 13 + 2 * (1 + 3 + 3 + 2)
        download = 13 + 2 * (4 + 4)
        self.assertEqual(refined["activation"]["refine_bytes"], 3 * (upload + download))
        self.assertEqual(refined_bytes - plain_bytes, 3 * (upload + download))
        self.assertEqual(plain["activation"]["refine_bytes"], 0)

    def test_shared_basis_installs_one_basis_per_input_and_charges_it_once(self):
        data = TinyData()
        clients = tiny_clients(data)
        common = dict(
            spec=RunSpec("sst2", "gev0_a_ef", 361, 0.1),
            train_data=data,
            client_train=clients,
            client_sizes=[len(c) for c in clients],
            sparse_k=3,
            adapter_numel=16,
            device=torch.device("cpu"),
            fingerprints={},
        )
        with tempfile.TemporaryDirectory() as tmp:
            separate, separate_bytes = _install_cab_basis(
                cfg=tiny_cfg(client_rank=2, basis_refine="gram"), model=TinySharedModel(), cache_path=Path(tmp) / "a.pt", **common
            )
            model = TinySharedModel()
            shared, shared_bytes = _install_cab_basis(
                cfg=tiny_cfg(client_rank=2, basis_refine="gram", share_basis=True),
                model=model,
                cache_path=Path(tmp) / "b.pt",
                **common,
            )
        self.assertTrue(torch.equal(model.attn.q.fixed_A, model.attn.v.fixed_A))
        self.assertTrue(shared["activation"]["share_basis"])
        self.assertEqual(shared["activation"]["broadcast_units"], 1)
        self.assertEqual(separate["activation"]["broadcast_units"], 2)
        # gradients: one rank-2 factor pair of the stacked 8 x 6 gradient vs. two pairs
        self.assertEqual(shared["uplink_factor_bytes_per_client"], [13 + 2 * (1 + 2 * 8 + 2 * 6)] * 3)
        self.assertEqual(separate["uplink_factor_bytes_per_client"], [13 + 2 * (1 + 2 * 5 + 2 * 6 + 2 * 3 + 2 * 6)] * 3)
        self.assertEqual(shared["downlink_basis_bytes_per_client"], 13 + 2 * (2 * 6))
        self.assertLess(shared_bytes, separate_bytes)
        self.assertAlmostEqual(shared["summary"]["energy_ratio_installed_mean"], 1.0, delta=2e-3)

    def test_dense_gev_installs_without_a_wire_ratio(self):
        data = TinyData()
        clients = tiny_clients(data)
        with tempfile.TemporaryDirectory() as tmp:
            record, setup = _install_cab_basis(
                cfg=tiny_cfg(),
                model=TinyModel(),
                spec=RunSpec("sst2", "gev0_a_dense", 361, None),
                train_data=data,
                client_train=clients,
                client_sizes=[len(c) for c in clients],
                sparse_k=18,
                adapter_numel=18,
                device=torch.device("cpu"),
                cache_path=Path(tmp) / "cab_basis.pt",
                fingerprints={},
            )
        self.assertEqual(record["kappa"], 0.0)
        self.assertEqual(record["omega"], 0.0)
        self.assertGreater(setup, 0)


class CabRegistrationTests(unittest.TestCase):
    def test_methods_are_dcef_synchronised_fixed_a_variants(self):
        from fedpeft.cab import CAB_EF_METHODS
        from fedpeft.engine import _adapter_construction_method

        self.assertEqual(CAB_METHODS, CONFIG_CAB_METHODS)
        self.assertTrue(CAB_KAPPA_ZERO_METHODS.issubset(CAB_METHODS))
        self.assertTrue((CAB_METHODS - {"gev0_a_dense"}).issubset(EF_METHODS))
        from fedpeft.cab import CAB_DENSE_METHODS
        from fedpeft.engine import DENSE_METHODS

        self.assertTrue((CAB_METHODS - CAB_EF_METHODS - CAB_DENSE_METHODS).issubset(DCEF_SERVER_DISCOUNT_METHODS))
        self.assertTrue(CAB_EF_METHODS.isdisjoint(DCEF_SERVER_DISCOUNT_METHODS))
        self.assertTrue(CAB_DENSE_METHODS.issubset(DENSE_METHODS))
        self.assertTrue(CAB_DENSE_METHODS.isdisjoint(EF_METHODS))
        for method in CAB_METHODS:
            expected = "random_a_dense" if method in CAB_DENSE_METHODS else "random_a_ef"
            self.assertEqual(_adapter_construction_method(method), expected)

    def test_setup_bytes_enter_the_delivered_total(self):
        tracker = WireTracker(schema_bytes=100, setup_bytes=1_000)
        tracker.upload(10)
        tracker.download(5, 2)
        self.assertEqual(tracker.total_delivered_bytes, 100 + 1_000 + 10 + 10)

    def test_configs_validate_and_provisional_gamma_is_gated(self):
        smoke = load_config(ROOT / "configs" / "roberta_cab_smoke.yaml")
        self.assertEqual(smoke["cab"]["gamma_source"], "provisional")
        with self.assertRaisesRegex(ValueError, "provisional"):
            load_config(ROOT / "configs" / "roberta_cab_dev.yaml")

        import yaml

        raw = yaml.safe_load((ROOT / "configs" / "roberta_cab_dev.yaml").read_text())
        raw["cab"]["gamma_source"] = "dcef_dev"
        validate_config(raw)
        self.assertEqual(matrix_run_count(raw), 117)
        self.assertEqual(raw["project"]["expected_runs"], 117)

        for mutation, message in (
            (lambda c: c["cab"].pop("client_rank"), "missing fields"),
            (lambda c: (c["cab"].update(client_rank=1), c["data"].update(num_clients=4, clients_per_round=4)), "at least the LoRA rank"),
            (lambda c: c["cab"].update(gamma={"sst2": 0.1, "rte": 1.5, "mnli": 0.1}), "gamma must lie"),
            (lambda c: c["cab"].update(kappa_multiplier=-1), "kappa_multiplier"),
            (lambda c: c["cab"].update(extra=1), "unsupported cab fields"),
            (lambda c: c["cab"].update(activation_rank=0), "activation_rank"),
            (lambda c: c["cab"].update(activation_ridge=0.0), "activation_ridge"),
            (lambda c: c["cab"].update(factor_dtype="bfloat16"), "factor_dtype"),
            (lambda c: c["cab"].update(basis_dtype="int4"), "basis_dtype"),
            (lambda c: c["cab"].update(basis_refine="exact"), "basis_refine"),
            (lambda c: c["cab"].update(share_basis="yes"), "share_basis"),
            (lambda c: c["training"]["method_overrides"].update(cab_a_dcef={"learning_rate": 0.01, "head_learning_rate": 0.01}), "matched fixed-A"),
        ):
            changed = copy.deepcopy(raw)
            mutation(changed)
            with self.assertRaisesRegex(ValueError, message):
                validate_config(changed)


if __name__ == "__main__":
    unittest.main()
