import copy
import unittest
from pathlib import Path

import torch

from fedpeft.aggregation import weighted_average
from fedpeft.adapters import TensorLayout
from fedpeft.compression import TopKCompressor
from fedpeft.config import SUPPORTED_METHODS, load_config, matrix_run_count, validate_config
from fedpeft.dcef import (
    DCEF_CLIENT_ONLY_METHODS,
    DCEF_CONSTRUCTION,
    DCEF_CONTROL_BYTES,
    DCEF_METHODS,
    DCEF_SERVER_DISCOUNT_METHODS,
    SecantGammaEstimator,
    dcef_server_input,
    dcef_virtual_start,
    encode_control_scalar,
    weighted_gamma,
)
from fedpeft.engine import (
    DENSE_METHODS,
    EF_METHODS,
    FULL_LORA_SPARSE_EF_METHODS,
    STABILITY_DIAGNOSTIC_METHODS,
    TOPK_NO_EF_METHODS,
    _adapter_construction_method,
    _build_adapter_compressor,
)


ROOT = Path(__file__).resolve().parents[1]


def simulate(
    rule: str,
    *,
    rounds: int,
    targets: list[torch.Tensor],
    gammas: list[float],
    weights: list[float],
    compressor: TopKCompressor,
    control: bool,
) -> dict:
    """Run the engine's client/server sequence on a federated quadratic.

    Local training from s is the exact linear map s -> s + gamma_i (c_i - s).
    ``rule`` is "ef" (standard bidirectional EF), "dcef" (virtual start plus
    server discount) or "dcef_client" (virtual start, ordinary server EF).
    Uses the real sparse wire codec, so FP16 rounding enters every residual.
    """
    dim = targets[0].numel()
    clients = len(targets)
    optimum = weighted_average(targets, weights)
    x = torch.zeros(dim)
    client_residuals = [torch.zeros(dim) for _ in range(clients)]
    server_residual = torch.zeros(dim)
    estimator = SecantGammaEstimator(clients) if rule == "dcef" else None
    client_in = [torch.zeros(dim, dtype=torch.float64) for _ in range(clients)]
    client_out = [torch.zeros(dim, dtype=torch.float64) for _ in range(clients)]
    server_in = torch.zeros(dim, dtype=torch.float64)
    server_out = torch.zeros(dim, dtype=torch.float64)
    errors, uplink_bytes, gamma_bars = [], [], []
    for _ in range(rounds):
        uploads, client_gammas = [], []
        for i in range(clients):
            if rule in {"dcef", "dcef_client"}:
                start = dcef_virtual_start(x, client_residuals[i])
            else:
                start = x.clone()
            end = start + gammas[i] * (targets[i] - start)
            delta = end - start
            corrected = delta + client_residuals[i]
            result = compressor.compress(corrected)
            client_residuals[i] = corrected - result.reconstructed
            client_in[i] += delta.double()
            client_out[i] += result.reconstructed.double()
            nbytes = result.packet.nbytes
            if estimator is not None:
                value = estimator.observe(i, start, delta)
                decoded, packet = encode_control_scalar(value)
                client_gammas.append(decoded)
                nbytes += packet.nbytes if control else 0
            uplink_bytes.append(nbytes)
            uploads.append(result.reconstructed)
        aggregate = weighted_average(uploads, weights)
        if rule == "dcef":
            gamma_bar = weighted_gamma(client_gammas, weights)
            corrected = dcef_server_input(aggregate, server_residual, gamma_bar)
            ledger_input = aggregate - gamma_bar * server_residual
            gamma_bars.append(gamma_bar)
        else:
            corrected = aggregate + server_residual
            ledger_input = aggregate
        down = compressor.compress(corrected)
        server_residual = corrected - down.reconstructed
        server_in += ledger_input.double()
        server_out += down.reconstructed.double()
        x = x + down.reconstructed
        errors.append(float(((x - optimum).norm() / optimum.norm()).item()))
    client_ledger = max(
        float((client_in[i] - client_out[i] - client_residuals[i].double()).abs().max())
        for i in range(clients)
    )
    server_ledger = float((server_in - server_out - server_residual.double()).abs().max())
    return {
        "errors": errors,
        "uplink_bytes": uplink_bytes,
        "client_ledger": client_ledger,
        "server_ledger": server_ledger,
        "gamma_bars": gamma_bars,
        "estimator": estimator,
    }


def quadratic_problem(dim=400, clients=4, heterogeneity=0.0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shared = torch.randn(dim, generator=generator)
    offsets = [heterogeneity * torch.randn(dim, generator=generator) for _ in range(clients)]
    mean = sum(offsets) / clients
    targets = [shared + offset - mean for offset in offsets]
    weights = [1.0] * clients
    return targets, weights


class DcefDispatchTests(unittest.TestCase):
    def test_methods_are_supported_sparse_ef_variants_with_matched_construction(self):
        self.assertEqual(DCEF_METHODS, DCEF_SERVER_DISCOUNT_METHODS | DCEF_CLIENT_ONLY_METHODS)
        self.assertTrue(DCEF_METHODS.issubset(SUPPORTED_METHODS))
        self.assertTrue(DCEF_METHODS.issubset(EF_METHODS))
        self.assertTrue(DCEF_METHODS.isdisjoint(DENSE_METHODS | TOPK_NO_EF_METHODS))
        self.assertTrue(DCEF_METHODS.isdisjoint(STABILITY_DIAGNOSTIC_METHODS))
        for method, construction in DCEF_CONSTRUCTION.items():
            self.assertEqual(_adapter_construction_method(method), construction)
            self.assertIn(construction, EF_METHODS)
        self.assertEqual(set(DCEF_CONSTRUCTION), DCEF_METHODS)
        self.assertTrue(
            {"full_lora_ef", "full_lora_dcef", "full_lora_dcef_client"}.issubset(
                FULL_LORA_SPARSE_EF_METHODS
            )
        )

    def test_control_scalar_is_one_fp16_dense_packet(self):
        self.assertEqual(DCEF_CONTROL_BYTES, 15)
        decoded, packet = encode_control_scalar(0.3)
        self.assertEqual(packet.nbytes, 15)
        self.assertEqual(decoded, float(torch.tensor(0.3).half().float()))
        self.assertAlmostEqual(decoded, 0.3, places=3)

    def test_uplink_budget_leaves_room_for_the_control_packet(self):
        layout = TensorLayout(
            names=["layer0.lora_A", "layer0.lora_B"],
            shapes=[(8, 768), (768, 8)],
            numels=[6144, 6144],
        )
        for adapter_cap in (21 + 6 * 100, 21 + 6 * 100 + 5, 59_292 - 3_089):
            shared = _build_adapter_compressor("full_lora_dcef", layout, adapter_cap)
            uplink = _build_adapter_compressor(
                "full_lora_dcef", layout, adapter_cap - DCEF_CONTROL_BYTES
            )
            self.assertLessEqual(21 + 6 * uplink.k + DCEF_CONTROL_BYTES, adapter_cap)
            self.assertGreaterEqual(uplink.k, shared.k - 3)
            self.assertLessEqual(uplink.k, shared.k)


class SecantGammaEstimatorTests(unittest.TestCase):
    def test_recovers_linear_contraction_and_checkpoints(self):
        generator = torch.Generator().manual_seed(7)
        target = torch.randn(50, generator=generator)
        gamma = 0.37
        estimator = SecantGammaEstimator(1)
        start = torch.zeros(50)
        seen = []
        for round_index in range(4):
            displacement = gamma * (target - start)
            seen.append(estimator.observe(0, start, displacement))
            start = start + 0.4 * displacement + 0.05 * torch.randn(50, generator=generator)
            if round_index == 1:
                restored = SecantGammaEstimator(1)
                restored.load_state_dict(copy.deepcopy(estimator.state_dict()), "cpu")
        self.assertEqual(seen[0], 0.0)
        for value in seen[1:]:
            self.assertAlmostEqual(value, gamma, places=5)
        self.assertEqual(estimator.observations, [3])
        self.assertEqual(restored.observations, [1])
        self.assertAlmostEqual(restored.estimate(0), gamma, places=5)

    def test_clips_to_unit_interval_and_rejects_malformed_state(self):
        estimator = SecantGammaEstimator(1)
        estimator.observe(0, torch.zeros(3), torch.zeros(3))
        estimator.observe(0, torch.ones(3), 5.0 * torch.ones(3))
        self.assertEqual(estimator.estimate(0), 0.0)
        estimator = SecantGammaEstimator(1)
        estimator.observe(0, torch.zeros(3), torch.zeros(3))
        estimator.observe(0, torch.ones(3), -5.0 * torch.ones(3))
        self.assertEqual(estimator.estimate(0), 1.0)

        state = estimator.state_dict()
        for mutate, message in (
            (lambda s: s.update(version=0), "unknown version"),
            (lambda s: s.update(num_clients=2), "client count"),
            (lambda s: s.update(prior=0.5), "prior"),
            (lambda s: s["previous_start"].__setitem__(0, None), "half of a"),
            (lambda s: s.update(numerator=[]), "malformed"),
        ):
            broken = copy.deepcopy(state)
            mutate(broken)
            with self.assertRaisesRegex(ValueError, message):
                SecantGammaEstimator(1).load_state_dict(broken, "cpu")

    def test_server_input_and_weighted_gamma(self):
        aggregate = torch.tensor([1.0, 2.0])
        residual = torch.tensor([4.0, -4.0])
        self.assertTrue(
            torch.equal(dcef_server_input(aggregate, residual, 0.25), torch.tensor([4.0, -1.0]))
        )
        self.assertTrue(torch.equal(dcef_server_input(aggregate, residual, 0.0), aggregate + residual))
        with self.assertRaisesRegex(ValueError, "gamma_bar"):
            dcef_server_input(aggregate, residual, 1.5)
        self.assertAlmostEqual(weighted_gamma([0.2, 0.8], [3.0, 1.0]), 0.35)


class DcefProtocolTests(unittest.TestCase):
    def setUp(self):
        self.targets, self.weights = quadratic_problem()
        # K = 10 of 400 coordinates: an average transmission delay of 40 rounds.
        self.cap = 21 + 6 * 10
        self.shared = TopKCompressor(400, self.cap)

    def test_standard_ef_overshoots_where_dcef_converges(self):
        gammas = [0.2] * 4  # gamma * d / K = 8, far above the EF limit of 2
        ef = simulate("ef", rounds=120, targets=self.targets, gammas=gammas,
                      weights=self.weights, compressor=self.shared, control=False)
        dcef = simulate("dcef", rounds=120, targets=self.targets, gammas=gammas,
                        weights=self.weights, compressor=self.shared, control=False)
        self.assertGreater(max(ef["errors"][-20:]), 1.0)
        self.assertLess(max(dcef["errors"][-20:]), 0.05)
        for value in dcef["gamma_bars"][1:]:
            self.assertAlmostEqual(value, 0.2, places=2)

    def test_dcef_tracks_dense_fedavg_when_local_progress_is_small(self):
        # gamma * d / K = 0.4.  Dense FedAvg from zero has relative error
        # (1 - gamma)^t on this homogeneous problem.  Standard EF is stable here
        # and can run slightly ahead of dense (its T-fold accumulation acts as a
        # larger step); DCEF follows the dense trajectory, delayed by sparsity.
        gamma, rounds = 0.01, 150
        dense_error = (1.0 - gamma) ** rounds
        ef = simulate("ef", rounds=rounds, targets=self.targets, gammas=[gamma] * 4,
                      weights=self.weights, compressor=self.shared, control=False)
        dcef = simulate("dcef", rounds=rounds, targets=self.targets, gammas=[gamma] * 4,
                        weights=self.weights, compressor=self.shared, control=False)
        self.assertLess(abs(dcef["errors"][-1] - dense_error), 0.1)
        self.assertLess(abs(ef["errors"][-1] - dcef["errors"][-1]), 0.1)
        self.assertLess(ef["errors"][-1], ef["errors"][0])

    def test_server_discount_matters_under_heterogeneity(self):
        targets, weights = quadratic_problem(heterogeneity=1.0, seed=3)
        gammas = [0.2] * 4
        runs = {
            rule: simulate(rule, rounds=150, targets=targets, gammas=gammas,
                           weights=weights, compressor=self.shared, control=False)
            for rule in ("dcef", "dcef_client")
        }
        tail = {rule: sum(run["errors"][-30:]) / 30 for rule, run in runs.items()}
        self.assertLess(tail["dcef"], tail["dcef_client"])

    def test_ledgers_close_and_uplink_respects_the_message_cap(self):
        gammas = [0.1, 0.2, 0.3, 0.4]
        uplink = TopKCompressor(400, self.cap - DCEF_CONTROL_BYTES)
        run = simulate("dcef", rounds=40, targets=self.targets, gammas=gammas,
                       weights=self.weights, compressor=uplink, control=True)
        self.assertLess(run["client_ledger"], 1e-4)
        self.assertLess(run["server_ledger"], 1e-4)
        self.assertLessEqual(max(run["uplink_bytes"]), self.cap)
        self.assertEqual(max(run["uplink_bytes"]), 21 + 6 * uplink.k + DCEF_CONTROL_BYTES)
        estimates = [run["estimator"].estimate(i) for i in range(4)]
        for estimate, gamma in zip(estimates, gammas):
            self.assertAlmostEqual(estimate, gamma, places=2)


class DcefConfigTests(unittest.TestCase):
    def test_dev_matrix_is_90_runs_on_fresh_seeds_and_smoke_is_valid(self):
        dev = load_config(ROOT / "configs" / "roberta_dcef_dev.yaml")
        self.assertEqual(matrix_run_count(dev), 90)
        self.assertEqual(dev["project"]["expected_runs"], 90)
        self.assertEqual(dev["data"]["partition_seeds"], [361, 362, 363])
        self.assertEqual(dev["evaluation"]["policy"], "train_holdout")
        self.assertEqual(
            dev["matrix"]["sparse_methods"],
            ["full_lora_ef", "full_lora_dcef", "full_lora_dcef_client"],
        )
        smoke = load_config(ROOT / "configs" / "roberta_dcef_smoke.yaml")
        self.assertEqual(smoke["project"]["phase"], "smoke")

    def test_dcef_must_share_learning_rates_with_its_ef_counterpart(self):
        original = load_config(ROOT / "configs" / "roberta_dcef_dev.yaml")
        for method, family in (("full_lora_dcef", "full-LoRA"), ("random_a_dcef", "fixed-A")):
            changed = copy.deepcopy(original)
            changed["training"]["method_overrides"][method] = {
                "learning_rate": 0.005,
                "head_learning_rate": 0.005,
            }
            with self.assertRaisesRegex(ValueError, f"matched {family} methods must share"):
                validate_config(changed)


if __name__ == "__main__":
    unittest.main()


class ConsensusDiagnosticsTests(unittest.TestCase):
    def test_eta_is_one_for_agreement_and_one_over_n_for_orthogonal_updates(self):
        from fedpeft.engine import _consensus_statistics

        same = [torch.tensor([1.0, 2.0, 0.0, 0.0])] * 4
        stats = _consensus_statistics(same, same, same, [], [1, 1, 1, 1], 4)
        self.assertAlmostEqual(stats["eta_local"], 1.0)
        self.assertAlmostEqual(stats["wire_recovers_local_consensus"], 1.0)
        orthogonal = [torch.eye(4)[i] for i in range(4)]
        stats = _consensus_statistics(orthogonal, orthogonal, orthogonal, [], [1, 1, 1, 1], 4)
        self.assertAlmostEqual(stats["eta_local"], 0.25)
        self.assertAlmostEqual(stats["eta_floor_orthogonal"], 0.25)
        opposite = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
        stats = _consensus_statistics(opposite, opposite, opposite, [], [1, 1], 2)
        self.assertAlmostEqual(stats["eta_local"], 0.0)

    def test_support_overlap_statistics(self):
        from fedpeft.engine import _consensus_statistics

        vecs = [torch.ones(6)] * 3
        supports = [torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([4, 5])]
        stats = _consensus_statistics(vecs, vecs, vecs, supports, [1, 1, 1], 6)
        # pairs: {0,1}&{1,2} -> 1/3; {0,1}&{4,5} -> 0; {1,2}&{4,5} -> 0
        self.assertAlmostEqual(stats["support_pairwise_jaccard_mean"], (1 / 3) / 3)
        self.assertAlmostEqual(stats["support_union_over_total"], 5 / 6)
        self.assertEqual(stats["support_selected_by_all_clients"], 0)
