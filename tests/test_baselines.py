import tempfile
import unittest
from pathlib import Path

import torch

from fedpeft import baselines
from fedpeft.adapters import FixedALinear, LoRALinear, LoRAXSLinear, TensorLayout, build_model
from fedpeft.config import validate_config
from fedpeft.engine import DENSE_METHODS, EF_METHODS, _adapter_construction_method
from fedpeft.wire import encode_dense
from test_decoder import tiny_qwen


def tiny_bundle(directory: Path, method: str, rank: int = 4):
    path = tiny_qwen(directory / "qwen")
    return build_model(
        model_path=str(path),
        num_labels=3,
        method=method,
        rank=rank,
        alpha=8.0,
        dropout=0.0,
        target_suffixes=["self_attn.q_proj", "self_attn.v_proj"],
        basis_seed=7,
        lora_xs_init_std=1e-5,
        spectral_cache_path=str(directory / "spectral.pt"),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def effective_weights(model) -> dict[str, torch.Tensor]:
    weights = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            weights[name] = module.base.weight.double() + module.scaling * (
                module.lora_B.double() @ module.lora_A.double()
            )
        elif isinstance(module, FixedALinear):
            weights[name] = module.base.weight.double() + module.scaling * (
                module.lora_B.double() @ module.fixed_A.double()
            )
    return weights


class BaselineDispatchTests(unittest.TestCase):
    def test_methods_dispatch_to_matching_constructions_and_paths(self):
        self.assertTrue(baselines.BASELINE_DENSE_METHODS.issubset(DENSE_METHODS))
        self.assertTrue(baselines.BASELINE_EF_METHODS.issubset(EF_METHODS))
        self.assertNotIn("fedsb", DENSE_METHODS | EF_METHODS)
        for method in ("flora_dense", "fedex_dense", "rolora_dense", "rolora_ef"):
            self.assertEqual(_adapter_construction_method(method), "fedavg_lora")
        for method in ("fedsvd_dense", "fedsvd_ef"):
            self.assertEqual(_adapter_construction_method(method), "ffa_lora")
        self.assertEqual([baselines.rolora_factor(i) for i in range(4)], ["lora_B", "lora_A"] * 2)

    def test_config_rejects_baselines_in_the_wrong_bucket(self):
        cfg = {
            "project": {"phase": "smoke", "root": "/tmp/x", "output_dir": "/tmp/x/docs/results/y"},
            "model": {},
            "data": {
                "tasks": ["sst2"],
                "num_clients": 2,
                "clients_per_round": 2,
                "dirichlet_alpha": 0.3,
                "local_holdout_fraction": 0.1,
                "partition_seeds": [1],
            },
            "training": {"learning_rate": 1e-3},
            "communication": {"wire_ratios": [0.1]},
            "matrix": {"dense_methods": ["fedsb"], "sparse_methods": []},
        }
        with self.assertRaisesRegex(ValueError, "wrong matrix bucket"):
            validate_config(cfg)
        cfg["matrix"] = {"dense_methods": [], "sparse_methods": ["flora_dense"]}
        with self.assertRaisesRegex(ValueError, "wrong matrix bucket"):
            validate_config(cfg)


class BaselineMechanicsTests(unittest.TestCase):
    def random_clients(self, layout: TensorLayout, count: int, seed: int) -> list[torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return [0.1 * torch.randn(layout.total_numel, generator=generator) for _ in range(count)]

    def test_flora_and_fedex_merge_the_exact_average_of_client_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = tiny_bundle(Path(tmp), "fedavg_lora")
            model, layout = bundle.model, bundle.adapter_layout
            clients = self.random_clients(layout, 3, 0)
            weights = [1.0, 2.0, 5.0]
            expected = {}
            for vector, weight in zip(clients, weights):
                layout.assign(model, vector)
                for name, value in effective_weights(model).items():
                    expected[name] = expected.get(name, 0) + weight / 8.0 * value
            base = {name: module.base.weight.detach().clone() for name, module in model.named_modules() if isinstance(module, LoRALinear)}

            # FedEx-LoRA: averaged adapter plus merged residual equals the average model.
            shared = sum(w / 8.0 * v for v, w in zip(clients, weights))
            baselines.merge_client_products(model, layout, clients, weights, shared)
            layout.assign(model, shared)
            for name, value in effective_weights(model).items():
                self.assertTrue(torch.allclose(value, expected[name], atol=1e-5), name)

            # FLoRA: the whole product is merged and the next LoRA starts from B = 0.
            for name, module in model.named_modules():
                if isinstance(module, LoRALinear):
                    module.base.weight.data.copy_(base[name])
            baselines.merge_client_products(model, layout, clients, weights, None)
            fresh = baselines.flora_fresh_adapter(model, layout, 7, 1)
            layout.assign(model, fresh)
            b_mask = baselines.factor_mask(layout, "lora_B", fresh.device)
            self.assertEqual(float(fresh[b_mask].abs().max()), 0.0)
            self.assertGreater(float(fresh[~b_mask].abs().max()), 0.0)
            for name, value in effective_weights(model).items():
                self.assertTrue(torch.allclose(value, expected[name], atol=1e-5), name)

    def test_merge_downlink_uses_the_cheaper_exact_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = tiny_bundle(Path(tmp), "fedavg_lora")
            layout = bundle.adapter_layout
            clients = self.random_clients(layout, 3, 1)
            stacked = encode_dense(torch.cat(clients)).nbytes + encode_dense(torch.ones(3)).nbytes
            self.assertEqual(
                baselines.stacked_downlink_bytes(layout, clients, [1.0] * 3, None),
                (stacked, "stacked_client_factors"),
            )
            # Tiny 32-dim layers: the dense residual is smaller than 3 clients' factors here.
            nbytes, encoding = baselines.stacked_downlink_bytes(layout, clients, [1.0] * 3, clients[0])
            self.assertLessEqual(nbytes, stacked)
            self.assertIn(encoding, {"stacked_client_factors", "averaged_adapter_plus_dense_residual"})

    def test_rolora_masks_and_trainable_flags_alternate_factors(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = tiny_bundle(Path(tmp), "fedavg_lora")
            model, layout = bundle.model, bundle.adapter_layout
            mask_b = baselines.factor_mask(layout, "lora_B", torch.device("cpu"))
            mask_a = baselines.factor_mask(layout, "lora_A", torch.device("cpu"))
            self.assertFalse(bool((mask_a & mask_b).any()))
            self.assertTrue(bool((mask_a | mask_b).all()))
            baselines.set_trainable_factor(model, "lora_B")
            trainable = [n for n, p in model.named_parameters() if p.requires_grad and "lora" in n]
            self.assertTrue(trainable and all(n.endswith("lora_B") for n in trainable))
            baselines.set_trainable_factor(model, None)
            self.assertTrue(all(p.requires_grad for n, p in model.named_parameters() if "lora" in n))

    def test_fedsvd_keeps_the_model_and_rotates_within_the_initial_subspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = tiny_bundle(Path(tmp), "ffa_lora")
            model, layout = bundle.model, bundle.adapter_layout
            modules = {n: m for n, m in model.named_modules() if isinstance(m, FixedALinear)}
            initial_a = {n: m.fixed_A.detach().double().clone() for n, m in modules.items()}
            global_adapter, residual = self.random_clients(layout, 2, 2)
            layout.assign(model, global_adapter)
            before = effective_weights(model)
            residual_before = residual.clone()
            residual_weight = {}
            offsets = baselines._slices(layout)
            for name, module in modules.items():
                b0, b1, shape = offsets[name + ".lora_B"]
                residual_weight[name] = residual[b0:b1].view(shape).double() @ module.fixed_A.double()
            error = baselines.fedsvd_reparameterize(model, layout, [global_adapter, residual])
            self.assertLess(error, 1e-5)
            layout.assign(model, global_adapter)
            after = effective_weights(model)
            for name, module in modules.items():
                self.assertTrue(torch.allclose(after[name], before[name], atol=1e-5), name)
                a = module.fixed_A.double()
                self.assertTrue(torch.allclose(a @ a.T, torch.eye(a.shape[0], dtype=torch.float64), atol=1e-5))
                # same row space as the initial Kaiming A
                projector = initial_a[name].T @ torch.linalg.pinv(initial_a[name].T)
                self.assertTrue(torch.allclose(a @ projector, a, atol=1e-5))
                b0, b1, shape = offsets[name + ".lora_B"]
                moved = residual[b0:b1].view(shape).double() @ a
                self.assertTrue(torch.allclose(moved, residual_weight[name], atol=1e-5), name)
            self.assertFalse(torch.equal(residual, residual_before))
            state = baselines.fixed_a_state(model)
            self.assertEqual(set(state), set(modules))

    def test_fedsb_rank_fits_the_budget_and_bases_reproduce_the_signed_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = tiny_bundle(Path(tmp), "fedsb")
            model = bundle.model
            targets = sorted(
                ((n, m) for n, m in model.named_modules() if isinstance(m, LoRAXSLinear)), key=lambda x: x[0]
            )
            count = len(targets)
            cap = 13 + 2 * count * 25 + 1
            self.assertEqual(baselines.fedsb_rank(model, cap), 5)
            self.assertEqual(baselines.fedsb_rank(model, cap - 2), 4)
            generator = torch.Generator().manual_seed(3)
            gradients = [torch.randn(m.base.out_features, m.base.in_features, generator=generator) for _, m in targets]
            bases, payload = baselines.install_fedsb(model, gradients, 3, 1e-3)
            self.assertEqual(len(bases), count)
            installed = sorted(
                ((n, m) for n, m in model.named_modules() if isinstance(m, LoRAXSLinear)), key=lambda x: x[0]
            )
            numel = 0
            for (name, module), gradient in zip(installed, gradients):
                self.assertEqual(module.rank, 3)
                self.assertEqual(module.scaling, 1.0)
                update = module.fixed_left @ module.lora_R @ torch.diag(module.fixed_singular) @ module.fixed_right
                u, s, v = torch.linalg.svd(-1e-3 * torch.sign(gradient.double()), full_matrices=False)
                best = (u[:, :3] * s[:3]) @ v[:3]
                self.assertTrue(torch.allclose(update.double(), best, atol=1e-6), name)
                numel += module.fixed_left.numel() + module.fixed_right.numel() + 3
            self.assertEqual(payload, 13 + 2 * numel)
            trainable = [p for n, p in model.named_parameters() if p.requires_grad and "lora_R" in n]
            self.assertEqual(sum(p.numel() for p in trainable), count * 9)


if __name__ == "__main__":
    unittest.main()
