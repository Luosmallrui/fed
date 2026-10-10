import unittest
import copy
from pathlib import Path

from fedpeft.config import (
    DIAGNOSTIC_RECIPE_VERSION,
    STABILITY_CANDIDATES,
    STABILITY_PROTOCOL_SHA256,
    STABILITY_V3_CANDIDATE,
    STABILITY_V3_PROTOCOL_SHA256,
    STABILITY_V4_CANDIDATE,
    STABILITY_V4_PROTOCOL_SHA256,
    SUPPORTED_METHODS,
    load_config,
    matrix_run_count,
    method_task_value,
    result_path,
    validate_config,
)
from scripts.run_lr_sweep import DENSE_METHODS as LR_SWEEP_DENSE_METHODS


def minimal_config(output_dir: str) -> dict:
    return {
        "project": {"root": "/tmp/fedpeft-project", "output_dir": output_dir, "phase": "smoke"},
        "model": {},
        "data": {
            "tasks": ["sst2"],
            "partition_seeds": [401],
            "clients_per_round": 2,
            "num_clients": 2,
            "dirichlet_alpha": 0.3,
            "local_holdout_fraction": 0.1,
        },
        "training": {"learning_rate": 3e-4, "head_learning_rate": 3e-4},
        "communication": {
            "wire_ratios": [0.1, 0.2],
            "value_dtype": "float16",
            "index_dtype": "uint32",
            "count_downlink_per_client": True,
        },
        "matrix": {
            "dense_methods": ["fedavg_lora"],
            "sparse_methods": ["flasc"],
            "diagnostic_dense_methods": [],
            "diagnostic_sparse_methods": [],
        },
    }


class ResultPathTests(unittest.TestCase):
    def test_stability_v4_stage_a_is_frozen_to_45_paired_runs(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_v4_stage_a.yaml")
        self.assertEqual(matrix_run_count(cfg), 45)
        self.assertEqual(cfg["project"]["expected_runs"], 45)
        self.assertEqual(cfg["data"]["tasks"], ["sst2", "mrpc", "qqp"])
        self.assertEqual(cfg["data"]["partition_seeds"], [351, 352, 353])
        self.assertEqual(cfg["stability"]["generation"], 4)
        self.assertEqual(cfg["stability"]["candidate"], STABILITY_V4_CANDIDATE)
        self.assertEqual(
            cfg["stability"]["protocol_sha256"], STABILITY_V4_PROTOCOL_SHA256
        )

    def test_stability_v4_rejects_task_seed_metric_and_schedule_drift(self):
        root = Path(__file__).resolve().parents[1]
        original = load_config(root / "configs" / "roberta_stability_v4_stage_a.yaml")
        for mutation, message in (
            (("data", "tasks", ["sst2", "mrpc", "rte"]), "stability-v4"),
            (("data", "partition_seeds", [351, 352, 354]), "frozen 45-run"),
            (("metrics", "qqp", "accuracy"), "metrics differ"),
            (("training", "schedule_overrides", {}), "frozen late_cosine"),
        ):
            changed = copy.deepcopy(original)
            section, key, value = mutation
            changed[section][key] = value
            with self.assertRaisesRegex(ValueError, message):
                validate_config(changed)

    def test_stability_v3_stage_a_is_frozen_to_30_paired_runs(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_v3_stage_a.yaml")
        self.assertEqual(matrix_run_count(cfg), 30)
        self.assertEqual(cfg["project"]["expected_runs"], 30)
        self.assertEqual(cfg["data"]["partition_seeds"], [331, 332, 333])
        self.assertEqual(cfg["communication"]["wire_ratios"], [0.1])
        self.assertEqual(cfg["stability"]["generation"], 3)
        self.assertEqual(cfg["stability"]["candidate"], STABILITY_V3_CANDIDATE)
        self.assertEqual(
            cfg["stability"]["protocol_sha256"], STABILITY_V3_PROTOCOL_SHA256
        )
        self.assertEqual(
            cfg["matrix"]["dense_methods"],
            ["fedavg_lora", "hybrid4_a_dense", "hybrid4_a_decay_dense"],
        )
        self.assertEqual(
            cfg["matrix"]["sparse_methods"],
            ["hybrid4_a_ef", "hybrid4_a_decay_ef"],
        )

    def test_stability_v3_rejects_schedule_or_seed_drift(self):
        root = Path(__file__).resolve().parents[1]
        original = load_config(root / "configs" / "roberta_stability_v3_stage_a.yaml")
        changed_schedule = copy.deepcopy(original)
        changed_schedule["training"]["schedule_overrides"][
            "hybrid4_a_decay_ef"
        ]["start_fraction"] = 0.7
        with self.assertRaisesRegex(ValueError, "frozen late_cosine"):
            validate_config(changed_schedule)

        changed_seed = copy.deepcopy(original)
        changed_seed["data"]["partition_seeds"] = [321, 332, 333]
        with self.assertRaisesRegex(ValueError, "frozen 30-run matrix"):
            validate_config(changed_seed)

        changed_protocol = copy.deepcopy(original)
        changed_protocol["stability"]["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "stability-v3 protocol_sha256"):
            validate_config(changed_protocol)

    def test_stability_v3_smoke_is_four_implementation_only_runs(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_v3_smoke.yaml")
        self.assertEqual(matrix_run_count(cfg), 4)
        self.assertEqual(cfg["data"]["partition_seeds"], [311])
        self.assertEqual(
            cfg["matrix"]["dense_methods"],
            ["hybrid4_a_dense", "hybrid4_a_decay_dense"],
        )
        self.assertEqual(
            cfg["matrix"]["sparse_methods"],
            ["hybrid4_a_ef", "hybrid4_a_decay_ef"],
        )

    def test_stability_stage_a_is_frozen_to_36_train_holdout_runs(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_stage_a.yaml")
        self.assertEqual(matrix_run_count(cfg), 36)
        self.assertEqual(cfg["project"]["expected_runs"], 36)
        self.assertEqual(cfg["project"]["phase"], "tune")
        self.assertEqual(cfg["evaluation"]["policy"], "train_holdout")
        self.assertEqual(cfg["data"]["tasks"], ["sst2", "rte"])
        self.assertEqual(cfg["data"]["partition_seeds"], [321, 322, 323])
        self.assertEqual(cfg["communication"]["wire_ratios"], [0.1])
        self.assertEqual(
            tuple(cfg["stability"]["candidates"]), STABILITY_CANDIDATES
        )
        self.assertEqual(
            cfg["stability"]["protocol_sha256"], STABILITY_PROTOCOL_SHA256
        )
        self.assertEqual(
            cfg["stability"]["diagnostic_recipe_version"],
            DIAGNOSTIC_RECIPE_VERSION,
        )
        self.assertEqual(
            cfg["matrix"]["diagnostic_sparse_methods"], ["random_block_ef"]
        )

    def test_stability_smoke_covers_three_candidates_and_diagnostic_control(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_smoke.yaml")
        self.assertEqual(matrix_run_count(cfg), 3)
        self.assertEqual(
            cfg["matrix"]["sparse_methods"], list(STABILITY_CANDIDATES)
        )
        self.assertEqual(
            cfg["matrix"]["diagnostic_sparse_methods"], ["random_block_ef"]
        )

    def test_random_block_is_diagnostic_only(self):
        cfg = minimal_config("docs/results/screen")
        cfg["matrix"]["sparse_methods"] = ["random_block_ef"]
        cfg["stability"] = {
            "stage": "smoke",
            "protocol_sha256": STABILITY_PROTOCOL_SHA256,
            "diagnostic_recipe_version": DIAGNOSTIC_RECIPE_VERSION,
            "candidates": list(STABILITY_CANDIDATES),
            "winner": None,
        }
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            validate_config(cfg)

    def test_stability_candidates_share_frozen_learning_rates(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_stage_a.yaml")
        cfg["training"]["method_overrides"]["spectral_rot_ef"] = {
            "learning_rate": {"sst2": 1e-3, "rte": 3e-4},
            "head_learning_rate": {"sst2": 3e-4, "rte": 3e-4},
        }
        with self.assertRaisesRegex(ValueError, "learning rate"):
            validate_config(cfg)

    def test_stability_stage_a_rejects_protocol_drift(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_stage_a.yaml")
        cfg["stability"]["protocol_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "preregistered protocol"):
            validate_config(cfg)

    def test_stability_protocol_freezes_partition_floor_and_gpu_residency(self):
        root = Path(__file__).resolve().parents[1]
        original = load_config(root / "configs" / "roberta_stability_stage_a.yaml")

        changed_minimum = copy.deepcopy(original)
        changed_minimum["data"]["min_client_examples"] = 16
        with self.assertRaisesRegex(ValueError, "data.min_client_examples"):
            validate_config(changed_minimum)

        cpu_resident = copy.deepcopy(original)
        cpu_resident["data"]["gpu_resident"] = False
        with self.assertRaisesRegex(ValueError, "data.gpu_resident"):
            validate_config(cpu_resident)

        smoke = load_config(root / "configs" / "roberta_stability_smoke.yaml")
        smoke["data"]["gpu_resident"] = False
        with self.assertRaisesRegex(ValueError, "data.gpu_resident"):
            validate_config(smoke)

    def test_stage_b_accepts_only_the_two_frozen_submatrices(self):
        root = Path(__file__).resolve().parents[1]
        base = load_config(root / "configs" / "roberta_stability_stage_a.yaml")
        first = copy.deepcopy(base)
        first["stability"].update(
            {"stage": "stage_b", "substage": "sst_rte_20", "winner": "spectral_rot_ef"}
        )
        first["communication"]["wire_ratios"] = [0.2]
        first["matrix"] = {
            "dense_methods": [],
            "sparse_methods": ["spectral_rot_ef"],
            "diagnostic_dense_methods": [],
            "diagnostic_sparse_methods": [],
        }
        first["project"]["expected_runs"] = 6
        validate_config(first)

        invalid = copy.deepcopy(first)
        invalid["communication"]["wire_ratios"] = [0.1]
        with self.assertRaisesRegex(ValueError, "frozen 6/9-run submatrix"):
            validate_config(invalid)

        second = copy.deepcopy(base)
        second["stability"].update(
            {"stage": "stage_b", "substage": "mnli_10_20", "winner": "spectral_rot_ef"}
        )
        second["data"]["tasks"] = ["mnli"]
        second["communication"]["wire_ratios"] = [0.1, 0.2]
        second["matrix"] = {
            "dense_methods": ["fedavg_lora"],
            "sparse_methods": ["spectral_rot_ef"],
            "diagnostic_dense_methods": [],
            "diagnostic_sparse_methods": [],
        }
        second["project"]["expected_runs"] = 9
        for key, value in (
            ("rounds", 50),
            ("local_steps", 16),
            ("batch_size", 128),
            ("eval_every", 5),
        ):
            second["training"][key]["mnli"] = value
        for method in ("fedavg_lora", "spectral_rot_ef"):
            for key in ("learning_rate", "head_learning_rate"):
                second["training"]["method_overrides"][method][key]["mnli"] = 0.001
        validate_config(second)

    def test_random_a_dense_is_classified_as_dense_by_lr_sweep(self):
        self.assertIn("random_a_dense", LR_SWEEP_DENSE_METHODS)
        self.assertIn("hybrid4_a_dense", LR_SWEEP_DENSE_METHODS)
        self.assertIn("hybrid4_a_decay_dense", LR_SWEEP_DENSE_METHODS)

    def test_full_lora_ef_is_supported_but_not_classified_as_dense(self):
        self.assertIn("full_lora_ef", SUPPORTED_METHODS)
        self.assertNotIn("full_lora_ef", LR_SWEEP_DENSE_METHODS)

    def test_formal_matrix_remains_81_runs_and_random_dense_is_optional(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "roberta_screen.yaml"
        cfg = load_config(path)
        main_runs = len(cfg["data"]["tasks"]) * len(cfg["data"]["partition_seeds"]) * (
            len(cfg["matrix"]["dense_methods"])
            + len(cfg["matrix"]["sparse_methods"])
            * len(cfg["communication"]["wire_ratios"])
        )
        self.assertEqual(main_runs, 81)
        self.assertNotIn("random_a_dense", cfg["matrix"]["dense_methods"])
        self.assertNotIn("random_a_dense", cfg["matrix"]["sparse_methods"])
        self.assertIn("random_a_dense", cfg["matrix"]["diagnostic_dense_methods"])

    def test_full_lora_wire_supplement_is_27_runs_and_inherits_frozen_protocol(self):
        root = Path(__file__).resolve().parents[1]
        formal = load_config(root / "configs" / "roberta_screen.yaml")
        supplement = load_config(root / "configs" / "roberta_full_lora_ef.yaml")
        main_runs = len(supplement["data"]["tasks"]) * len(
            supplement["data"]["partition_seeds"]
        ) * (
            len(supplement["matrix"]["dense_methods"])
            + len(supplement["matrix"]["sparse_methods"])
            * len(supplement["communication"]["wire_ratios"])
        )
        self.assertEqual(main_runs, 27)
        self.assertEqual(supplement["project"]["expected_runs"], 27)
        self.assertEqual(supplement["matrix"]["dense_methods"], ["fedavg_lora"])
        self.assertEqual(supplement["matrix"]["sparse_methods"], ["full_lora_ef"])
        for section, key in (
            ("data", "tasks"),
            ("data", "partition_seeds"),
            ("training", "rounds"),
            ("training", "local_steps"),
            ("training", "batch_size"),
            ("training", "eval_every"),
            ("communication", "wire_ratios"),
        ):
            self.assertEqual(supplement[section][key], formal[section][key])
        for task in formal["data"]["tasks"]:
            for key in ("learning_rate", "head_learning_rate"):
                expected = method_task_value(formal, key, task, "fedavg_lora")
                self.assertEqual(
                    method_task_value(supplement, key, task, "fedavg_lora"),
                    expected,
                )
                self.assertEqual(
                    method_task_value(supplement, key, task, "full_lora_ef"),
                    expected,
                )

    def test_validate_config_accepts_and_normalizes_result_directory(self):
        cfg = minimal_config("docs/results/screen")
        validate_config(cfg)
        self.assertEqual(
            cfg["project"]["output_dir"],
            str(Path("/tmp/fedpeft-project/docs/results/screen").resolve()),
        )

    def test_validate_config_rejects_output_outside_results_directory(self):
        cfg = minimal_config("artifacts/runs")
        with self.assertRaisesRegex(ValueError, "result path must be inside"):
            validate_config(cfg)

    def test_result_path_rejects_report_escape(self):
        cfg = minimal_config("docs/results/screen")
        with self.assertRaisesRegex(ValueError, "result path must be inside"):
            result_path(cfg, "RESULTS.md")

    def test_method_only_head_learning_rate_override_is_supported(self):
        cfg = minimal_config("docs/results/screen")
        cfg["training"].pop("head_learning_rate")
        cfg["training"]["method_overrides"] = {
            "fedavg_lora": {"head_learning_rate": 1e-3}
        }
        validate_config(cfg)

    def test_basis_pair_must_share_both_learning_rates(self):
        cfg = minimal_config("docs/results/screen")
        cfg["matrix"]["dense_methods"] = []
        cfg["matrix"]["sparse_methods"] = ["random_a_ef", "spectral_a_ef"]
        cfg["training"]["method_overrides"] = {
            "spectral_a_ef": {"head_learning_rate": 1e-3}
        }
        with self.assertRaisesRegex(ValueError, "share head learning rate"):
            validate_config(cfg)

    def test_dense_basis_pair_is_supported_and_must_share_learning_rates(self):
        cfg = minimal_config("docs/results/screen")
        cfg["matrix"]["dense_methods"] = []
        cfg["matrix"]["sparse_methods"] = []
        cfg["matrix"]["diagnostic_dense_methods"] = [
            "random_a_dense",
            "spectral_a_dense",
        ]
        validate_config(cfg)

        cfg["training"]["method_overrides"] = {
            "spectral_a_dense": {"learning_rate": 1e-3}
        }
        with self.assertRaisesRegex(ValueError, "share learning rate"):
            validate_config(cfg)

        cfg["training"]["method_overrides"] = {
            "spectral_a_dense": {"head_learning_rate": 1e-3}
        }
        with self.assertRaisesRegex(ValueError, "share head learning rate"):
            validate_config(cfg)

    def test_fixed_a_diagnostics_must_match_ef_learning_rates(self):
        cfg = minimal_config("docs/results/screen")
        cfg["matrix"]["dense_methods"] = []
        cfg["matrix"]["sparse_methods"] = ["random_a_ef", "spectral_a_ef"]
        cfg["matrix"]["diagnostic_dense_methods"] = [
            "random_a_dense",
            "spectral_a_dense",
        ]
        cfg["matrix"]["diagnostic_sparse_methods"] = [
            "random_a_topk",
            "spectral_a_topk",
        ]
        cfg["training"]["method_overrides"] = {
            "random_a_ef": {"learning_rate": 1e-3},
            "spectral_a_ef": {"learning_rate": 1e-3},
        }
        with self.assertRaisesRegex(ValueError, "must share learning rate"):
            validate_config(cfg)

        for method in (
            "random_a_topk",
            "spectral_a_topk",
            "random_a_dense",
            "spectral_a_dense",
        ):
            cfg["training"]["method_overrides"][method] = {
                "learning_rate": 1e-3
            }
        validate_config(cfg)

    def test_fixed_a_subset_must_share_adapter_and_head_learning_rates(self):
        cfg = minimal_config("docs/results/screen")
        cfg["matrix"]["dense_methods"] = []
        cfg["matrix"]["sparse_methods"] = []
        cfg["matrix"]["diagnostic_dense_methods"] = ["random_a_dense"]
        cfg["matrix"]["diagnostic_sparse_methods"] = ["random_a_topk"]
        cfg["training"]["method_overrides"] = {
            "random_a_topk": {"learning_rate": 1e-3}
        }
        with self.assertRaisesRegex(ValueError, "share learning rate"):
            validate_config(cfg)

        cfg["training"]["method_overrides"] = {
            "random_a_topk": {"head_learning_rate": 1e-3}
        }
        with self.assertRaisesRegex(ValueError, "share head learning rate"):
            validate_config(cfg)

    def test_full_lora_pair_must_share_adapter_and_head_learning_rates(self):
        cfg = minimal_config("docs/results/screen")
        cfg["matrix"]["sparse_methods"] = ["full_lora_ef"]
        cfg["training"]["method_overrides"] = {
            "full_lora_ef": {"learning_rate": 1e-3}
        }
        with self.assertRaisesRegex(
            ValueError, "full-LoRA methods must share learning rate"
        ):
            validate_config(cfg)

        cfg["training"]["method_overrides"] = {
            "full_lora_ef": {"head_learning_rate": 1e-3}
        }
        with self.assertRaisesRegex(
            ValueError, "full-LoRA methods must share head learning rate"
        ):
            validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
