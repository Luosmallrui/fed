import json
import tempfile
import unittest
from pathlib import Path

import yaml

from fedpeft.config import load_config, run_id
from scripts.build_dcef_report import expected_specs, paired_rows, stability_row


def _config(root: Path) -> Path:
    cfg = {
        "project": {
            "phase": "tune",
            "root": str(root),
            "output_dir": str(root / "docs" / "results" / "dcef_test"),
            "expected_runs": 3,
        },
        "model": {},
        "data": {
            "tasks": ["sst2"],
            "partition_seeds": [361],
            "num_clients": 2,
            "clients_per_round": 2,
            "dirichlet_alpha": 0.3,
            "local_holdout_fraction": 0.1,
        },
        "training": {"learning_rate": 1e-3, "head_learning_rate": 1e-3},
        "evaluation": {"policy": "train_holdout"},
        "communication": {
            "wire_ratios": [0.1],
            "value_dtype": "float16",
            "index_dtype": "uint32",
            "count_downlink_per_client": True,
        },
        "matrix": {
            "dense_methods": ["fedavg_lora"],
            "sparse_methods": ["full_lora_ef", "full_lora_dcef"],
        },
    }
    path = root / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _summary(method: str, scores: list[float], bytes_per_round: int, dcef: bool) -> dict:
    rounds = []
    for index, score in enumerate(scores, start=1):
        record = {
            "round": index,
            "primary_metric": score,
            "wire": {"total_delivered_bytes": index * bytes_per_round},
            "diagnostics": {},
        }
        if dcef:
            record["diagnostics"]["dcef"] = {"gamma_bar": None if index == 1 else 0.2}
        rounds.append(record)
    summary = {
        "status": "COMPLETE",
        "rounds": rounds,
        "final": rounds[-1],
        "model": {"adapter_parameters": 1000},
        "wire_budget": {"sparse_k": 10},
        "invariants": {
            "wire_cap_respected": True,
            "finite_metrics": True,
            "basis_unchanged": True,
            "ef_conservation_max": 1e-8,
        },
        "dcef": None,
    }
    if dcef:
        summary["dcef"] = {
            "uplink_sparse_k": 8,
            "downlink_sparse_k": 10,
            "uplink_control_bytes": 15,
        }
    return summary


class DcefReportTests(unittest.TestCase):
    def test_stability_endpoints_and_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(_config(root))
            specs = expected_specs(cfg)
            self.assertEqual(len(specs), 3)
            dense = _summary("fedavg_lora", [0.80, 0.90, 0.90, 0.90], 1000, dcef=False)
            # target = 0.855; EF reaches at round 2 then drops twice (reach-then-lose)
            ef = _summary("full_lora_ef", [0.70, 0.86, 0.80, 0.84], 100, dcef=False)
            dcef = _summary("full_lora_dcef", [0.70, 0.80, 0.86, 0.87], 100, dcef=True)
            ef_row = stability_row(("sst2", "full_lora_ef", 361, 0.1), ef, dense)
            dcef_row = stability_row(("sst2", "full_lora_dcef", 361, 0.1), dcef, dense)

            self.assertTrue(ef_row["technical_valid"])
            self.assertAlmostEqual(ef_row["target"], 0.855)
            self.assertEqual(ef_row["c95_round"], 2)
            self.assertEqual(ef_row["post_hit_violation_count"], 2)
            self.assertFalse(ef_row["strict_stable"])
            self.assertAlmostEqual(ef_row["speedup95"], 2000 / 200)

            self.assertEqual(dcef_row["c95_round"], 3)
            self.assertEqual(dcef_row["post_hit_violation_count"], 0)
            self.assertTrue(dcef_row["strict_stable"])
            self.assertAlmostEqual(dcef_row["delay_index_gamma_d_over_k"], 0.2 * 1000 / 10)

            pairs = paired_rows([ef_row, dcef_row])
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0]["comparison"], "full_lora_dcef-full_lora_ef")
            self.assertAlmostEqual(pairs[0]["delta_final"], 0.03)
            self.assertEqual(pairs[0]["direction"], "WIN")

    def test_audit_flags_missing_and_inconsistent_dcef_runs(self):
        missing = stability_row(("sst2", "full_lora_dcef", 361, 0.1), None, None)
        self.assertFalse(missing["technical_valid"])
        self.assertEqual(missing["technical_reasons"], "MISSING")
        broken = _summary("full_lora_dcef", [0.8, 0.9], 100, dcef=True)
        broken["dcef"]["uplink_sparse_k"] = 11
        broken["invariants"]["ef_conservation_max"] = 1e-3
        row = stability_row(("sst2", "full_lora_dcef", 361, 0.1), broken, None)
        self.assertFalse(row["technical_valid"])
        self.assertIn("DCEF_UPLINK_K", row["technical_reasons"])
        self.assertIn("EF_LEDGER", row["technical_reasons"])
        self.assertEqual(row["c95_status"], "NO_DENSE_REFERENCE")

    def test_run_ids_match_engine_layout(self):
        self.assertEqual(run_id("rte", "full_lora_dcef", 361, 0.05), "rte__full_lora_dcef__seed-361__wire-05")


if __name__ == "__main__":
    unittest.main()


class CabReportTests(unittest.TestCase):
    def _cab_summary(self, setup_counted=True):
        summary = _summary("cab_a_dcef", [0.70, 0.80, 0.86, 0.87], 100, dcef=True)
        setup = 50
        for record in summary["rounds"]:
            record["wire"]["setup_bytes"] = setup if setup_counted else 0
            record["wire"]["total_delivered_bytes"] += setup
        summary["final"] = summary["rounds"][-1]
        summary["invariants"]["before"] = {"fixed_basis_recipe": "cab_client_svd_fp16_v1"}
        summary["cab"] = {
            "recipe": "cab_client_svd_fp16_v1",
            "setup_bytes": setup,
            "kappa": 0.4,
            "gamma": 0.2,
            "n_eff": 7.5,
            "summary": {
                "objective_selected_mean": 0.30,
                "objective_spectral_mean": 0.60,
                "objective_random_mean": 0.90,
                "objective_gap_vs_exact_mean": 1.02,
            },
        }
        return summary

    def test_cab_rows_audit_setup_bytes_and_split_c95(self):
        dense = _summary("fedavg_lora", [0.80, 0.90, 0.90, 0.90], 1000, dcef=False)
        row = stability_row(("sst2", "cab_a_dcef", 361, 0.1), self._cab_summary(), dense)
        self.assertTrue(row["technical_valid"], row["technical_reasons"])
        self.assertEqual(row["setup_bytes"], 50)
        self.assertEqual(row["c95_bytes"], 350)
        self.assertEqual(row["c95_training_bytes"], 300)
        self.assertEqual(row["cab_objective_selected"], 0.30)
        broken = stability_row(
            ("sst2", "cab_a_dcef", 361, 0.1), self._cab_summary(setup_counted=False), dense
        )
        self.assertIn("CAB_SETUP_BYTES_NOT_COUNTED", broken["technical_reasons"])

    def test_calibration_prints_task_medians(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DCEF_RUNS.csv"
            path.write_text(
                "task,method,seed,wire,technical_valid,gamma_bar_mean,sparse_k\n"
                "sst2,random_a_dcef,361,0.1,True,0.10,9363\n"
                "sst2,random_a_dcef,362,0.1,True,0.30,9363\n"
                "sst2,random_a_dcef,363,0.1,True,0.20,9363\n"
                "sst2,random_a_dcef,363,0.05,False,0.90,4422\n"
                "sst2,full_lora_dcef,361,0.1,True,0.70,9363\n",
                encoding="utf-8",
            )
            root = Path(__file__).resolve().parents[1]
            result = subprocess.run(
                [sys.executable, str(root / "scripts" / "calibrate_cab_gamma.py"), "--runs", str(path)],
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertIn("gamma: {sst2: 0.2000}", result.stdout)
        self.assertIn("gamma_source: dcef_dev", result.stdout)
        self.assertIn("n=3", result.stdout)
