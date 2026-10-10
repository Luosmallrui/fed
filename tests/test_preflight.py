import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from fedpeft.config import load_config, matrix_run_count, required_run_count
from scripts import preflight as preflight_script


class _FakeSplit:
    def __init__(self, columns, label_names):
        self._columns = columns
        self.column_names = list(columns)
        self.features = {"label": SimpleNamespace(names=label_names)}

    def __getitem__(self, key):
        return self._columns[key]

    def __len__(self):
        return len(next(iter(self._columns.values())))


class _FakePartition:
    digest = "fixture-partition"
    train = [np.asarray([2, 3], dtype=np.int64)]
    holdout = [np.asarray([0, 1], dtype=np.int64)]


class PreflightRunCountTests(unittest.TestCase):
    def test_final_default_remains_81(self):
        cfg = {"project": {"phase": "final"}}
        self.assertEqual(required_run_count(cfg), 81)

    def test_supplement_can_require_27_runs(self):
        cfg = {
            "project": {"phase": "final", "expected_runs": 27},
            "data": {
                "tasks": ["sst2", "rte", "mnli"],
                "partition_seeds": [401, 402, 403],
            },
            "communication": {"wire_ratios": [0.1, 0.2]},
            "matrix": {
                "dense_methods": ["fedavg_lora"],
                "sparse_methods": ["full_lora_ef"],
            },
        }
        self.assertEqual(required_run_count(cfg), 27)
        self.assertEqual(matrix_run_count(cfg), 27)

    def test_stability_v3_requires_exactly_30_stage_a_runs(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs" / "roberta_stability_v3_stage_a.yaml")
        self.assertEqual(required_run_count(cfg), 30)
        self.assertEqual(matrix_run_count(cfg), 30)


class PreflightDatasetContractTests(unittest.TestCase):
    _LABEL_NAMES = {
        "mrpc": ["not_equivalent", "equivalent"],
        "qqp": ["not_duplicate", "duplicate"],
    }
    _TEXT_FIELDS = {
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
    }

    def _run_preflight(
        self,
        task,
        *,
        validation_labels=None,
        train_label_names=None,
        validation_label_names=None,
        missing_validation_field=None,
    ):
        expected_names = self._LABEL_NAMES[task]
        validation_labels = validation_labels or [0, 1, 0, 1]
        train_label_names = (
            expected_names if train_label_names is None else train_label_names
        )
        validation_label_names = (
            expected_names
            if validation_label_names is None
            else validation_label_names
        )
        train_columns = {
            field: [f"train-{field}"] * 4
            for field in self._TEXT_FIELDS[task]
        }
        train_columns["label"] = [0, 1, 0, 1]
        validation_columns = {
            field: [f"validation-{field}"] * 4
            for field in self._TEXT_FIELDS[task]
            if field != missing_validation_field
        }
        validation_columns["label"] = validation_labels
        dataset = {
            "train": _FakeSplit(train_columns, train_label_names),
            "validation": _FakeSplit(
                validation_columns,
                validation_label_names,
            ),
        }

        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            dataset_root = root / "datasets"
            model.mkdir()
            (dataset_root / task).mkdir(parents=True)
            cfg = {
                "_config_path": str(root / "fixture.yaml"),
                "project": {
                    "root": str(root),
                    "output_dir": str(root / "output"),
                    "phase": "smoke",
                    "expected_runs": 1,
                },
                "model": {"path": str(model)},
                "data": {
                    "root": str(dataset_root),
                    "tasks": [task],
                    "partition_seeds": [351],
                    "num_clients": 1,
                    "dirichlet_alpha": 0.3,
                    "local_holdout_fraction": 0.1,
                    "min_client_examples": 1,
                },
                "training": {
                    "mixed_precision": "bfloat16",
                    "schedule_overrides": {},
                },
            }
            dumped = []
            with contextlib.ExitStack() as stack:
                patchers = [
                    patch.object(
                        preflight_script,
                        "load_config",
                        return_value=cfg,
                    ),
                    patch.object(
                        preflight_script,
                        "load_from_disk",
                        return_value=dataset,
                    ),
                    patch.object(
                        preflight_script,
                        "create_dirichlet_partition",
                        return_value=_FakePartition(),
                    ),
                    patch.dict(
                        preflight_script.TASK_CANONICAL_ROWS,
                        {task: {"train": 4, "validation": 4}},
                    ),
                    patch.object(
                        preflight_script.AutoConfig,
                        "from_pretrained",
                        return_value=SimpleNamespace(model_type="roberta"),
                    ),
                    patch.object(
                        preflight_script.AutoTokenizer,
                        "from_pretrained",
                        return_value=SimpleNamespace(),
                    ),
                    patch.object(
                        preflight_script,
                        "cached_directory_sha256",
                        return_value="model-hash",
                    ),
                    patch.object(
                        preflight_script,
                        "source_python_sha256",
                        return_value="source-hash",
                    ),
                    patch.object(
                        preflight_script,
                        "file_sha256",
                        return_value="config-hash",
                    ),
                    patch.object(
                        preflight_script,
                        "matrix_run_count",
                        return_value=1,
                    ),
                    patch.object(
                        preflight_script,
                        "required_run_count",
                        return_value=1,
                    ),
                    patch.object(
                        preflight_script,
                        "atomic_json_dump",
                        side_effect=lambda value, _path: dumped.append(value),
                    ),
                    patch.object(
                        preflight_script.torch.cuda,
                        "is_available",
                        return_value=True,
                    ),
                    patch.object(
                        preflight_script.torch.cuda,
                        "is_bf16_supported",
                        return_value=True,
                    ),
                    patch.object(
                        preflight_script.torch.cuda,
                        "get_device_name",
                        return_value="fixture-gpu",
                    ),
                    patch.object(
                        preflight_script.torch.cuda,
                        "get_device_properties",
                        return_value=SimpleNamespace(total_memory=1),
                    ),
                    patch(
                        "sys.argv",
                        ["preflight.py", "--config", str(root / "fixture.yaml")],
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ]
                for patcher in patchers:
                    stack.enter_context(patcher)
                exit_code = None
                try:
                    preflight_script.main()
                except SystemExit as exc:
                    exit_code = exc.code
            self.assertEqual(len(dumped), 1)
            return dumped[0], exit_code

    def test_exact_mrpc_and_qqp_label_order_is_required(self):
        for task, expected_names in self._LABEL_NAMES.items():
            with self.subTest(task=task, order="canonical"):
                result, exit_code = self._run_preflight(task)
                self.assertEqual(result["status"], "PASS")
                self.assertIsNone(exit_code)
            reversed_names = list(reversed(expected_names))
            for split, override in (
                ("train", {"train_label_names": reversed_names}),
                (
                    "validation",
                    {"validation_label_names": reversed_names},
                ),
            ):
                with self.subTest(task=task, order="reversed", split=split):
                    result, exit_code = self._run_preflight(task, **override)
                    self.assertEqual(result["status"], "FAIL")
                    self.assertEqual(exit_code, 1)
                    self.assertIn(
                        f"{task}/{split} label names are {reversed_names}, "
                        f"expected {expected_names}",
                        result["failures"],
                    )

    def test_validation_rejects_illegal_labels_and_missing_text_schema(self):
        for task, fields in self._TEXT_FIELDS.items():
            with self.subTest(task=task, defect="illegal-label"):
                result, exit_code = self._run_preflight(
                    task,
                    validation_labels=[0, 2, 0, 2],
                )
                self.assertEqual(exit_code, 1)
                self.assertIn(
                    f"{task}/validation labels are [0, 2]",
                    result["failures"],
                )
            with self.subTest(task=task, defect="missing-field"):
                result, exit_code = self._run_preflight(
                    task,
                    missing_validation_field=fields[1],
                )
                self.assertEqual(exit_code, 1)
                self.assertIn(
                    f"{task}/validation missing fields: ['{fields[1]}']",
                    result["failures"],
                )


if __name__ == "__main__":
    unittest.main()
