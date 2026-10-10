import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fedpeft.provenance import (
    verified_stability_launch_provenance,
    verified_v3_launch_provenance,
)
from fedpeft.utils import file_sha256, source_python_sha256
from scripts.run_matrix import (
    _acquire_campaign_lock,
    _initialize_v3_campaign,
    _load_v3_resume_events,
    _release_campaign_lock,
)


def _write_passing_v3_preflight(directory: Path):
    root = directory / "project"
    output = root / "docs" / "results" / "stability_v3" / "development"
    source = root / "src" / "fedpeft" / "module.py"
    protocol = root / "docs" / "results" / "exploration" / "STABILITY_V3_PLAN.md"
    config = root / "configs" / "stage_a.yaml"
    source.parent.mkdir(parents=True)
    protocol.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    output.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    protocol.write_text("# frozen protocol\n", encoding="utf-8")
    config.write_text("frozen: true\n", encoding="utf-8")
    protocol_hash = file_sha256(protocol)
    cfg = {
        "_config_path": str(config),
        "project": {"root": str(root), "output_dir": str(output)},
        "stability": {"generation": 3, "stage": "stage_a", "protocol_sha256": protocol_hash},
    }
    preflight = {
        "status": "PASS",
        "checks": {
            "source_hash": source_python_sha256(root),
            "config_file_hash": file_sha256(config),
            "stability_protocol_hash": protocol_hash,
            "stability_protocol_file_hash": protocol_hash,
        },
        "failures": [],
    }
    preflight_path = output / "preflight.json"
    preflight_path.write_text(json.dumps(preflight) + "\n", encoding="utf-8")
    return cfg, output, source, protocol, config, preflight_path


class FrozenV3LaunchTests(unittest.TestCase):
    def test_stage_a_rejects_partial_launch_arguments(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{root / 'src'}:{root}" + (
            f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "run_matrix.py"),
                "--config",
                str(root / "configs" / "roberta_stability_v3_stage_a.yaml"),
                "--max-runs",
                "1",
            ],
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("complete frozen matrix", completed.stderr)

    def test_launch_provenance_requires_pass_and_current_exact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            cfg, output, source, protocol, config, preflight_path = (
                _write_passing_v3_preflight(Path(temporary))
            )
            provenance = verified_v3_launch_provenance(cfg, output)
            self.assertEqual(
                provenance["preflight_file_sha256"],
                hashlib.sha256(preflight_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                provenance["source_hash"],
                source_python_sha256(cfg["project"]["root"]),
            )

            mutations = (
                (source, "VALUE = 2\n", "source_hash"),
                (config, "frozen: false\n", "config_file_hash"),
                (protocol, "# changed protocol\n", "protocol"),
            )
            for path, replacement, message in mutations:
                with self.subTest(path=path.name):
                    original = path.read_bytes()
                    path.write_text(replacement, encoding="utf-8")
                    with self.assertRaisesRegex(RuntimeError, message):
                        verified_v3_launch_provenance(cfg, output)
                    path.write_bytes(original)

            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["status"] = "FAIL"
            preflight["failures"] = ["deliberate"]
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "status is not PASS"):
                verified_v3_launch_provenance(cfg, output)


class FrozenV4LaunchTests(unittest.TestCase):
    def test_minimal_self_declared_pass_cannot_bypass_v4_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            output = root / "docs" / "results" / "stability_v4" / "development"
            protocol = root / "docs" / "results" / "exploration" / "STABILITY_V4_PLAN.md"
            config = root / "configs" / "stage_a.yaml"
            source = root / "src" / "fedpeft" / "module.py"
            model = root / "model"
            for path in (output, protocol.parent, config.parent, source.parent, model):
                path.mkdir(parents=True, exist_ok=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            protocol.write_text("# frozen v4 protocol\n", encoding="utf-8")
            config.write_text("frozen: true\n", encoding="utf-8")
            (model / "weights.bin").write_bytes(b"model")
            dataset_root = root / "datasets"
            for task in ("sst2", "mrpc", "qqp"):
                task_root = dataset_root / task
                task_root.mkdir(parents=True)
                (task_root / "data.arrow").write_bytes(task.encode("utf-8"))
            protocol_hash = file_sha256(protocol)
            cfg = {
                "_config_path": str(config),
                "project": {
                    "root": str(root),
                    "output_dir": str(output),
                    "phase": "tune",
                    "expected_runs": 45,
                },
                "model": {"path": str(model)},
                "data": {
                    "root": str(dataset_root),
                    "tasks": ["sst2", "mrpc", "qqp"],
                    "partition_seeds": [351, 352, 353],
                    "num_clients": 10,
                },
                "training": {"schedule_overrides": {}},
                "communication": {"wire_ratios": [0.1]},
                "matrix": {
                    "dense_methods": [
                        "fedavg_lora",
                        "hybrid4_a_dense",
                        "hybrid4_a_decay_dense",
                    ],
                    "sparse_methods": [
                        "hybrid4_a_ef",
                        "hybrid4_a_decay_ef",
                    ],
                },
                "stability": {
                    "generation": 4,
                    "stage": "stage_a",
                    "protocol_sha256": protocol_hash,
                },
            }
            preflight = {
                "status": "PASS",
                "checks": {
                    "source_hash": source_python_sha256(root),
                    "config_file_hash": file_sha256(config),
                    "stability_protocol_hash": protocol_hash,
                    "stability_protocol_file_hash": protocol_hash,
                },
                "failures": [],
            }
            (output / "preflight.json").write_text(
                json.dumps(preflight) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "preflight schema"):
                verified_stability_launch_provenance(cfg, output)


class SourceHashTests(unittest.TestCase):
    def test_hash_binds_execution_scripts_but_not_report_only_scripts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "src" / "fedpeft" / "module.py"
            launcher = root / "scripts" / "run_matrix.py"
            report = root / "scripts" / "build_report.py"
            source.parent.mkdir(parents=True)
            launcher.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n", encoding="utf-8")
            launcher.write_text("LAUNCH = 1\n", encoding="utf-8")
            report.write_text("REPORT = 1\n", encoding="utf-8")
            original = source_python_sha256(root)
            launcher.write_text("LAUNCH = 2\n", encoding="utf-8")
            self.assertNotEqual(source_python_sha256(root), original)
            launcher.write_text("LAUNCH = 1\n", encoding="utf-8")
            report.write_text("REPORT = 2\n", encoding="utf-8")
            self.assertEqual(source_python_sha256(root), original)


class CampaignLockTests(unittest.TestCase):
    def test_lock_has_wrapper_owner_and_is_removed_on_release(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.lock"
            lock = _acquire_campaign_lock(path, resume=False)
            try:
                self.assertEqual(
                    path.read_text(encoding="utf-8").strip(),
                    f"{socket.gethostname()}:{os.getpid()}",
                )
            finally:
                _release_campaign_lock(lock)
            self.assertFalse(path.exists())

    def test_active_owner_is_refused_even_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.lock"
            lock = _acquire_campaign_lock(path, resume=False)
            try:
                with self.assertRaisesRegex(RuntimeError, "owner .* is active"):
                    _acquire_campaign_lock(path, resume=True)
            finally:
                _release_campaign_lock(lock)

    def test_resume_atomically_takes_over_only_confirmed_dead_same_host_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.lock"
            stale_pid = 87654321
            path.write_text(
                f"{socket.gethostname()}:{stale_pid}\n",
                encoding="utf-8",
            )
            with patch(
                "scripts.run_matrix.os.kill",
                side_effect=ProcessLookupError,
            ) as kill:
                lock = _acquire_campaign_lock(path, resume=True)
            try:
                kill.assert_called_once_with(stale_pid, 0)
                self.assertEqual(
                    path.read_text(encoding="utf-8").strip(),
                    f"{socket.gethostname()}:{os.getpid()}",
                )
            finally:
                _release_campaign_lock(lock)

    def test_resume_refuses_live_or_foreign_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "campaign.lock"
            path.write_text(
                f"{socket.gethostname()}:{os.getpid()}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "still alive"):
                _acquire_campaign_lock(path, resume=True)

            path.write_text("different-host:42\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "another host"):
                _acquire_campaign_lock(path, resume=True)

    def test_release_does_not_unlink_a_replaced_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "campaign.lock"
            replacement = directory / "replacement"
            lock = _acquire_campaign_lock(path, resume=False)
            replacement.write_text("different-host:42\n", encoding="utf-8")
            os.replace(replacement, path)
            _release_campaign_lock(lock)
            self.assertEqual(path.read_text(encoding="utf-8"), "different-host:42\n")

    def test_new_campaign_writes_config_then_atomically_creates_empty_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "campaign_config.json"
            events_path = directory / "campaign_events.jsonl"
            campaign_config = {"requested_run_ids": ["run-1"]}
            _initialize_v3_campaign(campaign_config, config_path, events_path)
            self.assertEqual(
                json.loads(config_path.read_text(encoding="utf-8")),
                campaign_config,
            )
            self.assertEqual(events_path.read_bytes(), b"")
            with self.assertRaises(FileExistsError):
                _initialize_v3_campaign(campaign_config, config_path, events_path)

    def test_resume_recovers_config_written_events_missing_window_only_if_pristine(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            config_path = directory / "campaign_config.json"
            events_path = directory / "campaign_events.jsonl"
            campaign_config = {"requested_run_ids": ["run-1"]}
            config_path.write_text(json.dumps(campaign_config), encoding="utf-8")
            (directory / "campaign.lock").write_text(
                f"{socket.gethostname()}:{os.getpid()}\n",
                encoding="utf-8",
            )
            (directory / "preflight.json").write_text("{}\n", encoding="utf-8")
            (directory / "campaign_launcher.log").write_text(
                "wrapper launched\n",
                encoding="utf-8",
            )

            self.assertEqual(
                _load_v3_resume_events(
                    campaign_config,
                    config_path,
                    events_path,
                    directory,
                ),
                [],
            )
            self.assertEqual(events_path.read_bytes(), b"")

            events_path.unlink()
            run_dir = directory / "run-1"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "after run artifacts exist"):
                _load_v3_resume_events(
                    campaign_config,
                    config_path,
                    events_path,
                    directory,
                )
            self.assertFalse(events_path.exists())

    def test_resume_rejects_replaced_preflight_even_when_checks_still_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cfg, output, _, _, _, preflight_path = _write_passing_v3_preflight(directory)
            original_campaign = {
                "requested_run_ids": ["run-1"],
                "launch_provenance": verified_v3_launch_provenance(cfg, output),
            }
            config_path = output / "campaign_config.json"
            events_path = output / "campaign_events.jsonl"
            config_path.write_text(json.dumps(original_campaign), encoding="utf-8")
            events_path.write_text("", encoding="utf-8")

            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["rewritten_after_launch"] = True
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            resumed_campaign = {
                **original_campaign,
                "launch_provenance": verified_v3_launch_provenance(cfg, output),
            }
            with self.assertRaisesRegex(RuntimeError, "configuration differs"):
                _load_v3_resume_events(
                    resumed_campaign,
                    config_path,
                    events_path,
                    output,
                )

    def test_resume_missing_events_treats_config_and_log_as_run_artifacts(self):
        for artifact_name in ("run_config.json", "launcher.log"):
            with (
                self.subTest(artifact_name=artifact_name),
                tempfile.TemporaryDirectory() as temporary,
            ):
                output = Path(temporary)
                config_path = output / "campaign_config.json"
                events_path = output / "campaign_events.jsonl"
                campaign = {"requested_run_ids": ["run-1"]}
                config_path.write_text(json.dumps(campaign), encoding="utf-8")
                run_dir = output / "run-1"
                run_dir.mkdir()
                (run_dir / artifact_name).write_text("partial\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "after run artifacts exist"):
                    _load_v3_resume_events(
                        campaign,
                        config_path,
                        events_path,
                        output,
                    )


if __name__ == "__main__":
    unittest.main()
