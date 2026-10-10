import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_fixed_a_ablation_report as report


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


class FixedAAblationReportAuditTests(unittest.TestCase):
    def test_frozen_inputs_bind_source_and_archived_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "frozen.yaml"
            protocol = root / "frozen.md"
            config.write_text("project: {}\n", encoding="utf-8")
            protocol.write_text("# frozen\n", encoding="utf-8")
            config_hash = hashlib.sha256(config.read_bytes()).hexdigest()
            protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
            provenance = root / "results" / "provenance"
            provenance.mkdir(parents=True)
            (provenance / "roberta_fixed_a_ablation.yaml").write_bytes(config.read_bytes())
            archived_protocol = provenance / "FORMAL_FIXED_A_ABLATION_PLAN.md"
            archived_protocol.write_bytes(protocol.read_bytes())

            with mock.patch.multiple(
                report,
                EXPECTED_CONFIG_SHA256=config_hash,
                EXPECTED_PROTOCOL_SHA256=protocol_hash,
            ):
                _, metadata, errors = report._audit_frozen_inputs(
                    root / "results", config, protocol
                )
                self.assertEqual(errors, [])
                self.assertEqual(metadata["config"]["archived_sha256"], config_hash)
                self.assertEqual(metadata["protocol"]["archived_sha256"], protocol_hash)

                archived_protocol.write_text("tampered\n", encoding="utf-8")
                _, _, errors = report._audit_frozen_inputs(root / "results", config, protocol)
                self.assertTrue(any("archived frozen protocol hash mismatch" in x for x in errors))

    def _campaign_fixture(self, root: Path):
        specs = (
            ("sst2", "random_a_dense", 401, None),
            ("sst2", "random_a_topk", 401, 0.1),
        )
        run_ids = [report._run_id(*spec) for spec in specs]
        frozen_config = {
            "project": {"output_dir": "copied/results", "expected_runs": 2},
            "matrix": {"dense_methods": ["random_a_dense"]},
        }
        manifest_config = json.loads(json.dumps(frozen_config))
        manifest_config["project"]["output_dir"] = str(root.resolve())
        _write_json(
            root / "campaign_config.json",
            {
                "config": manifest_config,
                "include_diagnostics": False,
                "requested_run_ids": run_ids,
            },
        )
        _write_json(
            root / "campaign_summary.json",
            {
                "completed_runs": run_ids,
                "failed_runs": [],
                "finished_at": "2026-09-14T01:00:00+00:00",
                "requested_runs": 2,
                "started_at": "2026-09-14T00:00:00+00:00",
                "status": "COMPLETE",
            },
        )
        events = []
        for index, run_id in enumerate(run_ids, start=1):
            events.extend(
                [
                    {"event": "START", "run_id": run_id, "index": index, "total": 2},
                    {
                        "event": "DONE",
                        "run_id": run_id,
                        "index": index,
                        "total": 2,
                        "returncode": 0,
                    },
                ]
            )
            _write_json(root / run_id / "summary.json", {"run": run_id})
        (root / "campaign_events.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
        )
        _write_json(
            root / "preflight.json",
            {
                "schema": "experiment-preflight-v1",
                "status": "PASS",
                "failures": [],
                "checks": {
                    "config_file_hash": report.EXPECTED_CONFIG_SHA256,
                    "source_hash": report.EXPECTED_PREFLIGHT_SOURCE_HASH,
                    "expected_runs": 2,
                    "output_dir": str(root.resolve()),
                },
            },
        )
        return specs, frozen_config, events

    def test_campaign_audit_requires_exact_manifest_events_and_summary_set(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs, frozen_config, events = self._campaign_fixture(root)
            metadata, errors = report._audit_campaign(root, frozen_config, specs)
            self.assertEqual(errors, [])
            self.assertEqual(metadata["requested_run_ids"], 2)
            self.assertEqual(metadata["event_rows"], 4)
            self.assertEqual(metadata["summary_files"], 2)

            _write_json(root / "unexpected__run" / "summary.json", {})
            (root / "campaign.lock").mkdir()
            (root / "failure.json").write_text("{}\n", encoding="utf-8")
            events.append(
                {"event": "DONE", "run_id": "extra", "index": 3, "total": 3, "returncode": 0}
            )
            (root / "campaign_events.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in events), encoding="utf-8"
            )
            _, errors = report._audit_campaign(root, frozen_config, specs)
            self.assertTrue(any("event count mismatch" in x for x in errors))
            self.assertIn("campaign.lock still exists", errors)
            self.assertIn("failure.json exists", errors)
            self.assertTrue(any("campaign has extra summaries" in x for x in errors))

    def test_subreport_requires_exact_valid_rows_and_records_artifact_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = (
                ("sst2", "random_a_dense", 401, None),
                ("sst2", "random_a_topk", 401, 0.1),
            )
            summaries = {}
            rows = []
            fields = [
                "run_id", "role", "audit_status", "reasons", "task", "method", "seed",
                "wire_target", "schema_version", "declared_status", "final_primary",
                "source_hash", "model_hash", "dataset_hash", "partition_hash", "layout_hash",
                "initial_head_hash", "summary_path",
            ]
            for index, spec in enumerate(specs, start=1):
                task, method, seed, ratio = spec
                run_id = report._run_id(*spec)
                summary = {
                    "final": {"primary_metric": 0.8 + 0.01 * index},
                    "model_hash": "model",
                    "dataset_hash": "dataset",
                    "partition_hash": "partition",
                    "layout_hash": "layout",
                    "initialization": {"initial_head_hash": "head"},
                }
                summaries[spec] = summary
                rows.append(
                    {
                        "run_id": run_id,
                        "role": "required",
                        "audit_status": "VALID",
                        "reasons": "",
                        "task": task,
                        "method": method,
                        "seed": str(seed),
                        "wire_target": "dense" if ratio is None else str(ratio),
                        "schema_version": "2",
                        "declared_status": "COMPLETE",
                        "final_primary": str(summary["final"]["primary_metric"]),
                        "source_hash": report.SOURCE_HASH,
                        "model_hash": "model",
                        "dataset_hash": "dataset",
                        "partition_hash": "partition",
                        "layout_hash": "layout",
                        "initial_head_hash": "head",
                        "summary_path": f"/remote/results/{run_id}/summary.json",
                    }
                )
            for name in report.SUBREPORT_ARTIFACTS:
                if name != "RUNS.csv":
                    (root / name).write_text(f"{name}\n", encoding="utf-8")

            def write_runs() -> None:
                with (root / "RUNS.csv").open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(rows)

            write_runs()
            metadata, errors = report._audit_subreport(root, specs, summaries)
            self.assertEqual(errors, [])
            self.assertEqual(metadata["rows"], 2)
            self.assertEqual(metadata["valid_required_rows"], 2)
            self.assertEqual(set(metadata["artifact_sha256"]), set(report.SUBREPORT_ARTIFACTS))

            rows[1]["audit_status"] = "INVALID"
            write_runs()
            _, errors = report._audit_subreport(root, specs, summaries)
            self.assertTrue(any("audit_status='INVALID'" in x for x in errors))


if __name__ == "__main__":
    unittest.main()
