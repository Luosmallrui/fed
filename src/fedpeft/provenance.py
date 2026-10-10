from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fedpeft.config import matrix_run_count, required_run_count
from fedpeft.utils import (
    cached_directory_sha256,
    file_sha256,
    source_python_sha256,
)


V3_LAUNCH_PROVENANCE_SCHEMA = "stability-v3-preflight-v1"
V3_PROTOCOL_RELATIVE_PATH = Path(
    "docs/results/exploration/STABILITY_V3_PLAN.md"
)
V4_LAUNCH_PROVENANCE_SCHEMA = "stability-v4-preflight-v1"
V4_PROTOCOL_RELATIVE_PATH = Path(
    "docs/results/exploration/STABILITY_V4_PLAN.md"
)

_V4_DATASET_ROWS = {
    "sst2": {"train": 67349, "validation": 872, "test": 1821},
    "mnli": {
        "train": 392702,
        "validation_matched": 9815,
        "validation_mismatched": 9832,
        "test_matched": 9796,
        "test_mismatched": 9847,
    },
    "mrpc": {"train": 3668, "validation": 408, "test": 1725},
    "qqp": {"train": 363846, "validation": 40430, "test": 390965},
}
_V4_NUM_LABELS = {"sst2": 2, "mnli": 3, "mrpc": 2, "qqp": 2}
_V4_TEXT_FIELDS = {
    "sst2": ["sentence"],
    "mnli": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    "qqp": ["question1", "question2"],
}
_V4_VALIDATION_SPLITS = {
    "sst2": ["validation"],
    "mnli": ["validation_matched", "validation_mismatched"],
    "mrpc": ["validation"],
    "qqp": ["validation"],
}
_V4_LABEL_NAMES = {
    "sst2": ["negative", "positive"],
    "mnli": ["entailment", "neutral", "contradiction"],
    "mrpc": ["not_equivalent", "equivalent"],
    "qqp": ["not_duplicate", "duplicate"],
}


def _v4_primary_metric_name(task: str) -> str:
    if task == "mnli":
        return "mean_matched_mismatched_accuracy"
    if task in {"mrpc", "qqp"}:
        return "glue_accuracy_f1_mean"
    return "accuracy"


def _validate_v4_preflight_contract(
    cfg: dict[str, Any], preflight: dict[str, Any], checks: dict[str, Any]
) -> list[str]:
    """Validate that the bound JSON is a complete V4 preflight result.

    Hash equality alone is insufficient: a hand-written ``PASS`` document
    could otherwise satisfy the launcher without proving that data,
    partition, metric, matrix, and accelerator checks actually ran.
    """

    errors: list[str] = []
    if preflight.get("schema") != V4_LAUNCH_PROVENANCE_SCHEMA:
        errors.append("preflight schema is not stability-v4-preflight-v1")

    tasks = [str(task) for task in cfg["data"]["tasks"]]
    seeds = [int(seed) for seed in cfg["data"]["partition_seeds"]]
    expected_checks = {
        "output_dir": cfg["project"]["output_dir"],
        "phase": cfg["project"]["phase"],
        "expected_runs": matrix_run_count(cfg),
        "required_runs": required_run_count(cfg),
        "primary_metrics": {
            task: _v4_primary_metric_name(task) for task in tasks
        },
        "schedule_overrides": cfg["training"].get("schedule_overrides", {}),
        "cuda_available": True,
        "gpu_bfloat16_supported": True,
        "model_type": "roberta",
    }
    for key, expected in expected_checks.items():
        if checks.get(key) != expected:
            errors.append(f"preflight {key} differs from the V4 contract")

    datasets = checks.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(tasks):
        errors.append("preflight datasets do not exactly match configured V4 tasks")
    else:
        for task in tasks:
            if datasets.get(task) != _V4_DATASET_ROWS.get(task):
                errors.append(f"preflight dataset rows differ for {task}")

    contracts = checks.get("dataset_contracts")
    if not isinstance(contracts, dict) or set(contracts) != set(tasks):
        errors.append("preflight dataset contracts do not exactly match V4 tasks")
    else:
        for task in tasks:
            contract = contracts.get(task)
            if not isinstance(contract, dict):
                errors.append(f"preflight dataset contract is malformed for {task}")
                continue
            if contract.get("required_text_fields") != _V4_TEXT_FIELDS[task]:
                errors.append(f"preflight text fields differ for {task}")
            split_contracts = contract.get("splits")
            expected_splits = {"train", *_V4_VALIDATION_SPLITS[task]}
            if (
                not isinstance(split_contracts, dict)
                or set(split_contracts) != expected_splits
            ):
                errors.append(f"preflight split contracts differ for {task}")
                continue
            for split in sorted(expected_splits):
                split_contract = split_contracts[split]
                if not isinstance(split_contract, dict):
                    errors.append(
                        f"preflight dataset contract {task}/{split} is malformed"
                    )
                    continue
                columns = split_contract.get("column_names")
                required_columns = {*_V4_TEXT_FIELDS[task], "label"}
                if not isinstance(columns, list) or not required_columns.issubset(
                    columns
                ):
                    errors.append(
                        f"preflight dataset contract {task}/{split} lacks fields"
                    )
                if split_contract.get("label_names") != _V4_LABEL_NAMES[task]:
                    errors.append(
                        f"preflight dataset contract {task}/{split} label order differs"
                    )
                if split_contract.get("label_values") != list(
                    range(_V4_NUM_LABELS[task])
                ):
                    errors.append(
                        f"preflight dataset contract {task}/{split} labels differ"
                    )

    previews = checks.get("partition_previews")
    expected_preview_keys = {
        f"{task}/seed-{seed}" for task in tasks for seed in seeds
    }
    if not isinstance(previews, dict) or set(previews) != expected_preview_keys:
        errors.append("preflight partition previews are incomplete or unexpected")
    else:
        clients = int(cfg["data"]["num_clients"])
        for key in sorted(expected_preview_keys):
            task = key.split("/", 1)[0]
            preview = previews[key]
            if not isinstance(preview, dict):
                errors.append(f"preflight partition preview {key} is malformed")
                continue
            digest = preview.get("digest")
            try:
                digest_is_valid = (
                    isinstance(digest, str)
                    and len(digest) == 64
                    and int(digest, 16) >= 0
                )
            except ValueError:
                digest_is_valid = False
            train_sizes = preview.get("train_sizes")
            holdout_sizes = preview.get("holdout_sizes")
            label_counts = preview.get("pooled_holdout_label_counts")
            expected_label_keys = {
                str(index) for index in range(_V4_NUM_LABELS[task])
            }
            if not digest_is_valid:
                errors.append(f"preflight partition preview {key} has invalid digest")
            for field, values in (
                ("train_sizes", train_sizes),
                ("holdout_sizes", holdout_sizes),
            ):
                if (
                    not isinstance(values, list)
                    or len(values) != clients
                    or any(not isinstance(value, int) or value <= 0 for value in values)
                ):
                    errors.append(
                        f"preflight partition preview {key}.{field} is invalid"
                    )
            if (
                not isinstance(label_counts, dict)
                or set(label_counts) != expected_label_keys
                or any(
                    not isinstance(value, int) or value <= 0
                    for value in label_counts.values()
                )
            ):
                errors.append(
                    f"preflight partition preview {key} lacks every holdout label"
                )

    for key in ("python", "torch", "transformers", "cuda_version", "gpu"):
        if not isinstance(checks.get(key), str) or not checks[key]:
            errors.append(f"preflight {key} is absent")
    model_hash = checks.get("model_hash")
    try:
        model_hash_is_valid = (
            isinstance(model_hash, str)
            and len(model_hash) == 64
            and int(model_hash, 16) >= 0
        )
    except ValueError:
        model_hash_is_valid = False
    if not model_hash_is_valid:
        errors.append("preflight model_hash is invalid")
    for key in ("gpu_memory_bytes", "disk_free_bytes"):
        value = checks.get(key)
        if not isinstance(value, int) or value <= 0:
            errors.append(f"preflight {key} is invalid")
    return errors


def verified_stability_launch_provenance(
    cfg: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    generation = cfg.get("stability", {}).get("generation")
    if generation == 3:
        schema = V3_LAUNCH_PROVENANCE_SCHEMA
        protocol_relative_path = V3_PROTOCOL_RELATIVE_PATH
    elif generation == 4:
        schema = V4_LAUNCH_PROVENANCE_SCHEMA
        protocol_relative_path = V4_PROTOCOL_RELATIVE_PATH
    else:
        raise RuntimeError("locked launch provenance requires stability generation 3 or 4")

    output_dir = Path(output_dir)
    preflight_path = output_dir / "preflight.json"
    label = f"stability-v{generation}"
    try:
        preflight_bytes = preflight_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"{label} Stage A requires readable {preflight_path}") from exc
    try:
        preflight = json.loads(preflight_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} preflight.json is not valid JSON") from exc
    if not isinstance(preflight, dict):
        raise RuntimeError(f"{label} preflight.json is not a JSON object")

    root = Path(cfg["project"]["root"])
    config_path = Path(cfg["_config_path"])
    protocol_path = root / protocol_relative_path
    try:
        current: dict[str, Any] = {
            "source_hash": source_python_sha256(root),
            "config_file_hash": file_sha256(config_path),
            "stability_protocol_file_hash": file_sha256(protocol_path),
        }
        if generation == 4:
            model_path = Path(cfg["model"]["path"])
            if not model_path.exists():
                raise FileNotFoundError(model_path)
            fingerprint_dir = root / "artifacts" / "fingerprints"
            model_cache_name = hashlib.sha256(
                str(model_path.resolve()).encode()
            ).hexdigest()[:16]
            current["model_hash"] = cached_directory_sha256(
                model_path,
                fingerprint_dir / f"model-{model_cache_name}.json",
            )
            dataset_hashes: dict[str, str] = {}
            for task in cfg["data"]["tasks"]:
                dataset_path = Path(cfg["data"]["root"]) / str(task)
                if not dataset_path.exists():
                    raise FileNotFoundError(dataset_path)
                dataset_cache_name = hashlib.sha256(
                    str(dataset_path.resolve()).encode()
                ).hexdigest()[:16]
                dataset_hashes[str(task)] = cached_directory_sha256(
                    dataset_path,
                    fingerprint_dir / f"dataset-{dataset_cache_name}.json",
                )
            current["dataset_hashes"] = dataset_hashes
    except OSError as exc:
        raise RuntimeError(
            f"cannot hash current {label} source/config/protocol bytes"
        ) from exc

    errors: list[str] = []
    if preflight.get("status") != "PASS":
        errors.append("preflight status is not PASS")
    if preflight.get("failures") != []:
        errors.append("preflight failures is not an empty list")
    checks = preflight.get("checks")
    if not isinstance(checks, dict):
        errors.append("preflight checks is not an object")
        checks = {}
    if generation == 4:
        errors.extend(_validate_v4_preflight_contract(cfg, preflight, checks))
    for key, value in current.items():
        if checks.get(key) != value:
            errors.append(f"preflight {key} differs from current bytes")

    declared_protocol_hash = cfg.get("stability", {}).get("protocol_sha256")
    if current["stability_protocol_file_hash"] != declared_protocol_hash:
        errors.append("current protocol file differs from the configured hash")
    if checks.get("stability_protocol_hash") != declared_protocol_hash:
        errors.append("preflight configured protocol hash differs from current config")
    if errors:
        raise RuntimeError(
            f"{label} Stage-A preflight provenance check failed: "
            + "; ".join(errors)
        )

    return {
        "schema": schema,
        "preflight_file": "preflight.json",
        "preflight_file_sha256": hashlib.sha256(preflight_bytes).hexdigest(),
        **current,
    }


def verified_v3_launch_provenance(
    cfg: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Verify the Stage-A preflight against current bytes and bind its file.

    The returned mapping is deliberately deterministic.  It is written into
    ``campaign_config.json`` on the first launch and reconstructed verbatim on
    resume, so replacing either the preflight or any checked input cannot
    silently continue the frozen campaign.
    """

    if cfg.get("stability", {}).get("generation") != 3:
        raise RuntimeError("verified_v3_launch_provenance requires generation 3")
    return verified_stability_launch_provenance(cfg, output_dir)
