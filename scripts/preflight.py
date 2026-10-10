#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer

from fedpeft.config import load_config, matrix_run_count, required_run_count
from fedpeft.data import (
    TASK_CANONICAL_ROWS,
    TASK_FIELDS,
    TASK_NUM_LABELS,
    TASK_VALIDATION_SPLITS,
    create_dirichlet_partition,
)
from fedpeft.metrics import primary_metric_name
from fedpeft.utils import (
    atomic_json_dump,
    cached_directory_sha256,
    file_sha256,
    source_python_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    failures: list[str] = []
    checks: dict[str, object] = {}
    actual_runs = matrix_run_count(cfg)
    required_runs = required_run_count(cfg)
    model_path = Path(cfg["model"]["path"])
    if not model_path.exists():
        failures.append(f"missing model: {model_path}")
    else:
        model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, use_fast=True)
        checks["model_type"] = model_config.model_type
        checks["tokenizer"] = tokenizer.__class__.__name__
        cache_name = hashlib.sha256(str(model_path.resolve()).encode()).hexdigest()[:16]
        checks["model_hash"] = cached_directory_sha256(
            model_path,
            Path(cfg["project"]["root"])
            / "artifacts"
            / "fingerprints"
            / f"model-{cache_name}.json",
        )
    datasets: dict[str, dict[str, int]] = {}
    dataset_hashes: dict[str, str] = {}
    dataset_contracts: dict[str, object] = {}
    task_metrics: dict[str, str] = {}
    partition_previews: dict[str, dict[str, object]] = {}
    for task in cfg["data"]["tasks"]:
        path = Path(cfg["data"]["root"]) / task
        if not path.exists():
            failures.append(f"missing dataset: {path}")
            continue
        loaded = load_from_disk(str(path))
        dataset_cache_name = hashlib.sha256(
            str(path.resolve()).encode()
        ).hexdigest()[:16]
        dataset_hashes[task] = cached_directory_sha256(
            path,
            Path(cfg["project"]["root"])
            / "artifacts"
            / "fingerprints"
            / f"dataset-{dataset_cache_name}.json",
        )
        missing = [x for x in ["train", *TASK_VALIDATION_SPLITS[task]] if x not in loaded]
        if missing:
            failures.append(f"{task} missing splits: {missing}")
        datasets[task] = {split: len(value) for split, value in loaded.items()}
        labels = set(loaded["train"]["label"])
        if labels != set(range(TASK_NUM_LABELS[task])):
            failures.append(f"{task} train labels are {sorted(labels)}")
        expected_rows = TASK_CANONICAL_ROWS.get(task)
        if expected_rows is not None:
            observed_rows = {split: len(value) for split, value in loaded.items()}
            if observed_rows != expected_rows:
                failures.append(
                    f"{task} split sizes are {observed_rows}, expected {expected_rows}"
                )
        required_text_fields = [field for field in TASK_FIELDS[task] if field]
        expected_label_names = {
            "sst2": ["negative", "positive"],
            "mnli": ["entailment", "neutral", "contradiction"],
            "mrpc": ["not_equivalent", "equivalent"],
            "qqp": ["not_duplicate", "duplicate"],
        }.get(task)
        contract_splits: dict[str, object] = {}
        for split in ["train", *TASK_VALIDATION_SPLITS[task]]:
            if split not in loaded:
                continue
            absent_fields = [
                field
                for field in [*required_text_fields, "label"]
                if field not in loaded[split].column_names
            ]
            if absent_fields:
                failures.append(f"{task}/{split} missing fields: {absent_fields}")
                continue
            for field in required_text_fields:
                values = loaded[split][field]
                if any(value is None or not str(value).strip() for value in values):
                    failures.append(f"{task}/{split}.{field} contains null/empty text")
            split_labels = set(loaded[split]["label"])
            if split_labels != set(range(TASK_NUM_LABELS[task])):
                failures.append(
                    f"{task}/{split} labels are {sorted(split_labels)}"
                )
            if expected_label_names is not None:
                label_feature = loaded[split].features.get("label")
                label_names = getattr(label_feature, "names", None)
                if label_names != expected_label_names:
                    failures.append(
                        f"{task}/{split} label names are {label_names}, "
                        f"expected {expected_label_names}"
                    )
            else:
                label_feature = loaded[split].features.get("label")
                label_names = getattr(label_feature, "names", None)
            contract_splits[split] = {
                "column_names": list(loaded[split].column_names),
                "label_names": label_names,
                "label_values": sorted(split_labels),
            }
        dataset_contracts[task] = {
            "required_text_fields": required_text_fields,
            "splits": contract_splits,
        }
        task_metrics[task] = primary_metric_name(task)
        labels_array = np.asarray(loaded["train"]["label"], dtype=np.int64)
        for seed in cfg["data"]["partition_seeds"]:
            try:
                partition = create_dirichlet_partition(
                    labels_array,
                    int(cfg["data"]["num_clients"]),
                    float(cfg["data"]["dirichlet_alpha"]),
                    float(cfg["data"]["local_holdout_fraction"]),
                    int(cfg["data"]["min_client_examples"]),
                    int(seed),
                )
            except RuntimeError as exc:
                failures.append(f"{task}/seed-{seed} partition failed: {exc}")
                continue
            pooled_holdout = np.concatenate(partition.holdout)
            holdout_labels = set(labels_array[pooled_holdout].tolist())
            if holdout_labels != set(range(TASK_NUM_LABELS[task])):
                failures.append(
                    f"{task}/seed-{seed} pooled holdout labels are "
                    f"{sorted(holdout_labels)}"
                )
            partition_previews[f"{task}/seed-{seed}"] = {
                "digest": partition.digest,
                "train_sizes": [len(value) for value in partition.train],
                "holdout_sizes": [len(value) for value in partition.holdout],
                "pooled_holdout_label_counts": {
                    str(label): int((labels_array[pooled_holdout] == label).sum())
                    for label in range(TASK_NUM_LABELS[task])
                },
            }
    checks.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "datasets": datasets,
            "dataset_hashes": dataset_hashes,
            "dataset_contracts": dataset_contracts,
            "primary_metrics": task_metrics,
            "partition_previews": partition_previews,
            "output_dir": cfg["project"]["output_dir"],
            "phase": cfg["project"]["phase"],
            "expected_runs": actual_runs,
            "required_runs": required_runs,
            "disk_free_bytes": shutil.disk_usage(cfg["project"]["root"]).free,
            "gpu_bfloat16_supported": torch.cuda.is_available()
            and torch.cuda.is_bf16_supported(),
            "source_hash": source_python_sha256(Path(cfg["project"]["root"])),
            "config_file_hash": file_sha256(cfg["_config_path"]),
            "stability_protocol_hash": cfg.get("stability", {}).get(
                "protocol_sha256"
            ),
            "schedule_overrides": cfg["training"].get(
                "schedule_overrides", {}
            ),
        }
    )
    stability_generation = cfg.get("stability", {}).get("generation")
    if stability_generation in {3, 4}:
        protocol_path = (
            Path(cfg["project"]["root"])
            / "docs"
            / "results"
            / "exploration"
            / f"STABILITY_V{stability_generation}_PLAN.md"
        )
        if not protocol_path.exists():
            failures.append(
                f"missing stability-v{stability_generation} protocol: {protocol_path}"
            )
        else:
            actual_protocol_hash = file_sha256(protocol_path)
            checks["stability_protocol_file_hash"] = actual_protocol_hash
            if actual_protocol_hash != cfg["stability"]["protocol_sha256"]:
                failures.append(
                    f"stability-v{stability_generation} protocol file hash "
                    "differs from the config"
                )
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    elif cfg["training"]["mixed_precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
        failures.append("configured BF16 is unsupported by this GPU")
    strict_matrix_size = cfg["project"]["phase"] == "final" or cfg.get(
        "stability", {}
    ).get("generation") in {3, 4}
    if strict_matrix_size and actual_runs != required_runs:
        failures.append(
            f"final matrix must contain {required_runs} runs, got {actual_runs}"
        )
    preflight_schema = (
        f"stability-v{stability_generation}-preflight-v1"
        if stability_generation in {3, 4}
        else "experiment-preflight-v1"
    )
    result = {
        "schema": preflight_schema,
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }
    output = Path(cfg["project"]["output_dir"]) / "preflight.json"
    atomic_json_dump(result, output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
