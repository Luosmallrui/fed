#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_from_disk
from transformers import AutoConfig, AutoTokenizer

from fedpeft.compression import largest_remainder_quotas
from fedpeft.config import (
    DIAGNOSTIC_RECIPE_VERSION,
    STABILITY_CANDIDATES,
    load_config,
    result_path,
    run_id,
    task_value,
)
from fedpeft.data import TASK_NUM_LABELS, TASK_VALIDATION_SPLITS
from fedpeft.metrics import primary_metric, primary_metric_name
from fedpeft.diagnostics import DiagnosticLayout
from fedpeft.utils import bytes_human, sha256_arrays

SCHEMA_VERSION = 2
MIN_PAIRED_SEEDS = 3
THRESHOLDS = {
    "sst2": 0.003,
    "mnli": 0.003,
    "rte": 0.01,
    "mrpc": 0.01,
    "qqp": 0.003,
}
STANDARD_LORA_METHODS = {"fedavg_lora", "flasc", "full_lora_ef"}
FIXED_A_METHODS = {
    "ffa_lora",
    "random_a_dense",
    "random_a_ef",
    "random_a_topk",
    "spectral_a_ef",
    "spectral_a_topk",
    "spectral_a_dense",
    "spectral_rot_ef",
    "spectral_block_ef",
    "hybrid4_a_ef",
    "hybrid4_a_dense",
    "hybrid4_a_decay_dense",
    "hybrid4_a_decay_ef",
    "random_block_ef",
}
LORA_XS_METHODS = {"lora_xs"}
STABILITY_DIAGNOSTIC_METHODS = {
    "spectral_a_ef",
    *STABILITY_CANDIDATES,
    "hybrid4_a_decay_ef",
    "random_block_ef",
}
BLOCK_BALANCED_METHODS = {"spectral_block_ef", "random_block_ef"}
COMMUNICATION_EFFICIENCY_BY_SEED_FIELDS = (
    "task",
    "method",
    "wire_target",
    "seed",
    "primary_metric_name",
    "primary_score",
    "reference_primary_score",
    "delta_primary_score_vs_fedavg",
    "primary_score_retention_vs_fedavg",
    "metric_semantics",
    "reference_method",
    "reference_status",
    "interpretation",
    "accuracy",
    "reference_accuracy",
    "delta_accuracy_vs_fedavg",
    "accuracy_retention_vs_fedavg",
    "normal_path_wire_bytes",
    "reference_normal_path_wire_bytes",
    "wire_fraction_vs_fedavg",
    "saving_vs_fedavg",
    "descriptive_cer",
    "c95_dense_target_accuracy",
    "c95_dense_target_primary_score",
    "c95_dense_status",
    "c95_dense_round",
    "c95_dense_wire_bytes",
    "fedavg_c95_dense_round",
    "fedavg_c95_dense_wire_bytes",
    "speedup95_vs_fedavg",
)
COMMUNICATION_EFFICIENCY_FIELDS = (
    "task",
    "method",
    "wire_target",
    "n",
    "expected_n",
    "complete",
    "metric_semantics",
    "reference_method",
    "paired_reference_n",
    "reference_complete",
    "interpretation",
    "primary_metric_name",
    "mean_primary_score",
    "sd_primary_score",
    "mean_accuracy",
    "sd_accuracy",
    "mean_normal_path_wire_bytes",
    "sd_normal_path_wire_bytes",
    "mean_wire_fraction_vs_fedavg",
    "sd_wire_fraction_vs_fedavg",
    "mean_saving_vs_fedavg",
    "sd_saving_vs_fedavg",
    "mean_delta_accuracy_vs_fedavg",
    "sd_delta_accuracy_vs_fedavg",
    "mean_delta_primary_score_vs_fedavg",
    "sd_delta_primary_score_vs_fedavg",
    "mean_accuracy_retention_vs_fedavg",
    "sd_accuracy_retention_vs_fedavg",
    "mean_primary_score_retention_vs_fedavg",
    "sd_primary_score_retention_vs_fedavg",
    "mean_descriptive_cer",
    "sd_descriptive_cer",
    "c95_dense_reached_n",
    "mean_c95_dense_round",
    "sd_c95_dense_round",
    "mean_c95_dense_wire_bytes",
    "sd_c95_dense_wire_bytes",
    "mean_fedavg_c95_dense_wire_bytes",
    "sd_fedavg_c95_dense_wire_bytes",
    "mean_speedup95_vs_fedavg",
    "sd_speedup95_vs_fedavg",
    "pareto_status",
)


@dataclass(frozen=True)
class ExpectedRun:
    task: str
    method: str
    seed: int
    wire_ratio: float | None
    role: str  # required | optional

    @property
    def run_id(self) -> str:
        return run_id(self.task, self.method, self.seed, self.wire_ratio)

    @property
    def run_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "method": self.method,
            "seed": self.seed,
            "wire_ratio": self.wire_ratio,
        }


@dataclass
class AuditRecord:
    run_id: str
    role: str
    path: Path
    spec: ExpectedRun | None = None
    summary: dict[str, Any] | None = None
    audit_status: str = "MISSING"
    reasons: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.audit_status == "VALID" and self.summary is not None


@dataclass(frozen=True)
class ModelExpectations:
    model_hash: str
    model_path: Path
    hidden_size: int
    num_layers: int
    target_names: tuple[str, ...]


@dataclass
class ValidationContext:
    source_hash: str
    model: ModelExpectations | None
    dataset_hashes: dict[str, str]
    dataset_rows: dict[str, int]
    dataset_label_hashes: dict[str, str]
    partition_hashes: dict[tuple[str, int], str]
    errors: list[str]


def _clean_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in cfg.items() if not key.startswith("_")}


def _method_task_value(cfg: dict[str, Any], key: str, task: str, method: str) -> Any:
    method_values = cfg["training"].get("method_overrides", {}).get(method, {})
    if key in method_values:
        value = method_values[key]
    else:
        value = cfg["training"][key]
    return value[task] if isinstance(value, dict) else value


def _learning_rates(cfg: dict[str, Any], task: str, method: str) -> tuple[float, float]:
    adapter_learning_rate = float(
        _method_task_value(cfg, "learning_rate", task, method)
    )
    try:
        head_learning_rate = float(
            _method_task_value(cfg, "head_learning_rate", task, method)
        )
    except KeyError:
        head_learning_rate = adapter_learning_rate
    return adapter_learning_rate, head_learning_rate


def _expected_lr_schedule(
    cfg: dict[str, Any], method: str, total_rounds: int
) -> dict[str, Any]:
    configured = cfg["training"].get("schedule_overrides", {}).get(method)
    if configured is None:
        return {
            "name": "constant",
            "start_fraction": None,
            "start_round": None,
            "final_multiplier": 1.0,
            "apply_to": ["adapter", "head"],
            "total_rounds": int(total_rounds),
        }
    return {
        "name": str(configured["name"]),
        "start_fraction": float(configured["start_fraction"]),
        "start_round": math.floor(
            float(configured["start_fraction"]) * int(total_rounds)
        ),
        "final_multiplier": float(configured["final_multiplier"]),
        "apply_to": list(configured["apply_to"]),
        "total_rounds": int(total_rounds),
    }


def _expected_round_learning_rate(
    cfg: dict[str, Any], spec: ExpectedRun, round_number: int, total_rounds: int
) -> dict[str, float | str]:
    adapter_lr, head_lr = _learning_rates(cfg, spec.task, spec.method)
    schedule = _expected_lr_schedule(cfg, spec.method, total_rounds)
    if schedule["name"] == "constant":
        multiplier = 1.0
    else:
        start_round = int(schedule["start_round"])
        if round_number <= start_round:
            multiplier = 1.0
        else:
            progress = (round_number - start_round) / (total_rounds - start_round)
            floor = float(schedule["final_multiplier"])
            multiplier = floor + (1.0 - floor) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
    return {
        "schedule_name": str(schedule["name"]),
        "multiplier": multiplier,
        "adapter_learning_rate": adapter_lr * multiplier,
        "head_learning_rate": head_lr * multiplier,
    }


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_hash(cfg: dict[str, Any], spec: ExpectedRun) -> str:
    payload = json.dumps(
        {"config": _clean_config(cfg), "run": spec.run_dict}, sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _artifact_directory_hash(root: Path) -> str:
    """Reproduce the engine's immutable model-content fingerprint without writing a cache."""
    root = root.resolve()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and path.suffix not in {".lock", ".tmp"}
    )
    if not files:
        raise FileNotFoundError(f"model artifact has no files: {root}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _dataset_hash(tokenized: Any) -> str:
    payload = [
        {
            "split": split,
            "rows": len(tokenized[split]),
            "fingerprint": str(getattr(tokenized[split], "_fingerprint", "")),
        }
        for split in sorted(tokenized.keys())
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _tokenizer_hash(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        serialized = backend.to_str()
    else:
        serialized = json.dumps(
            tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True
        )
    payload = {
        "backend": serialized,
        "special_tokens": {
            key: str(value) for key, value in tokenizer.special_tokens_map.items()
        },
        "model_input_names": list(tokenizer.model_input_names),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _raw_dataset_manifest(raw: Any) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "rows": len(raw[split]),
            "fingerprint": str(getattr(raw[split], "_fingerprint", "")),
            "features": str(raw[split].features),
        }
        for split in sorted(raw.keys())
    ]


def _tokenized_path(cfg: dict[str, Any], task: str) -> Path:
    """Find the one cache whose success manifest matches current raw/tokenizer inputs."""
    max_length = int(cfg["data"]["max_length"])
    raw = load_from_disk(str(Path(cfg["data"]["root"]) / task))
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["path"], local_files_only=True, use_fast=True
    )
    provenance = {
        "schema_version": 1,
        "task": task,
        "max_length": max_length,
        "padding": "max_length",
        "truncation": True,
        "tokenizer_hash": _tokenizer_hash(tokenizer),
        "raw_dataset": _raw_dataset_manifest(raw),
    }
    provenance_hash = hashlib.sha256(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_root = Path(cfg["project"]["tokenized_dir"])
    markers = sorted(
        cache_root.glob(
            f"roberta-base__{task}__len-{max_length}__source-*/_TOKENIZED_SUCCESS.json"
        )
    )
    matches: list[Path] = []
    for marker in markers:
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if recorded == provenance:
            matches.append(marker.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one current tokenized cache for {task}, found {len(matches)}"
        )
    expected_name = (
        f"roberta-base__{task}__len-{max_length}__source-{provenance_hash[:12]}"
    )
    if matches[0].name != expected_name:
        raise RuntimeError(
            f"tokenized cache name does not match its provenance: {matches[0].name} != {expected_name}"
        )
    return matches[0]


def _partition_path(
    cfg: dict[str, Any], task: str, seed: int, dataset_hash: str
) -> Path:
    data = cfg["data"]
    return Path(cfg["project"]["partition_dir"]) / (
        f"{task}__clients-{data['num_clients']}__alpha-{data['dirichlet_alpha']}"
        f"__holdout-{data['local_holdout_fraction']}"
        f"__min-{data['min_client_examples']}"
        f"__data-{dataset_hash[:12]}__seed-{seed}.npz"
    )


def _read_partition_hash(
    path: Path,
    num_clients: int,
    expected_rows: int,
    expected_metadata: dict[str, Any],
) -> str:
    if not path.exists():
        raise FileNotFoundError(f"partition is missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        for key, expected in expected_metadata.items():
            if key not in payload or payload[key].item() != expected:
                actual = None if key not in payload else payload[key].item()
                raise ValueError(
                    f"partition metadata mismatch for {key}: {actual!r} != {expected!r}"
                )
        arrays: list[np.ndarray] = []
        for client_id in range(num_clients):
            for prefix in ("train", "holdout"):
                key = f"{prefix}_{client_id}"
                if key not in payload:
                    raise ValueError(f"partition lacks {key}: {path}")
                values = np.asarray(payload[key], dtype=np.int64)
                if values.ndim != 1:
                    raise ValueError(f"partition {key} is not one-dimensional: {path}")
                arrays.append(values)
        embedded = str(payload["digest"].item()) if "digest" in payload else ""
    computed = sha256_arrays(arrays)
    if embedded != computed:
        raise ValueError(f"partition digest mismatch: {path}")
    combined = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
    if combined.size != expected_rows:
        raise ValueError(
            f"partition covers {combined.size} rows, expected {expected_rows}: {path}"
        )
    if combined.size:
        if int(combined.min()) < 0 or int(combined.max()) >= expected_rows:
            raise ValueError(f"partition contains an out-of-range index: {path}")
        if np.unique(combined).size != expected_rows:
            raise ValueError(f"partition contains duplicate or missing indices: {path}")
    return computed


def _build_context(
    cfg: dict[str, Any], specs: Iterable[ExpectedRun]
) -> ValidationContext:
    root = Path(cfg["project"]["root"]).resolve()
    errors: list[str] = []
    try:
        source_hash = _source_hash(root)
    except OSError as exc:  # surfaced as INVALID, never silently accepted
        source_hash = ""
        errors.append(f"cannot fingerprint source: {type(exc).__name__}: {exc}")

    model_expectations: ModelExpectations | None = None
    try:
        model_path = Path(cfg["model"]["path"]).resolve()
        model_config = AutoConfig.from_pretrained(
            str(model_path), local_files_only=True
        )
        if str(getattr(model_config, "model_type", "")) != "roberta":
            raise ValueError(f"expected RoBERTa, found {model_config.model_type}")
        hidden_size = int(model_config.hidden_size)
        num_layers = int(model_config.num_hidden_layers)
        target_names = tuple(
            f"encoder.encoder.layer.{layer}.{suffix}"
            for layer in range(num_layers)
            for suffix in cfg["model"]["target_suffixes"]
        )
        model_expectations = ModelExpectations(
            model_hash=_artifact_directory_hash(model_path),
            model_path=model_path,
            hidden_size=hidden_size,
            num_layers=num_layers,
            target_names=target_names,
        )
    except (AttributeError, OSError, ValueError) as exc:
        errors.append(f"cannot validate model artifact: {type(exc).__name__}: {exc}")

    tasks = sorted({spec.task for spec in specs})
    dataset_hashes: dict[str, str] = {}
    dataset_rows: dict[str, int] = {}
    dataset_label_hashes: dict[str, str] = {}
    for task in tasks:
        try:
            tokenized = load_from_disk(str(_tokenized_path(cfg, task)))
            dataset_hashes[task] = _dataset_hash(tokenized)
            dataset_rows[task] = len(tokenized["train"])
            labels = np.asarray(tokenized["train"]["labels"], dtype=np.int64)
            dataset_label_hashes[task] = sha256_arrays([labels])
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            errors.append(
                f"cannot validate tokenized dataset {task}: {type(exc).__name__}: {exc}"
            )

    partition_hashes: dict[tuple[str, int], str] = {}
    pairs = sorted({(spec.task, spec.seed) for spec in specs})
    for task, seed in pairs:
        if task not in dataset_hashes or task not in dataset_rows:
            continue
        try:
            path = _partition_path(cfg, task, seed, dataset_hashes[task])
            expected_metadata = {
                "seed": int(seed),
                "num_clients": int(cfg["data"]["num_clients"]),
                "alpha": float(cfg["data"]["dirichlet_alpha"]),
                "holdout_fraction": float(cfg["data"]["local_holdout_fraction"]),
                "min_client_examples": int(cfg["data"]["min_client_examples"]),
                "labels_hash": dataset_label_hashes[task],
            }
            partition_hashes[(task, seed)] = _read_partition_hash(
                path,
                int(cfg["data"]["num_clients"]),
                dataset_rows[task],
                expected_metadata,
            )
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            errors.append(
                f"cannot validate partition {task}/seed-{seed}: "
                f"{type(exc).__name__}: {exc}"
            )
    return ValidationContext(
        source_hash=source_hash,
        model=model_expectations,
        dataset_hashes=dataset_hashes,
        dataset_rows=dataset_rows,
        dataset_label_hashes=dataset_label_hashes,
        partition_hashes=partition_hashes,
        errors=errors,
    )


def _method_lists(cfg: dict[str, Any], diagnostic: bool) -> tuple[list[str], list[str]]:
    dense = list(cfg["matrix"].get("dense_methods", []))
    sparse = list(cfg["matrix"].get("sparse_methods", []))
    if diagnostic:
        dense = list(cfg["matrix"].get("diagnostic_dense_methods", []))
        sparse = list(cfg["matrix"].get("diagnostic_sparse_methods", []))
    return dense, sparse


def _specs(cfg: dict[str, Any], diagnostic: bool, role: str) -> list[ExpectedRun]:
    dense, sparse = _method_lists(cfg, diagnostic)
    values: list[ExpectedRun] = []
    for task in cfg["data"]["tasks"]:
        for raw_seed in cfg["data"]["partition_seeds"]:
            seed = int(raw_seed)
            values.extend(
                ExpectedRun(task, method, seed, None, role) for method in dense
            )
            for method in sparse:
                values.extend(
                    ExpectedRun(task, method, seed, float(ratio), role)
                    for ratio in cfg["communication"]["wire_ratios"]
                )
    return values


def _same_ratio(actual: Any, expected: float | None) -> bool:
    if expected is None:
        return actual is None
    try:
        return actual is not None and math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _is_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _expected_layout(
    cfg: dict[str, Any], model: ModelExpectations, spec: ExpectedRun
) -> tuple[str, int, int, int, list[str]]:
    rank = int(cfg["model"]["rank"])
    hidden = model.hidden_size
    labels = TASK_NUM_LABELS[spec.task]
    adapter: list[tuple[str, tuple[int, ...]]] = []
    for target in model.target_names:
        if spec.method in STANDARD_LORA_METHODS:
            adapter.extend(
                [
                    (f"{target}.lora_A", (rank, hidden)),
                    (f"{target}.lora_B", (hidden, rank)),
                ]
            )
        elif spec.method in FIXED_A_METHODS:
            adapter.append((f"{target}.lora_B", (hidden, rank)))
        elif spec.method in LORA_XS_METHODS:
            adapter.append((f"{target}.lora_R", (rank, rank)))
        else:
            raise ValueError(f"no report layout rule for method {spec.method}")
    adapter.sort(key=lambda item: item[0])
    head = sorted(
        [
            ("classifier.weight", (labels, hidden)),
            ("classifier.bias", (labels,)),
        ],
        key=lambda item: item[0],
    )
    payload = {"adapter": adapter, "head": head}
    layout_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    adapter_parameters = sum(math.prod(shape) for _, shape in adapter)
    head_parameters = hidden * labels + labels
    reference_adapter = len(model.target_names) * rank * (hidden + hidden)
    return (
        layout_hash,
        adapter_parameters,
        head_parameters,
        reference_adapter + head_parameters,
        list(model.target_names),
    )


def _canonical_wire_layout(
    method: str,
    task: str,
    target_names: Iterable[str],
    rank: int,
    hidden_size: int,
) -> tuple[list[tuple[str, tuple[int, ...]]], list[tuple[str, tuple[int, ...]]]]:
    """Reconstruct the canonical adapter/head manifest without loading a run.

    This deliberately mirrors the serialized tensor *contract*, rather than
    reading tensor names or shapes back from a summary.  Those summary fields
    are audit inputs and therefore cannot be the authority for their own wire
    ledger.
    """

    adapter: list[tuple[str, tuple[int, ...]]] = []
    for target in target_names:
        if method in STANDARD_LORA_METHODS:
            adapter.extend(
                [
                    (f"{target}.lora_A", (rank, hidden_size)),
                    (f"{target}.lora_B", (hidden_size, rank)),
                ]
            )
        elif method in FIXED_A_METHODS:
            adapter.append((f"{target}.lora_B", (hidden_size, rank)))
        elif method in LORA_XS_METHODS:
            adapter.append((f"{target}.lora_R", (rank, rank)))
        else:
            raise ValueError(f"no canonical wire layout rule for method {method}")
    adapter.sort(key=lambda item: item[0])
    labels = TASK_NUM_LABELS[task]
    head = sorted(
        [
            ("classifier.weight", (labels, hidden_size)),
            ("classifier.bias", (labels,)),
        ],
        key=lambda item: item[0],
    )
    return adapter, head


def _canonical_manifest_bytes(
    adapter: list[tuple[str, tuple[int, ...]]],
    head: list[tuple[str, tuple[int, ...]]],
) -> int:
    manifest = [
        {"name": name, "shape": list(shape)} for name, shape in adapter + head
    ]
    return len(
        json.dumps(
            manifest,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )


def _canonical_wire_dimensions(
    method: str,
    task: str,
    target_names: Iterable[str],
    rank: int,
    hidden_size: int,
) -> tuple[int, int, int, int]:
    """Return adapter, head, full-LoRA-reference and manifest byte counts."""

    names = tuple(target_names)
    adapter, head = _canonical_wire_layout(
        method, task, names, rank, hidden_size
    )
    adapter_parameters = sum(math.prod(shape) for _, shape in adapter)
    head_parameters = sum(math.prod(shape) for _, shape in head)
    reference_parameters = (
        len(names) * rank * (hidden_size + hidden_size) + head_parameters
    )
    return (
        adapter_parameters,
        head_parameters,
        reference_parameters,
        _canonical_manifest_bytes(adapter, head),
    )


def _append_mismatch(
    reasons: list[str], field: str, actual: Any, expected: Any
) -> None:
    if actual != expected:
        reasons.append(f"{field}: expected {expected!r}, got {actual!r}")


def _expected_metric_splits(cfg: dict[str, Any], task: str) -> set[str]:
    policy = cfg.get("evaluation", {}).get("policy", "official_validation")
    if policy == "train_holdout":
        return {"train_holdout"}
    return set(TASK_VALIDATION_SPLITS[task])


def _validate_rounds(
    summary: dict[str, Any], cfg: dict[str, Any], spec: ExpectedRun
) -> list[str]:
    reasons: list[str] = []
    expected_metric_splits = _expected_metric_splits(cfg, spec.task)
    rounds = summary.get("rounds")
    expected_count = int(task_value(cfg, "training", "rounds", spec.task))
    if not isinstance(rounds, list):
        return ["rounds is not a list"]
    if len(rounds) != expected_count:
        reasons.append(f"round count: expected {expected_count}, got {len(rounds)}")
    numbers = [value.get("round") for value in rounds if isinstance(value, dict)]
    if numbers != list(range(1, expected_count + 1)):
        reasons.append("round identifiers are not exactly 1..configured_rounds")
    for index, value in enumerate(rounds, start=1):
        if not isinstance(value, dict):
            reasons.append(f"round {index} is not an object")
            continue
        primary = value.get("primary_metric")
        if primary is not None and not _finite(primary):
            reasons.append(f"round {index} has a non-finite primary metric")
        elif primary is not None and (
            cfg.get("stability", {}).get("generation") == 4
            or spec.task in {"mrpc", "qqp"}
        ):
            try:
                split_values = value["splits"]
                if not isinstance(split_values, dict) or set(split_values) != expected_metric_splits:
                    raise ValueError(
                        f"expected splits {sorted(expected_metric_splits)}, got "
                        f"{sorted(split_values) if isinstance(split_values, dict) else type(split_values).__name__}"
                    )
                recomputed_primary = primary_metric(spec.task, split_values)
                if not math.isclose(
                    float(primary), recomputed_primary, rel_tol=0.0, abs_tol=1e-12
                ):
                    reasons.append(
                        f"round {index} primary metric does not match split metrics"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(f"round {index} split metrics are malformed: {exc}")
        if cfg.get("stability", {}).get("generation") in {3, 4}:
            expected_lr = _expected_round_learning_rate(
                cfg, spec, index, expected_count
            )
            if value.get("learning_rate") != expected_lr:
                reasons.append(
                    f"round {index} learning_rate differs from the frozen schedule"
                )
        timing = value.get("timing", {})
        if not isinstance(timing, dict):
            reasons.append(f"round {index} timing is not an object")
            continue
        for key in (
            "train_seconds",
            "compression_seconds",
            "eval_seconds",
            "round_seconds",
        ):
            if not _finite(timing.get(key)) or float(timing[key]) < 0:
                reasons.append(f"round {index} has invalid timing.{key}")
        if (
            not _finite(value.get("peak_gpu_memory_bytes"))
            or float(value.get("peak_gpu_memory_bytes", -1)) < 0
        ):
            reasons.append(f"round {index} has invalid peak_gpu_memory_bytes")
    final = summary.get("final")
    if not isinstance(final, dict):
        reasons.append("final is not an object")
        return reasons
    if rounds and final != rounds[-1]:
        reasons.append("final does not exactly equal the last round record")
    if final.get("round") != expected_count:
        reasons.append(f"final.round is not {expected_count}")
    if not _finite(final.get("primary_metric")):
        reasons.append("final primary metric is absent or non-finite")
        return reasons
    splits = final.get("splits", {})
    try:
        if not isinstance(splits, dict) or set(splits) != expected_metric_splits:
            raise ValueError(
                f"expected splits {sorted(expected_metric_splits)}, got "
                f"{sorted(splits) if isinstance(splits, dict) else type(splits).__name__}"
            )
        recomputed = primary_metric(spec.task, splits)
        if not math.isclose(
            float(final["primary_metric"]), recomputed, rel_tol=0.0, abs_tol=1e-12
        ):
            reasons.append(
                "final primary metric does not match its validation split metrics"
            )
    except (KeyError, TypeError, ValueError) as exc:
        reasons.append(f"final validation metrics are malformed: {exc}")
    server_primary = final.get("server_primary_metric")
    server_splits = final.get("server_splits")
    if spec.method == "flasc":
        if not _finite(server_primary):
            reasons.append(
                "FLASC final dense-server diagnostic is absent or non-finite"
            )
        elif not isinstance(server_splits, dict):
            reasons.append("FLASC final server_splits is not an object")
        else:
            actual_server_primary = float(server_primary)  # type: ignore[arg-type]
            try:
                if set(server_splits) != expected_metric_splits:
                    raise ValueError(
                        f"expected splits {sorted(expected_metric_splits)}, got "
                        f"{sorted(server_splits)}"
                    )
                recomputed_server = primary_metric(spec.task, server_splits)
                if not math.isclose(
                    actual_server_primary, recomputed_server, rel_tol=0.0, abs_tol=1e-12
                ):
                    reasons.append(
                        "FLASC dense-server diagnostic does not match server split metrics"
                    )
            except (KeyError, TypeError, ValueError) as exc:
                reasons.append(f"FLASC server validation metrics are malformed: {exc}")
    elif server_primary is not None or server_splits not in ({}, None):
        reasons.append("non-FLASC run unexpectedly contains a dense-server diagnostic")
    return reasons


def _validate_full_lora_factor_audit(
    summary: dict[str, Any], spec: ExpectedRun
) -> list[str]:
    """Require per-factor sparse-support diagnostics for full-LoRA EF runs."""
    if spec.method != "full_lora_ef":
        return []
    reasons: list[str] = []
    rounds = summary.get("rounds")
    model = summary.get("model")
    budget = summary.get("wire_budget")
    data = summary.get("data")
    if not isinstance(rounds, list):
        return ["full_lora_ef rounds are unavailable for adapter-factor audit"]
    if not isinstance(model, dict) or not isinstance(
        model.get("adapter_parameters"), int
    ) or int(model["adapter_parameters"]) <= 0:
        return ["full_lora_ef model.adapter_parameters is invalid"]
    if not isinstance(budget, dict) or not isinstance(budget.get("sparse_k"), int) or int(
        budget["sparse_k"]
    ) <= 0:
        return ["full_lora_ef wire_budget.sparse_k is invalid"]
    if not isinstance(data, dict) or not isinstance(data.get("num_clients"), int) or int(
        data["num_clients"]
    ) <= 0:
        return ["full_lora_ef data.num_clients is invalid"]
    adapter_parameters = int(model["adapter_parameters"])
    if adapter_parameters % 2 != 0:
        reasons.append("full_lora_ef adapter parameter count is not divisible by two")
    expected_factor_numel = adapter_parameters // 2
    sparse_k = int(budget["sparse_k"])
    num_clients = int(data["num_clients"])
    required_factors = {"lora_A", "lora_B"}
    norm_keys = (
        "aggregate_update_norm",
        "server_residual_norm",
        "client_residual_norm_mean",
    )
    for index, round_record in enumerate(rounds, start=1):
        if not isinstance(round_record, dict):
            reasons.append(f"round {index} cannot be adapter-factor audited")
            continue
        diagnostics = round_record.get("diagnostics")
        audit = (
            diagnostics.get("adapter_factor_audit")
            if isinstance(diagnostics, dict)
            else None
        )
        if not isinstance(audit, dict):
            reasons.append(f"round {index} adapter_factor_audit is absent")
            continue
        if set(audit) != required_factors:
            reasons.append(
                f"round {index} adapter_factor_audit factors are not exactly lora_A/lora_B"
            )
            continue
        logical_total = 0
        server_selected_total = 0
        client_selected_total = 0
        for factor in sorted(required_factors):
            values = audit.get(factor)
            if not isinstance(values, dict):
                reasons.append(f"round {index} {factor} audit is not an object")
                continue
            logical_numel = values.get("logical_numel")
            if not isinstance(logical_numel, int) or logical_numel <= 0:
                reasons.append(f"round {index} {factor}.logical_numel is not positive")
            else:
                logical_total += logical_numel
                if logical_numel != expected_factor_numel:
                    reasons.append(
                        f"round {index} {factor}.logical_numel differs from the "
                        "expected half of the square full-LoRA adapter"
                    )
            server_selected = values.get("server_transmitted_selected_count")
            if not isinstance(server_selected, int) or server_selected < 0:
                reasons.append(
                    f"round {index} {factor}.server_transmitted_selected_count is invalid"
                )
            else:
                server_selected_total += server_selected
                if isinstance(logical_numel, int) and server_selected > logical_numel:
                    reasons.append(
                        f"round {index} {factor}.server_transmitted_selected_count "
                        "exceeds that factor's logical support"
                    )
            client_selected = values.get("client_transmitted_selected_count_total")
            if not isinstance(client_selected, int) or client_selected < 0:
                reasons.append(
                    f"round {index} {factor}.client_transmitted_selected_count_total is invalid"
                )
            else:
                client_selected_total += client_selected
                if (
                    isinstance(logical_numel, int)
                    and client_selected > num_clients * logical_numel
                ):
                    reasons.append(
                        f"round {index} {factor}.client_transmitted_selected_count_total "
                        "exceeds num_clients times that factor's logical support"
                    )
                client_mean = values.get("client_transmitted_selected_count_mean")
                if not _finite(client_mean) or not math.isclose(
                    float(client_mean),
                    client_selected / num_clients,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    reasons.append(
                        f"round {index} {factor}.client_transmitted_selected_count_mean "
                        "does not equal total/num_clients"
                    )
            for key in norm_keys:
                value = values.get(key)
                if not _finite(value) or float(value) < 0.0:
                    reasons.append(f"round {index} {factor}.{key} is invalid")
        if logical_total != adapter_parameters:
            reasons.append(
                f"round {index} adapter factor logical_numel total differs from "
                "model.adapter_parameters"
            )
        if server_selected_total != sparse_k:
            reasons.append(
                f"round {index} server factor selected total differs from sparse_k"
            )
        if client_selected_total != num_clients * sparse_k:
            reasons.append(
                f"round {index} client factor selected total differs from "
                "num_clients*sparse_k"
            )
    return reasons


def _sum_nested_counts(value: Any) -> int | None:
    """Sum a diagnostic count tree, rejecting booleans and malformed leaves."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, dict):
        total = 0
        for child in value.values():
            subtotal = _sum_nested_counts(child)
            if subtotal is None:
                return None
            total += subtotal
        return total
    if isinstance(value, list):
        total = 0
        for child in value:
            subtotal = _sum_nested_counts(child)
            if subtotal is None:
                return None
            total += subtotal
        return total
    return None


def _validate_finite_diagnostic_tree(
    value: Any, path: str, reasons: list[str]
) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        reasons.append(f"{path} is non-finite")
    elif isinstance(value, dict):
        for key, child in value.items():
            _validate_finite_diagnostic_tree(child, f"{path}.{key}", reasons)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_finite_diagnostic_tree(child, f"{path}[{index}]", reasons)


def _validate_sparse_diagnostic_support(
    support: Any,
    expected_selected: int,
    path: str,
    reasons: list[str],
    *,
    expected_tensors: int | None = None,
    expected_rank: int | None = None,
    expected_tensor_names: tuple[str, ...] | None = None,
) -> None:
    if not isinstance(support, dict):
        reasons.append(f"{path} is not an object")
        return
    if support.get("selected_count") != expected_selected:
        reasons.append(
            f"{path}.selected_count differs from wire_budget.sparse_k"
        )
    for key in (
        "selected_by_tensor",
        "selected_by_rank",
        "selected_by_tensor_rank",
    ):
        counts = _sum_nested_counts(support.get(key))
        if counts != expected_selected:
            reasons.append(
                f"{path}.{key} counts do not sum to wire_budget.sparse_k"
            )
    by_tensor = support.get("selected_by_tensor")
    if expected_tensors is not None and (
        not isinstance(by_tensor, dict) or len(by_tensor) != expected_tensors
    ):
        reasons.append(f"{path}.selected_by_tensor has wrong tensor cardinality")
    if expected_tensor_names is not None and (
        not isinstance(by_tensor, dict)
        or set(by_tensor) != set(expected_tensor_names)
    ):
        reasons.append(f"{path}.selected_by_tensor keys differ from canonical LoRA-B layout")
    by_rank = support.get("selected_by_rank")
    if expected_rank is not None and (
        not isinstance(by_rank, list) or len(by_rank) != expected_rank
    ):
        reasons.append(f"{path}.selected_by_rank has wrong rank cardinality")
    by_tensor_rank = support.get("selected_by_tensor_rank")
    if expected_tensors is not None and (
        not isinstance(by_tensor_rank, dict)
        or len(by_tensor_rank) != expected_tensors
    ):
        reasons.append(
            f"{path}.selected_by_tensor_rank has wrong tensor cardinality"
        )
    elif isinstance(by_tensor_rank, dict) and expected_rank is not None and any(
        not isinstance(values, list) or len(values) != expected_rank
        for values in by_tensor_rank.values()
    ):
        reasons.append(f"{path}.selected_by_tensor_rank has wrong rank cardinality")
    if expected_tensor_names is not None and (
        not isinstance(by_tensor_rank, dict)
        or set(by_tensor_rank) != set(expected_tensor_names)
    ):
        reasons.append(
            f"{path}.selected_by_tensor_rank keys differ from canonical LoRA-B layout"
        )
    if (
        expected_tensor_names is not None
        and expected_rank is not None
        and isinstance(by_tensor, dict)
        and isinstance(by_rank, list)
        and len(by_rank) == expected_rank
        and isinstance(by_tensor_rank, dict)
        and set(by_tensor) == set(expected_tensor_names)
        and set(by_tensor_rank) == set(expected_tensor_names)
        and all(
            isinstance(by_tensor_rank[name], list)
            and len(by_tensor_rank[name]) == expected_rank
            for name in expected_tensor_names
        )
    ):
        for name in expected_tensor_names:
            if _sum_nested_counts(by_tensor_rank[name]) != by_tensor.get(name):
                reasons.append(
                    f"{path}.selected_by_tensor_rank row for {name} "
                    "does not equal selected_by_tensor"
                )
        for rank_index in range(expected_rank):
            column = _sum_nested_counts(
                [by_tensor_rank[name][rank_index] for name in expected_tensor_names]
            )
            if column != by_rank[rank_index]:
                reasons.append(
                    f"{path}.selected_by_tensor_rank column {rank_index} "
                    "does not equal selected_by_rank"
                )
    rounded = support.get("selected_values_rounded_to_zero")
    if (
        isinstance(rounded, bool)
        or not isinstance(rounded, int)
        or not 0 <= rounded <= expected_selected
    ):
        reasons.append(f"{path}.selected_values_rounded_to_zero is invalid")
    zero_support = support.get("zero_support_tensor_count")
    if (
        isinstance(zero_support, bool)
        or not isinstance(zero_support, int)
        or zero_support < 0
        or (expected_tensors is not None and zero_support > expected_tensors)
    ):
        reasons.append(f"{path}.zero_support_tensor_count is invalid")
    elif isinstance(by_tensor, dict) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in by_tensor.values()
    ):
        expected_zero_support = sum(value == 0 for value in by_tensor.values())
        if zero_support != expected_zero_support:
            reasons.append(
                f"{path}.zero_support_tensor_count disagrees with selected_by_tensor"
            )
    for key in ("cumulative_coverage", "selection_age", "adjacent_support_jaccard"):
        if not isinstance(support.get(key), dict):
            reasons.append(f"{path}.{key} is not an object")
    residual = support.get("residual")
    if residual is not None:
        reasons.append(f"{path}.residual must be a sibling of support")


def _validate_stability_protocol(
    summary: dict[str, Any], cfg: dict[str, Any], spec: ExpectedRun
) -> list[str]:
    """Validate protocol binding and the versioned, zero-wire diagnostic ledger."""
    stability = cfg.get("stability")
    if not isinstance(stability, dict):
        return []
    reasons: list[str] = []
    if summary.get("stability_protocol") != stability:
        reasons.append(
            "stability_protocol does not exactly match the frozen config section"
        )

    before = summary.get("invariants", {}).get("before", {})
    after = summary.get("invariants", {}).get("after", {})
    expected_recipe = {
        "spectral_a_dense": "stabilized_spectral_svd_v1",
        "spectral_a_ef": "stabilized_spectral_svd_v1",
        "spectral_block_ef": "stabilized_spectral_svd_v1",
        "spectral_rot_ef": "spectral_rot_hadamard8_cpu_float64_v1",
        "hybrid4_a_ef": "hybrid4_spectral_gaussian_qr_v1",
        "hybrid4_a_dense": "hybrid4_spectral_gaussian_qr_v1",
        "hybrid4_a_decay_dense": "hybrid4_spectral_gaussian_qr_v1",
        "hybrid4_a_decay_ef": "hybrid4_spectral_gaussian_qr_v1",
    }.get(spec.method)
    if expected_recipe is not None:
        for phase, values in (("before", before), ("after", after)):
            if not isinstance(values, dict):
                continue
            if values.get("fixed_basis_recipe") != expected_recipe:
                reasons.append(
                    f"invariants.{phase}.fixed_basis_recipe differs from "
                    f"{expected_recipe}"
                )
    if spec.method == "spectral_rot_ef":
        for phase, values in (("before", before), ("after", after)):
            error = values.get("spectral_rotation_projector_error_max")
            if not _finite(error) or float(error) > 1e-8:
                reasons.append(
                    f"invariants.{phase}.spectral_rotation_projector_error_max "
                    "is absent, non-finite, or exceeds 1e-8"
                )
    if spec.method in {
        "hybrid4_a_dense",
        "hybrid4_a_decay_dense",
        "hybrid4_a_ef",
        "hybrid4_a_decay_ef",
    }:
        seed = int(cfg["model"]["basis_seed"])
        targets = len(summary.get("model", {}).get("target_modules", []))
        expected_min = seed + 104729 + 3001
        expected_max = seed + 104729 * targets + 3001
        for phase, values in (("before", before), ("after", after)):
            for key in (
                "hybrid_leading_row_equality_max_error",
                "hybrid_mutual_orthogonality_max_error",
                "hybrid_row_orthogonality_max_error",
            ):
                error = values.get(key)
                if not _finite(error) or float(error) > 1e-4:
                    reasons.append(
                        f"invariants.{phase}.{key} is absent, non-finite, or "
                        "exceeds 1e-4"
                    )
            if values.get("hybrid_basis_seed_min") != expected_min:
                reasons.append(
                    f"invariants.{phase}.hybrid_basis_seed_min is inconsistent"
                )
            if values.get("hybrid_basis_seed_max") != expected_max:
                reasons.append(
                    f"invariants.{phase}.hybrid_basis_seed_max is inconsistent"
                )

    if spec.method not in STABILITY_DIAGNOSTIC_METHODS:
        return reasons
    rounds = summary.get("rounds")
    budget = summary.get("wire_budget", {})
    data = summary.get("data", {})
    if not isinstance(rounds, list):
        return [*reasons, "stability diagnostics cannot be checked without rounds"]
    sparse_k = budget.get("sparse_k") if isinstance(budget, dict) else None
    clients = data.get("num_clients") if isinstance(data, dict) else None
    model = summary.get("model", {})
    expected_tensors = (
        len(model.get("target_modules", [])) if isinstance(model, dict) else 0
    )
    expected_rank = model.get("rank") if isinstance(model, dict) else None
    expected_tensor_names: tuple[str, ...] = ()
    expected_block_numels: tuple[int, ...] = ()
    expected_layout_fingerprint: str | None = None
    target_modules = model.get("target_modules") if isinstance(model, dict) else None
    adapter_parameters = (
        model.get("adapter_parameters") if isinstance(model, dict) else None
    )
    if (
        isinstance(target_modules, list)
        and target_modules
        and all(isinstance(value, str) and value for value in target_modules)
        and isinstance(expected_rank, int)
        and expected_rank > 0
        and isinstance(adapter_parameters, int)
        and adapter_parameters > 0
        and adapter_parameters % (len(target_modules) * expected_rank) == 0
    ):
        hidden_size = adapter_parameters // (len(target_modules) * expected_rank)
        expected_tensor_names = tuple(
            sorted(f"{target}.lora_B" for target in target_modules)
        )
        expected_block_numels = tuple(
            hidden_size * expected_rank for _ in expected_tensor_names
        )
        expected_layout_fingerprint = DiagnosticLayout.build(
            expected_tensor_names,
            [(hidden_size, expected_rank) for _ in expected_tensor_names],
            expected_block_numels,
            expected_rank,
        ).fingerprint
    else:
        reasons.append(
            "cannot independently reconstruct the canonical diagnostic LoRA-B layout"
        )
    if not isinstance(sparse_k, int) or sparse_k <= 0:
        return [*reasons, "stability diagnostics require a positive sparse_k"]
    if not isinstance(clients, int) or clients <= 0:
        return [*reasons, "stability diagnostics require a positive num_clients"]
    expected_compressor = (
        "block_topk" if spec.method in BLOCK_BALANCED_METHODS else "global_topk"
    )
    if budget.get("compressor_kind") != expected_compressor:
        reasons.append(
            f"wire_budget.compressor_kind differs from {expected_compressor}"
        )
    if budget.get("diagnostic_recipe_version") != DIAGNOSTIC_RECIPE_VERSION:
        reasons.append(
            "wire_budget.diagnostic_recipe_version differs from the frozen recipe"
        )
    expected_sparse_adapter_bytes = 21 + 6 * sparse_k
    if budget.get("expected_sparse_adapter_bytes") != expected_sparse_adapter_bytes:
        reasons.append("wire_budget.expected_sparse_adapter_bytes is inconsistent")
    head_parameters = (
        model.get("head_parameters") if isinstance(model, dict) else None
    )
    if isinstance(head_parameters, int) and head_parameters > 0:
        expected_sparse_message_bytes = (
            expected_sparse_adapter_bytes + 13 + 2 * head_parameters
        )
        if budget.get("expected_sparse_message_bytes") != expected_sparse_message_bytes:
            reasons.append("wire_budget.expected_sparse_message_bytes is inconsistent")
    expected_quotas: tuple[int, ...] = ()
    if spec.method in BLOCK_BALANCED_METHODS:
        block_numels = budget.get("block_numels")
        quotas = budget.get("block_quotas")
        if (
            not isinstance(block_numels, list)
            or len(block_numels) != expected_tensors
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in block_numels
            )
            or sum(block_numels) != summary.get("model", {}).get(
                "adapter_parameters"
            )
        ):
            reasons.append("wire_budget.block_numels is inconsistent with the layout")
        if expected_block_numels and block_numels != list(expected_block_numels):
            reasons.append(
                "wire_budget.block_numels differs from the canonical LoRA-B layout"
            )
        if (
            not isinstance(quotas, list)
            or len(quotas) != expected_tensors
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in quotas
            )
            or sum(quotas) != sparse_k
        ):
            reasons.append("wire_budget.block_quotas is inconsistent with sparse_k")
        if expected_block_numels:
            try:
                expected_quotas = largest_remainder_quotas(
                    expected_block_numels, sparse_k
                )
            except (TypeError, ValueError) as exc:
                reasons.append(f"cannot independently recompute block quotas: {exc}")
            if expected_quotas and quotas != list(expected_quotas):
                reasons.append(
                    "wire_budget.block_quotas differs from the frozen "
                    "largest-remainder allocation"
                )
        if isinstance(block_numels, list) and isinstance(quotas, list):
            payload = json.dumps(
                {"block_numels": tuple(block_numels), "quotas": tuple(quotas)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            expected_quota_hash = hashlib.sha256(payload).hexdigest()
            if budget.get("block_quota_hash") != expected_quota_hash:
                reasons.append("wire_budget.block_quota_hash is inconsistent")
    elif any(
        budget.get(key) is not None
        for key in ("block_numels", "block_quotas", "block_quota_hash")
    ):
        reasons.append("global Top-k run unexpectedly declares block quotas")
    layout_fingerprint: str | None = None
    for index, round_record in enumerate(rounds, start=1):
        diagnostics = (
            round_record.get("diagnostics")
            if isinstance(round_record, dict)
            else None
        )
        stability_diagnostics = (
            diagnostics.get("stability_diagnostics")
            if isinstance(diagnostics, dict)
            else None
        )
        if not isinstance(stability_diagnostics, dict):
            reasons.append(f"round {index} stability_diagnostics is absent")
            continue
        path = f"round {index} stability_diagnostics"
        if stability_diagnostics.get("recipe_version") != DIAGNOSTIC_RECIPE_VERSION:
            reasons.append(f"{path}.recipe_version is not {DIAGNOSTIC_RECIPE_VERSION}")
        if stability_diagnostics.get("round") != index:
            reasons.append(f"{path}.round does not match the round record")
        fingerprint = stability_diagnostics.get("layout_fingerprint")
        if not _is_hash(fingerprint):
            reasons.append(f"{path}.layout_fingerprint is malformed")
        elif layout_fingerprint is None:
            layout_fingerprint = str(fingerprint)
        elif fingerprint != layout_fingerprint:
            reasons.append(f"{path}.layout_fingerprint changed across rounds")
        if (
            expected_layout_fingerprint is not None
            and fingerprint != expected_layout_fingerprint
        ):
            reasons.append(
                f"{path}.layout_fingerprint differs from the independently "
                "recomputed canonical layout"
            )
        client_values = stability_diagnostics.get("clients")
        server_values = stability_diagnostics.get("server")
        updates = stability_diagnostics.get("updates")
        if not isinstance(client_values, dict):
            reasons.append(f"{path}.clients is not an object")
        else:
            required_client_fields = {
                "per_client",
                "selected_count_total",
                "selected_by_tensor_total",
                "selected_by_rank_total",
                "selected_by_tensor_rank_total",
                "selected_values_rounded_to_zero_total",
                "zero_support_tensor_count",
                "cumulative_coverage_overall",
                "adjacent_support_jaccard_overall",
                "residual_to_update_ratio",
                "drained_residual_energy",
                "ef_conservation_max_abs",
            }
            missing = sorted(required_client_fields - set(client_values))
            if missing:
                reasons.append(f"{path}.clients is missing fields: {missing}")
            per_client = client_values.get("per_client")
            if not isinstance(per_client, list) or len(per_client) != clients:
                reasons.append(f"{path}.clients.per_client has wrong length")
            else:
                for client_index, client_value in enumerate(per_client):
                    support = (
                        client_value.get("support")
                        if isinstance(client_value, dict)
                        else None
                    )
                    _validate_sparse_diagnostic_support(
                        support,
                        sparse_k,
                        f"{path}.clients.per_client[{client_index}].support",
                        reasons,
                        expected_tensors=expected_tensors,
                        expected_rank=expected_rank
                        if isinstance(expected_rank, int)
                        else None,
                        expected_tensor_names=expected_tensor_names or None,
                    )
                    if (
                        spec.method in BLOCK_BALANCED_METHODS
                        and expected_quotas
                        and isinstance(support, dict)
                        and support.get("selected_by_tensor")
                        != dict(zip(expected_tensor_names, expected_quotas))
                    ):
                        reasons.append(
                            f"{path}.clients.per_client[{client_index}].support."
                            "selected_by_tensor differs from frozen block quotas"
                        )
                    if isinstance(client_value, dict) and not isinstance(
                        client_value.get("residual"), dict
                    ):
                        reasons.append(
                            f"{path}.clients.per_client[{client_index}].residual "
                            "is not an object"
                        )
                    if isinstance(client_value, dict) and client_value.get(
                        "client_id"
                    ) != client_index:
                        reasons.append(
                            f"{path}.clients.per_client[{client_index}].client_id "
                            "is inconsistent"
                        )
            if client_values.get("selected_count_total") != clients * sparse_k:
                reasons.append(
                    f"{path}.clients.selected_count_total differs from "
                    "num_clients*sparse_k"
                )
            selected_by_tensor = _sum_nested_counts(
                client_values.get("selected_by_tensor_total")
            )
            if selected_by_tensor != clients * sparse_k:
                reasons.append(
                    f"{path}.clients.selected_by_tensor_total does not sum to "
                    "num_clients*sparse_k"
                )
            if spec.method in BLOCK_BALANCED_METHODS and expected_quotas:
                expected_client_tensor_counts = {
                    name: clients * quota
                    for name, quota in zip(expected_tensor_names, expected_quotas)
                }
                if (
                    client_values.get("selected_by_tensor_total")
                    != expected_client_tensor_counts
                ):
                    reasons.append(
                        f"{path}.clients.selected_by_tensor_total differs from "
                        "num_clients times frozen block quotas"
                    )
            for key in ("selected_by_rank_total", "selected_by_tensor_rank_total"):
                if _sum_nested_counts(client_values.get(key)) != clients * sparse_k:
                    reasons.append(
                        f"{path}.clients.{key} does not sum to "
                        "num_clients*sparse_k"
                    )
            rounded_total = client_values.get(
                "selected_values_rounded_to_zero_total"
            )
            if (
                isinstance(rounded_total, bool)
                or not isinstance(rounded_total, int)
                or not 0 <= rounded_total <= clients * sparse_k
            ):
                reasons.append(
                    f"{path}.clients.selected_values_rounded_to_zero_total is invalid"
                )
            ef_error = client_values.get("ef_conservation_max_abs")
            if not _finite(ef_error) or float(ef_error) > 1e-5:
                reasons.append(
                    f"{path}.clients.ef_conservation_max_abs is absent, "
                    "non-finite, or exceeds 1e-5"
                )
            if spec.method in BLOCK_BALANCED_METHODS:
                zero_support = client_values.get("zero_support_tensor_count")
                maximum = (
                    zero_support.get("max")
                    if isinstance(zero_support, dict)
                    else None
                )
                if maximum != 0:
                    reasons.append(
                        f"{path}.clients block-balanced max zero-support count is not zero"
                    )
        if not isinstance(server_values, dict):
            reasons.append(f"{path}.server is not an object")
        else:
            _validate_sparse_diagnostic_support(
                server_values.get("support"),
                sparse_k,
                f"{path}.server.support",
                reasons,
                expected_tensors=expected_tensors,
                expected_rank=expected_rank
                if isinstance(expected_rank, int)
                else None,
                expected_tensor_names=expected_tensor_names or None,
            )
            if (
                spec.method in BLOCK_BALANCED_METHODS
                and expected_quotas
                and isinstance(server_values.get("support"), dict)
                and server_values["support"].get("selected_by_tensor")
                != dict(zip(expected_tensor_names, expected_quotas))
            ):
                reasons.append(
                    f"{path}.server.support.selected_by_tensor differs from "
                    "frozen block quotas"
                )
            residual = server_values.get("residual")
            if not isinstance(residual, dict):
                reasons.append(f"{path}.server.residual is not an object")
            else:
                overall = residual.get("overall")
                ef_error = (
                    overall.get("ef_conservation_max_abs")
                    if isinstance(overall, dict)
                    else None
                )
                if not _finite(ef_error) or float(ef_error) > 1e-5:
                    reasons.append(
                        f"{path}.server.residual.overall.ef_conservation_max_abs "
                        "is absent, non-finite, or exceeds 1e-5"
                    )
            if spec.method in BLOCK_BALANCED_METHODS:
                support = server_values.get("support")
                zero_support = (
                    support.get("zero_support_tensor_count")
                    if isinstance(support, dict)
                    else None
                )
                if zero_support != 0:
                    reasons.append(
                        f"{path}.server block-balanced zero-support is nonzero"
                    )
        if not isinstance(updates, dict):
            reasons.append(f"{path}.updates is not an object")
        else:
            required_update_keys = {
                "aggregate_norm",
                "transmitted_norm",
                "aggregate_norm_ema_beta_0_9",
                "transmitted_norm_ema_beta_0_9",
                "consecutive_aggregate_cosine",
                "consecutive_transmitted_cosine",
                "aggregate_to_transmitted_cosine",
                "transmitted_sign_flip_fraction",
                "transmitted_sign_flip_denominator",
                "client_cancellation_ratio",
                "client_to_aggregate_cosine",
            }
            missing = sorted(required_update_keys - set(updates))
            if missing:
                reasons.append(f"{path}.updates is missing fields: {missing}")
            for key in (
                "aggregate_norm",
                "transmitted_norm",
                "aggregate_norm_ema_beta_0_9",
                "transmitted_norm_ema_beta_0_9",
            ):
                if not _finite(updates.get(key)) or float(updates[key]) < 0.0:
                    reasons.append(f"{path}.updates.{key} is invalid")
            for key in (
                "aggregate_to_transmitted_cosine",
                "transmitted_sign_flip_fraction",
                "client_cancellation_ratio",
            ):
                if not _finite(updates.get(key)):
                    reasons.append(f"{path}.updates.{key} is non-finite")
            denominator = updates.get("transmitted_sign_flip_denominator")
            if (
                isinstance(denominator, bool)
                or not isinstance(denominator, int)
                or denominator < 0
            ):
                reasons.append(
                    f"{path}.updates.transmitted_sign_flip_denominator is invalid"
                )
        _validate_finite_diagnostic_tree(
            stability_diagnostics, path, reasons
        )
    if expected_layout_fingerprint is not None and budget.get(
        "diagnostic_layout_fingerprint"
    ) != expected_layout_fingerprint:
        reasons.append(
            "wire_budget.diagnostic_layout_fingerprint differs from the "
            "independently recomputed canonical layout"
        )
    return reasons


def _validate_invariants(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    invariants = summary.get("invariants")
    if not isinstance(invariants, dict):
        return ["invariants is not an object"]
    before = invariants.get("before", {})
    after = invariants.get("after", {})
    for label, values in (("before", before), ("after", after)):
        if not isinstance(values, dict):
            reasons.append(f"invariants.{label} is not an object")
            continue
        if values.get("all_base_parameters_frozen") is not True:
            reasons.append(f"invariants.{label}.all_base_parameters_frozen is not true")
        if values.get("unexpected_trainable_count") != 0:
            reasons.append(f"invariants.{label}.unexpected_trainable_count is not zero")
        if (
            not _finite(values.get("fixed_a_orthogonality_max_error"))
            or float(values.get("fixed_a_orthogonality_max_error", math.inf)) > 1e-4
        ):
            reasons.append(
                f"invariants.{label}.fixed_a_orthogonality_max_error exceeds 1e-4"
            )
        if not _is_hash(values.get("fixed_basis_hash")):
            reasons.append(f"invariants.{label}.fixed_basis_hash is malformed")
    if (
        isinstance(before, dict)
        and isinstance(after, dict)
        and before.get("fixed_basis_hash") != after.get("fixed_basis_hash")
    ):
        reasons.append("fixed basis hash changed")
    if invariants.get("basis_unchanged") is not True:
        reasons.append("basis_unchanged is not true")
    if invariants.get("finite_metrics") is not True:
        reasons.append("finite_metrics is not true")
    if invariants.get("wire_cap_respected") is not True:
        reasons.append("wire_cap_respected is not true")
    ef_error = invariants.get("ef_conservation_max")
    ef_numeric = float(ef_error) if _finite(ef_error) else math.inf  # type: ignore[arg-type]
    if ef_numeric > 1e-6:
        reasons.append("ef_conservation_max is absent, non-finite, or exceeds 1e-6")
    return reasons


def _validate_training_data_model(
    summary: dict[str, Any],
    cfg: dict[str, Any],
    spec: ExpectedRun,
    context: ValidationContext,
) -> list[str]:
    reasons: list[str] = []
    data = summary.get("data", {})
    if not isinstance(data, dict):
        reasons.append("data is not an object")
        data = {}
    _append_mismatch(reasons, "data.task", data.get("task"), spec.task)
    _append_mismatch(
        reasons, "data.partition_seed", data.get("partition_seed"), spec.seed
    )
    _append_mismatch(
        reasons,
        "data.num_clients",
        data.get("num_clients"),
        int(cfg["data"]["num_clients"]),
    )
    _append_mismatch(
        reasons,
        "data.dirichlet_alpha",
        data.get("dirichlet_alpha"),
        float(cfg["data"]["dirichlet_alpha"]),
    )
    _append_mismatch(
        reasons,
        "data.evaluation_policy",
        data.get("evaluation_policy"),
        cfg.get("evaluation", {}).get("policy", "official_validation"),
    )
    for key in ("client_train_sizes", "client_holdout_sizes"):
        values = data.get(key)
        if not isinstance(values, list) or len(values) != int(
            cfg["data"]["num_clients"]
        ):
            reasons.append(f"data.{key} does not contain one entry per client")
        elif any(not isinstance(value, int) or value < 0 for value in values):
            reasons.append(f"data.{key} contains an invalid size")

    training = summary.get("training", {})
    if not isinstance(training, dict):
        reasons.append("training is not an object")
        training = {}
    adapter_learning_rate, head_learning_rate = _learning_rates(
        cfg, spec.task, spec.method
    )
    expected_training = {
        "rounds": int(task_value(cfg, "training", "rounds", spec.task)),
        "local_steps": int(task_value(cfg, "training", "local_steps", spec.task)),
        "batch_size": int(task_value(cfg, "training", "batch_size", spec.task)),
        "eval_batch_size": int(cfg["training"]["eval_batch_size"]),
        "eval_every": int(task_value(cfg, "training", "eval_every", spec.task)),
        "adapter_learning_rate": adapter_learning_rate,
        "head_learning_rate": head_learning_rate,
        "weight_decay": float(cfg["training"]["weight_decay"]),
        "max_grad_norm": float(cfg["training"]["max_grad_norm"]),
        "mixed_precision": cfg["training"]["mixed_precision"],
    }
    if cfg.get("stability", {}).get("generation") in {3, 4}:
        expected_training["learning_rate_schedule"] = _expected_lr_schedule(
            cfg, spec.method, expected_training["rounds"]
        )
    for key, expected in expected_training.items():
        _append_mismatch(reasons, f"training.{key}", training.get(key), expected)

    if context.model is None:
        reasons.append("current model artifact could not be validated")
        return reasons
    model = summary.get("model", {})
    if not isinstance(model, dict):
        reasons.append("model is not an object")
        model = {}
    (
        expected_layout,
        adapter_parameters,
        head_parameters,
        reference_parameters,
        targets,
    ) = _expected_layout(cfg, context.model, spec)
    try:
        actual_model_path = Path(str(model.get("path"))).resolve()
    except (TypeError, ValueError):
        actual_model_path = Path("/")
    _append_mismatch(reasons, "model.path", actual_model_path, context.model.model_path)
    _append_mismatch(
        reasons, "model.rank", model.get("rank"), int(cfg["model"]["rank"])
    )
    _append_mismatch(
        reasons, "model.target_modules", model.get("target_modules"), targets
    )
    _append_mismatch(
        reasons,
        "model.adapter_parameters",
        model.get("adapter_parameters"),
        adapter_parameters,
    )
    _append_mismatch(
        reasons, "model.head_parameters", model.get("head_parameters"), head_parameters
    )
    _append_mismatch(
        reasons,
        "model.reference_full_lora_parameters",
        model.get("reference_full_lora_parameters"),
        reference_parameters,
    )
    _append_mismatch(
        reasons,
        "model.trainable_parameters",
        model.get("trainable_parameters"),
        adapter_parameters + head_parameters,
    )
    _append_mismatch(
        reasons, "layout_hash", summary.get("layout_hash"), expected_layout
    )
    return reasons


def _validate_wire(
    summary: dict[str, Any],
    spec: ExpectedRun,
    cfg: dict[str, Any] | None = None,
    model_expectations: ModelExpectations | None = None,
) -> list[str]:
    reasons: list[str] = []
    budget = summary.get("wire_budget", {})
    model = summary.get("model", {})
    if not isinstance(budget, dict):
        return ["wire_budget is not an object"]
    if not isinstance(model, dict):
        model = {}
    if not _same_ratio(budget.get("requested_ratio"), spec.wire_ratio):
        reasons.append("wire_budget.requested_ratio differs from the expected run")
    reference = budget.get("reference_message_bytes")
    cap = budget.get("message_cap_bytes")
    if not isinstance(reference, int) or reference <= 0:
        reasons.append("wire_budget.reference_message_bytes is invalid")
    elif spec.wire_ratio is None:
        if cap != reference:
            reasons.append("dense message cap differs from its reference message size")
    elif cap != math.floor(reference * spec.wire_ratio):
        reasons.append("sparse message cap is not floor(reference * requested_ratio)")
    if spec.wire_ratio is not None:
        sparse_k = budget.get("sparse_k")
        density = budget.get("adapter_density")
        if not isinstance(sparse_k, int) or sparse_k <= 0:
            reasons.append("wire_budget.sparse_k is invalid")
        density_numeric = float(density) if _finite(density) else math.nan  # type: ignore[arg-type]
        if not 0 < density_numeric <= 1:
            reasons.append("wire_budget.adapter_density is invalid")

    canonical_manifest_size: int | None = None
    if cfg is not None and model_expectations is not None:
        try:
            (
                adapter_parameters,
                head_parameters,
                reference_parameters,
                canonical_manifest_size,
            ) = _canonical_wire_dimensions(
                spec.method,
                spec.task,
                model_expectations.target_names,
                int(cfg["model"]["rank"]),
                model_expectations.hidden_size,
            )
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(
                "cannot independently reconstruct canonical wire layout: "
                f"{type(exc).__name__}: {exc}"
            )
            adapter_parameters = None
            head_parameters = None
            reference_parameters = None
    else:
        # Compatibility path for focused callers.  Production report audits
        # always pass the frozen config and independently loaded model shape.
        adapter_parameters = model.get("adapter_parameters")
        head_parameters = model.get("head_parameters")
        reference_parameters = model.get("reference_full_lora_parameters")
        # A complete summary still contains enough canonical metadata to run
        # the strict ledger audit when this helper is invoked directly.
        try:
            labels = TASK_NUM_LABELS[spec.task]
            declared_head_parameters = int(model["head_parameters"])
            hidden_numerator = declared_head_parameters - labels
            if hidden_numerator <= 0 or hidden_numerator % labels != 0:
                raise ValueError("classifier head cannot determine hidden size")
            target_names = model["target_modules"]
            if (
                not isinstance(target_names, list)
                or not target_names
                or any(not isinstance(name, str) or not name for name in target_names)
            ):
                raise ValueError("target_modules is not canonical")
            (
                adapter_parameters,
                head_parameters,
                reference_parameters,
                canonical_manifest_size,
            ) = _canonical_wire_dimensions(
                spec.method,
                spec.task,
                target_names,
                int(model["rank"]),
                hidden_numerator // labels,
            )
        except (KeyError, TypeError, ValueError):
            # Legacy unit fixtures intentionally omit layout metadata.  They
            # retain the packet-size checks above, while complete summaries
            # always take the strict path.
            pass
    expected_message_bytes: int | None = None
    if (
        isinstance(adapter_parameters, int)
        and adapter_parameters > 0
        and isinstance(head_parameters, int)
        and head_parameters > 0
        and isinstance(reference_parameters, int)
        and reference_parameters >= adapter_parameters + head_parameters
    ):
        expected_reference_bytes = 26 + 2 * reference_parameters
        if reference != expected_reference_bytes:
            reasons.append(
                "wire_budget.reference_message_bytes does not match the independently "
                "recomputed full-LoRA plus head packet size"
            )
        dense_head_bytes = 13 + 2 * head_parameters
        if spec.wire_ratio is None:
            expected_message_bytes = 26 + 2 * (adapter_parameters + head_parameters)
            expected_reference_adapter_bytes = (
                expected_reference_bytes - dense_head_bytes
            )
            if budget.get("adapter_cap_bytes") != expected_reference_adapter_bytes:
                reasons.append("dense wire_budget.adapter_cap_bytes is inconsistent")
        elif isinstance(cap, int) and cap > 0:
            expected_adapter_cap = cap - dense_head_bytes
            if budget.get("adapter_cap_bytes") != expected_adapter_cap:
                reasons.append("sparse wire_budget.adapter_cap_bytes is inconsistent")
            available = expected_adapter_cap - 21
            expected_k = (
                max(1, min(adapter_parameters, available // 6))
                if available >= 6
                else None
            )
            if expected_k is None:
                reasons.append("sparse adapter cap cannot carry one encoded entry")
            else:
                if budget.get("sparse_k") != expected_k:
                    reasons.append(
                        "wire_budget.sparse_k differs from the independently recomputed maximum"
                    )
                expected_density = expected_k / adapter_parameters
                if not _finite(budget.get("adapter_density")) or not math.isclose(
                    float(budget["adapter_density"]),
                    expected_density,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    reasons.append("wire_budget.adapter_density does not equal k/adapter_numel")
                expected_message_bytes = 21 + 6 * expected_k + dense_head_bytes
                if expected_message_bytes > cap:
                    reasons.append("independently recomputed sparse message exceeds cap")
    final = summary.get("final", {})
    final_wire = final.get("wire", {}) if isinstance(final, dict) else {}
    if not isinstance(final_wire, dict):
        return [*reasons, "final.wire is not an object"]
    ledger_fields = (
        "uplink_bytes",
        "downlink_emitted_bytes",
        "downlink_delivered_bytes",
        "uplink_messages",
        "downlink_messages",
        "schema_bytes",
        "total_delivered_bytes",
        "max_uplink_message_bytes",
        "max_downlink_message_bytes",
    )
    for key in ledger_fields:
        if type(final_wire.get(key)) is not int or final_wire[key] < 0:
            reasons.append(f"final.wire.{key} is invalid")
    if all(
        isinstance(final_wire.get(key), int)
        for key in (
            "uplink_bytes",
            "downlink_delivered_bytes",
            "schema_bytes",
            "total_delivered_bytes",
        )
    ):
        recomputed = (
            final_wire["uplink_bytes"]
            + final_wire["downlink_delivered_bytes"]
            + final_wire["schema_bytes"]
        )
        if final_wire["total_delivered_bytes"] != recomputed:
            reasons.append(
                "final.wire.total_delivered_bytes is internally inconsistent"
            )
    if expected_message_bytes is not None:
        for key in ("max_uplink_message_bytes", "max_downlink_message_bytes"):
            if final_wire.get(key) != expected_message_bytes:
                reasons.append(
                    f"final.wire.{key} differs from the independently recomputed packet size"
                )
        if spec.wire_ratio is not None and isinstance(cap, int):
            if expected_message_bytes > cap:
                reasons.append("independently recomputed recurrent message exceeds wire cap")

    # Full ledger audit.  Every cumulative field is derived from the canonical
    # packet format, configured client/round counts, and the canonical manifest;
    # no observed byte or message counter is used as an expectation.
    if expected_message_bytes is not None and canonical_manifest_size is not None:
        try:
            if cfg is not None and model_expectations is not None:
                clients = int(cfg["data"]["num_clients"])
                expected_rounds = int(
                    task_value(cfg, "training", "rounds", spec.task)
                )
                schema_once = (
                    cfg["communication"].get("schema_once_per_run") is True
                )
            else:
                clients = int(summary["data"]["num_clients"])
                expected_rounds = int(summary["training"]["rounds"])
                # All schema-v2 reports use the canonical one-time manifest.
                schema_once = True
        except (KeyError, TypeError, ValueError) as exc:
            reasons.append(
                "cannot independently reconstruct cumulative wire ledger: "
                f"{type(exc).__name__}: {exc}"
            )
            clients = 0
            expected_rounds = 0
            schema_once = False
        if clients <= 0:
            reasons.append("wire ledger client count is not positive")
        if expected_rounds <= 0:
            reasons.append("wire ledger round count is not positive")

        canonical_schema_bytes = (
            canonical_manifest_size * (1 + clients) if schema_once else 0
        )

        def expected_ledger(round_index: int) -> dict[str, int]:
            uplink = expected_message_bytes * clients * round_index
            emitted = expected_message_bytes * round_index
            delivered = expected_message_bytes * clients * round_index
            return {
                "uplink_bytes": uplink,
                "downlink_emitted_bytes": emitted,
                "downlink_delivered_bytes": delivered,
                "uplink_messages": clients * round_index,
                "downlink_messages": round_index,
                "schema_bytes": canonical_schema_bytes,
                "total_delivered_bytes": (
                    uplink + delivered + canonical_schema_bytes
                ),
                "max_uplink_message_bytes": expected_message_bytes,
                "max_downlink_message_bytes": expected_message_bytes,
            }

        rounds = summary.get("rounds")
        if not isinstance(rounds, list):
            reasons.append("wire ledger rounds is not a list")
        else:
            previous: dict[str, int] | None = None
            monotonic_fields = tuple(
                key for key in ledger_fields if key != "schema_bytes"
            )
            for round_index, round_record in enumerate(rounds, start=1):
                if not isinstance(round_record, dict):
                    reasons.append(f"round {round_index}.wire cannot be audited")
                    previous = None
                    continue
                round_wire = round_record.get("wire")
                if not isinstance(round_wire, dict):
                    reasons.append(f"round {round_index}.wire is not an object")
                    previous = None
                    continue
                expected = expected_ledger(round_index)
                for key in ledger_fields:
                    actual = round_wire.get(key)
                    if type(actual) is not int or actual != expected[key]:
                        reasons.append(
                            f"round {round_index}.wire.{key} differs from the "
                            "independently reconstructed cumulative ledger"
                        )
                if previous is not None:
                    for key in monotonic_fields:
                        before = previous.get(key)
                        after = round_wire.get(key)
                        if (
                            type(before) is int
                            and type(after) is int
                            and after < before
                        ):
                            reasons.append(
                                f"round {round_index}.wire.{key} is not monotonic"
                            )
                previous = round_wire

        expected_final = expected_ledger(expected_rounds)
        for key in ledger_fields:
            actual = final_wire.get(key)
            if type(actual) is not int or actual != expected_final[key]:
                reasons.append(
                    f"final.wire.{key} differs from the independently reconstructed "
                    "final ledger"
                )
    return reasons


def _validate_setup(
    summary: dict[str, Any],
    cfg: dict[str, Any],
    spec: ExpectedRun,
    context: ValidationContext,
) -> list[str]:
    reasons: list[str] = []
    setup = summary.get("setup_communication")
    if not isinstance(setup, dict):
        return ["setup_communication is not an object"]
    fixed_basis_numel = 0
    if context.model is not None:
        targets = len(context.model.target_names)
        rank = int(cfg["model"]["rank"])
        hidden = context.model.hidden_size
        if spec.method in FIXED_A_METHODS:
            fixed_basis_numel = targets * rank * hidden
        elif spec.method in LORA_XS_METHODS:
            fixed_basis_numel = targets * (hidden * rank + rank + rank * hidden)
    expected_handshake = 32 if fixed_basis_numel else 0
    expected_fallback = 13 + 2 * fixed_basis_numel if fixed_basis_numel else 0
    _append_mismatch(
        reasons,
        "setup_communication.assumed_basis_bytes",
        setup.get("assumed_basis_bytes"),
        0,
    )
    _append_mismatch(
        reasons,
        "setup_communication.basis_hash_handshake_bytes_per_client",
        setup.get("basis_hash_handshake_bytes_per_client"),
        expected_handshake,
    )
    _append_mismatch(
        reasons,
        "setup_communication.fallback_fp16_basis_bytes_per_client",
        setup.get("fallback_fp16_basis_bytes_per_client"),
        expected_fallback,
    )
    if not isinstance(setup.get("assumption"), str) or not setup["assumption"]:
        reasons.append("setup_communication.assumption is absent")
    return reasons


def _audit_summary(
    record: AuditRecord, cfg: dict[str, Any], context: ValidationContext
) -> None:
    assert record.spec is not None
    spec = record.spec
    try:
        summary = json.loads(record.path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError) as exc:
        record.audit_status = "INVALID"
        record.reasons.append(f"summary cannot be parsed: {type(exc).__name__}: {exc}")
        return
    if not isinstance(summary, dict):
        record.audit_status = "INVALID"
        record.reasons.append("summary root is not an object")
        return
    record.summary = summary
    reasons = record.reasons
    _append_mismatch(
        reasons, "schema_version", summary.get("schema_version"), SCHEMA_VERSION
    )
    _append_mismatch(reasons, "status", summary.get("status"), "COMPLETE")
    _append_mismatch(reasons, "run_id", summary.get("run_id"), spec.run_id)
    if record.path.parent.name != spec.run_id:
        reasons.append("summary directory name differs from the expected run_id")
    run = summary.get("run", {})
    if not isinstance(run, dict):
        reasons.append("run is not an object")
        run = {}
    _append_mismatch(reasons, "run.task", run.get("task"), spec.task)
    _append_mismatch(reasons, "run.method", run.get("method"), spec.method)
    _append_mismatch(reasons, "run.seed", run.get("seed"), spec.seed)
    if not _same_ratio(run.get("wire_ratio"), spec.wire_ratio):
        reasons.append("run.wire_ratio differs from the expected run")
    metric_metadata = summary.get("metrics")
    requires_metric_metadata = (
        cfg.get("stability", {}).get("generation") == 4
        or spec.task in {"mrpc", "qqp"}
    )
    if requires_metric_metadata and not isinstance(metric_metadata, dict):
        reasons.append("metrics metadata is not an object")
    elif isinstance(metric_metadata, dict):
        _append_mismatch(
            reasons,
            "metrics.schema_version",
            metric_metadata.get("schema_version"),
            1,
        )
        _append_mismatch(
            reasons,
            "metrics.primary_metric_name",
            metric_metadata.get("primary_metric_name"),
            primary_metric_name(spec.task),
        )
        _append_mismatch(
            reasons,
            "metrics.mrpc_qqp_positive_label",
            metric_metadata.get("mrpc_qqp_positive_label"),
            1 if spec.task in {"mrpc", "qqp"} else None,
        )

    expected_config_hash = _config_hash(cfg, spec)
    _append_mismatch(
        reasons, "config_hash", summary.get("config_hash"), expected_config_hash
    )
    if not context.source_hash:
        reasons.append("current source tree could not be fingerprinted")
    else:
        _append_mismatch(
            reasons, "source_hash", summary.get("source_hash"), context.source_hash
        )
    if context.model is None:
        reasons.append("current model hash is unavailable")
    else:
        _append_mismatch(
            reasons, "model_hash", summary.get("model_hash"), context.model.model_hash
        )
    expected_dataset_hash = context.dataset_hashes.get(spec.task)
    if expected_dataset_hash is None:
        reasons.append(f"current dataset hash is unavailable for {spec.task}")
    else:
        _append_mismatch(
            reasons, "dataset_hash", summary.get("dataset_hash"), expected_dataset_hash
        )
    expected_partition_hash = context.partition_hashes.get((spec.task, spec.seed))
    if expected_partition_hash is None:
        reasons.append(
            f"current partition hash is unavailable for {spec.task}/seed-{spec.seed}"
        )
    else:
        _append_mismatch(
            reasons,
            "partition_hash",
            summary.get("partition_hash"),
            expected_partition_hash,
        )

    snapshot_path = record.path.parent / "run_config.json"
    if not snapshot_path.exists():
        reasons.append("run_config.json is missing")
    else:
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            expected_snapshot = {
                "config": _clean_config(cfg),
                "run": spec.run_dict,
                "config_hash": expected_config_hash,
                "source_hash": context.source_hash,
            }
            if snapshot != expected_snapshot:
                reasons.append(
                    "run_config.json differs from the resolved expected configuration"
                )
        except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError) as exc:
            reasons.append(
                f"run_config.json cannot be parsed: {type(exc).__name__}: {exc}"
            )

    initialization = summary.get("initialization", {})
    if not isinstance(initialization, dict):
        reasons.append("initialization is not an object")
        initialization = {}
    for key in ("initial_adapter_hash", "initial_head_hash"):
        if not _is_hash(initialization.get(key)):
            reasons.append(f"initialization.{key} is malformed")
    for key in ("data_prepare_seconds", "spectral_seconds"):
        if (
            not _finite(initialization.get(key))
            or float(initialization.get(key, -1)) < 0
        ):
            reasons.append(f"initialization.{key} is absent, non-finite, or negative")
    resume_count = initialization.get("resume_count")
    if not isinstance(resume_count, int) or resume_count < 0:
        reasons.append("initialization.resume_count is absent or invalid")
    reasons.extend(_validate_training_data_model(summary, cfg, spec, context))
    reasons.extend(_validate_wire(summary, spec, cfg, context.model))
    reasons.extend(_validate_setup(summary, cfg, spec, context))
    reasons.extend(_validate_invariants(summary))
    reasons.extend(_validate_rounds(summary, cfg, spec))
    reasons.extend(_validate_full_lora_factor_audit(summary, spec))
    reasons.extend(_validate_stability_protocol(summary, cfg, spec))
    if (
        not _finite(summary.get("runtime_seconds"))
        or float(summary.get("runtime_seconds", -1)) < 0
    ):
        reasons.append("runtime_seconds is absent, non-finite, or negative")
    record.audit_status = "VALID" if not reasons else "INVALID"


def _apply_cross_run_gates(records: list[AuditRecord]) -> None:
    candidates = [
        record
        for record in records
        if record.valid and record.spec is not None and record.summary is not None
    ]

    task_seed_groups: dict[tuple[str, int], list[AuditRecord]] = defaultdict(list)
    task_method_groups: dict[tuple[str, str], list[AuditRecord]] = defaultdict(list)
    for record in candidates:
        assert record.spec is not None
        task_seed_groups[(record.spec.task, record.spec.seed)].append(record)
        task_method_groups[(record.spec.task, record.spec.method)].append(record)

    for (task, seed), group in task_seed_groups.items():
        partition_hashes = {
            str(record.summary["partition_hash"])
            for record in group
            if record.summary is not None
        }
        head_hashes = {
            str(record.summary["initialization"]["initial_head_hash"])
            for record in group
            if record.summary is not None
        }
        if len(partition_hashes) > 1:
            reason = (
                f"partition hash disagrees across methods in {task}/seed-{seed}: "
                f"{sorted(partition_hashes)}"
            )
            for record in group:
                record.reasons.append(reason)
        if len(head_hashes) > 1:
            reason = (
                f"initial head hash disagrees across methods in {task}/seed-{seed}: "
                f"{sorted(head_hashes)}"
            )
            for record in group:
                record.reasons.append(reason)
        fixed_a_group = [
            record
            for record in group
            if record.spec is not None and record.spec.method in FIXED_A_METHODS
        ]
        fixed_a_initial_hashes = {
            str(record.summary["initialization"]["initial_adapter_hash"])
            for record in fixed_a_group
            if record.summary is not None
        }
        if len(fixed_a_initial_hashes) > 1:
            reason = (
                f"fixed-A initial adapter hash disagrees across methods in "
                f"{task}/seed-{seed}: {sorted(fixed_a_initial_hashes)}"
            )
            for record in fixed_a_group:
                record.reasons.append(reason)
        spectral_equivalent_group = [
            record
            for record in group
            if record.spec is not None
            and record.spec.method
            in {"spectral_a_dense", "spectral_a_ef", "spectral_block_ef"}
        ]
        spectral_basis_hashes = {
            str(record.summary["invariants"]["before"]["fixed_basis_hash"])
            for record in spectral_equivalent_group
            if record.summary is not None
        }
        if len(spectral_basis_hashes) > 1:
            reason = (
                f"original/block-balanced spectral basis hash disagrees in "
                f"{task}/seed-{seed}: {sorted(spectral_basis_hashes)}"
            )
            for record in spectral_equivalent_group:
                record.reasons.append(reason)
        hybrid_group = [
            record
            for record in group
            if record.spec is not None
            and record.spec.method
            in {
                "hybrid4_a_dense",
                "hybrid4_a_decay_dense",
                "hybrid4_a_ef",
                "hybrid4_a_decay_ef",
            }
        ]
        hybrid_basis_hashes = {
            str(record.summary["invariants"]["before"]["fixed_basis_hash"])
            for record in hybrid_group
            if record.summary is not None
        }
        hybrid_layout_hashes = {
            str(record.summary["layout_hash"])
            for record in hybrid_group
            if record.summary is not None
        }
        if len(hybrid_basis_hashes) > 1 or len(hybrid_layout_hashes) > 1:
            reason = (
                f"Hybrid4 basis/layout differs across dense, EF, or decay aliases "
                f"in {task}/seed-{seed}"
            )
            for record in hybrid_group:
                record.reasons.append(reason)

        hybrid_ef_group = [
            record
            for record in group
            if record.spec is not None
            and record.spec.method in {"hybrid4_a_ef", "hybrid4_a_decay_ef"}
        ]
        if len(hybrid_ef_group) == 2:
            wire_signatures = {
                (
                    record.summary["wire_budget"].get("reference_message_bytes"),
                    record.summary["wire_budget"].get("message_cap_bytes"),
                    record.summary["wire_budget"].get("sparse_k"),
                    record.summary["wire_budget"].get("expected_sparse_message_bytes"),
                    record.summary["final"]["wire"].get("max_uplink_message_bytes"),
                    record.summary["final"]["wire"].get("max_downlink_message_bytes"),
                )
                for record in hybrid_ef_group
                if record.summary is not None
            }
            if len(wire_signatures) > 1:
                reason = (
                    f"Hybrid4 constant/decay EF wire signatures differ in "
                    f"{task}/seed-{seed}"
                )
                for record in hybrid_ef_group:
                    record.reasons.append(reason)

    for (task, method), group in task_method_groups.items():
        basis_hashes = {
            str(record.summary["invariants"]["before"]["fixed_basis_hash"])
            for record in group
            if record.summary is not None
        }
        if len(basis_hashes) > 1:
            reason = (
                f"fixed basis hash disagrees across seeds/budgets in {task}/{method}: "
                f"{sorted(basis_hashes)}"
            )
            for record in group:
                record.reasons.append(reason)

    for record in candidates:
        if record.reasons:
            record.audit_status = "INVALID"


def _collect_records(
    cfg: dict[str, Any], context: ValidationContext
) -> tuple[list[AuditRecord], list[ExpectedRun], list[ExpectedRun]]:
    output_dir = Path(cfg["project"]["output_dir"])
    required = _specs(cfg, diagnostic=False, role="required")
    optional = _specs(cfg, diagnostic=True, role="optional")
    records: list[AuditRecord] = []
    seen: set[str] = set()
    for spec in required:
        path = output_dir / spec.run_id / "summary.json"
        record = AuditRecord(spec.run_id, spec.role, path, spec=spec)
        seen.add(spec.run_id)
        if path.exists():
            _audit_summary(record, cfg, context)
        else:
            run_dir = path.parent
            if (run_dir / "failure.json").exists():
                record.audit_status = "INCOMPLETE"
                record.reasons.append(
                    "failure.json exists but no COMPLETE summary exists"
                )
            elif (run_dir / "checkpoint.pt").exists() or any(
                run_dir.glob("round-*.json")
            ):
                record.audit_status = "INCOMPLETE"
                record.reasons.append(
                    "partial artifacts exist but summary.json is missing"
                )
        records.append(record)
    for spec in optional:
        path = output_dir / spec.run_id / "summary.json"
        if not path.exists():
            continue
        record = AuditRecord(spec.run_id, spec.role, path, spec=spec)
        seen.add(spec.run_id)
        _audit_summary(record, cfg, context)
        records.append(record)
    for path in sorted(output_dir.glob("*/summary.json")):
        candidate_id = path.parent.name
        if candidate_id in seen:
            continue
        record = AuditRecord(candidate_id, "unexpected", path, audit_status="INVALID")
        record.reasons.append(
            "run is not part of the configured main or diagnostic matrix"
        )
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                record.summary = loaded
            else:
                record.reasons.append("summary root is not an object")
        except (json.JSONDecodeError, OSError, TypeError, UnicodeDecodeError) as exc:
            record.reasons.append(
                f"summary cannot be parsed: {type(exc).__name__}: {exc}"
            )
        records.append(record)
    _apply_cross_run_gates(records)
    return records, required, optional


def _split_scores(row: dict[str, Any]) -> tuple[float | None, float | None]:
    run = row.get("run", {})
    if not isinstance(run, dict) or run.get("task") != "mnli":
        return None, None
    final = row.get("final", {})
    splits = final.get("splits", {}) if isinstance(final, dict) else {}
    try:
        return (
            float(splits["validation_matched"]["accuracy"]),
            float(splits["validation_mismatched"]["accuracy"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, None


def _server_split_scores(row: dict[str, Any]) -> tuple[float | None, float | None]:
    run = row.get("run", {})
    if (
        not isinstance(run, dict)
        or run.get("task") != "mnli"
        or run.get("method") != "flasc"
    ):
        return None, None
    final = row.get("final", {})
    splits = final.get("server_splits", {}) if isinstance(final, dict) else {}
    try:
        return (
            float(splits["validation_matched"]["accuracy"]),
            float(splits["validation_mismatched"]["accuracy"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, None


def _binary_task_scores(row: dict[str, Any]) -> tuple[float | None, float | None]:
    run = row.get("run", {})
    if not isinstance(run, dict) or run.get("task") not in {"mrpc", "qqp"}:
        return None, None
    final = row.get("final", {})
    splits = final.get("splits", {}) if isinstance(final, dict) else {}
    values = splits.get("train_holdout", splits.get("validation", {}))
    try:
        return float(values["accuracy"]), float(values["f1"])
    except (KeyError, TypeError, ValueError):
        return None, None


def _effort(row: dict[str, Any]) -> dict[str, float | int]:
    rounds = row["rounds"]
    initialization = row["initialization"]
    setup = row["setup_communication"]
    clients = int(row["data"]["num_clients"])
    handshake_per_client = int(setup["basis_hash_handshake_bytes_per_client"])
    fallback_per_client = int(setup["fallback_fp16_basis_bytes_per_client"])
    return {
        "data_prepare_seconds": float(initialization["data_prepare_seconds"]),
        "spectral_seconds": float(initialization["spectral_seconds"]),
        "train_seconds": sum(
            float(value["timing"]["train_seconds"]) for value in rounds
        ),
        "compression_seconds": sum(
            float(value["timing"]["compression_seconds"]) for value in rounds
        ),
        "eval_seconds": sum(float(value["timing"]["eval_seconds"]) for value in rounds),
        "runtime_seconds": float(row["runtime_seconds"]),
        "resume_count": int(initialization["resume_count"]),
        "peak_gpu_memory_bytes": max(
            int(value["peak_gpu_memory_bytes"]) for value in rounds
        ),
        "basis_handshake_bytes_per_client": handshake_per_client,
        "basis_handshake_bytes_total": handshake_per_client * clients,
        "fallback_basis_bytes_per_client": fallback_per_client,
        "fallback_basis_bytes_total": fallback_per_client * clients,
    }


def _dense_fedavg_all_in_wire_bytes(row: dict[str, Any]) -> int:
    """Independently reconstruct the matched dense FedAvg-LoRA all-in wire.

    The reference uses the same task, model targets/rank, client count and round
    count as ``row``.  It does not trust ``wire_budget.reference_message_bytes``
    or reuse the candidate method's (potentially smaller) schema manifest.
    """

    run = row["run"]
    model = row["model"]
    task = str(run["task"])
    labels = TASK_NUM_LABELS[task]
    head_parameters = int(model["head_parameters"])
    hidden_numerator = head_parameters - labels
    if labels <= 0 or hidden_numerator <= 0 or hidden_numerator % labels != 0:
        raise ValueError("cannot infer hidden size from canonical classifier head")
    hidden_size = hidden_numerator // labels
    rank = int(model["rank"])
    target_names = model["target_modules"]
    if (
        rank <= 0
        or not isinstance(target_names, list)
        or not target_names
        or any(not isinstance(name, str) or not name for name in target_names)
    ):
        raise ValueError("cannot reconstruct dense FedAvg-LoRA wire layout")
    (
        _adapter_parameters,
        reconstructed_head_parameters,
        reference_parameters,
        manifest_size,
    ) = _canonical_wire_dimensions(
        "fedavg_lora",
        task,
        target_names,
        rank,
        hidden_size,
    )
    if reconstructed_head_parameters != head_parameters:
        raise ValueError("classifier head does not match canonical task layout")
    clients = int(row["data"]["num_clients"])
    rounds = int(row["training"]["rounds"])
    if clients <= 0 or rounds <= 0:
        raise ValueError("client and round counts must be positive")
    dense_message_bytes = 26 + 2 * reference_parameters
    dense_schema_bytes = manifest_size * (1 + clients)
    return dense_message_bytes * clients * rounds * 2 + dense_schema_bytes


def _actual_wire_ratio(row: dict[str, Any]) -> float:
    reference_total = _dense_fedavg_all_in_wire_bytes(row)
    return _normal_path_wire_bytes(row) / float(reference_total)


def _normal_path_wire_bytes(
    row: dict[str, Any], round_record: dict[str, Any] | None = None
) -> float:
    """Return recurrent delivered bytes plus the actually used basis handshake."""
    record = row["final"] if round_record is None else round_record
    recurrent = float(record["wire"]["total_delivered_bytes"])
    setup = row.get("setup_communication", {})
    data = row.get("data", {})
    if not isinstance(setup, dict) or not isinstance(data, dict):
        return recurrent
    handshake_per_client = setup.get("basis_hash_handshake_bytes_per_client", 0)
    clients = data.get("num_clients", 0)
    try:
        return recurrent + float(handshake_per_client) * int(clients)
    except (TypeError, ValueError):
        return recurrent


def _communication_to_accuracy(
    row: dict[str, Any], target_accuracy: float
) -> tuple[int, float] | None:
    """Return the first evaluated round and cumulative wire reaching a target."""
    for round_record in row.get("rounds", []):
        if not isinstance(round_record, dict):
            continue
        primary = round_record.get("primary_metric")
        if primary is None:
            continue
        if float(primary) >= target_accuracy:
            return (
                int(round_record["round"]),
                _normal_path_wire_bytes(row, round_record),
            )
    return None


def _mean_or_blank(values: list[float]) -> float | str:
    return statistics.mean(values) if values else ""


def _sd_or_blank(values: list[float]) -> float | str:
    return statistics.stdev(values) if len(values) > 1 else ""


def _communication_efficiency_seed_rows(
    cfg: dict[str, Any], valid_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Build paired endpoint and C95 quantities before cross-seed aggregation."""
    main_methods = {
        *cfg["matrix"].get("dense_methods", []),
        *cfg["matrix"].get("sparse_methods", []),
    }
    selected = [
        row
        for row in valid_rows
        if row["run"]["method"] in main_methods
        and row["run"]["task"] in cfg["data"]["tasks"]
    ]
    references = {
        (row["run"]["task"], int(row["run"]["seed"])): row
        for row in selected
        if row["run"]["method"] == "fedavg_lora"
        and row["run"]["wire_ratio"] is None
    }
    output: list[dict[str, Any]] = []
    for row in sorted(
        selected,
        key=lambda value: (
            value["run"]["task"],
            value["run"]["method"],
            -1
            if value["run"]["wire_ratio"] is None
            else value["run"]["wire_ratio"],
            int(value["run"]["seed"]),
        ),
    ):
        task = str(row["run"]["task"])
        method = str(row["run"]["method"])
        seed = int(row["run"]["seed"])
        ratio = row["run"]["wire_ratio"]
        score = float(row["final"]["primary_metric"])
        task_accuracy, _ = _binary_task_scores(row)
        endpoint_accuracy = score if task_accuracy is None else task_accuracy
        delivered = _normal_path_wire_bytes(row)
        metric_semantics = (
            "masked_server" if method == "flasc" else "communicated_global"
        )
        if method == "fedavg_lora":
            interpretation = "REFERENCE"
        elif method == "flasc":
            interpretation = "DESCRIPTIVE_ONLY_METRIC_STATE_MISMATCH"
        else:
            interpretation = "SAME_METRIC_DESCRIPTIVE"
        values: dict[str, Any] = {
            "task": task,
            "method": method,
            "wire_target": "dense" if ratio is None else ratio,
            "seed": seed,
            "primary_metric_name": primary_metric_name(task),
            "primary_score": score,
            "reference_primary_score": "",
            "delta_primary_score_vs_fedavg": "",
            "primary_score_retention_vs_fedavg": "",
            "metric_semantics": metric_semantics,
            "reference_method": "fedavg_lora",
            "reference_status": "MISSING",
            "interpretation": interpretation,
            "accuracy": endpoint_accuracy,
            "reference_accuracy": "",
            "delta_accuracy_vs_fedavg": "",
            "accuracy_retention_vs_fedavg": "",
            "normal_path_wire_bytes": delivered,
            "reference_normal_path_wire_bytes": "",
            "wire_fraction_vs_fedavg": "",
            "saving_vs_fedavg": "",
            "descriptive_cer": "",
            "c95_dense_target_accuracy": "",
            "c95_dense_target_primary_score": "",
            "c95_dense_status": "MISSING_REFERENCE",
            "c95_dense_round": "",
            "c95_dense_wire_bytes": "",
            "fedavg_c95_dense_round": "",
            "fedavg_c95_dense_wire_bytes": "",
            "speedup95_vs_fedavg": "",
        }
        reference = references.get((task, seed))
        if reference is None:
            output.append(values)
            continue
        reference_score = float(reference["final"]["primary_metric"])
        reference_task_accuracy, _ = _binary_task_scores(reference)
        reference_endpoint_accuracy = (
            reference_score
            if reference_task_accuracy is None
            else reference_task_accuracy
        )
        reference_wire = _normal_path_wire_bytes(reference)
        values["reference_accuracy"] = reference_endpoint_accuracy
        values["reference_primary_score"] = reference_score
        values["reference_normal_path_wire_bytes"] = reference_wire
        if reference_score == 0.0 or reference_wire <= 0.0:
            values["reference_status"] = "INVALID"
            values["c95_dense_status"] = "INVALID_REFERENCE"
            output.append(values)
            continue

        wire_fraction = delivered / reference_wire
        retention = score / reference_score
        accuracy_retention = (
            endpoint_accuracy / reference_endpoint_accuracy
            if reference_endpoint_accuracy != 0.0
            else math.nan
        )
        target_accuracy = 0.95 * reference_score
        reached = _communication_to_accuracy(row, target_accuracy)
        reference_reached = _communication_to_accuracy(reference, target_accuracy)
        values.update(
            {
                "reference_status": "COMPLETE",
                "delta_accuracy_vs_fedavg": endpoint_accuracy
                - reference_endpoint_accuracy,
                "delta_primary_score_vs_fedavg": score - reference_score,
                "accuracy_retention_vs_fedavg": accuracy_retention,
                "primary_score_retention_vs_fedavg": retention,
                "wire_fraction_vs_fedavg": wire_fraction,
                "saving_vs_fedavg": 1.0 - wire_fraction,
                "descriptive_cer": ""
                if wire_fraction <= 0.0
                else retention / wire_fraction,
                # This legacy column is meaningful only when the task primary
                # is accuracy-based.  MRPC/QQP use the frozen GLUE composite,
                # which belongs exclusively in the primary-score column.
                "c95_dense_target_accuracy": ""
                if task in {"mrpc", "qqp"}
                else target_accuracy,
                "c95_dense_target_primary_score": target_accuracy,
                "c95_dense_status": "NOT_REACHED",
            }
        )
        if reference_reached is not None:
            values["fedavg_c95_dense_round"] = reference_reached[0]
            values["fedavg_c95_dense_wire_bytes"] = reference_reached[1]
        if reached is not None and reference_reached is not None:
            values["c95_dense_status"] = "REACHED"
            values["c95_dense_round"] = reached[0]
            values["c95_dense_wire_bytes"] = reached[1]
            if reached[1] > 0.0:
                values["speedup95_vs_fedavg"] = (
                    reference_reached[1] / reached[1]
                )
        output.append(values)
    return output


def _communication_efficiency_rows(
    cfg: dict[str, Any], valid_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate main-matrix primary score and actual wire against paired FedAvg runs.

    Retention, delta, saving, and CER are computed per task/seed before averaging.
    CER is descriptive primary-score retention divided by the FedAvg wire fraction.
    Optional diagnostic methods are intentionally excluded from the Pareto set.
    """
    expected_seeds = {int(seed) for seed in cfg["data"]["partition_seeds"]}
    seed_rows = _communication_efficiency_seed_rows(cfg, valid_rows)
    grouped: dict[tuple[str, str, str | float], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in seed_rows:
        grouped[(row["task"], row["method"], row["wire_target"])].append(row)

    output: list[dict[str, Any]] = []
    for (task, method, ratio), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1 if item[0][2] == "dense" else item[0][2],
        ),
    ):
        scores = [float(row["primary_score"]) for row in group]
        accuracies = [float(row["accuracy"]) for row in group]
        wire_bytes = [float(row["normal_path_wire_bytes"]) for row in group]
        group_seeds = {int(row["seed"]) for row in group}
        paired = [row for row in group if row["reference_status"] == "COMPLETE"]
        reached = [row for row in paired if row["c95_dense_status"] == "REACHED"]
        matched_reference_seeds = {int(row["seed"]) for row in paired}

        def paired_values(key: str) -> list[float]:
            return [float(row[key]) for row in paired if row[key] != ""]

        wire_fractions = paired_values("wire_fraction_vs_fedavg")
        savings = paired_values("saving_vs_fedavg")
        accuracy_deltas = paired_values("delta_accuracy_vs_fedavg")
        primary_deltas = paired_values("delta_primary_score_vs_fedavg")
        accuracy_retentions = paired_values("accuracy_retention_vs_fedavg")
        primary_retentions = paired_values(
            "primary_score_retention_vs_fedavg"
        )
        cers = paired_values("descriptive_cer")
        target_rounds = [float(row["c95_dense_round"]) for row in reached]
        target_wire_bytes = [float(row["c95_dense_wire_bytes"]) for row in reached]
        reference_target_wire_bytes = [
            float(row["fedavg_c95_dense_wire_bytes"]) for row in reached
        ]
        target_speedups = [
            float(row["speedup95_vs_fedavg"])
            for row in reached
            if row["speedup95_vs_fedavg"] != ""
        ]

        complete = group_seeds == expected_seeds and len(group) == len(expected_seeds)
        reference_complete = matched_reference_seeds == expected_seeds
        metric_semantics = group[0]["metric_semantics"]
        interpretation = group[0]["interpretation"]
        output.append(
            {
                "task": task,
                "method": method,
                "wire_target": ratio,
                "n": len(group),
                "expected_n": len(expected_seeds),
                "complete": complete,
                "metric_semantics": metric_semantics,
                "reference_method": "fedavg_lora",
                "paired_reference_n": len(matched_reference_seeds),
                "reference_complete": reference_complete,
                "interpretation": interpretation,
                "primary_metric_name": primary_metric_name(task),
                "mean_primary_score": statistics.mean(scores),
                "sd_primary_score": _sd_or_blank(scores),
                "mean_accuracy": statistics.mean(accuracies),
                "sd_accuracy": _sd_or_blank(accuracies),
                "mean_normal_path_wire_bytes": statistics.mean(wire_bytes),
                "sd_normal_path_wire_bytes": _sd_or_blank(wire_bytes),
                "mean_wire_fraction_vs_fedavg": _mean_or_blank(wire_fractions),
                "sd_wire_fraction_vs_fedavg": _sd_or_blank(wire_fractions),
                "mean_saving_vs_fedavg": _mean_or_blank(savings),
                "sd_saving_vs_fedavg": _sd_or_blank(savings),
                "mean_delta_accuracy_vs_fedavg": _mean_or_blank(accuracy_deltas),
                "sd_delta_accuracy_vs_fedavg": _sd_or_blank(accuracy_deltas),
                "mean_delta_primary_score_vs_fedavg": _mean_or_blank(primary_deltas),
                "sd_delta_primary_score_vs_fedavg": _sd_or_blank(primary_deltas),
                "mean_accuracy_retention_vs_fedavg": _mean_or_blank(
                    accuracy_retentions
                ),
                "sd_accuracy_retention_vs_fedavg": _sd_or_blank(
                    accuracy_retentions
                ),
                "mean_primary_score_retention_vs_fedavg": _mean_or_blank(
                    primary_retentions
                ),
                "sd_primary_score_retention_vs_fedavg": _sd_or_blank(
                    primary_retentions
                ),
                "mean_descriptive_cer": _mean_or_blank(cers),
                "sd_descriptive_cer": _sd_or_blank(cers),
                "c95_dense_reached_n": len(target_wire_bytes),
                "mean_c95_dense_round": _mean_or_blank(target_rounds),
                "sd_c95_dense_round": _sd_or_blank(target_rounds),
                "mean_c95_dense_wire_bytes": _mean_or_blank(target_wire_bytes),
                "sd_c95_dense_wire_bytes": _sd_or_blank(target_wire_bytes),
                "mean_fedavg_c95_dense_wire_bytes": _mean_or_blank(
                    reference_target_wire_bytes
                ),
                "sd_fedavg_c95_dense_wire_bytes": _sd_or_blank(
                    reference_target_wire_bytes
                ),
                "mean_speedup95_vs_fedavg": _mean_or_blank(target_speedups),
                "sd_speedup95_vs_fedavg": _sd_or_blank(target_speedups),
                "pareto_status": "INELIGIBLE",
            }
        )

    for task in cfg["data"]["tasks"]:
        for semantics, status_prefix in (
            ("communicated_global", ""),
            ("masked_server", "REFERENCE_STYLE_"),
        ):
            eligible = [
                row
                for row in output
                if row["task"] == task
                and row["complete"] is True
                and row["metric_semantics"] == semantics
            ]
            for candidate in eligible:
                candidate_accuracy = float(candidate["mean_primary_score"])
                candidate_wire = float(candidate["mean_normal_path_wire_bytes"])
                dominated = any(
                    other is not candidate
                    and float(other["mean_normal_path_wire_bytes"])
                    <= candidate_wire
                    and float(other["mean_primary_score"]) >= candidate_accuracy
                    and (
                        float(other["mean_normal_path_wire_bytes"])
                        < candidate_wire
                        or float(other["mean_primary_score"]) > candidate_accuracy
                    )
                    for other in eligible
                )
                candidate["pareto_status"] = (
                    f"{status_prefix}{'NO' if dominated else 'YES'}"
                )
    return output


def _mean_sd(values: list[float], digits: int = 6) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.{digits}f} (n=1)"
    return (
        f"{statistics.mean(values):.{digits}f} ± {statistics.stdev(values):.{digits}f}"
    )


def _stored_mean_sd(
    row: dict[str, Any],
    stem: str,
    digits: int = 6,
    scale: float = 1.0,
    suffix: str = "",
) -> str:
    mean = row.get(f"mean_{stem}", "")
    if mean == "":
        return "—"
    sd = row.get(f"sd_{stem}", "")
    if sd == "":
        return f"{float(mean) * scale:.{digits}f}{suffix} (n=1)"
    return (
        f"{float(mean) * scale:.{digits}f}{suffix} ± "
        f"{float(sd) * scale:.{digits}f}{suffix}"
    )


def _stored_bytes_mean_sd(row: dict[str, Any], stem: str) -> str:
    mean = row.get(f"mean_{stem}", "")
    if mean == "":
        return "—"
    sd = row.get(f"sd_{stem}", "")
    if sd == "":
        return f"{bytes_human(float(mean))} (n=1)"
    return f"{bytes_human(float(mean))} ± {bytes_human(float(sd))}"


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _valid_rows(records: list[AuditRecord]) -> list[dict[str, Any]]:
    return [
        record.summary
        for record in records
        if record.valid and record.summary is not None
    ]


def _record_csv_row(record: AuditRecord) -> dict[str, Any]:
    row = record.summary or {}
    spec = record.spec
    declared = row.get("run", {}) if isinstance(row.get("run"), dict) else {}
    task = spec.task if spec else declared.get("task", "")
    method = spec.method if spec else declared.get("method", "")
    seed = spec.seed if spec else declared.get("seed", "")
    ratio = spec.wire_ratio if spec else declared.get("wire_ratio")
    matched, mismatched = _split_scores(row) if row else (None, None)
    final = row.get("final", {}) if isinstance(row.get("final"), dict) else {}
    wire = final.get("wire", {}) if isinstance(final.get("wire"), dict) else {}
    effort = _effort(row) if record.valid else {}
    initialization = (
        row.get("initialization", {})
        if isinstance(row.get("initialization"), dict)
        else {}
    )
    training = row.get("training", {}) if isinstance(row.get("training"), dict) else {}
    stability = (
        row.get("stability_protocol", {})
        if isinstance(row.get("stability_protocol"), dict)
        else {}
    )
    wire_budget = (
        row.get("wire_budget", {})
        if isinstance(row.get("wire_budget"), dict)
        else {}
    )
    server_matched, server_mismatched = (
        _server_split_scores(row) if row else (None, None)
    )
    task_accuracy, task_positive_f1 = (
        _binary_task_scores(row) if row else (None, None)
    )
    return {
        "run_id": record.run_id,
        "role": record.role,
        "audit_status": record.audit_status,
        "reasons": "; ".join(record.reasons),
        "task": task,
        "method": method,
        "seed": seed,
        "wire_target": "dense" if ratio is None else ratio,
        "schema_version": row.get("schema_version", ""),
        "declared_status": row.get("status", ""),
        "round_count": len(row.get("rounds", []))
        if isinstance(row.get("rounds"), list)
        else "",
        "adapter_learning_rate": training.get("adapter_learning_rate", ""),
        "head_learning_rate": training.get("head_learning_rate", ""),
        "final_primary": final.get("primary_metric", ""),
        "primary_metric_name": primary_metric_name(task) if task else "",
        "task_accuracy": "" if task_accuracy is None else task_accuracy,
        "task_positive_f1": "" if task_positive_f1 is None else task_positive_f1,
        "flasc_dense_server_primary": final.get("server_primary_metric", ""),
        "mnli_matched": "" if matched is None else matched,
        "mnli_mismatched": "" if mismatched is None else mismatched,
        "flasc_dense_server_mnli_matched": ""
        if server_matched is None
        else server_matched,
        "flasc_dense_server_mnli_mismatched": ""
        if server_mismatched is None
        else server_mismatched,
        "delivered_wire_bytes": wire.get("total_delivered_bytes", ""),
        "actual_reference_ratio": _actual_wire_ratio(row) if record.valid else "",
        "data_prepare_seconds": effort.get("data_prepare_seconds", ""),
        "spectral_seconds": effort.get("spectral_seconds", ""),
        "train_seconds": effort.get("train_seconds", ""),
        "compression_seconds": effort.get("compression_seconds", ""),
        "eval_seconds": effort.get("eval_seconds", ""),
        "runtime_seconds": effort.get("runtime_seconds", ""),
        "resume_count": effort.get("resume_count", ""),
        "peak_gpu_memory_bytes": effort.get("peak_gpu_memory_bytes", ""),
        "stability_stage": stability.get("stage", ""),
        "protocol_sha256": stability.get("protocol_sha256", ""),
        "diagnostic_recipe_version": stability.get(
            "diagnostic_recipe_version", ""
        ),
        "compressor_kind": wire_budget.get("compressor_kind", ""),
        "block_quota_hash": wire_budget.get("block_quota_hash", ""),
        "diagnostic_layout_fingerprint": wire_budget.get(
            "diagnostic_layout_fingerprint", ""
        ),
        "basis_handshake_bytes_per_client": effort.get(
            "basis_handshake_bytes_per_client", ""
        ),
        "basis_handshake_bytes_total": effort.get("basis_handshake_bytes_total", ""),
        "fallback_basis_bytes_per_client": effort.get(
            "fallback_basis_bytes_per_client", ""
        ),
        "fallback_basis_bytes_total": effort.get("fallback_basis_bytes_total", ""),
        "config_hash": row.get("config_hash", ""),
        "source_hash": row.get("source_hash", ""),
        "model_hash": row.get("model_hash", ""),
        "dataset_hash": row.get("dataset_hash", ""),
        "partition_hash": row.get("partition_hash", ""),
        "layout_hash": row.get("layout_hash", ""),
        "initial_head_hash": initialization.get("initial_head_hash", ""),
        "summary_path": str(record.path),
    }


def _csv_text(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _pair_rows(
    cfg: dict[str, Any], valid_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    index = {
        (
            row["run"]["task"],
            int(row["run"]["seed"]),
            row["run"]["method"],
            row["run"]["wire_ratio"],
        ): row
        for row in valid_rows
    }
    comparisons = [
        ("spectral_vs_random", "spectral_a_ef", "random_a_ef"),
        ("proposed_vs_flasc", "spectral_a_ef", "flasc"),
        ("random_ef_vs_noef", "random_a_ef", "random_a_topk"),
        ("spectral_ef_vs_noef", "spectral_a_ef", "spectral_a_topk"),
    ]
    configured_dense = {
        *cfg["matrix"].get("dense_methods", []),
        *cfg["matrix"].get("diagnostic_dense_methods", []),
    }
    configured_sparse = {
        *cfg["matrix"].get("sparse_methods", []),
        *cfg["matrix"].get("diagnostic_sparse_methods", []),
    }
    known_methods = configured_dense | configured_sparse | {
        str(row["run"]["method"]) for row in valid_rows
    }
    comparisons = [
        comparison
        for comparison in comparisons
        if comparison[1] in known_methods and comparison[2] in known_methods
    ]
    include_full_lora_compression_pair = (
        "fedavg_lora" in configured_dense and "full_lora_ef" in configured_sparse
    )
    include_dense_basis_pair = {
        "random_a_dense",
        "spectral_a_dense",
    }.issubset(configured_dense)
    output: list[dict[str, Any]] = []
    for task in cfg["data"]["tasks"]:
        threshold = THRESHOLDS[task]
        for raw_ratio in cfg["communication"]["wire_ratios"]:
            ratio = float(raw_ratio)
            for raw_seed in cfg["data"]["partition_seeds"]:
                seed = int(raw_seed)
                for comparison, left_method, right_method in comparisons:
                    left = index.get((task, seed, left_method, ratio))
                    right = index.get((task, seed, right_method, ratio))
                    left_score = (
                        None if left is None else float(left["final"]["primary_metric"])
                    )
                    right_score = (
                        None
                        if right is None
                        else float(right["final"]["primary_metric"])
                    )
                    delta = (
                        None
                        if left_score is None or right_score is None
                        else left_score - right_score
                    )
                    comparable = comparison != "proposed_vs_flasc"
                    output.append(
                        {
                            "task": task,
                            "wire_ratio": ratio,
                            "seed": seed,
                            "comparison": comparison,
                            "left_method": left_method,
                            "right_method": right_method,
                            "left_metric_semantics": "masked_server"
                            if left_method == "flasc"
                            else "communicated_global",
                            "right_metric_semantics": "masked_server"
                            if right_method == "flasc"
                            else "communicated_global",
                            "pair_status": "COMPLETE"
                            if delta is not None
                            else "MISSING",
                            "comparable": comparable,
                            "interpretation": "COMPARABLE"
                            if comparable
                            else "DESCRIPTIVE_ONLY",
                            "left_score": "" if left_score is None else left_score,
                            "right_score": "" if right_score is None else right_score,
                            "delta": "" if delta is None else delta,
                            "threshold": threshold if comparable else "",
                        }
                    )
                if include_full_lora_compression_pair:
                    left = index.get((task, seed, "full_lora_ef", ratio))
                    right = index.get((task, seed, "fedavg_lora", None))
                    left_score = (
                        None if left is None else float(left["final"]["primary_metric"])
                    )
                    right_score = (
                        None
                        if right is None
                        else float(right["final"]["primary_metric"])
                    )
                    delta = (
                        None
                        if left_score is None or right_score is None
                        else left_score - right_score
                    )
                    output.append(
                        {
                            "task": task,
                            "wire_ratio": ratio,
                            "seed": seed,
                            "comparison": "full_lora_ef_vs_fedavg",
                            "left_method": "full_lora_ef",
                            "right_method": "fedavg_lora",
                            "left_metric_semantics": "communicated_global",
                            "right_metric_semantics": "communicated_global",
                            "pair_status": "COMPLETE"
                            if delta is not None
                            else "MISSING",
                            "comparable": True,
                            "interpretation": "COMPARABLE",
                            "left_score": "" if left_score is None else left_score,
                            "right_score": "" if right_score is None else right_score,
                            "delta": "" if delta is None else delta,
                            "threshold": threshold,
                        }
                    )
        if include_dense_basis_pair:
            for raw_seed in cfg["data"]["partition_seeds"]:
                seed = int(raw_seed)
                left = index.get((task, seed, "spectral_a_dense", None))
                right = index.get((task, seed, "random_a_dense", None))
                left_score = (
                    None if left is None else float(left["final"]["primary_metric"])
                )
                right_score = (
                    None if right is None else float(right["final"]["primary_metric"])
                )
                delta = (
                    None
                    if left_score is None or right_score is None
                    else left_score - right_score
                )
                output.append(
                    {
                        "task": task,
                        "wire_ratio": None,
                        "seed": seed,
                        "comparison": "dense_spectral_vs_random",
                        "left_method": "spectral_a_dense",
                        "right_method": "random_a_dense",
                        "left_metric_semantics": "communicated_global",
                        "right_metric_semantics": "communicated_global",
                        "pair_status": "COMPLETE" if delta is not None else "MISSING",
                        "comparable": True,
                        "interpretation": "COMPARABLE",
                        "left_score": "" if left_score is None else left_score,
                        "right_score": "" if right_score is None else right_score,
                        "delta": "" if delta is None else delta,
                        "threshold": threshold,
                    }
                )
    return output


def _judgement(
    deltas: list[float], threshold: float, expected_n: int
) -> tuple[str, str]:
    required = max(MIN_PAIRED_SEEDS, expected_n)
    if len(deltas) < required:
        return "数据不足", f"只有 {len(deltas)}/{required} 个有效配对，不能下结论"
    mean = statistics.mean(deltas)
    positives = sum(value > 0 for value in deltas)
    negatives = sum(value < 0 for value in deltas)
    if abs(mean) < threshold:
        return "TIE", f"均值落在 ±{threshold:.3f} 的实际等效区间内"
    if mean >= threshold and positives == len(deltas):
        return "SUPPORTED", "全部配对为正且均值达到预注册阈值"
    if mean <= -threshold and negatives == len(deltas):
        return "HURT", "全部配对为负且均值低于负向阈值"
    return "MIXED", "方向不一致，不能作统一正向或负向结论"


def _results_markdown(
    cfg: dict[str, Any], records: list[AuditRecord], context: ValidationContext
) -> str:
    required_records = [record for record in records if record.role == "required"]
    valid_required = sum(record.valid for record in required_records)
    valid_rows = _valid_rows(records)
    efficiency_rows = _communication_efficiency_rows(cfg, valid_rows)
    invalid = [record for record in records if record.audit_status == "INVALID"]
    missing = [
        record for record in records if record.audit_status in {"MISSING", "INCOMPLETE"}
    ]
    full_lora_records = [
        record
        for record in records
        if record.spec is not None and record.spec.method == "full_lora_ef"
    ]
    full_lora_factor_pass = sum(
        record.summary is not None
        and record.spec is not None
        and not _validate_full_lora_factor_audit(record.summary, record.spec)
        for record in full_lora_records
    )
    evaluation_policy = str(
        cfg.get("evaluation", {}).get("policy", "official_validation")
    )
    evaluation_description = (
        "pooled train-only holdout"
        if evaluation_policy == "train_holdout"
        else "official validation"
    )
    lines = [
        "# RoBERTa Federated PEFT Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Validation summary",
        "",
        f"- Required valid runs: {valid_required}/{len(required_records)}",
        f"- Valid optional diagnostic runs: {sum(record.valid and record.role == 'optional' for record in records)}",
        f"- Invalid summaries: {len(invalid)}",
        f"- Missing/incomplete required runs: {len(missing)}",
        f"- Matrix status: **{'COMPLETE' if valid_required == len(required_records) and not invalid else 'INCOMPLETE OR INVALID'}**",
        *(
            [
                "- Full-LoRA A/B factor-audit gate: "
                f"{full_lora_factor_pass}/{len(full_lora_records)} expected runs pass."
            ]
            if full_lora_records
            else []
        ),
        "",
        (
            "Only schema-v2 runs that match the current config, source, model, dataset, "
            "partition, layout, head initialization, invariants, and configured round "
            "count enter statistics."
        ),
    ]
    if context.errors:
        lines += ["", "Validator asset errors:", ""] + [
            f"- {_markdown_escape(error)}" for error in context.errors
        ]
    if invalid or missing:
        lines += [
            "",
            "## Invalid and unavailable runs",
            "",
            "| Run | Role | Audit status | Reason |",
            "|---|---|---|---|",
        ]
        for record in [*invalid, *missing]:
            reason = "; ".join(record.reasons) or "summary.json not found"
            lines.append(
                f"| `{_markdown_escape(record.run_id)}` | {record.role} | "
                f"{record.audit_status} | {_markdown_escape(reason)} |"
            )

    protocol_specs = {
        (spec.task, spec.method, spec.wire_ratio): spec
        for spec in _specs(cfg, diagnostic=False, role="required")
    }
    lines += [
        "",
        "## Configured protocol",
        "",
        "| Task | Method | Wire | Rounds | Local steps | Batch | Eval every | Adapter LR | Head LR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(
        protocol_specs,
        key=lambda value: (value[0], value[1], -1 if value[2] is None else value[2]),
    ):
        task, method, ratio = key
        adapter_lr, head_lr = _learning_rates(cfg, task, method)
        lines.append(
            f"| {task} | {method} | {'dense' if ratio is None else f'{ratio:.0%}'} | "
            f"{int(task_value(cfg, 'training', 'rounds', task))} | "
            f"{int(task_value(cfg, 'training', 'local_steps', task))} | "
            f"{int(task_value(cfg, 'training', 'batch_size', task))} | "
            f"{int(task_value(cfg, 'training', 'eval_every', task))} | "
            f"{adapter_lr:.3g} | {head_lr:.3g} |"
        )

    lines += [
        "",
        "## Raw per-seed results",
        "",
        (
            "For FLASC, `Primary` and MNLI m/mm are masked-server metrics. "
            "`FLASC dense diagnostic` evaluates the unmasked dense server adapter; "
            "it is diagnostic only and is never substituted into paired tests."
        ),
        (
            "`All-in/reference` uses normal-path bytes (the method's recurrent "
            "ledger, canonical one-time schema, and any basis-hash handshake) in "
            "the numerator and an independently reconstructed matched dense "
            "FedAvg-LoRA recurrent ledger plus its own canonical schema in the "
            "denominator. Sparse recurrent packets are independently gated "
            "against the exact 10%/20% serialized per-message cap."
        ),
        (
            f"`Evaluation` uses the configured `{evaluation_policy}` policy "
            f"({evaluation_description}). Accuracy and positive-class F1 are "
            "computed on that same split; `train_holdout` values are not "
            "official-validation results."
        ),
        "",
        "| Task | Method | Wire | Seed | Adapter LR | Head LR | Primary | Evaluation accuracy | Evaluation positive F1 | FLASC dense diagnostic | MNLI m/mm | FLASC dense m/mm | Delivered wire | All-in/reference | Runtime | Peak GPU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        valid_rows,
        key=lambda value: (
            value["run"]["task"],
            value["run"]["method"],
            -1 if value["run"]["wire_ratio"] is None else value["run"]["wire_ratio"],
            value["run"]["seed"],
        ),
    ):
        matched, mismatched = _split_scores(row)
        server_matched, server_mismatched = _server_split_scores(row)
        effort = _effort(row)
        ratio = row["run"]["wire_ratio"]
        server_primary = row["final"].get("server_primary_metric")
        task_accuracy, task_positive_f1 = _binary_task_scores(row)
        lines.append(
            f"| {row['run']['task']} | {row['run']['method']} | "
            f"{'dense' if ratio is None else f'{float(ratio):.0%}'} | {row['run']['seed']} | "
            f"{float(row['training']['adapter_learning_rate']):.3g} | "
            f"{float(row['training']['head_learning_rate']):.3g} | "
            f"{float(row['final']['primary_metric']):.6f} | "
            f"{'—' if task_accuracy is None else f'{task_accuracy:.6f}'} | "
            f"{'—' if task_positive_f1 is None else f'{task_positive_f1:.6f}'} | "
            f"{'—' if server_primary is None else f'{float(server_primary):.6f}'} | "
            f"{'—' if matched is None else f'{matched:.6f} / {mismatched:.6f}'} | "
            f"{'—' if server_matched is None else f'{server_matched:.6f} / {server_mismatched:.6f}'} | "
            f"{bytes_human(float(row['final']['wire']['total_delivered_bytes']))} | "
            f"{_actual_wire_ratio(row):.4f} | {float(effort['runtime_seconds']):.1f}s | "
            f"{bytes_human(float(effort['peak_gpu_memory_bytes']))} |"
        )

    grouped: dict[tuple[str, str, float | None], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in valid_rows:
        grouped[
            (row["run"]["task"], row["run"]["method"], row["run"]["wire_ratio"])
        ].append(row)
    lines += [
        "",
        "## Mean ± SD across partition seeds",
        "",
        (
            "FLASC aggregate `Primary` remains the masked-server value. Its unmasked "
            "dense-server diagnostic is shown separately and is not client-visible."
        ),
        "",
        "| Task | Method | Wire | n | Adapter LR | Head LR | Primary | Evaluation accuracy | Evaluation positive F1 | FLASC dense diagnostic | MNLI matched | MNLI mismatched | FLASC dense matched | FLASC dense mismatched | Delivered wire | All-in/reference |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(
        grouped,
        key=lambda value: (value[0], value[1], -1 if value[2] is None else value[2]),
    ):
        task, method, ratio = key
        group = grouped[key]
        primary = [float(row["final"]["primary_metric"]) for row in group]
        split_pairs = [_split_scores(row) for row in group]
        matched_values = [value[0] for value in split_pairs if value[0] is not None]
        mismatched_values = [value[1] for value in split_pairs if value[1] is not None]
        server_primary_values = [
            float(row["final"]["server_primary_metric"])
            for row in group
            if row["final"].get("server_primary_metric") is not None
        ]
        server_split_pairs = [_server_split_scores(row) for row in group]
        binary_pairs = [_binary_task_scores(row) for row in group]
        task_accuracy_values = [
            value[0] for value in binary_pairs if value[0] is not None
        ]
        task_positive_f1_values = [
            value[1] for value in binary_pairs if value[1] is not None
        ]
        server_matched_values = [
            value[0] for value in server_split_pairs if value[0] is not None
        ]
        server_mismatched_values = [
            value[1] for value in server_split_pairs if value[1] is not None
        ]
        delivered = [
            float(row["final"]["wire"]["total_delivered_bytes"]) for row in group
        ]
        actual_ratios = [_actual_wire_ratio(row) for row in group]
        adapter_lrs = sorted(
            {float(row["training"]["adapter_learning_rate"]) for row in group}
        )
        head_lrs = sorted(
            {float(row["training"]["head_learning_rate"]) for row in group}
        )
        lines.append(
            f"| {task} | {method} | {'dense' if ratio is None else f'{float(ratio):.0%}'} | "
            f"{len(group)} | {','.join(f'{value:.3g}' for value in adapter_lrs)} | "
            f"{','.join(f'{value:.3g}' for value in head_lrs)} | "
            f"{_mean_sd(primary)} | {_mean_sd(task_accuracy_values)} | "
            f"{_mean_sd(task_positive_f1_values)} | {_mean_sd(server_primary_values)} | "
            f"{_mean_sd(matched_values)} | {_mean_sd(mismatched_values)} | "
            f"{_mean_sd(server_matched_values)} | {_mean_sd(server_mismatched_values)} | "
            f"{bytes_human(statistics.mean(delivered))} | "
            f"{_mean_sd(actual_ratios, 4)} |"
        )

    lines += [
        "",
        "## Communication efficiency: C@95%-DenseScore and primary-score–wire frontier",
        "",
        (
            "The reference is the same-task, same-partition-seed natural dense "
            "FedAvg-LoRA run. `Wire/FedAvg` uses actual normal-path bytes: recurrent "
            "delivered bytes (including schema) plus the basis-hash handshake; the "
            "unused full-basis fallback is excluded."
        ),
        (
            "Saving = 1 − wire fraction; Δ primary = method − FedAvg; retention "
            "= method primary score / FedAvg primary score. These endpoint quantities are "
            "computed seed-pair first and then summarized by mean ± sample SD."
        ),
        (
            "`C@95%-DenseScore` is the cumulative normal-path wire at the first recorded "
            "evaluation reaching 95% of the matched FedAvg final primary score; no values "
            "are interpolated between evaluation rounds. Speedup95 is paired FedAvg "
            "C@95 divided by method C@95 and is blank when the target is not reached. "
            "These are the primary scalar communication-efficiency summaries."
        ),
        (
            "Pareto status is a complementary descriptive frontier computed within "
            "each task from complete main-matrix groups using mean primary score and mean "
            "delivered bytes; it is not a scalar metric or winner ranking. FLASC is "
            "excluded from the comparable-state frontier because its "
            "masked-server metric is not "
            "state-equivalent; `REFERENCE_STYLE_*` compares only FLASC budget points."
        ),
        (
            "A study-specific retention-per-wire `descriptive_cer` is emitted only in "
            "the communication CSV files. It is not shown or ranked here and is not "
            "presented as a standard or state-of-the-art metric."
        ),
        "",
        "| Task | Method | Wire | n | Primary score | Normal-path wire | Wire/FedAvg | Saving | Δ primary | Retention | C@95%-DenseScore | Reached | Speedup95 | Pareto | Interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in efficiency_rows:
        target = row["wire_target"]
        wire_label = "dense" if target == "dense" else f"{float(target):.0%}"
        lines.append(
            f"| {row['task']} | {row['method']} | {wire_label} | "
            f"{row['n']}/{row['expected_n']} ({row['paired_reference_n']} paired) | "
            f"{_stored_mean_sd(row, 'primary_score')} | "
            f"{_stored_bytes_mean_sd(row, 'normal_path_wire_bytes')} | "
            f"{_stored_mean_sd(row, 'wire_fraction_vs_fedavg', 2, 100.0, '%')} | "
            f"{_stored_mean_sd(row, 'saving_vs_fedavg', 2, 100.0, '%')} | "
            f"{_stored_mean_sd(row, 'delta_primary_score_vs_fedavg')} | "
            f"{_stored_mean_sd(row, 'primary_score_retention_vs_fedavg', 2, 100.0, '%')} | "
            f"{_stored_bytes_mean_sd(row, 'c95_dense_wire_bytes')} | "
            f"{row['c95_dense_reached_n']}/{row['expected_n']} | "
            f"{_stored_mean_sd(row, 'speedup95_vs_fedavg', 3, suffix='×')} | "
            f"{row['pareto_status']} | {row['interpretation']} |"
        )

    by_effort_group: dict[tuple[str, str, float | None], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in valid_rows:
        by_effort_group[
            (row["run"]["task"], row["run"]["method"], row["run"]["wire_ratio"])
        ].append(row)
    lines += [
        "",
        "## Compute effort",
        "",
        (
            "Setup columns are total bytes for all configured clients. The handshake "
            "is the normal path; fallback bytes apply only when local basis "
            "reconstruction or hash agreement fails."
        ),
        "",
        "| Task | Method | Wire | n | Data prepare | Spectral | Train | Compress | Eval | Runtime | Resume count | Peak GPU | Basis handshake | Basis fallback |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(
        by_effort_group,
        key=lambda value: (value[0], value[1], -1 if value[2] is None else value[2]),
    ):
        task, method, ratio = key
        efforts = [_effort(row) for row in by_effort_group[key]]
        lines.append(
            f"| {task} | {method} | {'dense' if ratio is None else f'{float(ratio):.0%}'} | "
            f"{len(efforts)} | "
            f"{_mean_sd([float(value['data_prepare_seconds']) for value in efforts], 1)}s | "
            f"{_mean_sd([float(value['spectral_seconds']) for value in efforts], 1)}s | "
            f"{_mean_sd([float(value['train_seconds']) for value in efforts], 1)}s | "
            f"{_mean_sd([float(value['compression_seconds']) for value in efforts], 1)}s | "
            f"{_mean_sd([float(value['eval_seconds']) for value in efforts], 1)}s | "
            f"{_mean_sd([float(value['runtime_seconds']) for value in efforts], 1)}s | "
            f"{_mean_sd([float(value['resume_count']) for value in efforts], 1)} | "
            f"{bytes_human(max(float(value['peak_gpu_memory_bytes']) for value in efforts))} | "
            f"{bytes_human(statistics.mean(float(value['basis_handshake_bytes_total']) for value in efforts))} | "
            f"{bytes_human(statistics.mean(float(value['fallback_basis_bytes_total']) for value in efforts))} |"
        )
    lines += [
        "",
        (
            "Detailed validation/provenance is in `RUNS.csv`; seed-level comparisons "
            "are in `PAIRED_DELTAS.csv`; paired communication quantities are in "
            "`COMMUNICATION_EFFICIENCY_BY_SEED.csv` and their mean ± sample SD in "
            "`COMMUNICATION_EFFICIENCY.csv`; Chinese decisions are in `ANALYSIS.md`."
        ),
    ]
    return "\n".join(lines) + "\n"


def _analysis_markdown(
    cfg: dict[str, Any], records: list[AuditRecord], pair_rows: list[dict[str, Any]]
) -> str:
    valid_required = sum(
        record.valid and record.role == "required" for record in records
    )
    required_total = sum(record.role == "required" for record in records)
    invalid = sum(record.audit_status == "INVALID" for record in records)
    expected_n = len(cfg["data"]["partition_seeds"])
    lines = [
        "# 实验分析",
        "",
        "## 有效性门禁",
        "",
        f"- 正式主矩阵有效运行：{valid_required}/{required_total}",
        f"- INVALID 结果：{invalid}；这些结果没有进入任何均值、标准差或配对分析。",
        f"- 最少需要 {max(MIN_PAIRED_SEEDS, expected_n)} 个完整配对；不足时不下结论。",
        "- 实际意义阈值："
        + "，".join(
            f"{task}={THRESHOLDS[task]:.3f}" for task in cfg["data"]["tasks"]
        )
        + "。",
    ]
    labels = {
        "spectral_vs_random": "Spectral-A+EF − Random-A+EF",
        "proposed_vs_flasc": "Spectral-A+EF − FLASC（参考口径，仅描述）",
    }
    core_comparisons = {"spectral_vs_random", "proposed_vs_flasc"}
    if any(row["comparison"] in core_comparisons for row in pair_rows):
        lines += [
            "",
            "## 核心方法比较",
            "",
            "| Task | Wire | 比较 | 有效配对 | Mean ± SD delta | 正/负 | 判断 | 依据 |",
            "|---|---:|---|---:|---:|---:|---|---|",
        ]
        for task in cfg["data"]["tasks"]:
            threshold = THRESHOLDS[task]
            for raw_ratio in cfg["communication"]["wire_ratios"]:
                ratio = float(raw_ratio)
                for comparison in ("spectral_vs_random", "proposed_vs_flasc"):
                    if not any(
                        row["comparison"] == comparison for row in pair_rows
                    ):
                        continue
                    selected = [
                        row
                        for row in pair_rows
                        if row["task"] == task
                        and row["comparison"] == comparison
                        and row["wire_ratio"] is not None
                        and math.isclose(float(row["wire_ratio"]), ratio)
                        and row["pair_status"] == "COMPLETE"
                    ]
                    deltas = [float(row["delta"]) for row in selected]
                    if comparison == "proposed_vs_flasc":
                        judgement = "口径不一致"
                        reason = (
                            "FLASC 使用 reference-style masked-server 指标，"
                            "与 communicated-global 状态不作直接优劣归因"
                        )
                    else:
                        judgement, reason = _judgement(
                            deltas, threshold, expected_n
                        )
                    direction = (
                        f"{sum(value > 0 for value in deltas)}/"
                        f"{sum(value < 0 for value in deltas)}"
                    )
                    lines.append(
                        f"| {task} | {ratio:.0%} | {labels[comparison]} | "
                        f"{len(deltas)}/{expected_n} | {_mean_sd(deltas)} | "
                        f"{direction} | **{judgement}** | {reason} |"
                    )

    if any(row["comparison"] == "full_lora_ef_vs_fedavg" for row in pair_rows):
        lines += [
            "",
            "## Full-LoRA 压缩成本",
            "",
            (
                "该表使用同 task、seed 的 communicated-global 指标；"
                "DeltaAcc = FedAvg-LoRA+Top-k+EF − dense FedAvg-LoRA。"
            ),
            "",
            "| Task | Wire | 有效配对 | Mean ± SD DeltaAcc | 正/负 | 判断 | 依据 |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for task in cfg["data"]["tasks"]:
            threshold = THRESHOLDS[task]
            for raw_ratio in cfg["communication"]["wire_ratios"]:
                ratio = float(raw_ratio)
                selected = [
                    row
                    for row in pair_rows
                    if row["task"] == task
                    and row["comparison"] == "full_lora_ef_vs_fedavg"
                    and row["wire_ratio"] is not None
                    and math.isclose(float(row["wire_ratio"]), ratio)
                    and row["pair_status"] == "COMPLETE"
                ]
                deltas = [float(row["delta"]) for row in selected]
                judgement, reason = _judgement(deltas, threshold, expected_n)
                direction = (
                    f"{sum(value > 0 for value in deltas)}/"
                    f"{sum(value < 0 for value in deltas)}"
                )
                lines.append(
                    f"| {task} | {ratio:.0%} | {len(deltas)}/{expected_n} | "
                    f"{_mean_sd(deltas)} | {direction} | **{judgement}** | {reason} |"
                )

    if any(
        row["comparison"] == "dense_spectral_vs_random" for row in pair_rows
    ):
        lines += [
            "",
            "## 无压缩基底诊断",
            "",
            "该配对只用于机制诊断，不属于正式六方法主张。",
            "",
        ]
        for task in cfg["data"]["tasks"]:
            threshold = THRESHOLDS[task]
            selected = [
                row
                for row in pair_rows
                if row["task"] == task
                and row["comparison"] == "dense_spectral_vs_random"
                and row["pair_status"] == "COMPLETE"
            ]
            deltas = [float(row["delta"]) for row in selected]
            judgement, reason = _judgement(deltas, threshold, expected_n)
            lines.append(
                f"- **{task}**：Spectral-A dense − Random-A dense = {_mean_sd(deltas)}；"
                f"有效配对 {len(deltas)}/{expected_n}，判断 **{judgement}**（{reason}）。"
            )

    ef_comparisons = {"random_ef_vs_noef", "spectral_ef_vs_noef"}
    if any(row["comparison"] in ef_comparisons for row in pair_rows):
        lines += [
            "",
            "## Error-feedback 归因",
            "",
        ]
        for task in cfg["data"]["tasks"]:
            for raw_ratio in cfg["communication"]["wire_ratios"]:
                ratio = float(raw_ratio)
                random_pairs = [
                    row
                    for row in pair_rows
                    if row["task"] == task
                    and row["comparison"] == "random_ef_vs_noef"
                    and row["wire_ratio"] is not None
                    and math.isclose(float(row["wire_ratio"]), ratio)
                    and row["pair_status"] == "COMPLETE"
                ]
                spectral_pairs = [
                    row
                    for row in pair_rows
                    if row["task"] == task
                    and row["comparison"] == "spectral_ef_vs_noef"
                    and row["wire_ratio"] is not None
                    and math.isclose(float(row["wire_ratio"]), ratio)
                    and row["pair_status"] == "COMPLETE"
                ]
                required = max(MIN_PAIRED_SEEDS, expected_n)
                if not random_pairs or not spectral_pairs:
                    lines.append(
                        f"- **{task} / {ratio:.0%}：不能归因于 EF。** "
                        f"Random matched no-EF 配对 {len(random_pairs)}/{required}，"
                        f"Spectral matched no-EF 配对 {len(spectral_pairs)}/{required}；"
                        "缺少同 task、seed、wire 的 no-EF 对照。"
                    )
                elif len(random_pairs) < required or len(spectral_pairs) < required:
                    lines.append(
                        f"- **{task} / {ratio:.0%}：已有 EF/no-EF 对照，但种子不足。** "
                        f"Random 配对 {len(random_pairs)}/{required}，"
                        f"Spectral 配对 {len(spectral_pairs)}/{required}；"
                        "仅作开发诊断，不能作正式效果归因。"
                    )
                else:
                    random_delta = [float(row["delta"]) for row in random_pairs]
                    spectral_delta = [float(row["delta"]) for row in spectral_pairs]
                    lines.append(
                        f"- {task} / {ratio:.0%}：no-EF 配对完整；"
                        f"Random EF delta {_mean_sd(random_delta)}，"
                        f"Spectral EF delta {_mean_sd(spectral_delta)}。"
                    )
    mnli_boundary = (
        "- MNLI 的主指标是 matched/mismatched accuracy 的算术平均，两个 split 的原始结果同时保留在 `RESULTS.md`。"
        if cfg.get("evaluation", {}).get("policy", "official_validation")
        == "official_validation"
        else "- 本开发配置使用 pooled train-holdout；MNLI matched/mismatched 仅在 official-validation 正式配置中报告。"
    )
    lines += [
        "",
        "## 解释边界",
        "",
        "- `SUPPORTED` 只表示三个配置内划分 seed 的方向一致且均值超过预注册阈值，不等于统计显著或跨数据集泛化。",
        "- `TIE` 表示均值位于实际等效区间；`MIXED` 表示 seed 方向不一致；二者都不能写成方法稳定获益。",
        mnli_boundary,
        "- FLASC 的主指标遵循参考实现的 masked-server convention（聚合后参数乘本轮下发 mask）；它不是客户端已持有状态。dense-server 指标只诊断服务器未掩码状态。Spectral−FLASC 数值仅作透明描述，不进入同状态语义的优劣判断。",
        "- Dense reference 不伪装成 10%/20% 等预算方法；通信结论以实际序列化 delivered bytes 为准。",
    ]
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _report_directory(cfg: dict[str, Any], raw_output: str | None) -> Path:
    if raw_output is None:
        return result_path(cfg, cfg["project"]["output_dir"])
    candidate = result_path(cfg, raw_output)
    if candidate.suffix:
        if candidate.name != "RESULTS.md":
            raise ValueError("--output file, when used, must be named RESULTS.md")
        return candidate.parent
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output",
        help="report directory under docs/results, or a path ending in RESULTS.md",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    required = _specs(cfg, diagnostic=False, role="required")
    optional = _specs(cfg, diagnostic=True, role="optional")
    context = _build_context(cfg, [*required, *optional])
    records, _, _ = _collect_records(cfg, context)
    valid_rows = _valid_rows(records)
    pairs = _pair_rows(cfg, valid_rows)
    efficiency_by_seed = _communication_efficiency_seed_rows(cfg, valid_rows)
    efficiency = _communication_efficiency_rows(cfg, valid_rows)
    report_dir = _report_directory(cfg, args.output)

    run_rows = [_record_csv_row(record) for record in records]
    run_fields = (
        list(run_rows[0]) if run_rows else ["run_id", "audit_status", "reasons"]
    )
    pair_fields = (
        list(pairs[0])
        if pairs
        else [
            "task",
            "wire_ratio",
            "seed",
            "comparison",
            "left_method",
            "right_method",
            "left_metric_semantics",
            "right_metric_semantics",
            "pair_status",
            "comparable",
            "interpretation",
            "left_score",
            "right_score",
            "delta",
            "threshold",
        ]
    )
    efficiency_by_seed_fields = list(COMMUNICATION_EFFICIENCY_BY_SEED_FIELDS)
    efficiency_fields = list(COMMUNICATION_EFFICIENCY_FIELDS)
    outputs = {
        "RESULTS.md": _results_markdown(cfg, records, context),
        "RUNS.csv": _csv_text(run_rows, run_fields),
        "PAIRED_DELTAS.csv": _csv_text(pairs, pair_fields),
        "COMMUNICATION_EFFICIENCY_BY_SEED.csv": _csv_text(
            efficiency_by_seed, efficiency_by_seed_fields
        ),
        "COMMUNICATION_EFFICIENCY.csv": _csv_text(efficiency, efficiency_fields),
        "ANALYSIS.md": _analysis_markdown(cfg, records, pairs),
    }
    for name, content in outputs.items():
        path = result_path(cfg, report_dir / name)
        _atomic_write(path, content)
        print(path)


if __name__ == "__main__":
    main()
