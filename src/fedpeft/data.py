from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk

from .utils import atomic_json_dump, sha256_arrays


TASK_FIELDS = {
    "sst2": ("sentence", None),
    "rte": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
    # Web-service tasks built by scripts/prepare_web_tasks.py
    "banking77": ("text", None),
    "moderation": ("text", None),
}

# Decoder-only tokenizers have no separator/classification tokens: pairs are joined
# by a task prompt and the classifier reads the final token, so truncation keeps the
# end of the prompt (left truncation) and padding goes to the right.
DECODER_TEMPLATES = {
    "sst2": "Review: {0}\nSentiment:",
    "rte": "Premise: {0}\nHypothesis: {1}\nRelation:",
    "mnli": "Premise: {0}\nHypothesis: {1}\nRelation:",
    "mrpc": "Sentence 1: {0}\nSentence 2: {1}\nParaphrase:",
    "qqp": "Question 1: {0}\nQuestion 2: {1}\nDuplicate:",
    "qnli": "Question: {0}\nSentence: {1}\nAnswer:",
    "banking77": "Customer query: {0}\nIntent:",
    "moderation": "Comment: {0}\nToxic:",
}


def is_decoder_tokenizer(tokenizer: Any) -> bool:
    """True for tokenizers without CLS/SEP tokens (decoder-only LMs such as Qwen2)."""
    return getattr(tokenizer, "cls_token", None) is None and getattr(tokenizer, "sep_token", None) is None


TASK_NUM_LABELS = {"sst2": 2, "rte": 2, "mnli": 3, "mrpc": 2, "qqp": 2, "qnli": 2, "banking77": 77, "moderation": 2}
TASK_VALIDATION_SPLITS = {
    "sst2": ["validation"],
    "rte": ["validation"],
    "mnli": ["validation_matched", "validation_mismatched"],
    "mrpc": ["validation"],
    "qqp": ["validation"],
    "qnli": ["validation"],
    "banking77": ["validation"],
    # one held-out set per platform, so the metric can weight platforms equally
    "moderation": ["validation_civil", "validation_twitter"],
}
TASK_CANONICAL_ROWS = {
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
    "qnli": {"train": 104743, "validation": 5463, "test": 5463},
}


@dataclass
class ClientPartition:
    train: list[np.ndarray]
    holdout: list[np.ndarray]
    digest: str


class GpuTensorDataset:
    def __init__(self, tensors: dict[str, torch.Tensor], device: torch.device):
        self.tensors = {key: value.to(device, non_blocking=True) for key, value in tensors.items()}
        lengths = {value.shape[0] for value in self.tensors.values()}
        if len(lengths) != 1:
            raise ValueError(f"tensor lengths disagree: {lengths}")
        self.size = next(iter(lengths))
        self.device = device

    def batch(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        return {key: value.index_select(0, indices) for key, value in self.tensors.items()}

    def sequential_batches(self, batch_size: int):
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            idx = torch.arange(start, end, device=self.device)
            yield self.batch(idx)


def _tokenizer_hash(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        serialized = backend.to_str()
    else:
        serialized = json.dumps(tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True)
    payload = {
        "backend": serialized,
        "special_tokens": {key: str(value) for key, value in tokenizer.special_tokens_map.items()},
        "model_input_names": list(tokenizer.model_input_names),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _raw_dataset_manifest(raw: DatasetDict) -> list[dict[str, Any]]:
    return [
        {
            "split": split,
            "rows": len(raw[split]),
            "fingerprint": str(getattr(raw[split], "_fingerprint", "")),
            "features": str(raw[split].features),
        }
        for split in sorted(raw.keys())
    ]


def prepare_tokenized_dataset(
    dataset_root: str | Path,
    cache_root: str | Path,
    task: str,
    tokenizer: Any,
    max_length: int,
    num_proc: int,
) -> DatasetDict:
    if task not in TASK_FIELDS:
        raise ValueError(f"unsupported task: {task}")
    raw = load_from_disk(str(Path(dataset_root) / task))
    decoder = is_decoder_tokenizer(tokenizer)
    if decoder:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
    provenance = {
        "schema_version": 1,
        "task": task,
        "max_length": int(max_length),
        "padding": "max_length",
        "truncation": True,
        "tokenizer_hash": _tokenizer_hash(tokenizer),
        "raw_dataset": _raw_dataset_manifest(raw),
    }
    if decoder:
        # Only decoder caches record these, so existing encoder caches keep their hash.
        provenance.update(
            {"decoder_template": DECODER_TEMPLATES[task], "truncation_side": "left", "padding_side": "right"}
        )
    provenance_hash = hashlib.sha256(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    prefix = (
        "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in Path(str(tokenizer.name_or_path)).name)
        if decoder
        else "roberta-base"
    )
    cache_path = Path(cache_root) / (
        f"{prefix}__{task}__len-{max_length}__source-{provenance_hash[:12]}"
    )
    marker = cache_path / "_TOKENIZED_SUCCESS.json"
    if cache_path.exists() and marker.exists():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded == provenance:
            return load_from_disk(str(cache_path))
    if cache_path.exists():
        shutil.rmtree(cache_path)
    first, second = TASK_FIELDS[task]

    def tokenize(batch: dict[str, list[Any]]) -> dict[str, Any]:
        if decoder:
            template = DECODER_TEMPLATES[task]
            columns = [batch[first]] if second is None else [batch[first], batch[second]]
            args = ([template.format(*values) for values in zip(*columns)],)
        else:
            args = (batch[first],) if second is None else (batch[first], batch[second])
        encoded = tokenizer(
            *args,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        encoded["labels"] = batch["label"]
        return encoded

    # "group" (the platform of a Web-service task) is kept for the client partition.
    keep = {"input_ids", "attention_mask", "token_type_ids", "labels", "group"}
    tokenized = raw.map(
        tokenize,
        batched=True,
        num_proc=max(1, int(num_proc)),
        desc=f"Tokenizing {task}",
    )
    for split in tokenized:
        remove = [column for column in tokenized[split].column_names if column not in keep]
        if remove:
            tokenized[split] = tokenized[split].remove_columns(remove)
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{cache_path.name}.", dir=Path(cache_root))
    )
    try:
        tokenized.save_to_disk(str(temporary))
        atomic_json_dump(provenance, temporary / marker.name)
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    # Reload the published cache so the Dataset fingerprints are identical for
    # the first writer and every later reader.  Hugging Face may assign new
    # fingerprints across a save/load boundary even when tensor content is
    # unchanged.
    return load_from_disk(str(cache_path))


def dataset_to_tensors(dataset: Dataset) -> dict[str, torch.Tensor]:
    columns = [key for key in ("input_ids", "attention_mask", "token_type_ids", "labels") if key in dataset.column_names]
    formatted = dataset.with_format("torch", columns=columns)
    values = formatted[:]
    result: dict[str, torch.Tensor] = {}
    for key in columns:
        tensor = values[key]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        result[key] = tensor.long().contiguous()
    return result


def _dirichlet_allocation(
    labels: np.ndarray,
    positions: np.ndarray,
    num_clients: int,
    alpha: float,
    min_client_examples: int,
    rng: np.random.Generator,
    max_attempts: int,
) -> list[list[int]]:
    """Per-class Dirichlet split of ``positions`` (indices into the train set) over clients."""
    classes = np.unique(labels[positions])
    for _ in range(max_attempts):
        candidate: list[list[int]] = [[] for _ in range(num_clients)]
        for label in classes:
            class_indices = positions[labels[positions] == label]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha, dtype=np.float64))
            counts = rng.multinomial(len(class_indices), proportions)
            cursor = 0
            for client_id, count in enumerate(counts):
                candidate[client_id].extend(class_indices[cursor : cursor + count].tolist())
                cursor += count
        if min(len(items) for items in candidate) >= min_client_examples:
            return candidate
    raise RuntimeError(
        f"failed to produce {num_clients} clients with at least "
        f"{min_client_examples} examples after {max_attempts} attempts"
    )


def create_dirichlet_partition(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    holdout_fraction: float,
    min_client_examples: int,
    seed: int,
    max_attempts: int = 1000,
    groups: np.ndarray | None = None,
) -> ClientPartition:
    """Dirichlet label skew over clients.  With ``groups`` (e.g. the platform of each example),
    the clients are divided evenly among the groups and each group is split on its own, so a
    client only holds data of its own platform."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if groups is None:
        allocations = _dirichlet_allocation(
            labels, np.arange(len(labels)), num_clients, alpha, min_client_examples, rng, max_attempts
        )
    else:
        groups = np.asarray(groups, dtype=np.int64)
        names = np.unique(groups)
        if num_clients % len(names):
            raise ValueError(f"{num_clients} clients cannot be divided evenly among {len(names)} groups")
        allocations = []
        for name in names:
            allocations.extend(
                _dirichlet_allocation(
                    labels, np.flatnonzero(groups == name), num_clients // len(names), alpha,
                    min_client_examples, rng, max_attempts,
                )
            )

    train: list[np.ndarray] = []
    holdout: list[np.ndarray] = []
    for items in allocations:
        values = np.asarray(items, dtype=np.int64)
        rng.shuffle(values)
        n_holdout = int(round(len(values) * holdout_fraction))
        n_holdout = min(max(n_holdout, 1 if holdout_fraction > 0 else 0), len(values) - 1)
        holdout.append(np.sort(values[:n_holdout]))
        train.append(np.sort(values[n_holdout:]))
    all_arrays = [array for pair in zip(train, holdout) for array in pair]
    return ClientPartition(train=train, holdout=holdout, digest=sha256_arrays(all_arrays))


def load_or_create_partition(
    labels: np.ndarray,
    output_path: str | Path,
    num_clients: int,
    alpha: float,
    holdout_fraction: float,
    min_client_examples: int,
    seed: int,
    groups: np.ndarray | None = None,
) -> ClientPartition:
    output_path = Path(output_path)
    expected_metadata = {
        "seed": int(seed),
        "num_clients": int(num_clients),
        "alpha": float(alpha),
        "holdout_fraction": float(holdout_fraction),
        "min_client_examples": int(min_client_examples),
        "labels_hash": sha256_arrays([np.asarray(labels, dtype=np.int64)]),
    }
    if groups is not None:
        expected_metadata["groups_hash"] = sha256_arrays([np.asarray(groups, dtype=np.int64)])
    if output_path.exists():
        payload = np.load(output_path, allow_pickle=False)
        for key, expected in expected_metadata.items():
            if key not in payload or payload[key].item() != expected:
                actual = None if key not in payload else payload[key].item()
                raise ValueError(
                    f"partition metadata mismatch for {key}: {actual!r} != {expected!r}"
                )
        train = [payload[f"train_{i}"] for i in range(num_clients)]
        holdout = [payload[f"holdout_{i}"] for i in range(num_clients)]
        digest = str(payload["digest"].item())
        current = sha256_arrays([array for pair in zip(train, holdout) for array in pair])
        if current != digest:
            raise ValueError(f"partition digest mismatch: {output_path}")
        return ClientPartition(train=train, holdout=holdout, digest=digest)
    partition = create_dirichlet_partition(
        labels=labels,
        num_clients=num_clients,
        alpha=alpha,
        holdout_fraction=holdout_fraction,
        min_client_examples=min_client_examples,
        seed=seed,
        groups=groups,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {
        "digest": np.asarray(partition.digest),
        **{key: np.asarray(value) for key, value in expected_metadata.items()},
    }
    for i, (train_indices, holdout_indices) in enumerate(zip(partition.train, partition.holdout)):
        values[f"train_{i}"] = train_indices
        values[f"holdout_{i}"] = holdout_indices
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".npz", dir=output_path.parent
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary_name, **values)
        os.replace(temporary_name, output_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    metadata = {
        **expected_metadata,
        "sizes": [int(len(x)) for x in partition.train],
        "holdout_sizes": [int(len(x)) for x in partition.holdout],
        "digest": partition.digest,
    }
    atomic_json_dump(metadata, output_path.with_suffix(".json"))
    return partition


def load_task_to_device(
    tokenized: DatasetDict, device: torch.device
) -> tuple[GpuTensorDataset, dict[str, GpuTensorDataset]]:
    train = GpuTensorDataset(dataset_to_tensors(tokenized["train"]), device)
    validation = {
        split: GpuTensorDataset(dataset_to_tensors(tokenized[split]), device)
        for split in tokenized.keys()
        if split.startswith("validation")
    }
    return train, validation
