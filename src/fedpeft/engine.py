from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from torch import nn
from transformers import AutoTokenizer

from .aggregation import weighted_average
from . import baselines
from .baselines import (
    BASELINE_CONSTRUCTION,
    BASELINE_DENSE_METHODS,
    BASELINE_EF_METHODS,
    FEDEX_METHODS,
    FEDSB_CACHE_VERSION,
    FEDSB_METHODS,
    FEDSVD_METHODS,
    FLORA_METHODS,
    MERGE_METHODS,
    ROLORA_METHODS,
)
from .adapters import (
    FixedALinear,
    ModelBundle,
    TensorLayout,
    build_model,
    fixed_basis_numel,
    full_lora_reference_numel,
    model_invariants,
    trainable_parameter_count,
)
from .cab import (
    ACTIVATION_DEFAULT_RANK,
    ACTIVATION_DEFAULT_RIDGE,
    ACTIVATION_METHODS,
    ACTIVATION_RECIPE,
    CAB_CACHE_VERSION,
    CAB_DENSE_METHODS,
    CAB_EF_METHODS,
    CAB_KAPPA_ZERO_METHODS,
    CAB_METHODS,
    CAB_RECIPE,
    EVA_METHODS,
    GEV_METHODS,
    GRADIENT_FREE_METHODS,
    ORACLE_METHODS,
    LayerStatistics,
    activation_sketch,
    anisotropy,
    broadcast_basis,
    client_input_moments,
    client_weight_gradients,
    effective_clients,
    encode_client_payload,
    encode_factor_payload,
    generalized_top_rows,
    gram_refine_mix,
    kappa_from_budget,
    kfac_capture,
    orthonormalize_rows,
    scale_to_reference,
    sketch_matrix,
    subspace_overlap,
    top_eigen_rows,
    truncated_factors,
    whiten_rows,
    WHITENED_METHODS,
)
from .compression import BlockTopKCompressor, TopKCompressor
from .config import SUPPORTED_METHODS, method_task_value, result_path, run_id, task_value
from .dcef import (
    DCEF_CONSTRUCTION,
    DCEF_CONTROL_BYTES,
    DCEF_METHODS,
    DCEF_SERVER_DISCOUNT_METHODS,
    SecantGammaEstimator,
    dcef_server_input,
    dcef_virtual_start,
    encode_control_scalar,
    weighted_gamma,
)
from .data import (
    TASK_NUM_LABELS,
    GpuTensorDataset,
    load_or_create_partition,
    load_task_to_device,
    prepare_tokenized_dataset,
)
from .metrics import (
    aggregate_client_metrics,
    confusion_matrix,
    metrics_from_confusion,
    primary_metric,
    primary_metric_name,
)
from .diagnostics import DIAGNOSTIC_RECIPE_VERSION, SparseDiagnostics
from .utils import atomic_json_dump, cached_directory_sha256, seed_everything, tensor_sha256
from .wire import (
    WirePacket,
    decode_packet,
    decode_sparse_indices,
    encode_dense,
    schema_bytes,
    sparse_k_for_cap,
)


DENSE_METHODS = {
    "fedavg_lora",
    "ffa_lora",
    "lora_xs",
    "random_a_dense",
    "spectral_a_dense",
    "hybrid4_a_dense",
    "hybrid4_a_decay_dense",
    *CAB_DENSE_METHODS,
    *BASELINE_DENSE_METHODS,
}
STABILITY_CANDIDATE_METHODS = {
    "spectral_rot_ef",
    "spectral_block_ef",
    "hybrid4_a_ef",
    "hybrid4_a_decay_ef",
}
BLOCK_TOPK_EF_METHODS = {"spectral_block_ef", "random_block_ef"}
STABILITY_DIAGNOSTIC_METHODS = {
    "spectral_a_ef",
    *STABILITY_CANDIDATE_METHODS,
    "random_block_ef",
}
EF_METHODS = {
    "random_a_ef",
    "spectral_a_ef",
    "full_lora_ef",
    *STABILITY_CANDIDATE_METHODS,
    "random_block_ef",
    *DCEF_METHODS,
    *CAB_EF_METHODS,
    *BASELINE_EF_METHODS,
}
TOPK_NO_EF_METHODS = {"random_a_topk", "spectral_a_topk"}
FULL_LORA_SPARSE_EF_METHODS = {"full_lora_ef", "full_lora_dcef", "full_lora_dcef_client"}


@dataclass(frozen=True)
class RunSpec:
    task: str
    method: str
    seed: int
    wire_ratio: float | None

    @property
    def id(self) -> str:
        return run_id(self.task, self.method, self.seed, self.wire_ratio)


@dataclass
class WireTracker:
    uplink_bytes: int = 0
    downlink_emitted_bytes: int = 0
    downlink_delivered_bytes: int = 0
    uplink_messages: int = 0
    downlink_messages: int = 0
    schema_bytes: int = 0
    max_uplink_message_bytes: int = 0
    max_downlink_message_bytes: int = 0
    # One-time protocol traffic other than the schema (CAB factors and basis).
    setup_bytes: int = 0

    def upload(self, nbytes: int) -> None:
        self.uplink_bytes += int(nbytes)
        self.uplink_messages += 1
        self.max_uplink_message_bytes = max(self.max_uplink_message_bytes, int(nbytes))

    def download(self, nbytes: int, clients: int) -> None:
        self.downlink_emitted_bytes += int(nbytes)
        self.downlink_delivered_bytes += int(nbytes) * int(clients)
        self.downlink_messages += 1
        self.max_downlink_message_bytes = max(self.max_downlink_message_bytes, int(nbytes))

    @property
    def total_delivered_bytes(self) -> int:
        return (
            self.uplink_bytes
            + self.downlink_delivered_bytes
            + self.schema_bytes
            + self.setup_bytes
        )


def _dtype(name: str) -> torch.dtype:
    values = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in values:
        raise ValueError(f"unsupported dtype: {name}")
    return values[name]


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _config_hash(cfg: dict[str, Any], spec: RunSpec) -> str:
    clean = {key: value for key, value in cfg.items() if not key.startswith("_")}
    payload = json.dumps({"config": clean, "run": asdict(spec)}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _layout_hash(bundle: ModelBundle) -> str:
    payload = json.dumps(
        {
            "adapter": list(zip(bundle.adapter_layout.names, bundle.adapter_layout.shapes)),
            "head": list(zip(bundle.head_layout.names, bundle.head_layout.shapes)),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _lora_factor_slices(layout: TensorLayout) -> dict[str, tuple[slice, ...]]:
    """Map canonical flattened-layout ranges to the trainable LoRA factors."""
    ranges: dict[str, list[slice]] = {"lora_A": [], "lora_B": []}
    offset = 0
    for name, count in zip(layout.names, layout.numels):
        stop = offset + count
        for factor in ranges:
            if name.endswith(f".{factor}"):
                ranges[factor].append(slice(offset, stop))
                break
        offset = stop
    return {factor: tuple(values) for factor, values in ranges.items() if values}


def _factor_selected_counts(
    packet: WirePacket,
    factor_slices: dict[str, tuple[slice, ...]],
    factor_index_map: torch.Tensor | None = None,
) -> dict[str, int]:
    """Count transmitted packet indices by factor, including explicit zeros."""
    factors = tuple(factor_slices)
    if factor_index_map is None:
        factor_index_map = _factor_index_map(factor_slices, packet.logical_numel)
    indices = decode_sparse_indices(packet, device="cpu")
    selected_factors = factor_index_map.index_select(0, indices).to(torch.int64)
    if bool((selected_factors < 0).any().item()):
        raise RuntimeError("sparse packet selected an unaudited adapter coordinate")
    counts = torch.bincount(selected_factors, minlength=len(factors))
    return {factor: int(counts[index].item()) for index, factor in enumerate(factors)}


def _factor_index_map(
    factor_slices: dict[str, tuple[slice, ...]], total_numel: int
) -> torch.Tensor:
    """Build an O(1) packet-index lookup from canonical factor slices."""
    mapping = torch.full((total_numel,), -1, dtype=torch.int8, device="cpu")
    for factor_index, regions in enumerate(factor_slices.values()):
        for region in regions:
            mapping[region] = factor_index
    return mapping


def _factor_vector_norm(vector: torch.Tensor, regions: tuple[slice, ...]) -> float:
    values = [vector[region].reshape(-1) for region in regions]
    if not values:
        return 0.0
    flattened = values[0] if len(values) == 1 else torch.cat(values)
    return float(torch.linalg.vector_norm(flattened.float()).item())


def _adapter_factor_audit(
    factor_slices: dict[str, tuple[slice, ...]],
    aggregate_update: torch.Tensor,
    server_residual: torch.Tensor,
    client_residuals: list[torch.Tensor],
    server_selected_counts: dict[str, int],
    client_selected_counts: dict[str, int],
) -> dict[str, dict[str, float | int]]:
    values: dict[str, dict[str, float | int]] = {}
    for factor, regions in factor_slices.items():
        client_total = int(client_selected_counts.get(factor, 0))
        client_norms = [
            _factor_vector_norm(residual, regions) for residual in client_residuals
        ]
        values[factor] = {
            "logical_numel": sum(region.stop - region.start for region in regions),
            "aggregate_update_norm": _factor_vector_norm(aggregate_update, regions),
            "server_residual_norm": _factor_vector_norm(server_residual, regions),
            "server_transmitted_selected_count": int(
                server_selected_counts.get(factor, 0)
            ),
            "client_transmitted_selected_count_total": client_total,
            "client_transmitted_selected_count_mean": client_total
            / max(len(client_residuals), 1),
            "client_residual_norm_mean": sum(client_norms)
            / max(len(client_norms), 1),
        }
    return values


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


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _acquire_run_lock(lock_path: Path, force_unlock: bool) -> None:
    """Acquire a run lock, recovering only provably stale same-host locks."""
    if force_unlock:
        lock_path.unlink(missing_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()}:{os.getpid()}\n".encode())
            os.close(fd)
            return
        except FileExistsError as exc:
            try:
                host, raw_pid = lock_path.read_text(encoding="utf-8").strip().split(":", 1)
                pid = int(raw_pid)
            except (OSError, ValueError):
                raise RuntimeError(f"run has an unreadable lock: {lock_path}") from exc
            if host != socket.gethostname():
                raise RuntimeError(f"run is locked on host {host}: {lock_path}") from exc
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                lock_path.unlink(missing_ok=True)
                continue
            except PermissionError:
                pass
            raise RuntimeError(f"run is already locked by live pid {pid}: {lock_path}") from exc
    raise RuntimeError(f"failed to recover stale run lock: {lock_path}")


def _dense_roundtrip(vector: torch.Tensor):
    packet = encode_dense(vector)
    decoded = decode_packet(packet, vector.device)
    return decoded, packet


def _train_local(
    model: nn.Module,
    dataset: GpuTensorDataset,
    client_indices: torch.Tensor,
    steps: int,
    batch_size: int,
    adapter_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    mixed_precision: str,
    seed: int,
) -> tuple[float, float]:
    # Derive all stochasticity from (partition seed, round, client), so a
    # resumed run follows the same dropout and batch sequence as a clean run.
    torch.manual_seed(seed)
    if client_indices.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    named_trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    trainable = [parameter for _, parameter in named_trainable]
    adapter_parameters = [parameter for name, parameter in named_trainable if not name.startswith("classifier.")]
    head_parameters = [parameter for name, parameter in named_trainable if name.startswith("classifier.")]
    if not adapter_parameters or not head_parameters:
        raise RuntimeError("local optimizer requires both adapter and classifier parameters")
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter_parameters, "lr": adapter_learning_rate},
            {"params": head_parameters, "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    generator = torch.Generator(device=client_indices.device).manual_seed(seed)
    model.train()
    loss_sum = 0.0
    started = time.perf_counter()
    amp_dtype = _dtype(mixed_precision)
    permutation = torch.randperm(client_indices.numel(), generator=generator, device=client_indices.device)
    cursor = 0
    for _ in range(steps):
        pieces = []
        needed = batch_size
        while needed:
            available = permutation.numel() - cursor
            take = min(needed, available)
            pieces.append(permutation[cursor : cursor + take])
            cursor += take
            needed -= take
            if cursor == permutation.numel():
                permutation = torch.randperm(
                    client_indices.numel(), generator=generator, device=client_indices.device
                )
                cursor = 0
        positions = torch.cat(pieces)
        batch = dataset.batch(client_indices.index_select(0, positions))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=client_indices.device.type,
            dtype=amp_dtype,
            enabled=client_indices.device.type == "cuda",
        ):
            output = model(**batch)
            loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite local loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        loss_sum += float(loss.detach().item())
    if client_indices.device.type == "cuda":
        torch.cuda.synchronize(client_indices.device)
    return loss_sum / max(steps, 1), time.perf_counter() - started


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    datasets: dict[str, GpuTensorDataset],
    num_labels: int,
    batch_size: int,
    mixed_precision: str,
) -> tuple[dict[str, dict[str, float]], float]:
    model.eval()
    amp_dtype = _dtype(mixed_precision)
    results: dict[str, dict[str, float]] = {}
    started = time.perf_counter()
    for split, dataset in datasets.items():
        matrix = torch.zeros((num_labels, num_labels), dtype=torch.int64, device=dataset.device)
        loss_weighted = 0.0
        count = 0
        for batch in dataset.sequential_batches(batch_size):
            with torch.autocast(
                device_type=dataset.device.type,
                dtype=amp_dtype,
                enabled=dataset.device.type == "cuda",
            ):
                output = model(**batch)
            predictions = output.logits.argmax(dim=-1)
            matrix += confusion_matrix(predictions, batch["labels"], num_labels)
            size = int(batch["labels"].numel())
            loss_weighted += float(output.loss.item()) * size
            count += size
        values = metrics_from_confusion(matrix)
        values["loss"] = loss_weighted / max(count, 1)
        values["examples"] = count
        results[split] = values
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    return results, time.perf_counter() - started


@torch.inference_mode()
def _evaluate_clients(
    model: nn.Module,
    dataset: GpuTensorDataset,
    client_holdouts: list[torch.Tensor],
    num_labels: int,
    batch_size: int,
    mixed_precision: str,
) -> dict[str, float]:
    model.eval()
    amp_dtype = _dtype(mixed_precision)
    client_metrics: list[dict[str, float]] = []
    for indices in client_holdouts:
        matrix = torch.zeros((num_labels, num_labels), dtype=torch.int64, device=dataset.device)
        for start in range(0, indices.numel(), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = dataset.batch(batch_indices)
            with torch.autocast(
                device_type=dataset.device.type,
                dtype=amp_dtype,
                enabled=dataset.device.type == "cuda",
            ):
                output = model(**batch)
            matrix += confusion_matrix(output.logits.argmax(-1), batch["labels"], num_labels)
        client_metrics.append(metrics_from_confusion(matrix))
    return aggregate_client_metrics(client_metrics)


@torch.inference_mode()
def _evaluate_indices(
    model: nn.Module,
    dataset: GpuTensorDataset,
    indices: torch.Tensor,
    num_labels: int,
    batch_size: int,
    mixed_precision: str,
) -> tuple[dict[str, float], float]:
    model.eval()
    amp_dtype = _dtype(mixed_precision)
    matrix = torch.zeros((num_labels, num_labels), dtype=torch.int64, device=dataset.device)
    loss_weighted = 0.0
    count = 0
    started = time.perf_counter()
    for start in range(0, indices.numel(), batch_size):
        batch = dataset.batch(indices[start : start + batch_size])
        with torch.autocast(
            device_type=dataset.device.type,
            dtype=amp_dtype,
            enabled=dataset.device.type == "cuda",
        ):
            output = model(**batch)
        matrix += confusion_matrix(output.logits.argmax(-1), batch["labels"], num_labels)
        size = int(batch["labels"].numel())
        loss_weighted += float(output.loss.item()) * size
        count += size
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize(dataset.device)
    values = metrics_from_confusion(matrix)
    values["loss"] = loss_weighted / max(count, 1)
    values["examples"] = count
    return values, time.perf_counter() - started


def _combined_dense_roundtrip(adapter: torch.Tensor, head: torch.Tensor):
    adapter_decoded, adapter_packet = _dense_roundtrip(adapter)
    head_decoded, head_packet = _dense_roundtrip(head)
    return adapter_decoded, head_decoded, adapter_packet.nbytes + head_packet.nbytes


def _build_adapter_compressor(
    method: str,
    adapter_layout: TensorLayout,
    adapter_cap: int,
) -> TopKCompressor:
    if method in BLOCK_TOPK_EF_METHODS:
        return BlockTopKCompressor(adapter_layout.numels, adapter_cap)
    return TopKCompressor(adapter_layout.total_numel, adapter_cap)


def _adapter_construction_method(method: str) -> str:
    """Map a compressor-only control to its matched adapter construction."""
    if method in BASELINE_CONSTRUCTION:
        return BASELINE_CONSTRUCTION[method]
    if method in DCEF_CONSTRUCTION:
        return DCEF_CONSTRUCTION[method]
    if method in CAB_DENSE_METHODS:
        # Dense CAB-protocol bases replace the dense random-A placeholder.
        return "random_a_dense"
    if method in CAB_EF_METHODS:
        # CAB replaces the random-A placeholder before anything is hashed.
        return "random_a_ef"
    if method == "random_block_ef":
        return "random_a_ef"
    if method in {
        "hybrid4_a_dense",
        "hybrid4_a_decay_dense",
        "hybrid4_a_ef",
        "hybrid4_a_decay_ef",
    }:
        return "hybrid4_a_ef"
    return method


def _resolve_lr_schedule(
    cfg: dict[str, Any], method: str, total_rounds: int
) -> dict[str, Any]:
    """Resolve the shared per-round LR rule without mutable scheduler state."""
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
    if configured.get("name") != "late_cosine_v1":
        raise ValueError(f"unsupported LR schedule for {method}: {configured}")
    start_fraction = float(configured["start_fraction"])
    final_multiplier = float(configured["final_multiplier"])
    if not 0.0 < start_fraction < 1.0:
        raise ValueError("LR schedule start_fraction must be in (0, 1)")
    if not 0.0 < final_multiplier <= 1.0:
        raise ValueError("LR schedule final_multiplier must be in (0, 1]")
    if list(configured.get("apply_to", [])) != ["adapter", "head"]:
        raise ValueError("late_cosine_v1 must apply to adapter and head together")
    start_round = math.floor(start_fraction * int(total_rounds))
    if start_round >= total_rounds:
        raise ValueError("LR schedule must leave at least one decay round")
    return {
        "name": "late_cosine_v1",
        "start_fraction": start_fraction,
        "start_round": start_round,
        "final_multiplier": final_multiplier,
        "apply_to": ["adapter", "head"],
        "total_rounds": int(total_rounds),
    }


def _lr_multiplier(round_number: int, schedule: dict[str, Any]) -> float:
    """Return the one-indexed, deterministic LR multiplier for a round."""
    total_rounds = int(schedule["total_rounds"])
    if not 1 <= int(round_number) <= total_rounds:
        raise ValueError(
            f"round_number must be in [1, {total_rounds}], got {round_number}"
        )
    if schedule["name"] == "constant":
        return 1.0
    if schedule["name"] != "late_cosine_v1":
        raise ValueError(f"unsupported resolved LR schedule: {schedule['name']}")
    start_round = int(schedule["start_round"])
    if round_number <= start_round:
        return 1.0
    progress = (round_number - start_round) / (total_rounds - start_round)
    floor = float(schedule["final_multiplier"])
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _round_learning_rates(
    base_adapter_learning_rate: float,
    base_head_learning_rate: float,
    round_number: int,
    schedule: dict[str, Any],
) -> dict[str, float | str]:
    multiplier = _lr_multiplier(round_number, schedule)
    return {
        "schedule_name": str(schedule["name"]),
        "multiplier": multiplier,
        "adapter_learning_rate": base_adapter_learning_rate * multiplier,
        "head_learning_rate": base_head_learning_rate * multiplier,
    }


def _build_sparse_diagnostics(
    method: str,
    adapter_layout: TensorLayout,
    rank: int,
    num_clients: int,
    recipe_version: str,
) -> SparseDiagnostics | None:
    if method not in STABILITY_DIAGNOSTIC_METHODS:
        return None
    return SparseDiagnostics(
        adapter_layout.names,
        adapter_layout.shapes,
        adapter_layout.numels,
        rank,
        num_clients,
        recipe_version,
    )


def _compressor_metadata(
    compressor: TopKCompressor | None,
    diagnostics: SparseDiagnostics | None,
) -> dict[str, Any]:
    block_numels: tuple[int, ...] | None = None
    block_quotas: tuple[int, ...] | None = None
    block_quota_hash: str | None = None
    if isinstance(compressor, BlockTopKCompressor):
        block_numels = compressor.block_numels
        block_quotas = compressor.quotas
        payload = json.dumps(
            {"block_numels": block_numels, "quotas": block_quotas},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        block_quota_hash = hashlib.sha256(payload).hexdigest()
    return {
        "compressor_kind": (
            "block_topk"
            if isinstance(compressor, BlockTopKCompressor)
            else "global_topk"
            if compressor is not None
            else "dense"
        ),
        "block_numels": None if block_numels is None else list(block_numels),
        "block_quotas": None if block_quotas is None else list(block_quotas),
        "block_quota_hash": block_quota_hash,
        "expected_sparse_adapter_bytes": (
            None if compressor is None else 21 + 6 * compressor.k
        ),
        "diagnostic_recipe_version": (
            None if diagnostics is None else diagnostics.recipe_version
        ),
        "diagnostic_layout_fingerprint": (
            None if diagnostics is None else diagnostics.layout_fingerprint
        ),
    }


def _restore_sparse_diagnostics(
    diagnostics: SparseDiagnostics | None,
    state: Any,
    completed_round: int,
) -> None:
    if diagnostics is None:
        if state is not None:
            raise RuntimeError(
                "non-diagnostic checkpoint unexpectedly contains stability diagnostics"
            )
        return
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint is missing complete stability diagnostic state")
    try:
        diagnostics.load_state_dict(state)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"stability diagnostic checkpoint is invalid: {exc}") from exc
    if diagnostics.completed_round != int(completed_round):
        raise RuntimeError(
            "stability diagnostic checkpoint round differs from completed_round"
        )


def _consensus_statistics(
    local: list[torch.Tensor],
    corrected: list[torch.Tensor],
    uploads: list[torch.Tensor],
    supports: list[torch.Tensor],
    weights: list[float],
    numel: int,
) -> dict[str, Any]:
    """How much of what clients compute and send survives weighted averaging.

    eta = ||sum_i p_i v_i||^2 / sum_i p_i ||v_i||^2 for v = the fresh local
    update, the EF-corrected vector the client tries to send, and the decoded
    upload.  eta = 1 when all clients agree; about 1/N_eff when their vectors
    are orthogonal.  Also reports how much of the dense consensus the decoded
    aggregate recovers this round, and how much client supports overlap.
    """
    total = float(sum(weights))
    p = [float(w) / total for w in weights]
    n_eff = 1.0 / sum(x * x for x in p)

    def eta(vectors: list[torch.Tensor]) -> tuple[float, torch.Tensor]:
        mean = sum(pi * v.double() for pi, v in zip(p, vectors))
        spread = sum(pi * float(v.double().pow(2).sum().item()) for pi, v in zip(p, vectors))
        value = float(mean.pow(2).sum().item()) / spread if spread > 0 else float("nan")
        return value, mean

    eta_local, local_mean = eta(local)
    eta_corrected, _ = eta(corrected)
    eta_wire, wire_mean = eta(uploads)
    consensus_norm2 = float(local_mean.pow(2).sum().item())
    out: dict[str, Any] = {
        "n_eff": n_eff,
        "eta_local": eta_local,
        "eta_corrected": eta_corrected,
        "eta_wire": eta_wire,
        "eta_floor_orthogonal": 1.0 / n_eff,
        "wire_recovers_local_consensus": float((wire_mean * local_mean).sum().item()) / consensus_norm2
        if consensus_norm2 > 0
        else float("nan"),
    }
    if supports:
        counts = torch.zeros(numel, dtype=torch.int64, device=supports[0].device)
        for index in supports:
            counts[index] += 1
        sizes = [int(index.numel()) for index in supports]
        masks = []
        for index in supports:
            mask = torch.zeros(numel, dtype=torch.bool, device=index.device)
            mask[index] = True
            masks.append(mask)
        jaccards = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                union = int((masks[i] | masks[j]).sum().item())
                if union:
                    jaccards.append(int((masks[i] & masks[j]).sum().item()) / union)
        out.update(
            {
                "support_pairwise_jaccard_mean": sum(jaccards) / len(jaccards) if jaccards else float("nan"),
                "support_union_over_total": int((counts > 0).sum().item()) / max(sum(sizes), 1),
                "support_selected_by_all_clients": int((counts == len(supports)).sum().item()),
            }
        )
    return out


def _cab_settings(cfg: dict[str, Any], task: str) -> dict[str, Any]:
    section = cfg["cab"]
    override = section.get("kappa_override")
    return {
        "gamma": float(task_value(cfg, "cab", "gamma", task)),
        "gamma_source": str(section["gamma_source"]),
        "kappa_multiplier": float(section.get("kappa_multiplier", 1.0)),
        "kappa_override": None if override is None else float(override),
        "client_rank": int(section["client_rank"]),
        "gradient_examples": int(task_value(cfg, "cab", "gradient_examples", task)),
        "gradient_batch_size": int(task_value(cfg, "cab", "gradient_batch_size", task)),
        "activation_rank": int(section.get("activation_rank", ACTIVATION_DEFAULT_RANK)),
        "activation_ridge": float(section.get("activation_ridge", ACTIVATION_DEFAULT_RIDGE)),
        "factor_dtype": str(section.get("factor_dtype", "float16")),
        "basis_dtype": str(section.get("basis_dtype", "float16")),
        "basis_refine": str(section.get("basis_refine", "none")),
        "share_basis": bool(section.get("share_basis", False)),
    }


def _install_cab_basis(
    *,
    cfg: dict[str, Any],
    spec: RunSpec,
    model: nn.Module,
    train_data: Any,
    client_train: list[torch.Tensor],
    client_sizes: list[int],
    sparse_k: int,
    adapter_numel: int,
    device: torch.device,
    cache_path: Path,
    fingerprints: dict[str, str],
) -> tuple[dict[str, Any], int]:
    """Compute the CAB basis, install it into every FixedALinear, return (record, setup bytes).

    The basis is cached beside the run so a resumed run reuses the identical
    bytes instead of recomputing GPU gradients.
    """
    settings = _cab_settings(cfg, spec.task)
    activation = spec.method in ACTIVATION_METHODS
    whitened = spec.method in WHITENED_METHODS
    recipe = ACTIVATION_RECIPE if activation else CAB_RECIPE
    targets = sorted(
        (
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, FixedALinear)
        ),
        key=lambda item: item[0],
    )
    if not targets:
        raise RuntimeError("CAB requires FixedALinear target modules")
    rank = int(targets[0][1].rank)
    n_eff = effective_clients(client_sizes)
    total_size = float(sum(client_sizes))
    weights = [float(size) / total_size for size in client_sizes]
    if settings["kappa_override"] is not None:
        kappa_theory = settings["kappa_override"]
    else:
        kappa_theory = kappa_from_budget(settings["gamma"], adapter_numel, sparse_k, n_eff)
    kappa = (
        0.0
        if spec.method in CAB_KAPPA_ZERO_METHODS
        else kappa_theory * settings["kappa_multiplier"]
    )
    expected_cache = {
        "version": CAB_CACHE_VERSION,
        "fingerprints": fingerprints,
        "settings": settings,
        "kappa": kappa,
        "sparse_k": int(sparse_k),
        "target_names": [name for name, _ in targets],
    }

    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        mismatches = {
            key: (cached.get(key), value)
            for key, value in expected_cache.items()
            if cached.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                f"stale CAB basis cache {cache_path}; remove it to recompute: {mismatches}"
            )
        with torch.no_grad():
            for (_, module), rows in zip(targets, cached["bases"]):
                module.fixed_A.copy_(rows.to(device=module.fixed_A.device, dtype=torch.float32))
                module.basis_audit = {"fixed_basis_recipe": recipe}
        return cached["record"], int(cached["setup_bytes"])

    started = time.perf_counter()
    placeholders = [module.fixed_A.detach().double().cpu().clone() for _, module in targets]
    stats = [
        LayerStatistics(module.base.out_features, module.base.in_features, device)
        for _, module in targets
    ]
    base_weights = [module.base.weight for _, module in targets]
    uplink_bytes: list[int] = []
    reported_energy: list[float] = []
    # Activation-metric methods: modules sharing one input (query and value of a
    # layer) form one group whose input second moment is sketched once.
    group_keys = [name.rsplit(".", 1)[0] for name, _ in targets]
    group_ids = {key: index for index, key in enumerate(dict.fromkeys(group_keys))}
    groups = [group_ids[key] for key in group_keys]
    group_dims: dict[int, int] = {}
    for (_, module), group in zip(targets, groups):
        group_dims.setdefault(group, int(module.base.in_features))
    dims = [group_dims[group] for group in range(len(group_ids))]
    sigma_exact = [torch.zeros(d, d, dtype=torch.float64) for d in dims] if activation else []
    sigma_reported = [torch.zeros(d, d, dtype=torch.float64) for d in dims] if activation else []
    mean_exact = [torch.zeros(d, dtype=torch.float64) for d in dims] if activation else []
    activation_bytes: list[int] = []
    sketch_error: list[float] = []
    refine = whitened and settings["basis_refine"] == "gram"
    # Modules reading one input (query and value of a layer) share the K-FAC input factor
    # Sigma_x; with one shared A the objective sums over them, so each group needs one
    # generalized eigenproblem (sum_m Gbar_m^T Gbar_m, Sigma_x), one broadcast and one Gram.
    share = activation and settings["share_basis"]
    members = [[i for i, g in enumerate(groups) if g == group] for group in range(len(group_ids))]
    # Per-client exact input second moments, kept only for the Gram refinement round.
    client_seconds: list[list[torch.Tensor]] = []
    for client_id, indices in enumerate(client_train):
        if activation:
            seconds, firsts = client_input_moments(
                model,
                [module for _, module in targets],
                groups,
                train_data,
                indices,
                settings["gradient_examples"],
                settings["gradient_batch_size"],
                spec.seed * 1_000_003 - 10_007 + client_id,
                _dtype(cfg["training"]["mixed_precision"]),
            )
            parts: list[torch.Tensor] = []
            for second in seconds:
                parts.extend(activation_sketch(second, settings["activation_rank"]))
            decoded_sketch, sketch_nbytes = encode_client_payload(parts)
            worst = 0.0
            for group, (second, first) in enumerate(zip(seconds, firsts)):
                second = second.double().cpu()
                approx = sketch_matrix(decoded_sketch[2 * group], decoded_sketch[2 * group + 1])
                sigma_exact[group] += weights[client_id] * second
                sigma_reported[group] += weights[client_id] * approx
                mean_exact[group] += weights[client_id] * first.double().cpu()
                worst = max(worst, float(torch.linalg.norm(approx - second) / torch.linalg.norm(second)))
            activation_bytes.append(sketch_nbytes)
            if refine:
                client_seconds.append([second.detach().float().cpu() for second in seconds])
            sketch_error.append(worst)
            del seconds, firsts, parts, decoded_sketch
        gradients = client_weight_gradients(
            model,
            base_weights,
            train_data,
            indices,
            settings["gradient_examples"],
            settings["gradient_batch_size"],
            # Round index -1 of the training seed formula: disjoint from every
            # minibatch seed used by local training.
            spec.seed * 1_000_003 - 10_007 + client_id,
            _dtype(cfg["training"]["mixed_precision"]),
        )
        factors: list[torch.Tensor] = []
        if share:
            # One truncated SVD of the output-stacked gradients of each group: the server
            # only needs sum_m G_m^T G_m, which the stacked factors reproduce.
            for group_members in members:
                stacked = torch.cat([gradients[i] for i in group_members], dim=0)
                factors.extend(truncated_factors(stacked, settings["client_rank"]))
        else:
            for grad in gradients:
                factors.extend(truncated_factors(grad, settings["client_rank"]))
        decoded, nbytes = encode_factor_payload(factors, settings["factor_dtype"])
        if share:
            per_module: list[tuple[torch.Tensor, torch.Tensor]] = [None] * len(targets)  # type: ignore[list-item]
            for group, group_members in enumerate(members):
                left, right = decoded[2 * group], decoded[2 * group + 1]
                offset = 0
                for i in group_members:
                    rows_out = gradients[i].shape[0]
                    per_module[i] = (left[offset : offset + rows_out], right)
                    offset += rows_out
        else:
            per_module = [(decoded[2 * i], decoded[2 * i + 1]) for i in range(len(targets))]
        exact_energy = reported = 0.0
        for index, (stat, grad) in enumerate(zip(stats, gradients)):
            left, right = per_module[index]
            stat.add_exact(grad, weights[client_id])
            stat.add_reported(left, right, weights[client_id])
            exact_energy += float(grad.double().pow(2).sum().item())
            reported += float((left @ right).pow(2).sum().item())
        reported_energy.append(reported / exact_energy if exact_energy > 0 else float("nan"))
        # Gradient-free ablations keep the gradients as simulation-only diagnostics.
        uplink_bytes.append(0 if spec.method in GRADIENT_FREE_METHODS else nbytes)
        del gradients, factors, decoded

    selected_rows: list[torch.Tensor] = []
    layers: list[dict[str, Any]] = []
    ridge = settings["activation_ridge"]
    all_matrices = [stat.matrices(kappa, n_eff) for stat in stats]
    if share:
        # Each module selects from its group's summed objective; members then hold one basis.
        summed = []
        for group_members in members:
            summed.append(
                {key: sum(all_matrices[i][key] for i in group_members) for key in ("reported_m", "exact_m")}
            )
    for index, ((name, module), stat, placeholder) in enumerate(zip(targets, stats, placeholders)):
        matrices = dict(all_matrices[index])
        group = groups[index]
        if share:
            matrices.update(summed[group])
            placeholder = placeholders[members[group][0]]
        oracle = spec.method in ORACLE_METHODS
        # Oracle arms select from the exact statistics (simulation-only upper bound).
        basis_m = matrices["exact_m"] if oracle else matrices["reported_m"]
        basis_sigma = sigma_exact[group] if oracle else (sigma_reported[group] if activation else None)
        if spec.method in GEV_METHODS:
            rows, values = generalized_top_rows(basis_m, basis_sigma, rank, ridge)
            exact_rows, _ = generalized_top_rows(matrices["exact_m"], sigma_exact[group], rank, ridge)
        elif spec.method == "cab0wh_a_ef":
            euclidean, values = top_eigen_rows(matrices["reported_m"], rank)
            rows = whiten_rows(euclidean, sigma_reported[group], ridge)
            exact_rows = whiten_rows(top_eigen_rows(matrices["exact_m"], rank)[0], sigma_exact[group], ridge)
        elif spec.method == "randomwh_a_ef":
            values = top_eigen_rows(matrices["reported_m"], rank)[1]
            rows = whiten_rows(placeholder, sigma_reported[group], ridge)
            exact_rows = whiten_rows(placeholder, sigma_exact[group], ridge)
        elif spec.method in EVA_METHODS:
            # EVA: directions of largest input second moment, row-orthonormal.
            rows, values = top_eigen_rows(sigma_reported[group], rank)
            exact_rows, _ = top_eigen_rows(sigma_exact[group], rank)
        else:
            rows, values = top_eigen_rows(matrices["reported_m"], rank)
            exact_rows, _ = top_eigen_rows(matrices["exact_m"], rank)
        if whitened:
            rows = scale_to_reference(rows, basis_sigma)
            exact_rows = scale_to_reference(exact_rows, sigma_exact[group])
        # Subspace diagnostics use orthonormal representatives of the row spaces.
        selected_space = orthonormalize_rows(rows) if whitened else rows
        exact_space = orthonormalize_rows(exact_rows) if whitened else exact_rows
        consensus_rows, _ = top_eigen_rows(matrices["exact_consensus"], rank)
        spectral_rows = torch.linalg.svd(
            module.base.weight.detach().double().cpu(), full_matrices=False
        )[2][:rank]
        consensus = matrices["exact_consensus"].cpu()
        het = matrices["exact_het"].cpu()
        trace_consensus = float(torch.trace(consensus).item())
        n = module.base.in_features

        def stationary_objective(basis: torch.Tensor) -> float:
            # J(P)/tr(S) under the exact statistics, with this method's kappa
            projector = basis.double().cpu().T @ basis.double().cpu()
            captured = float(torch.trace(projector @ consensus).item())
            floor = float(torch.trace(projector @ het).item())
            if trace_consensus <= 0:
                return float("nan")
            return (trace_consensus - captured + kappa * floor) / trace_consensus

        selected_rows.append(rows)
        layers.append(
            {
                "name": name,
                "positive_eigenvalues": int((values > 0).sum().item()),
                "top_eigenvalues": [float(v) for v in values[: rank + 2].tolist()],
                "het_to_consensus_trace": float(torch.trace(het).item()) / trace_consensus
                if trace_consensus > 0
                else float("nan"),
                "overlap_reported_vs_exact": subspace_overlap(selected_space, exact_space),
                "overlap_vs_consensus_basis": subspace_overlap(selected_space, consensus_rows),
                "overlap_vs_spectral": subspace_overlap(selected_space, spectral_rows),
                "overlap_vs_random_placeholder": subspace_overlap(selected_space, placeholder),
                "random_expected_overlap": rank / n,
                "objective": {
                    "selected": stationary_objective(selected_space),
                    "exact_optimum": stationary_objective(exact_space),
                    "spectral": stationary_objective(spectral_rows),
                    "random": stationary_objective(placeholder),
                },
            }
        )
        if activation:
            # Zero-training predictions under the exact statistics: the share of the
            # optimal one-step K-FAC decrease (kappa = 0 consensus) each basis attains,
            # and how much of its projected input energy is the shared mean.
            second, mean = sigma_exact[group], mean_exact[group]
            gev0_rows, generalized = generalized_top_rows(consensus, second, rank, ridge)
            best = float(generalized[:rank].clamp_min(0.0).sum().item())
            candidates = {
                "selected": rows,
                "random": placeholder,
                "spectral": spectral_rows,
                "consensus": consensus_rows,
                "eva": top_eigen_rows(second, rank)[0],
                "gev0_exact": gev0_rows,
            }
            layers[-1]["kfac_capture"] = {
                key: kfac_capture(value, consensus, second, ridge, best) for key, value in candidates.items()
            }
            layers[-1]["anisotropy"] = {
                key: anisotropy(value, second, mean) for key, value in candidates.items()
            }
            layers[-1]["overlap_vs_eva"] = subspace_overlap(selected_space, candidates["eva"])
    units = [group_members[0] for group_members in members] if share else list(range(len(targets)))
    unit_of = [groups[i] for i in range(len(targets))] if share else list(range(len(targets)))
    used_units, downlink_nbytes = broadcast_basis(
        [selected_rows[i] for i in units], orthonormalize=not whitened, dtype=settings["basis_dtype"]
    )
    used_rows = [used_units[unit_of[i]] for i in range(len(targets))]
    installed_rows = used_rows
    refine_bytes = 0
    if refine:
        # Second, tiny round: every client reports the Gram A Sigma_i A^T (upper triangle)
        # of each broadcast basis on its own inputs plus tr(Sigma_i) per input group, as one
        # FP16 packet; the server returns an r x r mix per basis so that
        # A Sigma_x A^T = (tr(Sigma_x) / d) I holds exactly.  The sketched Sigma only has
        # to pick the subspace; the normalisation no longer depends on its accuracy.
        unit_groups = [groups[i] for i in units]
        uppers = [torch.triu_indices(rows.shape[0], rows.shape[0]) for rows in used_units]
        grams = [torch.zeros(rows.shape[0], rows.shape[0], dtype=torch.float64) for rows in used_units]
        traces = [0.0] * len(group_ids)
        for client_id, seconds_by_group in enumerate(client_seconds):
            parts = []
            for unit, rows in enumerate(used_units):
                gram = rows.double() @ seconds_by_group[unit_groups[unit]].double() @ rows.double().T
                parts.append(gram[uppers[unit][0], uppers[unit][1]])
            parts.append(torch.tensor([float(torch.trace(x.double())) for x in seconds_by_group], dtype=torch.float64))
            decoded_parts, nbytes = encode_client_payload(parts)
            refine_bytes += nbytes
            for unit in range(len(used_units)):
                full = torch.zeros_like(grams[unit])
                full[uppers[unit][0], uppers[unit][1]] = decoded_parts[unit]
                grams[unit] += weights[client_id] * (full + full.T - torch.diag(torch.diagonal(full)))
            for group, value in enumerate(decoded_parts[-1].tolist()):
                traces[group] += weights[client_id] * value
        mixes = [
            gram_refine_mix(grams[unit], traces[unit_groups[unit]], rows.shape[1])
            for unit, rows in enumerate(used_units)
        ]
        decoded_mixes, mix_nbytes = broadcast_basis(mixes, orthonormalize=False, dtype="float16")
        refine_bytes += mix_nbytes * len(client_train)
        installed_units = [mix @ rows.double() for mix, rows in zip(decoded_mixes, used_units)]
        installed_rows = [installed_units[unit_of[i]] for i in range(len(targets))]
        del client_seconds
    with torch.no_grad():
        for (_, module), rows in zip(targets, installed_rows):
            module.fixed_A.copy_(rows.to(device=module.fixed_A.device, dtype=torch.float32))
            module.basis_audit = {"fixed_basis_recipe": recipe}
    if activation:
        for index, (layer, rows) in enumerate(zip(layers, installed_rows)):
            installed = anisotropy(rows, sigma_exact[groups[index]], mean_exact[groups[index]])
            layer["installed"] = installed
    for layer, used, selected in zip(layers, used_rows, selected_rows):
        layer["overlap_broadcast_vs_selected"] = subspace_overlap(
            orthonormalize_rows(used), orthonormalize_rows(selected)
        ) if whitened else subspace_overlap(used, selected)
    setup_bytes = int(
        sum(uplink_bytes) + sum(activation_bytes) + downlink_nbytes * len(client_train) + refine_bytes
    )

    def layer_mean(path: tuple[str, ...]) -> float:
        values = []
        for layer in layers:
            value: Any = layer
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    activation_summary: dict[str, float] = {}
    if activation:
        for key in ("selected", "random", "spectral", "consensus", "eva", "gev0_exact"):
            activation_summary[f"kfac_capture_{key}_mean"] = layer_mean(("kfac_capture", key))
            activation_summary[f"common_mode_{key}_mean"] = layer_mean(("anisotropy", key, "common_mode"))
            activation_summary[f"energy_ratio_{key}_mean"] = layer_mean(("anisotropy", key, "energy_ratio"))
        activation_summary["overlap_vs_eva_mean"] = layer_mean(("overlap_vs_eva",))
        activation_summary["energy_ratio_installed_mean"] = layer_mean(("installed", "energy_ratio"))
        activation_summary["common_mode_installed_mean"] = layer_mean(("installed", "common_mode"))
    record = {
        "recipe": recipe,
        "method": spec.method,
        **settings,
        "kappa_theory": kappa_theory,
        "kappa": kappa,
        "n_eff": n_eff,
        "client_weights": weights,
        "sparse_k": int(sparse_k),
        "adapter_numel": int(adapter_numel),
        "omega": int(adapter_numel) / int(sparse_k) - 1.0,
        "rank": rank,
        "setup_bytes": setup_bytes,
        "uplink_factor_bytes_per_client": uplink_bytes,
        "reported_energy_fraction_by_client": reported_energy,
        "downlink_basis_bytes_per_client": int(downlink_nbytes),
        "activation": {
            "groups": len(group_ids),
            "uplink_sketch_bytes_per_client": activation_bytes,
            "sketch_relative_error_by_client": sketch_error,
            "gradient_upload_counted": spec.method not in GRADIENT_FREE_METHODS,
            "oracle_exact_statistics": spec.method in ORACLE_METHODS,
            "basis_refine": settings["basis_refine"] if whitened else "none",
            "refine_bytes": int(refine_bytes),
            "share_basis": bool(share),
            "broadcast_units": len(units),
        }
        if activation
        else None,
        "seconds": time.perf_counter() - started,
        "summary": {
            "overlap_reported_vs_exact_mean": layer_mean(("overlap_reported_vs_exact",)),
            "reported_energy_fraction_min": min(reported_energy),
            "objective_gap_vs_exact_mean": layer_mean(("objective", "selected"))
            / max(layer_mean(("objective", "exact_optimum")), 1e-300),
            "overlap_vs_consensus_basis_mean": layer_mean(("overlap_vs_consensus_basis",)),
            "overlap_vs_spectral_mean": layer_mean(("overlap_vs_spectral",)),
            "overlap_vs_random_placeholder_mean": layer_mean(("overlap_vs_random_placeholder",)),
            "positive_eigenvalues_min": min(layer["positive_eigenvalues"] for layer in layers),
            "het_to_consensus_trace_mean": layer_mean(("het_to_consensus_trace",)),
            "objective_selected_mean": layer_mean(("objective", "selected")),
            "objective_exact_optimum_mean": layer_mean(("objective", "exact_optimum")),
            "objective_spectral_mean": layer_mean(("objective", "spectral")),
            "objective_random_mean": layer_mean(("objective", "random")),
            **activation_summary,
        },
        "layers": layers,
    }
    _atomic_torch_save(
        {
            **expected_cache,
            "bases": [rows.float() for rows in installed_rows],
            "record": record,
            "setup_bytes": setup_bytes,
        },
        cache_path,
    )
    return record, setup_bytes


def _install_fedsb(
    *,
    cfg: dict[str, Any],
    spec: RunSpec,
    model: nn.Module,
    bundle: ModelBundle,
    train_data: Any,
    client_train: list[torch.Tensor],
    device: torch.device,
    cache_path: Path,
    fingerprints: dict[str, str],
) -> tuple[dict[str, Any], int]:
    """Fed-SB: budget-matched rank, LoRA-SB bases from the summed client gradients, cached
    beside the run so a resume installs identical bytes.  Returns (record, setup bytes)."""
    if spec.wire_ratio is None:
        raise ValueError("Fed-SB takes its rank from the wire budget")
    head_bytes = encode_dense(bundle.head_layout.flatten(model)).nbytes
    reference_bytes = 13 + 2 * full_lora_reference_numel(model, int(cfg["model"]["rank"])) + head_bytes
    adapter_cap = math.floor(reference_bytes * spec.wire_ratio) - head_bytes
    rank = baselines.fedsb_rank(model, adapter_cap)
    learning_rate = float(method_task_value(cfg, "learning_rate", spec.task, spec.method))
    section = cfg.get("baselines", {})
    examples = int(section.get("fedsb_gradient_examples", 1024))
    batch_size = int(section.get("fedsb_gradient_batch_size", 64))
    expected = {
        "version": FEDSB_CACHE_VERSION,
        "fingerprints": fingerprints,
        "rank": rank,
        "learning_rate": learning_rate,
        "examples": examples,
    }
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        mismatches = {key: (cached.get(key), value) for key, value in expected.items() if cached.get(key) != value}
        if mismatches:
            raise RuntimeError(f"stale Fed-SB basis cache {cache_path}; remove it to recompute: {mismatches}")
        bases, payload_bytes = cached["bases"], int(cached["payload_bytes"])
        baselines.load_fedsb_bases(model, bases)
    else:
        targets = sorted(
            (
                (name, module)
                for name, module in model.named_modules()
                if isinstance(module, baselines.LoRAXSLinear)
            ),
            key=lambda item: item[0],
        )
        weights = [module.base.weight for _, module in targets]
        sums = [torch.zeros_like(weight, dtype=torch.float32) for weight in weights]
        for client_id, indices in enumerate(client_train):
            gradients = client_weight_gradients(
                model,
                weights,
                train_data,
                indices,
                examples,
                batch_size,
                spec.seed * 1_000_003 - 10_007 + client_id,
                _dtype(cfg["training"]["mixed_precision"]),
            )
            count = float(min(examples, int(indices.numel())))
            for total, gradient in zip(sums, gradients):
                total += count * gradient
        bases, payload_bytes = baselines.install_fedsb(model, sums, rank, learning_rate)
        del sums
        _atomic_torch_save({**expected, "bases": bases, "payload_bytes": payload_bytes}, cache_path)
    adapter_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_R" in name]
    bundle.adapter_layout = TensorLayout.from_model(model, adapter_names)
    num_clients = len(client_train)
    record = baselines.summary_record(
        spec.method,
        rank=rank,
        adapter_cap_bytes=adapter_cap,
        dense_adapter_bytes=13 + 2 * bundle.adapter_layout.total_numel,
        learning_rate_in_init=learning_rate,
        gradient_examples_per_client=examples,
        basis_broadcast_bytes_per_client=payload_bytes,
        setup_bytes=payload_bytes * num_clients,
        gradient_upload_counted=False,
        assumption="only the basis broadcast is charged (lower bound on Fed-SB setup traffic)",
    )
    return record, payload_bytes * num_clients


def run_experiment(
    cfg: dict[str, Any],
    spec: RunSpec,
    resume: bool = True,
    force_unlock: bool = False,
) -> dict[str, Any]:
    if spec.task not in cfg["data"]["tasks"]:
        raise ValueError(f"task {spec.task} is not configured")
    if spec.method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method: {spec.method}")
    configured_methods = {
        method
        for key in (
            "dense_methods",
            "sparse_methods",
            "diagnostic_dense_methods",
            "diagnostic_sparse_methods",
        )
        for method in cfg["matrix"].get(key, [])
    }
    if spec.method not in configured_methods:
        raise ValueError(f"method {spec.method} is not enabled by this config")
    if spec.seed not in {int(seed) for seed in cfg["data"]["partition_seeds"]}:
        raise ValueError(f"seed {spec.seed} is not enabled by this config")
    if spec.method in DENSE_METHODS and spec.wire_ratio is not None:
        raise ValueError("dense methods must not specify a wire ratio")
    if spec.method not in DENSE_METHODS and spec.wire_ratio is None:
        raise ValueError("sparse methods require a wire ratio")
    if spec.wire_ratio is not None and not any(
        math.isclose(spec.wire_ratio, float(ratio), rel_tol=0.0, abs_tol=1e-12)
        for ratio in cfg["communication"]["wire_ratios"]
    ):
        raise ValueError(f"wire ratio {spec.wire_ratio} is not enabled by this config")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real experiment runner")

    root = Path(cfg["project"]["root"])
    config_hash = _config_hash(cfg, spec)
    source_hash = _source_hash(root)
    run_dir = result_path(cfg, Path(cfg["project"]["output_dir"]) / spec.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    completed_summary: dict[str, Any] | None = None
    if resume and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE":
            completed_summary = existing
    lock_path = run_dir / ".lock"
    _acquire_run_lock(lock_path, force_unlock)

    started_run = time.perf_counter()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    seed_everything(spec.seed)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        clean_config = {key: value for key, value in cfg.items() if not key.startswith("_")}
        resolved_snapshot = {
            "config": clean_config,
            "run": asdict(spec),
            "config_hash": config_hash,
            "source_hash": source_hash,
        }
        snapshot_path = run_dir / "run_config.json"
        if snapshot_path.exists():
            existing_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            if existing_snapshot != resolved_snapshot:
                raise RuntimeError("run_config.json differs from the requested run")
        else:
            atomic_json_dump(resolved_snapshot, snapshot_path)
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True, use_fast=True)
        prep_started = time.perf_counter()
        tokenized = prepare_tokenized_dataset(
            cfg["data"]["root"],
            cfg["project"]["tokenized_dir"],
            spec.task,
            tokenizer,
            int(cfg["data"]["max_length"]),
            int(cfg["data"]["preprocessing_workers"]),
        )
        dataset_hash = _dataset_hash(tokenized)
        train_data, validation_data = load_task_to_device(tokenized, device)
        data_prepare_seconds = time.perf_counter() - prep_started

        labels = np.asarray(tokenized["train"]["labels"], dtype=np.int64)
        partition_path = Path(cfg["project"]["partition_dir"]) / (
            f"{spec.task}__clients-{cfg['data']['num_clients']}__alpha-{cfg['data']['dirichlet_alpha']}"
            f"__holdout-{cfg['data']['local_holdout_fraction']}"
            f"__min-{cfg['data']['min_client_examples']}"
            f"__data-{dataset_hash[:12]}__seed-{spec.seed}.npz"
        )
        partition = load_or_create_partition(
            labels,
            partition_path,
            int(cfg["data"]["num_clients"]),
            float(cfg["data"]["dirichlet_alpha"]),
            float(cfg["data"]["local_holdout_fraction"]),
            int(cfg["data"]["min_client_examples"]),
            spec.seed,
        )
        client_train = [torch.from_numpy(x).to(device=device, dtype=torch.long) for x in partition.train]
        client_holdout = [torch.from_numpy(x).to(device=device, dtype=torch.long) for x in partition.holdout]
        pooled_holdout = torch.cat(client_holdout).sort().values
        evaluation_policy = cfg.get("evaluation", {}).get("policy", "official_validation")

        def evaluate_primary() -> tuple[dict[str, dict[str, float]], float, float]:
            if evaluation_policy == "train_holdout":
                values, elapsed = _evaluate_indices(
                    model,
                    train_data,
                    pooled_holdout,
                    TASK_NUM_LABELS[spec.task],
                    int(cfg["training"]["eval_batch_size"]),
                    cfg["training"]["mixed_precision"],
                )
                split_values = {"train_holdout": values}
                return split_values, primary_metric(spec.task, split_values), elapsed
            split_values, elapsed = _evaluate(
                model,
                validation_data,
                TASK_NUM_LABELS[spec.task],
                int(cfg["training"]["eval_batch_size"]),
                cfg["training"]["mixed_precision"],
            )
            return split_values, primary_metric(spec.task, split_values), elapsed

        # random_block_ef is the diagnostic matched-random control: it uses
        # the exact Random-A construction and differs only in its compressor.
        adapter_method = _adapter_construction_method(spec.method)
        bundle = build_model(
            model_path=cfg["model"]["path"],
            num_labels=TASK_NUM_LABELS[spec.task],
            method=adapter_method,
            rank=int(cfg["model"]["rank"]),
            alpha=float(cfg["model"]["alpha"]),
            dropout=float(cfg["model"]["dropout"]),
            target_suffixes=list(cfg["model"]["target_suffixes"]),
            basis_seed=int(cfg["model"]["basis_seed"]),
            lora_xs_init_std=float(cfg["model"]["lora_xs_init_std"]),
            spectral_cache_path=cfg["project"]["spectral_cache"],
            device=device,
            dtype=_dtype(cfg["model"]["torch_dtype"]),
        )
        model = bundle.model
        model_cache_name = hashlib.sha256(str(Path(cfg["model"]["path"]).resolve()).encode()).hexdigest()[:16]
        model_hash = cached_directory_sha256(
            cfg["model"]["path"],
            root / "artifacts" / "fingerprints" / f"model-{model_cache_name}.json",
        )
        adapter_layout = bundle.adapter_layout
        head_layout = bundle.head_layout
        factor_slices = (
            _lora_factor_slices(adapter_layout)
            if spec.method in FULL_LORA_SPARSE_EF_METHODS
            else {}
        )
        factor_index_map = (
            _factor_index_map(factor_slices, adapter_layout.total_numel)
            if factor_slices
            else None
        )
        if spec.method in FULL_LORA_SPARSE_EF_METHODS:
            if set(factor_slices) != {"lora_A", "lora_B"}:
                raise RuntimeError(f"{spec.method} requires both trainable LoRA factors")
            factor_numel = sum(
                region.stop - region.start
                for regions in factor_slices.values()
                for region in regions
            )
            if factor_numel != adapter_layout.total_numel:
                raise RuntimeError(f"{spec.method} factor slices do not cover the adapter layout")
        cab_record: dict[str, Any] | None = None
        cab_setup_bytes = 0
        if spec.method in CAB_METHODS:
            if spec.method in CAB_DENSE_METHODS:
                # Uncompressed synchronisation: K = d (omega = 0, so kappa = 0).
                cab_sparse_k = adapter_layout.total_numel
            else:
                if spec.wire_ratio is None:
                    raise ValueError("CAB methods require a wire ratio")
                # Same budget arithmetic as the compressor built below; checked there.
                cab_head_bytes = encode_dense(head_layout.flatten(model)).nbytes
                cab_reference_bytes = (
                    13
                    + 2 * full_lora_reference_numel(model, int(cfg["model"]["rank"]))
                    + cab_head_bytes
                )
                cab_sparse_k = sparse_k_for_cap(
                    adapter_layout.total_numel,
                    math.floor(cab_reference_bytes * spec.wire_ratio) - cab_head_bytes,
                )
            cab_record, cab_setup_bytes = _install_cab_basis(
                cfg=cfg,
                spec=spec,
                model=model,
                train_data=train_data,
                client_train=client_train,
                client_sizes=[len(x) for x in partition.train],
                sparse_k=cab_sparse_k,
                adapter_numel=adapter_layout.total_numel,
                device=device,
                cache_path=run_dir / "cab_basis.pt",
                fingerprints={
                    "config_hash": config_hash,
                    "source_hash": source_hash,
                    "model_hash": model_hash,
                    "dataset_hash": dataset_hash,
                    "partition_hash": partition.digest,
                },
            )
        baseline_record: dict[str, Any] | None = None
        baseline_setup_bytes = 0
        if spec.method in FEDSB_METHODS:
            baseline_record, baseline_setup_bytes = _install_fedsb(
                cfg=cfg,
                spec=spec,
                model=model,
                bundle=bundle,
                train_data=train_data,
                client_train=client_train,
                device=device,
                cache_path=run_dir / "fedsb_basis.pt",
                fingerprints={
                    "config_hash": config_hash,
                    "source_hash": source_hash,
                    "model_hash": model_hash,
                    "dataset_hash": dataset_hash,
                    "partition_hash": partition.digest,
                },
            )
            adapter_layout = bundle.adapter_layout
        if (cab_record is not None or baseline_record is not None) and device.type == "cuda":
            # Return the basis-construction gradient buffers to the driver so concurrent
            # runs on the same GPU do not see this run's install-phase peak.
            torch.cuda.empty_cache()
        layout_hash = _layout_hash(bundle)
        invariants_before = model_invariants(model, spec.method)
        if not invariants_before["all_base_parameters_frozen"]:
            raise RuntimeError("unexpected trainable backbone parameters")
        if float(invariants_before["fixed_a_orthogonality_max_error"]) > 1e-4:
            raise RuntimeError("fixed orthogonal A invariant failed")

        global_adapter = adapter_layout.flatten(model)
        global_head = head_layout.flatten(model)
        initial_adapter = global_adapter.clone()
        initial_head = global_head.clone()
        initial_adapter_hash = tensor_sha256(initial_adapter)
        initial_head_hash = tensor_sha256(initial_head)

        if completed_summary is not None:
            completed_expected = {
                "config_hash": config_hash,
                "source_hash": source_hash,
                "model_hash": model_hash,
                "dataset_hash": dataset_hash,
                "partition_hash": partition.digest,
                "layout_hash": layout_hash,
            }
            mismatches: dict[str, tuple[Any, Any]] = {
                key: (completed_summary.get(key), expected)
                for key, expected in completed_expected.items()
                if completed_summary.get(key) != expected
            }
            initialization = completed_summary.get("initialization", {})
            for key, expected in {
                "initial_adapter_hash": initial_adapter_hash,
                "initial_head_hash": initial_head_hash,
            }.items():
                if initialization.get(key) != expected:
                    mismatches[f"initialization.{key}"] = (initialization.get(key), expected)
            completed_basis_hash = (
                completed_summary.get("invariants", {}).get("before", {}).get("fixed_basis_hash")
            )
            if completed_basis_hash != invariants_before["fixed_basis_hash"]:
                mismatches["invariants.before.fixed_basis_hash"] = (
                    completed_basis_hash,
                    invariants_before["fixed_basis_hash"],
                )
            if mismatches:
                raise RuntimeError(
                    "completed run provenance differs from current assets; "
                    f"use a new output directory: {mismatches}"
                )
            return completed_summary

        reference_adapter_numel = full_lora_reference_numel(model, int(cfg["model"]["rank"]))
        reference_adapter_bytes = 13 + 2 * reference_adapter_numel
        reference_head_bytes = encode_dense(global_head).nbytes
        reference_message_bytes = reference_adapter_bytes + reference_head_bytes
        num_clients = int(cfg["data"]["num_clients"])
        adapter_compressor: TopKCompressor | None = None
        if spec.wire_ratio is not None:
            message_cap = math.floor(reference_message_bytes * spec.wire_ratio)
            adapter_cap = message_cap - reference_head_bytes
            if spec.method not in FEDSB_METHODS:
                # Fed-SB's dense payload was sized to fit the cap; every packet is still checked.
                adapter_compressor = _build_adapter_compressor(
                    spec.method, adapter_layout, adapter_cap
                )
        else:
            message_cap = reference_message_bytes
            adapter_cap = reference_adapter_bytes
        # The uplink compressor equals the shared one except for DCEF with the
        # server discount, whose upload also carries one control packet; the
        # sparse budget shrinks so the whole message still fits the same cap.
        uplink_compressor = adapter_compressor
        uplink_control_bytes = 0
        if spec.method in DCEF_SERVER_DISCOUNT_METHODS:
            if adapter_compressor is None:
                raise RuntimeError(f"{spec.method} requires a wire ratio")
            uplink_control_bytes = DCEF_CONTROL_BYTES
            uplink_compressor = _build_adapter_compressor(
                spec.method, adapter_layout, adapter_cap - uplink_control_bytes
            )
        if cab_record is not None and spec.method not in CAB_DENSE_METHODS:
            assert adapter_compressor is not None
            if adapter_compressor.k != int(cab_record["sparse_k"]):
                raise RuntimeError("CAB kappa used a different sparse K than the compressor")

        diagnostic_recipe_version = str(
            cfg.get("stability", {}).get(
                "diagnostic_recipe_version", DIAGNOSTIC_RECIPE_VERSION
            )
        )
        sparse_diagnostics = _build_sparse_diagnostics(
            spec.method,
            adapter_layout,
            int(cfg["model"]["rank"]),
            num_clients,
            diagnostic_recipe_version,
        )
        compressor_metadata = _compressor_metadata(
            adapter_compressor, sparse_diagnostics
        )
        if spec.method in DCEF_METHODS:
            assert uplink_compressor is not None and adapter_compressor is not None
            compressor_metadata = {
                **compressor_metadata,
                "dcef_variant": (
                    "client_virtual_start+server_discount"
                    if spec.method in DCEF_SERVER_DISCOUNT_METHODS
                    else "client_virtual_start"
                ),
                "uplink_sparse_k": uplink_compressor.k,
                "downlink_sparse_k": adapter_compressor.k,
                "uplink_control_bytes": uplink_control_bytes,
            }

        tracker = WireTracker()
        if bool(cfg["communication"].get("schema_once_per_run", True)):
            manifest_size = schema_bytes(
                adapter_layout.names + head_layout.names,
                adapter_layout.shapes + head_layout.shapes,
            )
            tracker.schema_bytes = manifest_size * (1 + int(cfg["data"]["num_clients"]))
        tracker.setup_bytes = cab_setup_bytes + baseline_setup_bytes

        rounds = int(task_value(cfg, "training", "rounds", spec.task))
        local_steps = int(task_value(cfg, "training", "local_steps", spec.task))
        batch_size = int(task_value(cfg, "training", "batch_size", spec.task))
        eval_every = int(task_value(cfg, "training", "eval_every", spec.task))
        if eval_every < 1:
            raise ValueError("eval_every must be at least 1")
        base_adapter_learning_rate = float(
            method_task_value(cfg, "learning_rate", spec.task, spec.method)
        )
        try:
            base_head_learning_rate = float(
                method_task_value(cfg, "head_learning_rate", spec.task, spec.method)
            )
        except KeyError:
            base_head_learning_rate = base_adapter_learning_rate
        learning_rate_schedule = _resolve_lr_schedule(cfg, spec.method, rounds)
        consensus_diagnostics = bool(cfg.get("diagnostics", {}).get("consensus", False))
        client_residuals = [torch.zeros_like(global_adapter) for _ in range(num_clients)]
        server_residual = torch.zeros_like(global_adapter)
        gamma_estimator = (
            SecantGammaEstimator(num_clients)
            if spec.method in DCEF_SERVER_DISCOUNT_METHODS
            else None
        )
        ef_audit_width = 4
        ef_audit_weights: torch.Tensor | None = None
        client_ef_inputs: torch.Tensor | None = None
        client_ef_transmitted: torch.Tensor | None = None
        server_ef_inputs: torch.Tensor | None = None
        server_ef_transmitted: torch.Tensor | None = None
        if spec.method in EF_METHODS:
            audit_generator = torch.Generator(device="cpu").manual_seed(91_827)
            ef_audit_weights = torch.randn(
                ef_audit_width,
                global_adapter.numel(),
                generator=audit_generator,
                dtype=torch.float64,
            )
            ef_audit_weights /= torch.linalg.vector_norm(ef_audit_weights, dim=1, keepdim=True)
            ef_audit_weights = ef_audit_weights.to(device)
            client_ef_inputs = torch.zeros(
                (num_clients, ef_audit_width), device=device, dtype=torch.float64
            )
            client_ef_transmitted = torch.zeros_like(client_ef_inputs)
            server_ef_inputs = torch.zeros(ef_audit_width, device=device, dtype=torch.float64)
            server_ef_transmitted = torch.zeros_like(server_ef_inputs)

        def audit_projection(vector: torch.Tensor) -> torch.Tensor:
            if ef_audit_weights is None:
                raise RuntimeError("EF audit projection requested for a non-EF method")
            return ef_audit_weights @ vector.double()
        completed_round = 0
        prior_runtime_seconds = 0.0
        cumulative_data_prepare_seconds = data_prepare_seconds
        cumulative_spectral_seconds = bundle.spectral_seconds
        resume_count = 0
        checkpoint_path = run_dir / "checkpoint.pt"
        if resume and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            expected_fingerprints = {
                "config_hash": config_hash,
                "source_hash": source_hash,
                "model_hash": model_hash,
                "dataset_hash": dataset_hash,
                "partition_hash": partition.digest,
                "layout_hash": layout_hash,
                "fixed_basis_hash": str(invariants_before["fixed_basis_hash"]),
                "initial_adapter_hash": initial_adapter_hash,
                "initial_head_hash": initial_head_hash,
                "compressor_metadata": compressor_metadata,
                "learning_rate_schedule": learning_rate_schedule,
            }
            mismatches = {
                key: (checkpoint.get(key), expected)
                for key, expected in expected_fingerprints.items()
                if checkpoint.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(f"checkpoint provenance differs: {mismatches}")
            checkpoint_completed_round = int(checkpoint["completed_round"])
            global_adapter = checkpoint["global_adapter"].to(device)
            global_head = checkpoint["global_head"].to(device)
            client_residuals = [x.to(device) for x in checkpoint["client_residuals"]]
            server_residual = checkpoint["server_residual"].to(device)
            baseline_state = checkpoint.get("baseline_state")
            if spec.method in MERGE_METHODS:
                baselines.load_merged_weight_state(model, baseline_state["merged_weights"])
            elif spec.method in FEDSVD_METHODS:
                baselines.load_fixed_a_state(model, baseline_state["fixed_A"])
            elif baseline_state is not None:
                raise RuntimeError("checkpoint unexpectedly contains baseline state")
            if spec.method in EF_METHODS:
                ef_state = checkpoint.get("ef_audit")
                if not isinstance(ef_state, dict):
                    raise RuntimeError("EF checkpoint is missing the cumulative audit ledger")
                client_ef_inputs = ef_state["client_inputs"].to(device)
                client_ef_transmitted = ef_state["client_transmitted"].to(device)
                server_ef_inputs = ef_state["server_inputs"].to(device)
                server_ef_transmitted = ef_state["server_transmitted"].to(device)
            if gamma_estimator is not None:
                try:
                    gamma_estimator.load_state_dict(checkpoint.get("dcef_state"), device)
                except ValueError as exc:
                    raise RuntimeError(f"DCEF checkpoint state is invalid: {exc}") from exc
            elif checkpoint.get("dcef_state") is not None:
                raise RuntimeError("non-DCEF checkpoint unexpectedly contains DCEF state")
            _restore_sparse_diagnostics(
                sparse_diagnostics,
                checkpoint.get("stability_diagnostics"),
                checkpoint_completed_round,
            )
            tracker = WireTracker(**checkpoint["wire_tracker"])
            completed_round = checkpoint_completed_round
            prior_runtime_seconds = float(checkpoint.get("runtime_seconds", 0.0))
            cumulative_data_prepare_seconds += float(checkpoint.get("data_prepare_seconds", 0.0))
            cumulative_spectral_seconds += float(checkpoint.get("spectral_seconds", 0.0))
            resume_count = int(checkpoint.get("resume_count", 0)) + 1

        per_round: list[dict[str, Any]] = []
        for old_path in sorted(run_dir.glob("round-*.json")):
            record = json.loads(old_path.read_text(encoding="utf-8"))
            if int(record["round"]) <= completed_round:
                per_round.append(record)
        restored_rounds = [int(record["round"]) for record in per_round]
        if restored_rounds != list(range(1, completed_round + 1)):
            raise RuntimeError(
                "checkpoint and per-round records disagree: "
                f"checkpoint={completed_round}, records={restored_rounds}"
            )
        for record in per_round:
            expected_learning_rate = _round_learning_rates(
                base_adapter_learning_rate,
                base_head_learning_rate,
                int(record["round"]),
                learning_rate_schedule,
            )
            if record.get("learning_rate") != expected_learning_rate:
                raise RuntimeError(
                    "resumed round learning-rate provenance differs: "
                    f"round={record['round']}, expected={expected_learning_rate}, "
                    f"observed={record.get('learning_rate')}"
                )
        if sparse_diagnostics is not None:
            for record in per_round:
                restored = record.get("diagnostics", {}).get(
                    "stability_diagnostics"
                )
                if not isinstance(restored, dict):
                    raise RuntimeError(
                        "resumed round record is missing stability diagnostics"
                    )
                expected = {
                    "recipe_version": sparse_diagnostics.recipe_version,
                    "layout_fingerprint": sparse_diagnostics.layout_fingerprint,
                    "round": int(record["round"]),
                }
                observed = {key: restored.get(key) for key in expected}
                if observed != expected:
                    raise RuntimeError(
                        "resumed round diagnostic provenance differs: "
                        f"expected={expected}, observed={observed}"
                    )

        for round_index in range(completed_round, rounds):
            round_started = time.perf_counter()
            round_learning_rate = _round_learning_rates(
                base_adapter_learning_rate,
                base_head_learning_rate,
                round_index + 1,
                learning_rate_schedule,
            )
            adapter_compression_seconds = 0.0
            residual_conservation_max = 0.0
            residual_conservation_absolute_max = 0.0
            client_factor_selected_counts = {
                factor: 0 for factor in factor_slices
            }
            server_factor_selected_counts = {
                factor: 0 for factor in factor_slices
            }
            server_transmitted_adapter: torch.Tensor | None = None
            round_baseline: dict[str, Any] = {}
            rolora_mask: torch.Tensor | None = None
            if spec.method in ROLORA_METHODS:
                active_factor = baselines.rolora_factor(round_index)
                rolora_mask = baselines.factor_mask(adapter_layout, active_factor, device)
                baselines.set_trainable_factor(model, active_factor)
                round_baseline["rolora_factor"] = active_factor

            if spec.method == "flasc":
                assert adapter_compressor is not None
                down_result = adapter_compressor.compress(global_adapter)
                start_adapter = down_result.reconstructed
                flasc_mask = torch.zeros_like(global_adapter, dtype=torch.bool)
                flasc_mask[decode_sparse_indices(down_result.packet, global_adapter.device)] = True
                start_head, head_down_packet = _dense_roundtrip(global_head)
                tracker.download(down_result.packet.nbytes + head_down_packet.nbytes, num_clients)
                if down_result.packet.nbytes + head_down_packet.nbytes > message_cap:
                    raise AssertionError("FLASC downlink exceeded the configured wire cap")
                adapter_compression_seconds += down_result.seconds
            else:
                start_adapter = global_adapter.clone()
                start_head = global_head.clone()

            client_adapter_updates: list[torch.Tensor] = []
            client_head_updates: list[torch.Tensor] = []
            client_weights: list[float] = []
            client_losses: list[float] = []
            client_gammas: list[float] = []
            virtual_offset_norms: list[float] = []
            # Opt-in, zero-wire consensus diagnostics (cfg diagnostics.consensus).
            consensus_local: list[torch.Tensor] = []
            consensus_corrected: list[torch.Tensor] = []
            consensus_supports: list[torch.Tensor] = []
            gamma_bar: float | None = None
            train_seconds = 0.0

            for client_id, indices in enumerate(client_train):
                if spec.method in DCEF_METHODS:
                    # Train from the client's virtual model x + e_i.  The upload
                    # (local end - x) then carries (I - Gamma) e_i implicitly.
                    client_start_adapter = dcef_virtual_start(
                        start_adapter, client_residuals[client_id]
                    )
                    virtual_offset_norms.append(
                        float(torch.linalg.vector_norm(client_residuals[client_id]).item())
                    )
                else:
                    client_start_adapter = start_adapter
                adapter_layout.assign(model, client_start_adapter)
                head_layout.assign(model, start_head)
                loss, elapsed = _train_local(
                    model,
                    train_data,
                    indices,
                    local_steps,
                    batch_size,
                    float(round_learning_rate["adapter_learning_rate"]),
                    float(round_learning_rate["head_learning_rate"]),
                    float(cfg["training"]["weight_decay"]),
                    float(cfg["training"]["max_grad_norm"]),
                    cfg["training"]["mixed_precision"],
                    spec.seed * 1_000_003 + round_index * 10_007 + client_id,
                )
                client_losses.append(loss)
                train_seconds += elapsed
                # Local displacement from the training start; for DCEF the start
                # is virtual, so delta + residual = local end - global adapter.
                delta_adapter = adapter_layout.flatten(model) - client_start_adapter
                delta_head = head_layout.flatten(model) - start_head
                decoded_head, head_packet = _dense_roundtrip(delta_head)

                if spec.method in DENSE_METHODS or spec.method in FEDSB_METHODS:
                    if rolora_mask is not None:
                        # RoLoRA sends only the factor it trained (the other one did not move).
                        sent, adapter_packet = _dense_roundtrip(delta_adapter[rolora_mask])
                        decoded_adapter = torch.zeros_like(delta_adapter)
                        decoded_adapter[rolora_mask] = sent
                    else:
                        decoded_adapter, adapter_packet = _dense_roundtrip(delta_adapter)
                    packet_bytes = adapter_packet.nbytes + head_packet.nbytes
                elif spec.method == "flasc" or spec.method in TOPK_NO_EF_METHODS:
                    assert adapter_compressor is not None
                    compressed = adapter_compressor.compress(delta_adapter)
                    decoded_adapter = compressed.reconstructed
                    packet_bytes = compressed.packet.nbytes + head_packet.nbytes
                    adapter_compression_seconds += compressed.seconds
                elif spec.method in EF_METHODS:
                    assert uplink_compressor is not None
                    assert client_ef_inputs is not None
                    assert client_ef_transmitted is not None
                    old_residual = client_residuals[client_id]
                    corrected = delta_adapter + old_residual
                    compressed = uplink_compressor.compress(
                        corrected if rolora_mask is None else corrected * rolora_mask
                    )
                    decoded_adapter = compressed.reconstructed
                    if factor_slices:
                        selected = _factor_selected_counts(
                            compressed.packet, factor_slices, factor_index_map
                        )
                        for factor, count in selected.items():
                            client_factor_selected_counts[factor] += count
                    client_residuals[client_id] = corrected - decoded_adapter
                    if sparse_diagnostics is not None:
                        sparse_diagnostics.observe_client(
                            round_index + 1,
                            client_id,
                            compressed.packet,
                            raw_update=delta_adapter,
                            corrected_update=corrected,
                            transmitted_update=decoded_adapter,
                            next_residual=client_residuals[client_id],
                        )
                    client_ef_inputs[client_id] += audit_projection(delta_adapter)
                    client_ef_transmitted[client_id] += audit_projection(decoded_adapter)
                    ledger_error = (
                        client_ef_inputs[client_id]
                        - client_ef_transmitted[client_id]
                        - audit_projection(client_residuals[client_id])
                    ).abs().max()
                    ledger_scale = client_ef_inputs[client_id].abs().max().clamp_min(1e-12)
                    residual_conservation_absolute_max = max(
                        residual_conservation_absolute_max, float(ledger_error.item())
                    )
                    residual_conservation_max = max(
                        residual_conservation_max, float((ledger_error / ledger_scale).item())
                    )
                    packet_bytes = compressed.packet.nbytes + head_packet.nbytes
                    adapter_compression_seconds += compressed.seconds
                    if gamma_estimator is not None:
                        gamma_value = gamma_estimator.observe(
                            client_id, client_start_adapter, delta_adapter
                        )
                        decoded_gamma, gamma_packet = encode_control_scalar(gamma_value)
                        if gamma_packet.nbytes != uplink_control_bytes:
                            raise AssertionError("DCEF control packet size changed")
                        client_gammas.append(decoded_gamma)
                        packet_bytes += gamma_packet.nbytes
                else:
                    raise ValueError(spec.method)

                tracker.upload(packet_bytes)
                if spec.wire_ratio is not None and packet_bytes > message_cap:
                    raise AssertionError(f"uplink packet {packet_bytes} exceeds cap {message_cap}")
                if consensus_diagnostics:
                    consensus_local.append(delta_adapter.detach().float())
                    consensus_corrected.append(
                        (delta_adapter + old_residual).detach().float()
                        if spec.method in EF_METHODS
                        else delta_adapter.detach().float()
                    )
                    if spec.method not in DENSE_METHODS:
                        consensus_supports.append(
                            decode_sparse_indices(compressed.packet, delta_adapter.device)
                        )
                client_adapter_updates.append(decoded_adapter)
                client_head_updates.append(decoded_head)
                client_weights.append(float(len(partition.train[client_id])))

            if spec.method in ROLORA_METHODS:
                baselines.set_trainable_factor(model, None)
            aggregate_adapter = weighted_average(client_adapter_updates, client_weights)
            aggregate_head = weighted_average(client_head_updates, client_weights)

            if spec.method == "flasc":
                global_adapter = global_adapter + aggregate_adapter
                global_head = start_head + aggregate_head
            else:
                decoded_head_down, head_down_packet = _dense_roundtrip(aggregate_head)
                if spec.method in DENSE_METHODS or spec.method in FEDSB_METHODS:
                    if rolora_mask is not None:
                        sent, adapter_down_packet = _dense_roundtrip(aggregate_adapter[rolora_mask])
                        decoded_adapter_down = torch.zeros_like(aggregate_adapter)
                        decoded_adapter_down[rolora_mask] = sent
                    else:
                        decoded_adapter_down, adapter_down_packet = _dense_roundtrip(aggregate_adapter)
                    global_adapter = start_adapter + decoded_adapter_down
                    packet_bytes = adapter_down_packet.nbytes + head_down_packet.nbytes
                    if spec.method in MERGE_METHODS:
                        client_adapters = [start_adapter + update for update in client_adapter_updates]
                        shared = global_adapter if spec.method in FEDEX_METHODS else None
                        baselines.merge_client_products(
                            model, adapter_layout, client_adapters, client_weights, shared
                        )
                        merge_bytes, merge_encoding = baselines.stacked_downlink_bytes(
                            adapter_layout, client_adapters, client_weights, shared
                        )
                        packet_bytes = merge_bytes + head_down_packet.nbytes
                        round_baseline["downlink_encoding"] = merge_encoding
                        if spec.method in FLORA_METHODS:
                            # The product now lives in the frozen weights; restart from a fresh LoRA.
                            global_adapter = baselines.flora_fresh_adapter(
                                model, adapter_layout, int(cfg["model"]["basis_seed"]), round_index + 1
                            )
                elif spec.method in TOPK_NO_EF_METHODS:
                    assert adapter_compressor is not None
                    compressed = adapter_compressor.compress(aggregate_adapter)
                    global_adapter = start_adapter + compressed.reconstructed
                    packet_bytes = compressed.packet.nbytes + head_down_packet.nbytes
                    adapter_compression_seconds += compressed.seconds
                else:
                    assert adapter_compressor is not None
                    assert server_ef_inputs is not None
                    assert server_ef_transmitted is not None
                    if spec.method in DCEF_SERVER_DISCOUNT_METHODS:
                        # Clients that never saw e_s request gamma_bar * e_s again
                        # in this aggregate; carry only the remainder forward.
                        gamma_bar = weighted_gamma(client_gammas, client_weights)
                        corrected = dcef_server_input(
                            aggregate_adapter, server_residual, gamma_bar
                        )
                        server_ledger_input = aggregate_adapter - gamma_bar * server_residual
                    else:
                        corrected = aggregate_adapter + server_residual
                        server_ledger_input = aggregate_adapter
                    compressed = adapter_compressor.compress(
                        corrected if rolora_mask is None else corrected * rolora_mask
                    )
                    if factor_slices:
                        server_factor_selected_counts = _factor_selected_counts(
                            compressed.packet, factor_slices, factor_index_map
                        )
                    server_residual = corrected - compressed.reconstructed
                    server_transmitted_adapter = compressed.reconstructed
                    if sparse_diagnostics is not None:
                        sparse_diagnostics.observe_server(
                            round_index + 1,
                            compressed.packet,
                            raw_update=aggregate_adapter,
                            corrected_update=corrected,
                            transmitted_update=server_transmitted_adapter,
                            next_residual=server_residual,
                        )
                    server_ef_inputs += audit_projection(server_ledger_input)
                    server_ef_transmitted += audit_projection(compressed.reconstructed)
                    ledger_error = (
                        server_ef_inputs
                        - server_ef_transmitted
                        - audit_projection(server_residual)
                    ).abs().max()
                    ledger_scale = server_ef_inputs.abs().max().clamp_min(1e-12)
                    residual_conservation_absolute_max = max(
                        residual_conservation_absolute_max, float(ledger_error.item())
                    )
                    residual_conservation_max = max(
                        residual_conservation_max, float((ledger_error / ledger_scale).item())
                    )
                    global_adapter = start_adapter + compressed.reconstructed
                    packet_bytes = compressed.packet.nbytes + head_down_packet.nbytes
                    adapter_compression_seconds += compressed.seconds
                global_head = start_head + decoded_head_down
                tracker.download(packet_bytes, num_clients)
                if spec.wire_ratio is not None and packet_bytes > message_cap:
                    raise AssertionError(f"downlink packet {packet_bytes} exceeds cap {message_cap}")
                if spec.method in FEDSVD_METHODS:
                    # Every client refactorises the same B A locally: no extra bytes.  EF residuals
                    # live in B coordinates and move with the basis; the audit ledger restarts from
                    # the rotated residuals so per-round conservation stays checkable.
                    round_baseline["fedsvd_refactor_error"] = baselines.fedsvd_reparameterize(
                        model, adapter_layout, [global_adapter, server_residual, *client_residuals]
                    )
                    if spec.method in EF_METHODS:
                        assert client_ef_inputs is not None and client_ef_transmitted is not None
                        assert server_ef_inputs is not None and server_ef_transmitted is not None
                        for client_id in range(num_clients):
                            client_ef_inputs[client_id] = audit_projection(client_residuals[client_id])
                        client_ef_transmitted.zero_()
                        server_ef_inputs.copy_(audit_projection(server_residual))
                        server_ef_transmitted.zero_()

            should_evaluate = (round_index + 1) % eval_every == 0 or round_index + 1 == rounds
            adapter_layout.assign(model, global_adapter)
            head_layout.assign(model, global_head)
            server_split_metrics: dict[str, dict[str, float]] = {}
            server_primary_metric: float | None = None
            server_eval_seconds = 0.0
            if spec.method == "flasc" and round_index + 1 == rounds:
                # Match the reference implementation's default masked-server
                # evaluation and retain the dense-server value as a diagnostic.
                # This is not labelled client-visible: the mask came from the
                # pre-aggregation downlink while values include this update.
                server_split_metrics, server_primary_metric, server_eval_seconds = evaluate_primary()
            if spec.method == "flasc" and should_evaluate:
                adapter_layout.assign(model, global_adapter * flasc_mask)
            split_metrics: dict[str, dict[str, float]] = {}
            round_primary_metric: float | None = None
            eval_seconds = 0.0
            if should_evaluate:
                split_metrics, round_primary_metric, eval_seconds = evaluate_primary()
            client_metrics = {}
            if round_index + 1 == rounds:
                client_metrics = _evaluate_clients(
                    model,
                    train_data,
                    client_holdout,
                    TASK_NUM_LABELS[spec.task],
                    int(cfg["training"]["eval_batch_size"]),
                    cfg["training"]["mixed_precision"],
                )
            if spec.method == "flasc":
                adapter_layout.assign(model, global_adapter)
            round_diagnostics: dict[str, Any] = {
                "server_residual_norm": float(torch.linalg.vector_norm(server_residual).item()),
                "mean_client_residual_norm": float(
                    sum(torch.linalg.vector_norm(x).item() for x in client_residuals)
                    / len(client_residuals)
                ),
                "residual_conservation_max": residual_conservation_max,
                "residual_conservation_absolute_max": residual_conservation_absolute_max,
                "adapter_update_norm": float(torch.linalg.vector_norm(aggregate_adapter).item()),
                "head_update_norm": float(torch.linalg.vector_norm(aggregate_head).item()),
            }
            if round_baseline:
                round_diagnostics["baseline"] = round_baseline
            if consensus_diagnostics:
                round_diagnostics["consensus"] = _consensus_statistics(
                    consensus_local,
                    consensus_corrected,
                    client_adapter_updates,
                    consensus_supports,
                    client_weights,
                    adapter_layout.total_numel,
                )
            if spec.method in DCEF_METHODS:
                assert uplink_compressor is not None
                round_diagnostics["dcef"] = {
                    "variant": compressor_metadata["dcef_variant"],
                    "virtual_offset_norm_mean": sum(virtual_offset_norms)
                    / len(virtual_offset_norms),
                    "virtual_offset_norm_max": max(virtual_offset_norms),
                    "gamma_hat_by_client": client_gammas if gamma_estimator is not None else None,
                    "gamma_observations_by_client": (
                        list(gamma_estimator.observations) if gamma_estimator is not None else None
                    ),
                    "gamma_bar": gamma_bar,
                    "server_residual_retention": 1.0 if gamma_bar is None else 1.0 - gamma_bar,
                    "uplink_sparse_k": uplink_compressor.k,
                    "uplink_control_bytes": uplink_control_bytes,
                }
            if factor_slices:
                assert adapter_compressor is not None and uplink_compressor is not None
                if sum(client_factor_selected_counts.values()) != (
                    num_clients * uplink_compressor.k
                ):
                    raise RuntimeError("client packet factor counts do not cover sparse support")
                if sum(server_factor_selected_counts.values()) != adapter_compressor.k:
                    raise RuntimeError("server packet factor counts do not cover sparse support")
                round_diagnostics["adapter_factor_audit"] = _adapter_factor_audit(
                    factor_slices=factor_slices,
                    aggregate_update=aggregate_adapter,
                    server_residual=server_residual,
                    client_residuals=client_residuals,
                    server_selected_counts=server_factor_selected_counts,
                    client_selected_counts=client_factor_selected_counts,
                )
            if sparse_diagnostics is not None:
                if server_transmitted_adapter is None:
                    raise RuntimeError(
                        "stability diagnostics are missing the decoded server packet"
                    )
                round_diagnostics["stability_diagnostics"] = (
                    sparse_diagnostics.finish_round(
                        round_index + 1,
                        client_adapter_updates,
                        client_weights,
                        aggregate_adapter,
                        server_transmitted_adapter,
                    )
                )
            record = {
                "round": round_index + 1,
                "primary_metric": round_primary_metric,
                "splits": split_metrics,
                "server_splits": server_split_metrics,
                "server_primary_metric": server_primary_metric,
                "client_metrics": client_metrics,
                "mean_train_loss": sum(client_losses) / len(client_losses),
                "learning_rate": round_learning_rate,
                "wire": asdict(tracker) | {"total_delivered_bytes": tracker.total_delivered_bytes},
                "diagnostics": round_diagnostics,
                "timing": {
                    "train_seconds": train_seconds,
                    "compression_seconds": adapter_compression_seconds,
                    "eval_seconds": eval_seconds + server_eval_seconds,
                    "round_seconds": time.perf_counter() - round_started,
                },
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            }
            atomic_json_dump(record, run_dir / f"round-{round_index + 1:03d}.json")
            per_round.append(record)
            if (round_index + 1) % int(cfg["training"]["checkpoint_every"]) == 0:
                ef_audit_checkpoint = None
                if spec.method in EF_METHODS:
                    assert client_ef_inputs is not None
                    assert client_ef_transmitted is not None
                    assert server_ef_inputs is not None
                    assert server_ef_transmitted is not None
                    ef_audit_checkpoint = {
                        "client_inputs": client_ef_inputs.cpu(),
                        "client_transmitted": client_ef_transmitted.cpu(),
                        "server_inputs": server_ef_inputs.cpu(),
                        "server_transmitted": server_ef_transmitted.cpu(),
                    }
                _atomic_torch_save(
                    {
                        "config_hash": config_hash,
                        "source_hash": source_hash,
                        "model_hash": model_hash,
                        "dataset_hash": dataset_hash,
                        "partition_hash": partition.digest,
                        "layout_hash": layout_hash,
                        "fixed_basis_hash": str(invariants_before["fixed_basis_hash"]),
                        "initial_adapter_hash": initial_adapter_hash,
                        "initial_head_hash": initial_head_hash,
                        "compressor_metadata": compressor_metadata,
                        "learning_rate_schedule": learning_rate_schedule,
                        "completed_round": round_index + 1,
                        "global_adapter": global_adapter.cpu(),
                        "global_head": global_head.cpu(),
                        "client_residuals": [x.cpu() for x in client_residuals],
                        "server_residual": server_residual.cpu(),
                        "baseline_state": {"merged_weights": baselines.merged_weight_state(model)}
                        if spec.method in MERGE_METHODS
                        else {"fixed_A": baselines.fixed_a_state(model)}
                        if spec.method in FEDSVD_METHODS
                        else None,
                        "wire_tracker": asdict(tracker),
                        "ef_audit": ef_audit_checkpoint,
                        "dcef_state": None
                        if gamma_estimator is None
                        else gamma_estimator.state_dict(),
                        "stability_diagnostics": None
                        if sparse_diagnostics is None
                        else sparse_diagnostics.state_dict(),
                        "runtime_seconds": prior_runtime_seconds + time.perf_counter() - started_run,
                        "data_prepare_seconds": cumulative_data_prepare_seconds,
                        "spectral_seconds": cumulative_spectral_seconds,
                        "resume_count": resume_count,
                    },
                    checkpoint_path,
                )

        final = per_round[-1]
        if final["primary_metric"] is None:
            raise RuntimeError("the final round must include an evaluation")
        invariants_after = model_invariants(model, spec.method)
        if (
            invariants_before["fixed_basis_hash"] != invariants_after["fixed_basis_hash"]
            and spec.method not in FEDSVD_METHODS
        ):
            raise RuntimeError("a frozen basis changed during training")
        summary = {
            "schema_version": 2,
            "status": "COMPLETE",
            "run": asdict(spec),
            "run_id": spec.id,
            "metrics": {
                "schema_version": 1,
                "primary_metric_name": primary_metric_name(spec.task),
                "mrpc_qqp_positive_label": 1
                if spec.task in {"mrpc", "qqp"}
                else None,
            },
            "stability_protocol": clean_config.get("stability"),
            "config_hash": config_hash,
            "source_hash": source_hash,
            "layout_hash": layout_hash,
            "partition_hash": partition.digest,
            "dataset_hash": dataset_hash,
            "model_hash": model_hash,
            "model": {
                "path": cfg["model"]["path"],
                "rank": int(cfg["model"]["rank"]),
                "target_modules": bundle.target_names,
                "trainable_parameters": trainable_parameter_count(model),
                "adapter_parameters": adapter_layout.total_numel,
                "head_parameters": head_layout.total_numel,
                "reference_full_lora_parameters": reference_adapter_numel + head_layout.total_numel,
            },
            "wire_budget": {
                "requested_ratio": spec.wire_ratio,
                "reference_message_bytes": reference_message_bytes,
                "message_cap_bytes": message_cap,
                "adapter_cap_bytes": adapter_cap,
                "sparse_k": None if adapter_compressor is None else adapter_compressor.k,
                "adapter_density": None
                if adapter_compressor is None
                else adapter_compressor.k / adapter_layout.total_numel,
                "expected_sparse_message_bytes": None
                if compressor_metadata["expected_sparse_adapter_bytes"] is None
                else int(compressor_metadata["expected_sparse_adapter_bytes"])
                + reference_head_bytes,
                **compressor_metadata,
            },
            "dcef": None
            if spec.method not in DCEF_METHODS
            else {
                "variant": compressor_metadata["dcef_variant"],
                "construction_method": _adapter_construction_method(spec.method),
                "uplink_sparse_k": compressor_metadata["uplink_sparse_k"],
                "downlink_sparse_k": compressor_metadata["downlink_sparse_k"],
                "uplink_control_bytes": compressor_metadata["uplink_control_bytes"],
                "final_gamma_hat_by_client": None
                if gamma_estimator is None
                else [gamma_estimator.estimate(i) for i in range(num_clients)],
                "gamma_observations_by_client": None
                if gamma_estimator is None
                else list(gamma_estimator.observations),
                "gamma_prior": None if gamma_estimator is None else gamma_estimator.prior,
            },
            "setup_communication": {
                "assumed_basis_bytes": 0,
                "basis_hash_handshake_bytes_per_client": 32
                if fixed_basis_numel(model) > 0
                else 0,
                "fallback_fp16_basis_bytes_per_client": (
                    13 + 2 * fixed_basis_numel(model) if fixed_basis_numel(model) > 0 else 0
                ),
                "assumption": (
                    "fixed basis reconstructed locally and accepted only after SHA-256 agreement"
                    if fixed_basis_numel(model) > 0
                    else "no fixed basis"
                ),
            }
            if cab_record is None and baseline_record is None
            else {
                "assumed_basis_bytes": int(baseline_record["basis_broadcast_bytes_per_client"]),
                "basis_hash_handshake_bytes_per_client": 0,
                "fallback_fp16_basis_bytes_per_client": 0,
                "setup_bytes_in_delivered_total": int(baseline_record["setup_bytes"]),
                "assumption": str(baseline_record["assumption"]),
            }
            if cab_record is None
            else {
                "assumed_basis_bytes": int(cab_record["downlink_basis_bytes_per_client"]),
                "basis_hash_handshake_bytes_per_client": 0,
                "fallback_fp16_basis_bytes_per_client": 0,
                "setup_bytes_in_delivered_total": int(cab_record["setup_bytes"]),
                "assumption": (
                    "CAB: one FP16 rank-k gradient factorisation uploaded per client and the basis "
                    "broadcast once as FP16; both counted in total_delivered_bytes"
                ),
            },
            "cab": cab_record,
            "baseline": baseline_record
            if baseline_record is not None
            else {"method": spec.method, "construction": BASELINE_CONSTRUCTION[spec.method]}
            if spec.method in BASELINE_CONSTRUCTION
            else None,
            "data": {
                "task": spec.task,
                "num_clients": num_clients,
                "partition_seed": spec.seed,
                "dirichlet_alpha": float(cfg["data"]["dirichlet_alpha"]),
                "client_train_sizes": [len(x) for x in partition.train],
                "client_holdout_sizes": [len(x) for x in partition.holdout],
                "evaluation_policy": evaluation_policy,
            },
            "training": {
                "rounds": rounds,
                "local_steps": local_steps,
                "batch_size": batch_size,
                "eval_batch_size": int(cfg["training"]["eval_batch_size"]),
                "eval_every": eval_every,
                "adapter_learning_rate": base_adapter_learning_rate,
                "head_learning_rate": base_head_learning_rate,
                "learning_rate_schedule": learning_rate_schedule,
                "weight_decay": float(cfg["training"]["weight_decay"]),
                "max_grad_norm": float(cfg["training"]["max_grad_norm"]),
                "mixed_precision": cfg["training"]["mixed_precision"],
            },
            "initialization": {
                "initial_adapter_norm": float(torch.linalg.vector_norm(initial_adapter).item()),
                "initial_adapter_hash": initial_adapter_hash,
                "initial_head_norm": float(torch.linalg.vector_norm(initial_head).item()),
                "initial_head_hash": initial_head_hash,
                "spectral_seconds": cumulative_spectral_seconds,
                "data_prepare_seconds": cumulative_data_prepare_seconds,
                "resume_count": resume_count,
            },
            "invariants": {
                "before": invariants_before,
                "after": invariants_after,
                "basis_unchanged": invariants_before["fixed_basis_hash"]
                == invariants_after["fixed_basis_hash"],
                "finite_metrics": all(
                    round_value["primary_metric"] is None
                    or math.isfinite(float(round_value["primary_metric"]))
                    for round_value in per_round
                ),
                "wire_cap_respected": spec.wire_ratio is None
                or (
                    tracker.max_uplink_message_bytes <= message_cap
                    and tracker.max_downlink_message_bytes <= message_cap
                ),
                "ef_conservation_max": max(
                    (float(round_value["diagnostics"]["residual_conservation_max"]) for round_value in per_round),
                    default=0.0,
                ),
                "ef_conservation_absolute_max": max(
                    (
                        float(
                            round_value["diagnostics"][
                                "residual_conservation_absolute_max"
                            ]
                        )
                        for round_value in per_round
                    ),
                    default=0.0,
                ),
                "ef_audit": "four deterministic random projections accumulated across rounds",
            },
            "final": final,
            "rounds": per_round,
            "runtime_seconds": prior_runtime_seconds + time.perf_counter() - started_run,
            "environment": {
                "hostname": socket.gethostname(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
        }
        atomic_json_dump(summary, summary_path)
        return summary
    except Exception as exc:
        atomic_json_dump(
            {
                "schema_version": 2,
                "status": "FAILED",
                "run": asdict(spec),
                "run_id": spec.id,
                "config_hash": config_hash,
                "source_hash": source_hash,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            run_dir / "failure.json",
        )
        raise
    finally:
        lock_path.unlink(missing_ok=True)
