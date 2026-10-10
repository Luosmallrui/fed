from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .wire import WirePacket, decode_packet, decode_sparse_indices


DIAGNOSTIC_RECIPE_VERSION = "v1"


def _quantiles(
    values: Sequence[float] | torch.Tensor, include_p10: bool = False
) -> dict[str, float] | None:
    if isinstance(values, torch.Tensor):
        tensor = values.detach().to(device="cpu", dtype=torch.float64).flatten()
    else:
        if not values:
            return None
        tensor = torch.tensor(list(values), dtype=torch.float64)
    if tensor.numel() == 0:
        return None
    result = {
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "max": float(tensor.max().item()),
    }
    if include_p10:
        result = {
            "min": float(tensor.min().item()),
            "p10": float(torch.quantile(tensor, 0.10).item()),
            **result,
        }
    return result


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    if left.numel() != right.numel():
        raise ValueError("cosine vectors must have the same length")
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        return 0.0
    return float(torch.dot(left, right).div(left_norm * right_norm).item())


@dataclass(frozen=True)
class DiagnosticLayout:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    numels: tuple[int, ...]
    rank: int
    offsets: tuple[int, ...]
    fingerprint: str

    @classmethod
    def build(
        cls,
        names: Sequence[str],
        shapes: Sequence[Sequence[int]],
        numels: Sequence[int],
        rank: int,
    ) -> "DiagnosticLayout":
        canonical_names = tuple(str(value) for value in names)
        canonical_shapes = tuple(tuple(int(item) for item in value) for value in shapes)
        canonical_numels = tuple(int(value) for value in numels)
        rank = int(rank)
        if not canonical_names or not (
            len(canonical_names) == len(canonical_shapes) == len(canonical_numels)
        ):
            raise ValueError("diagnostic layout fields must be non-empty and aligned")
        if list(canonical_names) != sorted(canonical_names) or len(set(canonical_names)) != len(
            canonical_names
        ):
            raise ValueError("diagnostic tensor names must be unique and in canonical order")
        if rank <= 0:
            raise ValueError("rank must be positive")
        for name, shape, count in zip(canonical_names, canonical_shapes, canonical_numels):
            if not name.endswith(".lora_B"):
                raise ValueError(f"diagnostic layout contains a non-LoRA-B tensor: {name}")
            if not shape or shape[-1] != rank:
                raise ValueError(f"tensor {name} does not use rank as its final dimension")
            if math.prod(shape) != count or count <= 0:
                raise ValueError(f"tensor {name} has an inconsistent shape/numel")
        offsets = [0]
        for count in canonical_numels:
            offsets.append(offsets[-1] + count)
        payload = {
            "names": canonical_names,
            "shapes": canonical_shapes,
            "numels": canonical_numels,
            "rank": rank,
            "recipe_version": DIAGNOSTIC_RECIPE_VERSION,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return cls(
            canonical_names,
            canonical_shapes,
            canonical_numels,
            rank,
            tuple(offsets),
            fingerprint,
        )

    @property
    def total_numel(self) -> int:
        return self.offsets[-1]


class _SupportHistory:
    def __init__(self, logical_numel: int):
        self.previous = torch.zeros(logical_numel, dtype=torch.bool)
        self.ever = torch.zeros(logical_numel, dtype=torch.bool)
        self.last_selected = torch.full((logical_numel,), -1, dtype=torch.int32)
        self.has_previous = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "previous": self.previous.clone(),
            "ever": self.ever.clone(),
            "last_selected": self.last_selected.clone(),
            "has_previous": self.has_previous,
        }

    def load_state_dict(self, state: dict[str, Any], completed_round: int) -> None:
        required = {"previous", "ever", "last_selected", "has_previous"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("support diagnostic checkpoint is incomplete")
        previous = state["previous"]
        ever = state["ever"]
        last_selected = state["last_selected"]
        if not all(isinstance(value, torch.Tensor) for value in (previous, ever, last_selected)):
            raise ValueError("support diagnostic checkpoint tensors are malformed")
        expected_shape = self.previous.shape
        if previous.shape != expected_shape or ever.shape != expected_shape or last_selected.shape != expected_shape:
            raise ValueError("support diagnostic checkpoint shape differs from the layout")
        previous = previous.detach().to(device="cpu", dtype=torch.bool).clone()
        ever = ever.detach().to(device="cpu", dtype=torch.bool).clone()
        last_selected = last_selected.detach().to(device="cpu", dtype=torch.int32).clone()
        if bool((previous & ~ever).any().item()):
            raise ValueError("previous support is not a subset of cumulative support")
        if bool((last_selected < -1).any().item()) or bool(
            (last_selected > completed_round).any().item()
        ):
            raise ValueError("support diagnostic checkpoint contains an invalid selection round")
        if not torch.equal(last_selected >= 0, ever):
            raise ValueError("selection ages disagree with cumulative support")
        has_previous = state["has_previous"]
        if not isinstance(has_previous, bool) or has_previous != (completed_round > 0):
            raise ValueError("support diagnostic checkpoint has an invalid previous-support flag")
        self.previous = previous
        self.ever = ever
        self.last_selected = last_selected
        self.has_previous = has_previous


class SparseDiagnostics:
    """Checkpointable, zero-wire sparse-update diagnostics.

    All support statistics are derived from encoded packet indices.  The
    returned dictionaries contain only JSON-compatible values; ``state_dict``
    contains CPU tensors intended for the ordinary experiment checkpoint.
    """

    def __init__(
        self,
        tensor_names: Sequence[str],
        tensor_shapes: Sequence[Sequence[int]],
        tensor_numels: Sequence[int],
        rank: int,
        num_clients: int,
        recipe_version: str = DIAGNOSTIC_RECIPE_VERSION,
    ):
        if recipe_version != DIAGNOSTIC_RECIPE_VERSION:
            raise ValueError(
                f"unsupported diagnostic recipe {recipe_version!r}; "
                f"expected {DIAGNOSTIC_RECIPE_VERSION!r}"
            )
        if isinstance(num_clients, bool) or int(num_clients) != num_clients or int(num_clients) <= 0:
            raise ValueError("num_clients must be a positive integer")
        self.recipe_version = recipe_version
        self.layout = DiagnosticLayout.build(
            tensor_names, tensor_shapes, tensor_numels, rank
        )
        self.num_clients = int(num_clients)
        self.completed_round = 0
        self._client_support = [
            _SupportHistory(self.layout.total_numel) for _ in range(self.num_clients)
        ]
        self._server_support = _SupportHistory(self.layout.total_numel)
        self._pending_clients: dict[int, dict[str, Any]] = {}
        self._pending_server: dict[str, Any] | None = None
        self._previous_aggregate: torch.Tensor | None = None
        self._previous_transmitted: torch.Tensor | None = None
        self._aggregate_norm_ema: float | None = None
        self._transmitted_norm_ema: float | None = None

        tensor_ids = torch.empty(self.layout.total_numel, dtype=torch.int64)
        rank_ids = torch.empty(self.layout.total_numel, dtype=torch.int64)
        for tensor_id, (start, stop) in enumerate(
            zip(self.layout.offsets[:-1], self.layout.offsets[1:])
        ):
            tensor_ids[start:stop] = tensor_id
            rank_ids[start:stop] = torch.arange(stop - start, dtype=torch.int64) % self.layout.rank
        self._tensor_ids = tensor_ids
        self._rank_ids = rank_ids

    @property
    def layout_fingerprint(self) -> str:
        return self.layout.fingerprint

    def _check_vector(self, value: torch.Tensor, label: str) -> torch.Tensor:
        flat = value.detach().float().flatten()
        if flat.numel() != self.layout.total_numel:
            raise ValueError(
                f"{label} has {flat.numel()} values, expected {self.layout.total_numel}"
            )
        if not bool(torch.isfinite(flat).all().item()):
            raise ValueError(f"{label} contains a non-finite value")
        return flat

    @staticmethod
    def _norm(value: torch.Tensor) -> float:
        return float(torch.linalg.vector_norm(value.float()).item())

    def _group_counts(self, support: torch.Tensor) -> tuple[list[int], list[int], list[list[int]]]:
        indices = support.nonzero().flatten()
        tensor_counts = torch.bincount(
            self._tensor_ids.index_select(0, indices), minlength=len(self.layout.names)
        )
        rank_counts = torch.bincount(
            self._rank_ids.index_select(0, indices), minlength=self.layout.rank
        )
        combined = self._tensor_ids.index_select(0, indices) * self.layout.rank
        combined += self._rank_ids.index_select(0, indices)
        tensor_rank = torch.bincount(
            combined, minlength=len(self.layout.names) * self.layout.rank
        ).reshape(len(self.layout.names), self.layout.rank)
        return (
            [int(value) for value in tensor_counts.tolist()],
            [int(value) for value in rank_counts.tolist()],
            [[int(value) for value in row] for row in tensor_rank.tolist()],
        )

    @staticmethod
    def _jaccard(current: torch.Tensor, previous: torch.Tensor) -> float | None:
        union = int((current | previous).sum().item())
        if union == 0:
            return None
        return int((current & previous).sum().item()) / union

    @staticmethod
    def _age_summary(values: torch.Tensor) -> dict[str, float]:
        result = _quantiles(values)
        assert result is not None
        return result

    def _support_observation(
        self,
        history: _SupportHistory,
        indices: torch.Tensor,
        round_index: int,
    ) -> dict[str, Any]:
        current = torch.zeros(self.layout.total_numel, dtype=torch.bool)
        current[indices] = True
        previous = history.previous
        has_previous = history.has_previous
        history.ever |= current
        history.last_selected[indices] = round_index
        ages = torch.where(
            history.last_selected >= 0,
            round_index - history.last_selected,
            torch.full_like(history.last_selected, round_index),
        )

        tensor_counts, rank_counts, tensor_rank_counts = self._group_counts(current)
        coverage_by_tensor: dict[str, float] = {}
        age_by_tensor: dict[str, dict[str, float]] = {}
        jaccard_by_tensor: dict[str, float | None] = {}
        for name, start, stop in zip(
            self.layout.names, self.layout.offsets[:-1], self.layout.offsets[1:]
        ):
            coverage_by_tensor[name] = float(history.ever[start:stop].float().mean().item())
            age_by_tensor[name] = self._age_summary(ages[start:stop])
            jaccard_by_tensor[name] = (
                self._jaccard(current[start:stop], previous[start:stop])
                if has_previous
                else None
            )

        coverage_by_rank: list[float] = []
        age_by_rank: list[dict[str, float]] = []
        jaccard_by_rank: list[float | None] = []
        for rank_index in range(self.layout.rank):
            mask = self._rank_ids == rank_index
            coverage_by_rank.append(float(history.ever[mask].float().mean().item()))
            age_by_rank.append(self._age_summary(ages[mask]))
            jaccard_by_rank.append(
                self._jaccard(current[mask], previous[mask]) if has_previous else None
            )

        observation = {
            "selected_count": int(indices.numel()),
            "selected_by_tensor": dict(zip(self.layout.names, tensor_counts)),
            "selected_by_rank": rank_counts,
            "selected_by_tensor_rank": dict(zip(self.layout.names, tensor_rank_counts)),
            "zero_support_tensor_count": sum(value == 0 for value in tensor_counts),
            "cumulative_coverage": {
                "overall": float(history.ever.float().mean().item()),
                "by_tensor": coverage_by_tensor,
                "by_rank": coverage_by_rank,
            },
            "selection_age": {
                "overall": self._age_summary(ages),
                "by_tensor": age_by_tensor,
                "by_rank": age_by_rank,
            },
            "adjacent_support_jaccard": {
                "overall": self._jaccard(current, previous) if has_previous else None,
                "by_tensor": jaccard_by_tensor,
                "by_rank": jaccard_by_rank,
            },
        }
        history.previous = current
        history.has_previous = True
        return observation

    def _residual_observation(
        self,
        raw_update: torch.Tensor,
        corrected_update: torch.Tensor,
        transmitted_update: torch.Tensor,
        next_residual: torch.Tensor,
    ) -> dict[str, Any]:
        raw = self._check_vector(raw_update, "raw_update")
        corrected = self._check_vector(corrected_update, "corrected_update")
        transmitted = self._check_vector(transmitted_update, "transmitted_update")
        residual = self._check_vector(next_residual, "next_residual")

        def values(start: int, stop: int) -> dict[str, float]:
            raw_norm = self._norm(raw[start:stop])
            corrected_norm = self._norm(corrected[start:stop])
            residual_norm = self._norm(residual[start:stop])
            conservation = float(
                (corrected[start:stop] - transmitted[start:stop] - residual[start:stop])
                .abs()
                .max()
                .item()
            )
            return {
                "raw_update_norm": raw_norm,
                "corrected_update_norm": corrected_norm,
                "next_residual_norm": residual_norm,
                "residual_to_update_ratio": residual_norm / max(raw_norm, 1e-12),
                "drained_residual_energy": 1.0
                - residual_norm * residual_norm / max(corrected_norm * corrected_norm, 1e-24),
                "ef_conservation_max_abs": conservation,
            }

        return {
            "overall": values(0, self.layout.total_numel),
            "by_tensor": {
                name: values(start, stop)
                for name, start, stop in zip(
                    self.layout.names,
                    self.layout.offsets[:-1],
                    self.layout.offsets[1:],
                )
            },
        }

    def _observe(
        self,
        history: _SupportHistory,
        round_index: int,
        packet: WirePacket,
        raw_update: torch.Tensor,
        corrected_update: torch.Tensor,
        transmitted_update: torch.Tensor,
        next_residual: torch.Tensor,
    ) -> dict[str, Any]:
        if round_index != self.completed_round + 1:
            raise ValueError(
                f"diagnostics expected round {self.completed_round + 1}, got {round_index}"
            )
        if packet.kind != "sparse" or packet.logical_numel != self.layout.total_numel:
            raise ValueError("diagnostics require a sparse packet matching the adapter layout")
        indices = decode_sparse_indices(packet, device="cpu")
        if indices.numel() != packet.transmitted_numel:
            raise ValueError("packet support count differs from transmitted_numel")
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError("sparse diagnostic packet contains duplicate indices")
        transmitted = self._check_vector(transmitted_update, "transmitted_update")
        decoded = decode_packet(packet, device="cpu")
        if not torch.equal(decoded, transmitted.detach().to(device="cpu", dtype=torch.float32)):
            raise ValueError("transmitted_update differs from the actual packet payload")
        support = self._support_observation(history, indices, round_index)
        support["selected_values_rounded_to_zero"] = int(
            (decoded.index_select(0, indices) == 0).sum().item()
        )
        return {
            "support": support,
            "residual": self._residual_observation(
                raw_update,
                corrected_update,
                transmitted,
                next_residual,
            ),
        }

    def observe_client(
        self,
        round_index: int,
        client_id: int,
        packet: WirePacket,
        raw_update: torch.Tensor,
        corrected_update: torch.Tensor,
        transmitted_update: torch.Tensor,
        next_residual: torch.Tensor,
    ) -> dict[str, Any]:
        if not 0 <= int(client_id) < self.num_clients:
            raise ValueError(f"invalid client_id: {client_id}")
        client_id = int(client_id)
        if client_id in self._pending_clients:
            raise ValueError(f"client {client_id} was observed twice in round {round_index}")
        observation = self._observe(
            self._client_support[client_id],
            round_index,
            packet,
            raw_update,
            corrected_update,
            transmitted_update,
            next_residual,
        )
        self._pending_clients[client_id] = observation
        return observation

    def observe_server(
        self,
        round_index: int,
        packet: WirePacket,
        raw_update: torch.Tensor,
        corrected_update: torch.Tensor,
        transmitted_update: torch.Tensor,
        next_residual: torch.Tensor,
    ) -> dict[str, Any]:
        if self._pending_server is not None:
            raise ValueError(f"server was observed twice in round {round_index}")
        observation = self._observe(
            self._server_support,
            round_index,
            packet,
            raw_update,
            corrected_update,
            transmitted_update,
            next_residual,
        )
        self._pending_server = observation
        return observation

    def _client_aggregate(self, observations: list[dict[str, Any]]) -> dict[str, Any]:
        selected_by_tensor = {
            name: sum(value["support"]["selected_by_tensor"][name] for value in observations)
            for name in self.layout.names
        }
        selected_by_rank = [
            sum(value["support"]["selected_by_rank"][rank] for value in observations)
            for rank in range(self.layout.rank)
        ]
        selected_by_tensor_rank = {
            name: [
                sum(
                    value["support"]["selected_by_tensor_rank"][name][rank]
                    for value in observations
                )
                for rank in range(self.layout.rank)
            ]
            for name in self.layout.names
        }

        def residual_quantiles(key: str, tensor_name: str | None = None):
            values = []
            for observation in observations:
                residual = observation["residual"]
                source = residual["overall"] if tensor_name is None else residual["by_tensor"][tensor_name]
                values.append(float(source[key]))
            return _quantiles(values)

        return {
            "per_client": [
                {"client_id": client_id, **observations[client_id]}
                for client_id in range(self.num_clients)
            ],
            "selected_count_total": sum(
                value["support"]["selected_count"] for value in observations
            ),
            "selected_by_tensor_total": selected_by_tensor,
            "selected_by_rank_total": selected_by_rank,
            "selected_by_tensor_rank_total": selected_by_tensor_rank,
            "selected_values_rounded_to_zero_total": sum(
                value["support"]["selected_values_rounded_to_zero"]
                for value in observations
            ),
            "zero_support_tensor_count": _quantiles(
                [float(value["support"]["zero_support_tensor_count"]) for value in observations]
            ),
            "cumulative_coverage_overall": _quantiles(
                [float(value["support"]["cumulative_coverage"]["overall"]) for value in observations]
            ),
            "adjacent_support_jaccard_overall": _quantiles(
                [
                    float(value["support"]["adjacent_support_jaccard"]["overall"])
                    for value in observations
                    if value["support"]["adjacent_support_jaccard"]["overall"] is not None
                ]
            ),
            "residual_to_update_ratio": {
                "overall": residual_quantiles("residual_to_update_ratio"),
                "by_tensor": {
                    name: residual_quantiles("residual_to_update_ratio", name)
                    for name in self.layout.names
                },
            },
            "drained_residual_energy": {
                "overall": residual_quantiles("drained_residual_energy"),
                "by_tensor": {
                    name: residual_quantiles("drained_residual_energy", name)
                    for name in self.layout.names
                },
            },
            "ef_conservation_max_abs": max(
                float(value["residual"]["overall"]["ef_conservation_max_abs"])
                for value in observations
            ),
        }

    def finish_round(
        self,
        round_index: int,
        decoded_client_updates: Sequence[torch.Tensor],
        client_weights: Sequence[float],
        aggregate_update: torch.Tensor,
        transmitted_update: torch.Tensor,
    ) -> dict[str, Any]:
        if round_index != self.completed_round + 1:
            raise ValueError(
                f"diagnostics expected round {self.completed_round + 1}, got {round_index}"
            )
        if set(self._pending_clients) != set(range(self.num_clients)):
            raise ValueError("diagnostics are missing one or more client observations")
        if self._pending_server is None:
            raise ValueError("diagnostics are missing the server observation")
        if len(decoded_client_updates) != self.num_clients or len(client_weights) != self.num_clients:
            raise ValueError("client update diagnostics are not aligned with num_clients")
        updates = [self._check_vector(value, "decoded_client_update") for value in decoded_client_updates]
        aggregate = self._check_vector(aggregate_update, "aggregate_update")
        transmitted = self._check_vector(transmitted_update, "transmitted_update")

        weights = [float(value) for value in client_weights]
        if any(not math.isfinite(value) or value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("client diagnostic weights must be finite, non-negative, and non-zero")
        total_weight = sum(weights)
        normalized = [value / total_weight for value in weights]
        weighted_norm_sum = sum(
            weight * self._norm(update) for weight, update in zip(normalized, updates)
        )
        cancellation = 1.0 - self._norm(aggregate) / max(weighted_norm_sum, 1e-12)
        client_cosines = [_cosine(update, aggregate) for update in updates]

        aggregate_cpu = aggregate.to(device="cpu", dtype=torch.float32)
        transmitted_cpu = transmitted.to(device="cpu", dtype=torch.float32)
        aggregate_norm = self._norm(aggregate_cpu)
        transmitted_norm = self._norm(transmitted_cpu)
        aggregate_consecutive = (
            None
            if self._previous_aggregate is None
            else _cosine(aggregate_cpu, self._previous_aggregate)
        )
        transmitted_consecutive = (
            None
            if self._previous_transmitted is None
            else _cosine(transmitted_cpu, self._previous_transmitted)
        )
        sign_denominator = 0
        sign_flip_fraction = 0.0
        if self._previous_transmitted is not None:
            nonzero_intersection = (transmitted_cpu != 0) & (self._previous_transmitted != 0)
            sign_denominator = int(nonzero_intersection.sum().item())
            if sign_denominator:
                flips = (
                    torch.sign(transmitted_cpu[nonzero_intersection])
                    != torch.sign(self._previous_transmitted[nonzero_intersection])
                )
                sign_flip_fraction = float(flips.float().mean().item())

        beta = 0.9
        self._aggregate_norm_ema = (
            aggregate_norm
            if self._aggregate_norm_ema is None
            else beta * self._aggregate_norm_ema + (1.0 - beta) * aggregate_norm
        )
        self._transmitted_norm_ema = (
            transmitted_norm
            if self._transmitted_norm_ema is None
            else beta * self._transmitted_norm_ema + (1.0 - beta) * transmitted_norm
        )

        client_observations = [self._pending_clients[index] for index in range(self.num_clients)]
        output = {
            "recipe_version": self.recipe_version,
            "round": round_index,
            "layout_fingerprint": self.layout.fingerprint,
            "clients": self._client_aggregate(client_observations),
            "server": self._pending_server,
            "updates": {
                "aggregate_norm": aggregate_norm,
                "transmitted_norm": transmitted_norm,
                "aggregate_norm_ema_beta_0_9": self._aggregate_norm_ema,
                "transmitted_norm_ema_beta_0_9": self._transmitted_norm_ema,
                "consecutive_aggregate_cosine": aggregate_consecutive,
                "consecutive_transmitted_cosine": transmitted_consecutive,
                "aggregate_to_transmitted_cosine": _cosine(aggregate, transmitted),
                "transmitted_sign_flip_fraction": sign_flip_fraction,
                "transmitted_sign_flip_denominator": sign_denominator,
                "client_cancellation_ratio": cancellation,
                "client_to_aggregate_cosine": _quantiles(
                    client_cosines, include_p10=True
                ),
            },
        }
        self._previous_aggregate = aggregate_cpu.clone()
        self._previous_transmitted = transmitted_cpu.clone()
        self.completed_round = round_index
        self._pending_clients = {}
        self._pending_server = None
        return output

    def state_dict(self) -> dict[str, Any]:
        if self._pending_clients or self._pending_server is not None:
            raise RuntimeError("diagnostic state cannot be checkpointed mid-round")
        return {
            "recipe_version": self.recipe_version,
            "layout_fingerprint": self.layout.fingerprint,
            "num_clients": self.num_clients,
            "completed_round": self.completed_round,
            "client_support": [value.state_dict() for value in self._client_support],
            "server_support": self._server_support.state_dict(),
            "previous_aggregate": None
            if self._previous_aggregate is None
            else self._previous_aggregate.clone(),
            "previous_transmitted": None
            if self._previous_transmitted is None
            else self._previous_transmitted.clone(),
            "aggregate_norm_ema": self._aggregate_norm_ema,
            "transmitted_norm_ema": self._transmitted_norm_ema,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        required = {
            "recipe_version",
            "layout_fingerprint",
            "num_clients",
            "completed_round",
            "client_support",
            "server_support",
            "previous_aggregate",
            "previous_transmitted",
            "aggregate_norm_ema",
            "transmitted_norm_ema",
        }
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("diagnostic checkpoint is incomplete")
        if state["recipe_version"] != self.recipe_version:
            raise ValueError("diagnostic checkpoint recipe version differs")
        if state["layout_fingerprint"] != self.layout.fingerprint:
            raise ValueError("diagnostic checkpoint layout differs")
        if state["num_clients"] != self.num_clients:
            raise ValueError("diagnostic checkpoint client count differs")
        completed_round = state["completed_round"]
        if isinstance(completed_round, bool) or not isinstance(completed_round, int) or completed_round < 0:
            raise ValueError("diagnostic checkpoint completed_round is invalid")
        clients = state["client_support"]
        if not isinstance(clients, list) or len(clients) != self.num_clients:
            raise ValueError("diagnostic checkpoint client support is incomplete")
        for target, value in zip(self._client_support, clients):
            target.load_state_dict(value, completed_round)
        self._server_support.load_state_dict(state["server_support"], completed_round)

        def optional_vector(value: Any, label: str) -> torch.Tensor | None:
            if value is None:
                if completed_round != 0:
                    raise ValueError(f"diagnostic checkpoint {label} is missing")
                return None
            if not isinstance(value, torch.Tensor) or value.numel() != self.layout.total_numel:
                raise ValueError(f"diagnostic checkpoint {label} is malformed")
            value = value.detach().to(device="cpu", dtype=torch.float32).flatten().clone()
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"diagnostic checkpoint {label} is non-finite")
            return value

        aggregate = optional_vector(state["previous_aggregate"], "previous_aggregate")
        transmitted = optional_vector(state["previous_transmitted"], "previous_transmitted")

        def optional_ema(value: Any, label: str) -> float | None:
            if value is None:
                if completed_round != 0:
                    raise ValueError(f"diagnostic checkpoint {label} is missing")
                return None
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"diagnostic checkpoint {label} is invalid")
            return numeric

        self._previous_aggregate = aggregate
        self._previous_transmitted = transmitted
        self._aggregate_norm_ema = optional_ema(
            state["aggregate_norm_ema"], "aggregate_norm_ema"
        )
        self._transmitted_norm_ema = optional_ema(
            state["transmitted_norm_ema"], "transmitted_norm_ema"
        )
        self.completed_round = completed_round
        self._pending_clients = {}
        self._pending_server = None


# A descriptive alias for callers that prefer the longer name.
SparseDiagnosticTracker = SparseDiagnostics
