from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import torch

from .wire import (
    WirePacket,
    decode_packet,
    encode_sparse,
    sparse_k_for_cap,
    topk_packet,
)


@dataclass
class CompressionResult:
    reconstructed: torch.Tensor
    packet: WirePacket
    seconds: float
    residual_norm: float
    input_norm: float


class TopKCompressor:
    def __init__(self, logical_numel: int, byte_cap: int):
        self.logical_numel = int(logical_numel)
        self.byte_cap = int(byte_cap)
        self.k = sparse_k_for_cap(self.logical_numel, self.byte_cap)

    def compress(self, vector: torch.Tensor) -> CompressionResult:
        if vector.numel() != self.logical_numel:
            raise ValueError(f"expected {self.logical_numel} values, got {vector.numel()}")
        if vector.is_cuda:
            torch.cuda.synchronize(vector.device)
        started = perf_counter()
        reconstructed, packet = topk_packet(vector, self.k)
        if vector.is_cuda:
            torch.cuda.synchronize(vector.device)
        elapsed = perf_counter() - started
        if packet.nbytes > self.byte_cap:
            raise AssertionError(f"packet {packet.nbytes} exceeds cap {self.byte_cap}")
        residual = vector.float() - reconstructed
        return CompressionResult(
            reconstructed=reconstructed,
            packet=packet,
            seconds=elapsed,
            residual_norm=float(torch.linalg.vector_norm(residual).item()),
            input_norm=float(torch.linalg.vector_norm(vector.float()).item()),
        )


def largest_remainder_quotas(block_numels: Sequence[int], k: int) -> tuple[int, ...]:
    """Allocate exactly ``k`` entries over canonical blocks.

    This is the lower-bounded, capacity-aware largest-remainder rule frozen in
    ``STABILITY-FOLLOWUP-v1``.  All arithmetic is integer arithmetic, and
    canonical block order breaks equal-remainder ties.  When the budget can
    cover every non-empty block, every block first receives one entry.  The
    remaining entries are allocated in proportion to remaining capacity.
    """

    capacities = tuple(int(value) for value in block_numels)
    if not capacities:
        raise ValueError("block_numels must not be empty")
    if any(value <= 0 for value in capacities):
        raise ValueError("every block must contain at least one value")
    total = sum(capacities)
    if isinstance(k, bool) or int(k) != k or not 1 <= int(k) <= total:
        raise ValueError(f"k must be an integer in [1, {total}], got {k!r}")
    k = int(k)

    block_count = len(capacities)
    quotas = [1 if k >= block_count else 0 for _ in capacities]
    remaining = k - sum(quotas)

    # A loop keeps the rule correct even if a future layout contains very
    # small blocks that saturate during an allocation pass.
    while remaining:
        available = [capacity - quota for capacity, quota in zip(capacities, quotas)]
        active = [index for index, value in enumerate(available) if value > 0]
        if not active:
            raise AssertionError("quota allocation exhausted capacity before assigning k")
        total_available = sum(available[index] for index in active)
        pass_budget = min(remaining, total_available)

        floors: dict[int, int] = {}
        remainders: dict[int, int] = {}
        for index in active:
            numerator = pass_budget * available[index]
            quotient, remainder = divmod(numerator, total_available)
            floors[index] = min(quotient, available[index])
            remainders[index] = remainder

        assigned = sum(floors.values())
        for index, value in floors.items():
            quotas[index] += value
        remaining -= assigned

        if remaining:
            candidates = sorted(
                (
                    index
                    for index in active
                    if quotas[index] < capacities[index]
                ),
                key=lambda index: (-remainders[index], index),
            )
            if not candidates:
                continue
            take = min(remaining, len(candidates))
            for index in candidates[:take]:
                quotas[index] += 1
            remaining -= take

    if sum(quotas) != k:
        raise AssertionError("largest-remainder quotas do not sum to k")
    if any(not 0 <= quota <= capacity for quota, capacity in zip(quotas, capacities)):
        raise AssertionError("largest-remainder quota exceeds a block capacity")
    if k >= block_count and any(quota < 1 for quota in quotas):
        raise AssertionError("a non-empty block did not receive its guaranteed entry")
    return tuple(quotas)


class BlockTopKCompressor(TopKCompressor):
    """Top-k with a frozen per-tensor quota and the ordinary sparse codec.

    The layout is not serialized.  Both peers derive the same quotas from the
    canonical tensor sizes.  The resulting payload is therefore exactly one
    standard ``FSPR`` packet with the same global ``k`` as ``TopKCompressor``.
    """

    def __init__(self, block_numels: Sequence[int], byte_cap: int):
        self.block_numels = tuple(int(value) for value in block_numels)
        if not self.block_numels or any(value <= 0 for value in self.block_numels):
            raise ValueError("block_numels must contain positive block sizes")
        super().__init__(sum(self.block_numels), byte_cap)
        self.quotas = largest_remainder_quotas(self.block_numels, self.k)
        offsets = [0]
        for count in self.block_numels:
            offsets.append(offsets[-1] + count)
        self.block_offsets = tuple(offsets)

    def compress(self, vector: torch.Tensor) -> CompressionResult:
        if vector.numel() != self.logical_numel:
            raise ValueError(f"expected {self.logical_numel} values, got {vector.numel()}")
        if vector.is_cuda:
            torch.cuda.synchronize(vector.device)
        started = perf_counter()
        flat = vector.detach().float().flatten()
        selected: list[torch.Tensor] = []
        for offset, count, quota in zip(
            self.block_offsets[:-1], self.block_numels, self.quotas
        ):
            block = flat.narrow(0, offset, count)
            # Stable sorting makes magnitude ties resolve to the lower local
            # index.  Concatenation in canonical block order is deterministic.
            local = torch.argsort(block.abs(), descending=True, stable=True)[:quota]
            selected.append(local + offset)
        indices = torch.cat(selected)
        if indices.numel() != self.k:
            raise AssertionError(
                f"block Top-k selected {indices.numel()} entries, expected {self.k}"
            )
        values = flat.index_select(0, indices)
        packet = encode_sparse(indices, values, self.logical_numel)
        reconstructed = decode_packet(packet, device=flat.device)
        if vector.is_cuda:
            torch.cuda.synchronize(vector.device)
        elapsed = perf_counter() - started

        expected_bytes = 21 + 6 * self.k
        if packet.transmitted_numel != self.k:
            raise AssertionError("block Top-k packet transmitted a non-canonical count")
        if packet.nbytes != expected_bytes:
            raise AssertionError(
                f"block Top-k packet is {packet.nbytes} B, expected {expected_bytes} B"
            )
        if packet.nbytes > self.byte_cap:
            raise AssertionError(f"packet {packet.nbytes} exceeds cap {self.byte_cap}")
        residual = vector.float() - reconstructed
        return CompressionResult(
            reconstructed=reconstructed,
            packet=packet,
            seconds=elapsed,
            residual_norm=float(torch.linalg.vector_norm(residual).item()),
            input_norm=float(torch.linalg.vector_norm(vector.float()).item()),
        )


class ErrorFeedbackCompressor(TopKCompressor):
    def compress_with_residual(
        self, vector: torch.Tensor, residual: torch.Tensor
    ) -> tuple[CompressionResult, torch.Tensor]:
        corrected = vector.float() + residual.float()
        result = self.compress(corrected)
        next_residual = corrected - result.reconstructed
        return result, next_residual
