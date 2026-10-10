from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


MAGIC_DENSE = b"FDEN"
MAGIC_SPARSE = b"FSPR"
VERSION = 1
DENSE_HEADER = struct.Struct("<4sBQ")
SPARSE_HEADER = struct.Struct("<4sBQQ")


@dataclass(frozen=True)
class WirePacket:
    payload: bytes
    logical_numel: int
    transmitted_numel: int
    kind: str

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def encode_dense(vector: torch.Tensor) -> WirePacket:
    cpu = vector.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
    payload = DENSE_HEADER.pack(MAGIC_DENSE, VERSION, cpu.size) + cpu.tobytes(order="C")
    return WirePacket(payload, int(cpu.size), int(cpu.size), "dense")


def encode_sparse(indices: torch.Tensor, values: torch.Tensor, logical_numel: int) -> WirePacket:
    idx = indices.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
    if logical_numel >= 2**32:
        raise ValueError("uint32 wire indices cannot address this vector")
    if idx.size and (idx.min() < 0 or idx.max() >= logical_numel):
        raise ValueError("sparse index outside logical vector")
    idx = idx.astype(np.uint32, copy=False)
    vals = values.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
    if idx.size != vals.size:
        raise ValueError("indices and values have different lengths")
    payload = (
        SPARSE_HEADER.pack(MAGIC_SPARSE, VERSION, logical_numel, idx.size)
        + idx.tobytes(order="C")
        + vals.tobytes(order="C")
    )
    return WirePacket(payload, int(logical_numel), int(idx.size), "sparse")


def decode_packet(packet: WirePacket, device: torch.device | str = "cpu") -> torch.Tensor:
    payload = packet.payload
    magic = payload[:4]
    if magic == MAGIC_DENSE:
        _, version, size = DENSE_HEADER.unpack_from(payload)
        if version != VERSION:
            raise ValueError(f"unsupported wire version: {version}")
        array = np.frombuffer(payload, dtype=np.float16, offset=DENSE_HEADER.size, count=size).copy()
        return torch.from_numpy(array).to(device=device, dtype=torch.float32)
    if magic == MAGIC_SPARSE:
        _, version, size, count = SPARSE_HEADER.unpack_from(payload)
        if version != VERSION:
            raise ValueError(f"unsupported wire version: {version}")
        index_start = SPARSE_HEADER.size
        value_start = index_start + count * np.dtype(np.uint32).itemsize
        indices = np.frombuffer(payload, dtype=np.uint32, offset=index_start, count=count).copy()
        values = np.frombuffer(payload, dtype=np.float16, offset=value_start, count=count).copy()
        result = torch.zeros(int(size), dtype=torch.float32, device=device)
        if count:
            result[torch.from_numpy(indices.astype(np.int64)).to(device)] = torch.from_numpy(values).to(
                device=device, dtype=torch.float32
            )
        return result
    raise ValueError("invalid packet magic")


def decode_sparse_indices(packet: WirePacket, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the exact support carried by a sparse packet.

    Looking at non-zero reconstructed values is not equivalent: a selected
    value can round to zero on an FP16 wire.  FLASC needs the transmitted mask
    itself when evaluating the model visible to clients.
    """
    payload = packet.payload
    if payload[:4] != MAGIC_SPARSE:
        raise ValueError("packet is not sparse")
    _, version, size, count = SPARSE_HEADER.unpack_from(payload)
    if version != VERSION:
        raise ValueError(f"unsupported wire version: {version}")
    indices = np.frombuffer(
        payload,
        dtype=np.uint32,
        offset=SPARSE_HEADER.size,
        count=count,
    ).copy()
    if indices.size and int(indices.max()) >= int(size):
        raise ValueError("sparse packet contains an out-of-range index")
    return torch.from_numpy(indices.astype(np.int64)).to(device=device)


def schema_bytes(names: Iterable[str], shapes: Iterable[tuple[int, ...]]) -> int:
    payload = json.dumps(
        [{"name": name, "shape": list(shape)} for name, shape in zip(names, shapes)],
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return len(payload)


def sparse_k_for_cap(logical_numel: int, byte_cap: int) -> int:
    available = byte_cap - SPARSE_HEADER.size
    per_entry = np.dtype(np.uint32).itemsize + np.dtype(np.float16).itemsize
    if available < per_entry:
        raise ValueError(f"wire cap {byte_cap} B is too small for one sparse entry")
    return max(1, min(logical_numel, available // per_entry))


def topk_packet(vector: torch.Tensor, k: int) -> tuple[torch.Tensor, WirePacket]:
    flat = vector.detach().float().flatten()
    k = max(1, min(int(k), flat.numel()))
    # Stable sorting makes ties (especially the all-zero first round) resolve by
    # canonical flattened index rather than a backend-dependent CUDA choice.
    indices = torch.argsort(flat.abs(), descending=True, stable=True)[:k]
    values = flat.index_select(0, indices)
    packet = encode_sparse(indices, values, flat.numel())
    # Every algorithm consumes exactly what crossed the wire. In particular,
    # error feedback retains FP16 quantization error as well as sparsity error.
    reconstructed = decode_packet(packet, device=flat.device)
    return reconstructed, packet
