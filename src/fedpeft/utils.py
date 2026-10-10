from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_dump(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(value: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_arrays(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def tensor_sha256(tensor: "torch.Tensor") -> str:
    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def directory_sha256(root: str | Path) -> str:
    """Content fingerprint for a local model/tokenizer artifact directory."""
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of the exact bytes in one provenance file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_python_sha256(root: str | Path) -> str:
    """Fingerprint all Python code that can launch or execute an experiment.

    The original hash covered only ``src/``.  Locked stability campaigns also
    depend on the CLI/preflight orchestration under ``scripts/``; excluding it
    left a window where launch semantics could change after preflight without
    changing the recorded source hash.
    """
    root = Path(root)
    digest = hashlib.sha256()
    execution_entrypoints = (
        root / "scripts" / "preflight.py",
        root / "scripts" / "run_matrix.py",
        root / "scripts" / "run_experiment.py",
    )
    paths = [
        *(root / "src").rglob("*.py"),
        *(path for path in execution_entrypoints if path.is_file()),
    ]
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached_directory_sha256(root: str | Path, cache_path: str | Path) -> str:
    """Hash an immutable artifact tree once, reusing it while file metadata match."""
    root = Path(root).resolve()
    cache_path = Path(cache_path)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
        and path.suffix not in {".lock", ".tmp"}
    )
    manifest: list[dict[str, str | int]] = [
        {
            "path": str(path.relative_to(root)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("root") == str(root) and cached.get("manifest") == manifest:
                return str(cached["sha256"])
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    digest = hashlib.sha256()
    for metadata, path in zip(manifest, files):
        digest.update(str(metadata["path"]).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    value = digest.hexdigest()
    atomic_json_dump(
        {"root": str(root), "manifest": manifest, "sha256": value},
        cache_path,
    )
    return value


def bytes_human(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")
