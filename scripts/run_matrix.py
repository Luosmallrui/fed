#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fedpeft.config import load_config, result_path, run_id
from fedpeft.provenance import verified_stability_launch_provenance
from fedpeft.utils import append_jsonl, atomic_json_dump


def _csv(value: str | None, cast=str):
    return None if value is None else [cast(item.strip()) for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class _CampaignLock:
    path: Path
    fd: int
    owner: str
    device: int
    inode: int


def _write_lock_owner(fd: int, owner: str) -> None:
    payload = f"{owner}\n".encode("utf-8")
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    written = 0
    while written < len(payload):
        count = os.write(fd, payload[written:])
        if count == 0:
            raise OSError("short write while creating campaign.lock")
        written += count
    os.fsync(fd)


def _read_lock_fd(fd: int) -> tuple[str, os.stat_result]:
    metadata = os.fstat(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    payload = os.read(fd, 4097)
    if len(payload) > 4096:
        raise RuntimeError("campaign.lock is unreasonably large")
    try:
        owner = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("campaign.lock is not valid UTF-8") from exc
    return owner, metadata


def _read_lock_path(path: Path) -> tuple[str, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        return _read_lock_fd(fd)
    finally:
        os.close(fd)


def _parse_lock_owner(owner: str, path: Path) -> tuple[str, int]:
    try:
        host, raw_pid = owner.split(":", 1)
        pid = int(raw_pid)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"cannot validate campaign lock {path}") from exc
    if not host or pid <= 0:
        raise RuntimeError(f"cannot validate campaign lock {path}")
    return host, pid


def _create_campaign_lock(path: Path, owner: str) -> _CampaignLock:
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_lock_owner(fd, owner)
        metadata = os.fstat(fd)
        return _CampaignLock(path, fd, owner, metadata.st_dev, metadata.st_ino)
    except BaseException:
        try:
            metadata = os.fstat(fd)
            try:
                current = path.stat()
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == (metadata.st_dev, metadata.st_ino):
                    path.unlink(missing_ok=True)
        finally:
            os.close(fd)
        raise


def _acquire_campaign_lock(path: Path, *, resume: bool) -> _CampaignLock:
    """Atomically acquire the stability-v3 wrapper lock.

    A stale lock may be replaced only by an explicit resume, after the recorded
    same-host process is proven dead and the path is re-read unchanged while
    the stale inode is exclusively locked.  The retained descriptor prevents
    two resumptions from concurrently removing and replacing the same lock.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    owner = f"{socket.gethostname()}:{os.getpid()}"
    for _ in range(3):
        try:
            return _create_campaign_lock(path, owner)
        except FileExistsError as exc:
            if not resume:
                try:
                    existing_owner, _ = _read_lock_path(path)
                except (FileNotFoundError, OSError, RuntimeError):
                    existing_owner = "unreadable owner"
                raise RuntimeError(
                    f"stability-v3 campaign is already locked by {existing_owner}"
                ) from exc

            flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            try:
                stale_fd = os.open(path, flags)
            except FileNotFoundError:
                continue
            try:
                try:
                    fcntl.flock(stale_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as lock_exc:
                    try:
                        active_owner, _ = _read_lock_path(path)
                    except (FileNotFoundError, OSError, RuntimeError):
                        active_owner = "unreadable owner"
                    raise RuntimeError(
                        "cannot resume while stability-v3 campaign owner "
                        f"{active_owner} is active"
                    ) from lock_exc

                first_owner, first_metadata = _read_lock_fd(stale_fd)
                host, pid = _parse_lock_owner(first_owner, path)
                if host != socket.gethostname():
                    raise RuntimeError(
                        "cannot confirm that stability-v3 campaign owner "
                        f"{first_owner} on another host is dead"
                    )
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                except PermissionError as live_exc:
                    raise RuntimeError(
                        f"cannot resume while campaign wrapper pid {pid} is still alive"
                    ) from live_exc
                else:
                    raise RuntimeError(
                        f"cannot resume while campaign wrapper pid {pid} is still alive"
                    )

                try:
                    second_owner, second_metadata = _read_lock_path(path)
                except FileNotFoundError:
                    continue
                first_identity = (first_metadata.st_dev, first_metadata.st_ino)
                second_identity = (second_metadata.st_dev, second_metadata.st_ino)
                if second_owner != first_owner or second_identity != first_identity:
                    raise RuntimeError(
                        "campaign.lock changed while validating its stale owner"
                    )
                path.unlink()
                try:
                    return _create_campaign_lock(path, owner)
                except FileExistsError:
                    continue
            finally:
                try:
                    fcntl.flock(stale_fd, fcntl.LOCK_UN)
                finally:
                    os.close(stale_fd)
    raise RuntimeError("failed to atomically acquire stability-v3 campaign.lock")


def _release_campaign_lock(lock: _CampaignLock) -> None:
    """Release the descriptor and unlink only this wrapper's lock inode."""

    try:
        try:
            owner, metadata = _read_lock_path(lock.path)
        except (FileNotFoundError, OSError, RuntimeError):
            return
        if (
            owner == lock.owner
            and (metadata.st_dev, metadata.st_ino) == (lock.device, lock.inode)
        ):
            lock.path.unlink(missing_ok=True)
    finally:
        try:
            fcntl.flock(lock.fd, fcntl.LOCK_UN)
        finally:
            os.close(lock.fd)


def _atomic_create_empty(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _initialize_v3_campaign(
    campaign_config: dict[str, Any],
    campaign_config_path: Path,
    events_path: Path,
) -> None:
    atomic_json_dump(campaign_config, campaign_config_path)
    _atomic_create_empty(events_path)


def _existing_run_artifacts(output_dir: Path) -> list[Path]:
    return sorted(
        {
            path.parent
            for pattern in (
                "*/summary.json",
                "*/failure.json",
                "*/checkpoint.pt",
                "*/run_config.json",
                "*/launcher.log",
                "*/.lock",
                "*/round-*.json",
            )
            for path in output_dir.glob(pattern)
        }
    )


def _load_v3_resume_events(
    campaign_config: dict[str, Any],
    campaign_config_path: Path,
    events_path: Path,
    output_dir: Path,
) -> list[dict[str, object]]:
    if not campaign_config_path.exists():
        raise RuntimeError(
            "cannot resume stability-v3 without campaign_config.json"
        )
    existing_config = json.loads(campaign_config_path.read_text(encoding="utf-8"))
    if existing_config != campaign_config:
        raise RuntimeError(
            "cannot resume stability-v3 because campaign configuration differs"
        )
    if not events_path.exists():
        run_artifacts = _existing_run_artifacts(output_dir)
        if run_artifacts:
            raise RuntimeError(
                "cannot recover missing campaign_events.jsonl after run artifacts exist: "
                f"{run_artifacts}"
            )
        _atomic_create_empty(events_path)

    prior_events: list[dict[str, object]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if not isinstance(event, dict):
            raise RuntimeError("campaign event is not an object")
        prior_events.append(event)
    return prior_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--tasks", help="comma-separated subset")
    parser.add_argument("--methods", help="comma-separated subset")
    parser.add_argument("--seeds", help="comma-separated subset")
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--resume-campaign",
        action="store_true",
        help="resume an interrupted, provenance-identical locked stability campaign",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    is_locked_stage_a = (
        cfg.get("stability", {}).get("generation") in {3, 4}
        and cfg.get("stability", {}).get("stage") == "stage_a"
    )
    if is_locked_stage_a and any(
        value is not None
        for value in (args.tasks, args.methods, args.seeds, args.max_runs)
    ):
        raise ValueError(
            "locked stability Stage A must be launched as the complete frozen matrix; "
            "subset and max-runs options are prohibited"
        )
    if is_locked_stage_a and args.include_diagnostics:
        raise ValueError(
            "locked stability Stage A has no diagnostic-only launch subset"
        )
    if args.resume_campaign and not is_locked_stage_a:
        raise ValueError("--resume-campaign is supported only for locked stability Stage A")
    tasks = _csv(args.tasks) or list(cfg["data"]["tasks"])
    seeds = _csv(args.seeds, int) or [int(x) for x in cfg["data"]["partition_seeds"]]
    dense = list(cfg["matrix"]["dense_methods"])
    sparse = list(cfg["matrix"]["sparse_methods"])
    if args.include_diagnostics:
        dense += list(cfg["matrix"].get("diagnostic_dense_methods", []))
        sparse += list(cfg["matrix"].get("diagnostic_sparse_methods", []))
    allowed = set(_csv(args.methods) or dense + sparse)
    dense = [x for x in dense if x in allowed]
    sparse = [x for x in sparse if x in allowed]
    ratios = [float(x) for x in cfg["communication"]["wire_ratios"]]
    specs: list[tuple[str, str, int, float | None]] = []
    for task in tasks:
        for seed in seeds:
            specs.extend((task, method, seed, None) for method in dense)
            specs.extend((task, method, seed, ratio) for method in sparse for ratio in ratios)
    if args.max_runs is not None:
        specs = specs[: args.max_runs]

    root = Path(cfg["project"]["root"])
    output_dir = result_path(cfg, cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_prefix = "diagnostic_campaign" if args.include_diagnostics else "campaign"
    events_path = output_dir / f"{campaign_prefix}_events.jsonl"
    campaign_config_path = output_dir / f"{campaign_prefix}_config.json"
    campaign_summary_path = output_dir / f"{campaign_prefix}_summary.json"
    campaign_lock: _CampaignLock | None = None
    if is_locked_stage_a:
        campaign_lock = _acquire_campaign_lock(
            output_dir / "campaign.lock",
            resume=args.resume_campaign,
        )
    try:
        _run_campaign(
            args=args,
            cfg=cfg,
            is_locked_stage_a=is_locked_stage_a,
            specs=specs,
            root=root,
            output_dir=output_dir,
            events_path=events_path,
            campaign_config_path=campaign_config_path,
            campaign_summary_path=campaign_summary_path,
        )
    finally:
        if campaign_lock is not None:
            _release_campaign_lock(campaign_lock)


def _run_campaign(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    is_locked_stage_a: bool,
    specs: list[tuple[str, str, int, float | None]],
    root: Path,
    output_dir: Path,
    events_path: Path,
    campaign_config_path: Path,
    campaign_summary_path: Path,
) -> None:
    requested_run_ids = [
        run_id(task, method, seed, ratio) for task, method, seed, ratio in specs
    ]
    campaign_config = {
        "config": {key: value for key, value in cfg.items() if not key.startswith("_")},
        "requested_run_ids": requested_run_ids,
        "include_diagnostics": args.include_diagnostics,
    }
    if is_locked_stage_a:
        campaign_config["launch_provenance"] = verified_stability_launch_provenance(
            cfg, output_dir
        )
    prior_events: list[dict[str, object]] = []
    if is_locked_stage_a:
        existing_control_files = [
            path
            for path in (events_path, campaign_config_path, campaign_summary_path)
            if path.exists()
        ]
        existing_run_artifacts = _existing_run_artifacts(output_dir)
        if not args.resume_campaign and (existing_control_files or existing_run_artifacts):
            raise RuntimeError(
                "refusing to duplicate or mix the frozen stability campaign: "
                f"control_files={existing_control_files}, "
                f"run_directories={existing_run_artifacts}"
            )
        if args.resume_campaign:
            prior_events = _load_v3_resume_events(
                campaign_config,
                campaign_config_path,
                events_path,
                output_dir,
            )
            if any(event.get("event") == "FAIL" for event in prior_events):
                raise RuntimeError("a failed locked stability run cannot be resumed")
            for lock_path in output_dir.glob("*/.lock"):
                try:
                    host, raw_pid = lock_path.read_text(encoding="utf-8").strip().split(":", 1)
                    pid = int(raw_pid)
                except (OSError, ValueError) as exc:
                    raise RuntimeError(f"cannot validate campaign lock {lock_path}") from exc
                if host == socket.gethostname():
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        continue
                    except PermissionError:
                        pass
                    raise RuntimeError(
                        f"cannot resume while run process {pid} is still alive"
                    )
        else:
            _initialize_v3_campaign(
                campaign_config,
                campaign_config_path,
                events_path,
            )
    else:
        atomic_json_dump(campaign_config, campaign_config_path)
    run_script = root / "scripts" / "run_experiment.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root / 'src'}:{root / '.compat'}:{root / '.deps'}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env["TOKENIZERS_PARALLELISM"] = "false"
    if is_locked_stage_a:
        env["FEDPEFT_LOCKED_CAMPAIGN_TOKEN"] = str(
            campaign_config["launch_provenance"]["preflight_file_sha256"]
        )
    failures: list[str] = []
    completed: list[str] = []
    campaign_started = dt.datetime.now(dt.timezone.utc)
    for index, (task, method, seed, ratio) in enumerate(specs, start=1):
        identifier = run_id(task, method, seed, ratio)
        run_dir = result_path(cfg, output_dir / identifier)
        run_dir.mkdir(parents=True, exist_ok=True)
        prior_for_run = [
            event for event in prior_events if event.get("run_id") == identifier
        ]
        prior_done = any(event.get("event") == "DONE" for event in prior_for_run)
        summary_path = run_dir / "summary.json"
        if args.resume_campaign and summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict) and summary.get("status") == "COMPLETE":
                if not prior_done:
                    recovered = dt.datetime.now(dt.timezone.utc)
                    append_jsonl(
                        {
                            "event": "DONE",
                            "run_id": identifier,
                            "index": index,
                            "total": len(specs),
                            "returncode": 0,
                            "elapsed_seconds": 0.0,
                            "timestamp": recovered.isoformat(),
                            "recovered_from_summary": True,
                        },
                        events_path,
                    )
                completed.append(identifier)
                print(f"[{index}/{len(specs)}] SKIP {identifier} (complete)", flush=True)
                continue
        if prior_done:
            raise RuntimeError(
                f"campaign event says DONE but COMPLETE summary is unavailable: {identifier}"
            )
        command = [
            sys.executable,
            str(run_script),
            "--config",
            str(Path(cfg["_config_path"]).resolve()),
            "--task",
            task,
            "--method",
            method,
            "--seed",
            str(seed),
        ]
        if ratio is not None:
            command += ["--wire-ratio", str(ratio)]
        print(f"[{index}/{len(specs)}] RUN  {identifier}", flush=True)
        started = dt.datetime.now(dt.timezone.utc)
        append_jsonl(
            {
                "event": "RESUME" if prior_for_run else "START",
                "run_id": identifier,
                "index": index,
                "total": len(specs),
                "timestamp": started.isoformat(),
            },
            events_path,
        )
        with (run_dir / "launcher.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=root,
                env=env,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        finished = dt.datetime.now(dt.timezone.utc)
        append_jsonl(
            {
                "event": "DONE" if result.returncode == 0 else "FAIL",
                "run_id": identifier,
                "index": index,
                "total": len(specs),
                "returncode": result.returncode,
                "elapsed_seconds": (finished - started).total_seconds(),
                "timestamp": finished.isoformat(),
            },
            events_path,
        )
        if result.returncode:
            failures.append(identifier)
            print(f"[{index}/{len(specs)}] FAIL {identifier} rc={result.returncode}", flush=True)
            if args.fail_fast:
                break
        else:
            completed.append(identifier)
            print(f"[{index}/{len(specs)}] DONE {identifier}", flush=True)
    atomic_json_dump(
        {
            "status": "COMPLETE" if not failures else "FAILED",
            "started_at": campaign_started.isoformat(),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "requested_runs": len(specs),
            "completed_runs": completed,
            "failed_runs": failures,
        },
        campaign_summary_path,
    )
    if failures:
        print("FAILED RUNS:", *failures, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
