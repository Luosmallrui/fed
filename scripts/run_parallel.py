#!/usr/bin/env python3
"""Run a config's matrix with several concurrent run_experiment.py workers.

Same run set, command, environment and per-run logs as run_matrix.py, but
several runs share the GPU at once (each RoBERTa run uses a few GiB and does not
saturate an A800).  Completed runs are skipped by run_experiment.py itself, so
the launcher can be restarted at any time.  Bookkeeping goes to
parallel_events.jsonl / parallel_summary.json, never to run_matrix's files.

Before any worker starts, every task is tokenised once in this process:
prepare_tokenized_dataset publishes its cache by directory replacement, which is
not safe when two processes create the same cache concurrently.

Not for locked stability Stage A campaigns (use run_matrix.py there).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fedpeft.config import load_config, result_path, run_id, task_value  # noqa: E402
from fedpeft.utils import atomic_json_dump  # noqa: E402

# Relative per-run cost (seconds on one A800, single process) used to start
# the longest runs first so the tail of the campaign is not one long MNLI run.
TASK_COST = {"mnli": 640.0, "qqp": 600.0, "qnli": 600.0, "sst2": 190.0, "rte": 100.0, "mrpc": 60.0}


def _csv(value: str | None, cast=str):
    return None if value is None else [cast(item) for item in value.split(",") if item]


def build_specs(cfg, args) -> list[tuple[str, str, int, float | None]]:
    tasks = _csv(args.tasks) or list(cfg["data"]["tasks"])
    seeds = _csv(args.seeds, int) or [int(x) for x in cfg["data"]["partition_seeds"]]
    dense = list(cfg["matrix"]["dense_methods"])
    sparse = list(cfg["matrix"]["sparse_methods"])
    if args.include_diagnostics:
        dense += list(cfg["matrix"].get("diagnostic_dense_methods", []))
        sparse += list(cfg["matrix"].get("diagnostic_sparse_methods", []))
    allowed = set(_csv(args.methods) or dense + sparse)
    ratios = [float(x) for x in cfg["communication"]["wire_ratios"]]
    specs = []
    for task in tasks:
        for seed in seeds:
            specs.extend((task, m, seed, None) for m in dense if m in allowed)
            specs.extend((task, m, seed, r) for m in sparse if m in allowed for r in ratios)
    return specs


def prepare_tokenization(cfg, tasks: list[str]) -> None:
    from transformers import AutoTokenizer

    from fedpeft.data import prepare_tokenized_dataset

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["path"], local_files_only=True, use_fast=True)
    for task in tasks:
        prepare_tokenized_dataset(
            cfg["data"]["root"],
            cfg["project"]["tokenized_dir"],
            task,
            tokenizer,
            int(cfg["data"]["max_length"]),
            int(cfg["data"]["preprocessing_workers"]),
        )
        print(f"tokenized {task}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--threads-per-run",
        type=int,
        default=2,
        help="OMP/MKL threads per run; workers x threads should not exceed the CPU quota",
    )
    parser.add_argument("--tasks")
    parser.add_argument("--methods")
    parser.add_argument("--seeds")
    parser.add_argument("--include-diagnostics", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    stability = cfg.get("stability", {})
    if stability.get("generation") in {3, 4} and stability.get("stage") == "stage_a":
        raise SystemExit("locked stability Stage A must use run_matrix.py")
    specs = build_specs(cfg, args)
    specs.sort(key=lambda s: -TASK_COST.get(s[0], 100.0) * float(task_value(cfg, "training", "rounds", s[0])))
    output_dir = result_path(cfg, cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_tokenization(cfg, sorted({s[0] for s in specs}))

    events_path = output_dir / "parallel_events.jsonl"
    lock = threading.Lock()

    def event(record: dict) -> None:
        record["timestamp"] = dt.datetime.now(dt.timezone.utc).isoformat()
        with lock, events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Containers often expose every host core but enforce a smaller CPU quota;
    # BLAS/OpenMP then oversubscribes and CPU linear algebra can stall for
    # minutes.  Cap per-run threads so all workers fit the quota.
    threads = str(max(1, args.threads_per_run))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = threads

    def run(spec):
        task, method, seed, ratio = spec
        identifier = run_id(task, method, seed, ratio)
        run_dir = result_path(cfg, output_dir / identifier)
        run_dir.mkdir(parents=True, exist_ok=True)
        summary = run_dir / "summary.json"
        if summary.exists() and json.loads(summary.read_text()).get("status") == "COMPLETE":
            return identifier, "SKIP", 0.0
        command = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"), "--config",
                   str(Path(cfg["_config_path"]).resolve()), "--task", task, "--method", method,
                   "--seed", str(seed)]
        if ratio is not None:
            command += ["--wire-ratio", str(ratio)]
        event({"event": "START", "run_id": identifier})
        started = dt.datetime.now(dt.timezone.utc)
        with (run_dir / "launcher.log").open("a", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = (dt.datetime.now(dt.timezone.utc) - started).total_seconds()
        status = "DONE" if result.returncode == 0 else "FAIL"
        event({"event": status, "run_id": identifier, "returncode": result.returncode, "elapsed_seconds": elapsed})
        return identifier, status, elapsed

    started = dt.datetime.now(dt.timezone.utc)
    done, failed, skipped = [], [], []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run, spec) for spec in specs]
        for count, future in enumerate(as_completed(futures), start=1):
            identifier, status, elapsed = future.result()
            {"DONE": done, "FAIL": failed, "SKIP": skipped}[status].append(identifier)
            print(f"[{count}/{len(specs)}] {status} {identifier} ({elapsed:.0f}s)", flush=True)
    atomic_json_dump(
        {
            "status": "COMPLETE" if not failed else "FAILED",
            "config": str(Path(cfg["_config_path"]).resolve()),
            "workers": args.workers,
            "started_at": started.isoformat(),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "requested_runs": len(specs),
            "completed_runs": done,
            "skipped_complete_runs": skipped,
            "failed_runs": failed,
        },
        output_dir / "parallel_summary.json",
    )
    if failed:
        print("FAILED RUNS:", *failed, sep="\n  ", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
