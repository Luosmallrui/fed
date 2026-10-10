#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

import yaml

from fedpeft.config import load_config, result_path, validate_config


DENSE_METHODS = {
    "fedavg_lora",
    "ffa_lora",
    "lora_xs",
    "random_a_dense",
    "spectral_a_dense",
    "hybrid4_a_dense",
    "hybrid4_a_decay_dense",
}


def _csv(value: str, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _lr_slug(value: float) -> str:
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only holdout LR sweep")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--methods", required=True, help="comma-separated method names")
    parser.add_argument("--adapter-lrs", required=True, help="comma-separated learning rates")
    parser.add_argument(
        "--head-lr",
        type=float,
        help="optional method-specific classifier-head rate for every selected method",
    )
    parser.add_argument("--tasks", help="comma-separated task subset")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    base = load_config(args.base_config)
    if base["project"]["phase"] != "tune":
        raise ValueError("LR sweeps require a tune-phase base config")
    methods = _csv(args.methods)
    lrs = _csv(args.adapter_lrs, float)
    tasks = _csv(args.tasks) if args.tasks else list(base["data"]["tasks"])
    output_root = result_path(base, args.output_root)
    runner = Path(base["project"]["root"]) / "scripts" / "run_matrix.py"
    env = os.environ.copy()

    failures: list[str] = []
    for lr in lrs:
        cfg = copy.deepcopy({key: value for key, value in base.items() if not key.startswith("_")})
        cfg["data"]["tasks"] = tasks
        cfg["project"]["output_dir"] = str(output_root / f"adapter-lr-{_lr_slug(lr)}")
        cfg["matrix"]["dense_methods"] = [method for method in methods if method in DENSE_METHODS]
        cfg["matrix"]["sparse_methods"] = [method for method in methods if method not in DENSE_METHODS]
        cfg["matrix"]["diagnostic_dense_methods"] = []
        cfg["matrix"]["diagnostic_sparse_methods"] = []
        overrides = cfg["training"].setdefault("method_overrides", {})
        for method in methods:
            overrides.setdefault(method, {})["learning_rate"] = lr
            if args.head_lr is not None:
                overrides[method]["head_learning_rate"] = args.head_lr
        validate_config(cfg)
        trial_dir = result_path(cfg, cfg["project"]["output_dir"])
        trial_dir.mkdir(parents=True, exist_ok=True)
        config_path = trial_dir / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        command = [sys.executable, str(runner), "--config", str(config_path)]
        if args.fail_fast:
            command.append("--fail-fast")
        result = subprocess.run(command, cwd=base["project"]["root"], env=env, check=False)
        if result.returncode:
            failures.append(f"adapter_lr={lr:g}")
            if args.fail_fast:
                break
    if failures:
        raise SystemExit(f"failed LR trials: {', '.join(failures)}")


if __name__ == "__main__":
    main()
