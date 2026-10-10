#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

from fedpeft.config import load_config
from fedpeft.engine import RunSpec, run_experiment
from fedpeft.provenance import verified_stability_launch_provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--wire-ratio", type=float)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help="remove an existing lock before launch; use only after verifying no writer is alive",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    stability = cfg.get("stability", {})
    if (
        stability.get("generation") in {3, 4}
        and stability.get("stage") == "stage_a"
    ):
        output_dir = Path(cfg["project"]["output_dir"])
        provenance = verified_stability_launch_provenance(cfg, output_dir)
        expected_token = provenance["preflight_file_sha256"]
        if os.environ.get("FEDPEFT_LOCKED_CAMPAIGN_TOKEN") != expected_token:
            raise RuntimeError(
                "locked stability Stage A may only run through the verified "
                "full-matrix launcher"
            )
        campaign_lock = output_dir / "campaign.lock"
        expected_owner = f"{socket.gethostname()}:{os.getppid()}"
        try:
            actual_owner = campaign_lock.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RuntimeError(
                "locked stability Stage A requires the active campaign lock"
            ) from exc
        if actual_owner != expected_owner:
            raise RuntimeError(
                "locked stability Stage A campaign owner does not match parent launcher"
            )
    summary = run_experiment(
        cfg,
        RunSpec(args.task, args.method, args.seed, args.wire_ratio),
        resume=not args.no_resume,
        force_unlock=args.force_unlock,
    )
    print(json.dumps({"status": summary["status"], "run_id": summary["run_id"]}))


if __name__ == "__main__":
    main()
