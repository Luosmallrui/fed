"""Run the exact basis-construction path of one run and stop before training.

Zero-training protocol comparisons (setup bytes, installed energy ratio, overlap with
the exact optimum) use the same code as real runs: run_experiment executes data
preparation, model build and _install_cab_basis, then this script interrupts it at the
first post-install invariant check.  Output goes to the config's output_dir; delete it
afterwards (it holds an incomplete run).

Usage: python scripts/proposals/basis_install_dry.py --config C --task T --method M --seed S --wire-ratio W
"""

from __future__ import annotations

import argparse
import json

import fedpeft.engine as engine
from fedpeft.config import load_config


class StopAfterInstall(Exception):
    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--wire-ratio", type=float)
    args = parser.parse_args()
    cfg = load_config(args.config)
    captured = {}
    original = engine._install_cab_basis

    def install(**kwargs):
        record, setup = original(**kwargs)
        captured["record"], captured["setup"] = record, setup
        raise StopAfterInstall

    engine._install_cab_basis = install
    try:
        engine.run_experiment(cfg, engine.RunSpec(args.task, args.method, args.seed, args.wire_ratio), resume=False)
    except StopAfterInstall:
        pass
    except Exception:
        if "record" not in captured:
            raise
    record = captured["record"]
    summary = record["summary"]
    activation = record.get("activation") or {}
    print(
        json.dumps(
            {
                "setup_MB": round(captured["setup"] / 1e6, 3),
                "gradient_MB": round(sum(record["uplink_factor_bytes_per_client"]) / 1e6, 3),
                "sketch_MB": round(sum(activation.get("uplink_sketch_bytes_per_client", [0])) / 1e6, 3),
                "broadcast_MB": round(record["downlink_basis_bytes_per_client"] * len(record["client_weights"]) / 1e6, 3),
                "refine_MB": round(activation.get("refine_bytes", 0) / 1e6, 3),
                "energy_installed": round(summary.get("energy_ratio_installed_mean", float("nan")), 3),
                "common_installed": round(summary.get("common_mode_installed_mean", float("nan")), 4),
                "kfac_selected": round(summary.get("kfac_capture_selected_mean", float("nan")), 3),
                "overlap_vs_exact": round(summary["overlap_reported_vs_exact_mean"], 3),
                "sketch_error_max": round(max(activation.get("sketch_relative_error_by_client", [0])), 3),
            }
        )
    )


if __name__ == "__main__":
    main()
