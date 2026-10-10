"""Calibrate cab.gamma from the DCEF development stage.

CAB's kappa needs the local contraction gamma of B-only training.  DCEF already
estimates it online (client secant least squares, aggregated as gamma_bar), so
the calibrated value is the per-task median of gamma_bar_mean over the
technically valid random_a_dcef runs in DCEF_RUNS.csv (produced by
build_dcef_report.py after running the DCEF dev matrix with
--include-diagnostics).

The script also checks, per task and budget, whether the median lies inside the
region where the closed-form kappa is finite: gamma < 2 / (1 + omega / N_eff),
omega = d / K - 1.  N_eff <= num_clients, so using num_clients is optimistic;
values close to the bound may still fail at launch and then need
cab.kappa_override.

Prints a YAML snippet to paste into the cab section; it does not edit configs.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

FIXED_A_ROBERTA_QV_R8 = 147_456


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", required=True, type=Path, help="DCEF_RUNS.csv from the DCEF dev stage")
    parser.add_argument("--method", default="random_a_dcef")
    parser.add_argument("--adapter-numel", type=int, default=FIXED_A_ROBERTA_QV_R8)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="DCEF dev output dir; if given, N_eff per task is the smallest over its partitions",
    )
    args = parser.parse_args()
    n_eff_by_task: dict[str, float] = {}
    if args.results_dir is not None:
        import json

        for path in args.results_dir.glob("*/summary.json"):
            summary = json.loads(path.read_text(encoding="utf-8"))
            sizes = summary.get("data", {}).get("client_train_sizes")
            if not sizes:
                continue
            n_eff = sum(sizes) ** 2 / sum(x * x for x in sizes)
            task = summary["data"]["task"]
            n_eff_by_task[task] = min(n_eff, n_eff_by_task.get(task, float("inf")))

    gammas: dict[str, list[float]] = defaultdict(list)
    sparse_k: dict[tuple[str, str], int] = {}
    with args.runs.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != args.method or row.get("technical_valid") != "True":
                continue
            if row.get("gamma_bar_mean"):
                gammas[row["task"]].append(float(row["gamma_bar_mean"]))
            if row.get("sparse_k"):
                sparse_k[(row["task"], row["wire"])] = int(float(row["sparse_k"]))
    if not gammas:
        raise SystemExit(
            f"no valid {args.method} rows with gamma_bar_mean; run the DCEF dev matrix "
            "with --include-diagnostics and rebuild DCEF_RUNS.csv"
        )

    medians = {}
    for task, values in sorted(gammas.items()):
        medians[task] = statistics.median(values)
        print(
            f"{task}: median gamma_bar={medians[task]:.4f} "
            f"(n={len(values)}, min={min(values):.4f}, max={max(values):.4f})"
        )
        for (k_task, wire), k in sorted(sparse_k.items()):
            if k_task != task:
                continue
            omega = args.adapter_numel / k - 1.0
            n_eff = n_eff_by_task.get(task, float(args.num_clients))
            bound = 2.0 / (1.0 + omega / n_eff)
            status = "ok" if medians[task] < bound else "OUTSIDE: set cab.kappa_override"
            print(f"    wire {wire}: K={k}, omega={omega:.2f}, gamma bound (N_eff={n_eff:.2f}) {bound:.4f} -> {status}")

    print("\n# paste into the cab section:")
    print("  gamma: {" + ", ".join(f"{task}: {value:.4f}" for task, value in medians.items()) + "}")
    print("  gamma_source: dcef_dev")


if __name__ == "__main__":
    main()
