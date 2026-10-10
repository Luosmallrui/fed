"""Summarise the confirmation stage (configs/roberta_confirm_main.yaml and _eva.yaml).

Writes CONFIRM_SUMMARY.md and CONFIRM_RUNS.csv next to the confirm_main results:
per task and budget the mean +- SD of the final official-validation score, how many
seeds reach C95 (95% of that seed's dense FedAvg-LoRA final score, bytes include the
one-time basis setup) and the median bytes when they do; paired seed-level
comparisons with win / tie / loss counts (|delta| <= 0.003 is a tie) and an exact
two-sided sign test over the non-tied pairs.

Usage (server):
  python scripts/summarize_confirm.py /data/home/202412012250/bishe_confirm/docs/results
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

TASKS = ("mnli", "qnli", "sst2")
WIRES = (0.05, 0.10, 0.20)
SPARSE = ("random_a_ef", "spectral_a_ef", "cab0_a_ef", "cab0wh_a_ef", "gev0_a_ef")
DENSE = ("fedavg_lora", "random_a_dense", "gev0_a_dense")
TIE = 0.003
PAIRS = (
    ("gev0_a_ef", "random_a_ef", WIRES),
    ("gev0_a_ef", "spectral_a_ef", WIRES),
    ("gev0_a_ef", "cab0wh_a_ef", WIRES),
    ("cab0wh_a_ef", "random_a_ef", WIRES),
    ("cab0_a_ef", "random_a_ef", WIRES),
    ("eva_a_ef", "random_a_ef", (0.10,)),
    ("gev0_a_dense", "random_a_dense", (None,)),
    ("gev0_a_dense", "fedavg_lora", (None,)),
)


def run_id(task: str, method: str, seed: int, wire: float | None) -> str:
    suffix = "dense" if wire is None else f"wire-{int(round(wire * 100)):02d}"
    return f"{task}__{method}__seed-{seed}__{suffix}"


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    root = Path(sys.argv[1])
    runs: dict[tuple[str, str, int, float | None], dict] = {}
    seeds: set[int] = set()
    for directory in ("confirm_main", "confirm_eva"):
        for path in sorted((root / directory).glob("*/summary.json")):
            summary = json.loads(path.read_text())
            spec = summary["run"]
            wire = spec.get("wire_ratio")
            key = (spec["task"], spec["method"], int(spec["seed"]), None if wire is None else round(float(wire), 2))
            runs[key] = summary
            seeds.add(int(spec["seed"]))
    seeds_sorted = sorted(seeds)

    rows = []
    for (task, method, seed, wire), summary in sorted(runs.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3] or 0)):
        dense = runs.get((task, "fedavg_lora", seed, None))
        target = 0.95 * dense["final"]["primary_metric"] if dense else math.nan
        hit = next(
            (r for r in summary["rounds"] if r.get("primary_metric") is not None and r["primary_metric"] >= target),
            None,
        )
        rows.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "wire": "dense" if wire is None else wire,
                "final": summary["final"]["primary_metric"],
                "c95_target": target,
                "c95_round": hit["round"] if hit else "",
                "c95_bytes": hit["wire"]["total_delivered_bytes"] if hit else "",
                "setup_bytes": summary["final"]["wire"].get("setup_bytes", 0),
                "total_bytes": summary["final"]["wire"]["total_delivered_bytes"],
            }
        )
    by_key = {(r["task"], r["method"], r["seed"], None if r["wire"] == "dense" else r["wire"]): r for r in rows}
    out_dir = root / "confirm_main"
    with (out_dir / "CONFIRM_RUNS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def cell(task: str, method: str, wire: float | None) -> str:
        members = [by_key[(task, method, s, wire)] for s in seeds_sorted if (task, method, s, wire) in by_key]
        if not members:
            return ""
        finals = [m["final"] for m in members]
        sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
        hits = [m["c95_bytes"] for m in members if m["c95_bytes"] != ""]
        reach = f"{len(hits)}/{len(members)}"
        if hits:
            reach += f", {statistics.median(hits) / 1e6:.1f} MB"
        return f"{statistics.fmean(finals):.3f} ± {sd:.3f} ({reach})"

    lines = [
        "# Confirmation stage summary",
        "",
        f"Official validation, seeds {seeds_sorted}, phase final.  Cells: mean ± SD of the final score "
        "(seeds reaching C95, median C95 bytes including the one-time basis setup).",
        "",
    ]
    for task in TASKS:
        lines += [f"## {task.upper()}", "", "| Method | dense | 5% | 10% | 20% |", "|---|---|---|---|---|"]
        for method in DENSE:
            lines.append(f"| {method} | {cell(task, method, None)} | | | |")
        for method in (*SPARSE, "eva_a_ef"):
            lines.append(f"| {method} | | " + " | ".join(cell(task, method, w) for w in WIRES) + " |")
        lines.append("")

    lines += [
        "## Paired comparisons (same task, seed and budget)",
        "",
        f"Win / tie / loss with |Δ| ≤ {TIE} as a tie; p = exact two-sided sign test over non-tied pairs.",
        "",
        "| Left − right | Task | n | mean Δ | W / T / L | sign-test p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for left, right, wires in PAIRS:
        overall = []
        for task in (*TASKS, "all"):
            deltas = []
            for t in (TASKS if task == "all" else (task,)):
                for wire in wires:
                    for seed in seeds_sorted:
                        a, b = by_key.get((t, left, seed, wire)), by_key.get((t, right, seed, wire))
                        if a and b:
                            deltas.append(a["final"] - b["final"])
            if not deltas:
                continue
            wins = sum(d > TIE for d in deltas)
            losses = sum(d < -TIE for d in deltas)
            ties = len(deltas) - wins - losses
            lines.append(
                f"| {left} − {right} | {task} | {len(deltas)} | {statistics.fmean(deltas):+.4f} | "
                f"{wins} / {ties} / {losses} | {sign_test(wins, losses):.2g} |"
            )
    (out_dir / "CONFIRM_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
