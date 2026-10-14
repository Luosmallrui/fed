"""Compare gev0 with published federated LoRA baselines on the confirmation protocol.

Pools run directories from several result roots (the confirmation runs of gev0 / random /
FedAvg-LoRA and the baseline runs), then writes BASELINES_SUMMARY.md and BASELINES_RUNS.csv:
per task the mean +- SD of the final official-validation score, seeds reaching C95 (95% of the
same seed's dense FedAvg-LoRA final score; bytes include any one-time setup) with the median
C95 bytes, the mean delivered bytes per round, and paired seed-level comparisons of gev0
against every baseline (|delta| <= 0.003 is a tie; exact two-sided sign test).

Usage (server):
  python scripts/summarize_baselines.py --out <dir> [--override <run_root>] <run_root> [<run_root> ...]

--override loads one more root whose runs replace same-key runs from the others (used for the
gev0 rerun at its own swept learning rates); every replaced run is listed in the summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

TASKS = ("mnli", "qnli", "sst2")
WIRES = (0.05, 0.10, 0.20)
TIE = 0.003
NAMES = {
    "fedavg_lora": "FedIT (FedAvg-LoRA)",
    "random_a_dense": "FFA-LoRA (orthonormal A)",
    "flora_dense": "FLoRA",
    "fedex_dense": "FedEx-LoRA",
    "rolora_dense": "RoLoRA",
    "fedsvd_dense": "FedSVD",
    "gev0_a_dense": "gev0 (ours)",
    "random_a_ef": "FFA-LoRA + Top-k/EF",
    "full_lora_ef": "FedIT + Top-k/EF",
    "rolora_ef": "RoLoRA + Top-k/EF",
    "fedsvd_ef": "FedSVD + Top-k/EF",
    "fedsb": "Fed-SB (rank matched to budget)",
    "gev0_a_ef": "gev0 + Top-k/EF (ours)",
}
DENSE = ("fedavg_lora", "random_a_dense", "flora_dense", "fedex_dense", "rolora_dense", "fedsvd_dense", "gev0_a_dense")
SPARSE = ("random_a_ef", "full_lora_ef", "rolora_ef", "fedsvd_ef", "fedsb", "gev0_a_ef")
PAIRS = [("gev0_a_dense", m, (None,)) for m in DENSE if m != "gev0_a_dense"] + [
    ("gev0_a_ef", m, WIRES) for m in SPARSE if m != "gev0_a_ef"
]


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--override")
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    runs: dict[tuple[str, str, int, float | None], dict] = {}
    replaced: list[tuple[str, str, int, float | None]] = []
    sources = [(root, False) for root in args.roots] + ([(args.override, True)] if args.override else [])
    for root, overriding in sources:
        for path in sorted(Path(root).glob("*/summary.json")):
            summary = json.loads(path.read_text())
            if summary.get("status") != "COMPLETE":
                continue
            spec = summary["run"]
            wire = spec.get("wire_ratio")
            key = (spec["task"], spec["method"], int(spec["seed"]), None if wire is None else round(float(wire), 2))
            if key in runs:
                if not overriding:
                    raise SystemExit(f"duplicate run {key} in {path}")
                replaced.append(key)
            runs[key] = summary
    seeds = sorted({key[2] for key in runs})

    rows = []
    for (task, method, seed, wire), summary in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3] or 0)):
        dense = runs.get((task, "fedavg_lora", seed, None))
        target = 0.95 * dense["final"]["primary_metric"] if dense else math.nan
        hit = next((r for r in summary["rounds"] if r.get("primary_metric") is not None and r["primary_metric"] >= target), None)
        evaluated = [r["primary_metric"] for r in summary["rounds"] if r.get("primary_metric") is not None]
        total = summary["final"]["wire"]["total_delivered_bytes"]
        setup = summary["final"]["wire"].get("setup_bytes", 0)
        rows.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "wire": "dense" if wire is None else wire,
                "final": summary["final"]["primary_metric"],
                "last3": statistics.fmean(evaluated[-3:]),
                "c95_round": hit["round"] if hit else "",
                "c95_bytes": hit["wire"]["total_delivered_bytes"] if hit else "",
                "setup_bytes": setup,
                "bytes_per_round": (total - setup) / len(summary["rounds"]),
                "fedsb_rank": (summary.get("baseline") or {}).get("rank", ""),
            }
        )
    by_key = {(r["task"], r["method"], r["seed"], None if r["wire"] == "dense" else r["wire"]): r for r in rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "BASELINES_RUNS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def members(task: str, method: str, wire: float | None) -> list[dict]:
        return [by_key[(task, method, s, wire)] for s in seeds if (task, method, s, wire) in by_key]

    def cell(task: str, method: str, wire: float | None) -> str:
        group = members(task, method, wire)
        if not group:
            return ""
        finals = [m["final"] for m in group]
        sd = statistics.stdev(finals) if len(finals) > 1 else 0.0
        hits = [m["c95_bytes"] for m in group if m["c95_bytes"] != ""]
        reach = f"{len(hits)}/{len(group)}"
        if hits:
            reach += f", {statistics.median(hits) / 1e6:.1f} MB"
        return f"{statistics.fmean(finals):.3f} ± {sd:.3f} ({reach})"

    lines = [
        "# gev0 vs published federated LoRA baselines",
        "",
        f"Official validation, seeds {seeds}.  Cells: mean ± SD of the final score (seeds reaching C95, "
        "median C95 bytes including one-time setup).  C95 = 95% of the same seed's dense FedIT final score.",
        "",
    ]
    if replaced:
        lines += [
            f"Runs replaced from `{args.override}` ({len(replaced)}): "
            + ", ".join(sorted({f"{t}/{m}" for t, m, _, _ in replaced})),
            "",
        ]
    for task in TASKS:
        lines += [f"## {task.upper()}", "", "### Uncompressed", "", "| Method | final (C95) | MB / round |", "|---|---|---:|"]
        for method in DENSE:
            group = members(task, method, None)
            if group:
                per_round = statistics.fmean(m["bytes_per_round"] for m in group) / 1e6
                lines.append(f"| {NAMES[method]} | {cell(task, method, None)} | {per_round:.2f} |")
        lines += ["", "### Communication budget (fraction of a dense full-LoRA message)", "", "| Method | 5% | 10% | 20% |", "|---|---|---|---|"]
        for method in SPARSE:
            if any(members(task, method, w) for w in WIRES):
                lines.append(f"| {NAMES[method]} | " + " | ".join(cell(task, method, w) for w in WIRES) + " |")
        ranks = sorted({(m["wire"], m["fedsb_rank"]) for w in WIRES for m in members(task, "fedsb", w)})
        if ranks:
            lines.append("")
            lines.append("Fed-SB rank per budget: " + ", ".join(f"{int(w * 100)}% → r={r}" for w, r in ranks))
        lines.append("")

    lines += [
        "## Paired comparisons (same task, seed and budget)",
        "",
        f"Win / tie / loss with |Δ| ≤ {TIE} as a tie; p = exact two-sided sign test over non-tied pairs.",
        "",
        "| gev0 − baseline | Task | n | mean Δ final | mean Δ last-3 | W / T / L | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for left, right, wires in PAIRS:
        for task in (*TASKS, "all"):
            deltas, deltas3 = [], []
            for t in (TASKS if task == "all" else (task,)):
                for wire in wires:
                    for seed in seeds:
                        a, b = by_key.get((t, left, seed, wire)), by_key.get((t, right, seed, wire))
                        if a and b:
                            deltas.append(a["final"] - b["final"])
                            deltas3.append(a["last3"] - b["last3"])
            if not deltas:
                continue
            wins = sum(d > TIE for d in deltas)
            losses = sum(d < -TIE for d in deltas)
            lines.append(
                f"| {NAMES[left]} − {NAMES[right]} | {task} | {len(deltas)} | {statistics.fmean(deltas):+.4f} | "
                f"{statistics.fmean(deltas3):+.4f} | {wins} / {len(deltas) - wins - losses} / {losses} | "
                f"{sign_test(wins, losses):.2g} |"
            )
    (out / "BASELINES_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
