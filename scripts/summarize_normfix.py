"""Does curvature normalization fix other data-driven frozen bases?

Pools the confirmation runs (random / gradient-energy / normalized gradient-energy / ours from confirm_main,
Fed-SB from baselines_main) with the normfix runs (EVA, EVA-normalized, Fed-SB-normalized) and writes
NORMFIX_SUMMARY.md: per task and budget the mean +- SD of the final official-validation score with seeds
reaching C95, and paired comparisons (|delta| <= 0.003 is a tie; exact two-sided sign test).

Usage (server):
  python scripts/summarize_normfix.py --out <dir> [--override <run_root>] <run_root> [<run_root> ...]
  --override roots replace runs with the same (task, method, seed, budget), e.g. Fed-SB-normalized
  rerun at its own swept rate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

TASKS = ("mnli", "qnli", "sst2")
WIRES = (0.05, 0.10, 0.20)
TIE = 0.003
NAMES = {
    "random_a_ef": "Random (FFA-LoRA)",
    "cab0_a_ef": "Grad-energy",
    "cab0wh_a_ef": "Grad-energy + normalization",
    "eva_a_ef": "EVA",
    "evawh_a_ef": "EVA + normalization",
    "fedsb": "Fed-SB",
    "fedsb_norm": "Fed-SB + normalization",
    "gev0_a_ef": "Ours",
}
PAIRS = [
    ("cab0wh_a_ef", "cab0_a_ef"),
    ("evawh_a_ef", "eva_a_ef"),
    ("fedsb_norm", "fedsb"),
    ("cab0wh_a_ef", "random_a_ef"),
    ("evawh_a_ef", "random_a_ef"),
    ("fedsb_norm", "random_a_ef"),
    ("gev0_a_ef", "evawh_a_ef"),
    ("gev0_a_ef", "fedsb_norm"),
]


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("roots", nargs="+")
    args = parser.parse_args()
    runs: dict[tuple[str, str, int, float | None], dict] = {}
    replaced = 0
    for root in (*args.roots, *args.override):
        overriding = root in args.override
        for path in sorted(Path(root).glob("*/summary.json")):
            summary = json.loads(path.read_text())
            spec = summary["run"]
            if summary.get("status") != "COMPLETE" or spec["method"] not in NAMES | {"fedavg_lora": ""}:
                continue
            wire = spec.get("wire_ratio")
            key = (spec["task"], spec["method"], int(spec["seed"]), None if wire is None else round(float(wire), 2))
            if key in runs and not overriding:
                raise SystemExit(f"duplicate run {key} in {path}")
            replaced += key in runs
            runs[key] = summary
    seeds = sorted({key[2] for key in runs})

    def final(key):
        return runs[key]["final"]["primary_metric"]

    def reaches(key):
        task, _, seed, _ = key
        dense = runs.get((task, "fedavg_lora", seed, None))
        if dense is None:
            return None
        target = 0.95 * dense["final"]["primary_metric"]
        return any(r.get("primary_metric") is not None and r["primary_metric"] >= target for r in runs[key]["rounds"])

    lines = ["# Curvature normalization as a fix for data-driven frozen bases", "",
             f"Official validation, seeds {seeds}; bidirectional Top-k + EF.  Cells: mean ± SD of the final score "
             "(seeds reaching C95).", ""]
    if args.override:
        lines += [f"Runs from {', '.join(args.override)} replace {replaced} runs of the other roots.", ""]
    for task in TASKS:
        lines += [f"## {task.upper()}", "", "| Basis | 5% | 10% | 20% |", "|---|---|---|---|"]
        for method, name in NAMES.items():
            cells = []
            for wire in WIRES:
                keys = [(task, method, s, wire) for s in seeds if (task, method, s, wire) in runs]
                if not keys:
                    cells.append("")
                    continue
                values = [final(k) for k in keys]
                sd = statistics.stdev(values) if len(values) > 1 else 0.0
                hit = sum(bool(reaches(k)) for k in keys)
                cells.append(f"{statistics.fmean(values):.3f} ± {sd:.3f} ({hit}/{len(keys)})")
            if any(cells):
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Paired comparisons (same task, seed and budget)", "",
              "| Left − right | Task | n | mean Δ | W / T / L | p |", "|---|---|---:|---:|---:|---:|"]
    for left, right in PAIRS:
        for task in (*TASKS, "all"):
            deltas = []
            for t in (TASKS if task == "all" else (task,)):
                for wire in WIRES:
                    for seed in seeds:
                        a, b = (t, left, seed, wire), (t, right, seed, wire)
                        if a in runs and b in runs:
                            deltas.append(final(a) - final(b))
            if not deltas:
                continue
            wins = sum(d > TIE for d in deltas)
            losses = sum(d < -TIE for d in deltas)
            lines.append(f"| {NAMES[left]} − {NAMES[right]} | {task} | {len(deltas)} | {statistics.fmean(deltas):+.4f} | "
                         f"{wins} / {len(deltas) - wins - losses} / {losses} | {sign_test(wins, losses):.2g} |")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "NORMFIX_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
