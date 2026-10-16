"""Summary of the Web-service tasks (configs/roberta_web_main.yaml).

Per task and budget: mean +- SD of the final primary metric (Banking77: accuracy; moderation: mean of the
two platforms' macro-F1), seeds reaching C95 (95% of the same seed's final uncompressed FedIT score) and
the median C95 bytes; then paired comparisons of FoCuS (gev0_a_ef) against every other compressed method
(same task, seed and budget; |delta| <= 0.003 is a tie; exact two-sided sign test).

Usage (server):  python scripts/summarize_web.py --out <dir> <run_root>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

TASKS = ("banking77", "moderation")
WIRES = (0.05, 0.10)
TIE = 0.003
OURS = "gev0_a_ef"
NAMES = {
    "random_a_ef": "FFA-LoRA",
    "cab0_a_ef": "Grad-energy",
    "full_lora_ef": "FedIT",
    "rolora_ef": "RoLoRA",
    "fedsvd_ef": "FedSVD",
    "fedsb": "Fed-SB",
    "gev0_a_ef": "FoCuS",
}


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("root")
    args = parser.parse_args()
    runs = {}
    for path in sorted(Path(args.root).glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("status") != "COMPLETE":
            continue
        spec = summary["run"]
        wire = spec.get("wire_ratio")
        runs[(spec["task"], spec["method"], int(spec["seed"]), None if wire is None else round(float(wire), 2))] = summary
    seeds = sorted({key[2] for key in runs})

    def final(key):
        return runs[key]["final"]["primary_metric"]

    def c95(key):
        dense = runs.get((key[0], "fedavg_lora", key[2], None))
        if dense is None:
            return None
        target = 0.95 * dense["final"]["primary_metric"]
        for record in runs[key]["rounds"]:
            if record.get("primary_metric") is not None and record["primary_metric"] >= target:
                return int(record["wire"]["total_delivered_bytes"])
        return None

    table = {}
    lines = ["# Web-service tasks", "", f"Seeds {seeds}; final score x100, mean ± SD (seeds reaching C95; median C95 MB).", ""]
    for task in TASKS:
        dense = [final((task, "fedavg_lora", s, None)) for s in seeds if (task, "fedavg_lora", s, None) in runs]
        lines += [f"## {task}", "", f"Uncompressed FedIT: {100 * statistics.fmean(dense):.1f}" if dense else "", "",
                  "| Method | " + " | ".join(f"{int(w * 100)}%" for w in WIRES) + " |", "|---|" + "---|" * len(WIRES)]
        for method, name in NAMES.items():
            cells = []
            for wire in WIRES:
                keys = [(task, method, s, wire) for s in seeds if (task, method, s, wire) in runs]
                if not keys:
                    cells.append("")
                    continue
                values = [100 * final(k) for k in keys]
                reached = [c95(k) for k in keys]
                hits = [b for b in reached if b is not None]
                cell = {"mean": statistics.fmean(values), "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                        "seeds": len(values), "c95_hits": len(hits),
                        "c95_median_mb": statistics.median(hits) / 1e6 if hits else None}
                table.setdefault(task, {}).setdefault(method, {})[str(wire)] = cell
                mb = "never" if cell["c95_median_mb"] is None else f"{cell['c95_median_mb']:.1f} MB"
                cells.append(f"{cell['mean']:.1f} ± {cell['sd']:.1f} ({len(hits)}/{len(values)}; {mb})")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## FoCuS − method (same task, seed and budget)", "",
              "| Method | Task | n | mean Δ (points) | W / T / L | p |", "|---|---|---:|---:|---:|---:|"]
    pairs = {}
    for method, name in NAMES.items():
        if method == OURS:
            continue
        for task in (*TASKS, "all"):
            deltas = []
            for t in (TASKS if task == "all" else (task,)):
                for wire in WIRES:
                    for s in seeds:
                        a, b = (t, OURS, s, wire), (t, method, s, wire)
                        if a in runs and b in runs:
                            deltas.append(final(a) - final(b))
            if not deltas:
                continue
            wins = sum(d > TIE for d in deltas)
            losses = sum(d < -TIE for d in deltas)
            p = sign_test(wins, losses)
            pairs[f"{method}/{task}"] = {"n": len(deltas), "mean": 100 * statistics.fmean(deltas), "wins": wins,
                                         "ties": len(deltas) - wins - losses, "losses": losses, "p": p}
            lines.append(f"| {name} | {task} | {len(deltas)} | {100 * statistics.fmean(deltas):+.1f} | "
                         f"{wins} / {len(deltas) - wins - losses} / {losses} | {p:.2g} |")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "WEB_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "web_summary.json").write_text(json.dumps({"table": table, "pairs": pairs}, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
