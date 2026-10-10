"""Summarise the consensus-measurement runs (configs/roberta_consensus_*.yaml).

For every run, averages the per-round consensus diagnostics over the early
(first 20%), middle and late (last 20%) thirds of training and prints them next
to the final primary score, so the decision rules in
docs/proposals/RESEARCH_DIRECTIONS.md section 4 can be read off directly:
  * eta_wire << eta_local and eta varies with alpha  -> heterogeneity-driven;
  * eta insensitive to alpha but sensitive to batch -> minibatch-noise-driven;
  * eta high everywhere                              -> little cancellation.

Usage:
  python3 scripts/summarize_consensus.py docs/results/exploration/consensus_*
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

KEYS = ("eta_local", "eta_corrected", "eta_wire", "wire_recovers_local_consensus", "support_pairwise_jaccard_mean")


def phase_means(rounds: list[dict], key: str) -> tuple[float, float, float]:
    values = [r["diagnostics"]["consensus"].get(key) for r in rounds if "consensus" in r.get("diagnostics", {})]
    values = [float(v) for v in values if v is not None]
    if not values:
        return (float("nan"),) * 3
    n = len(values)
    cut = max(1, n // 5)
    early, late = values[:cut], values[-cut:]
    middle = values[cut:-cut] or values
    return statistics.fmean(early), statistics.fmean(middle), statistics.fmean(late)


def main() -> None:
    rows = []
    for directory in sys.argv[1:]:
        for path in sorted(Path(directory).glob("*/summary.json")):
            summary = json.loads(path.read_text())
            rounds = summary["rounds"]
            if not rounds or "consensus" not in rounds[0].get("diagnostics", {}):
                continue
            row = {
                "dir": Path(directory).name.replace("consensus_", ""),
                "task": summary["run"]["task"],
                "method": summary["run"]["method"],
                "final": summary["final"]["primary_metric"],
                "n_eff": rounds[0]["diagnostics"]["consensus"]["n_eff"],
                "batch": summary["training"]["batch_size"],
            }
            for key in KEYS:
                row[key] = phase_means(rounds, key)
            rows.append(row)
    rows.sort(key=lambda r: (r["task"], r["method"], r["dir"]))
    print("eta = ||weighted mean||^2 / weighted mean of ||v||^2 ; 1 = full agreement, 1/N_eff = orthogonal")
    print("values are early / middle / late averages over rounds\n")
    header = f"{'task':5s} {'method':13s} {'setting':10s} {'N_eff':>5s} {'final':>6s} | " + " | ".join(f"{k:>28s}" for k in KEYS)
    print(header)
    for r in rows:
        cells = []
        for key in KEYS:
            e, m, l = r[key]
            cells.append(f"{e:8.3f} {m:8.3f} {l:8.3f}  ")
        print(f"{r['task']:5s} {r['method']:13s} {r['dir']:10s} {r['n_eff']:5.2f} {r['final']:6.3f} | " + " | ".join(f"{c:>28s}" for c in cells))


if __name__ == "__main__":
    main()
